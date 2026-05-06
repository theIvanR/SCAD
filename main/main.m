clear; clc;

%% config
infile = 'distorted_audio.mp3';
outfile = 'recovered_clean.wav';

cfg = struct();
    cfg.Lh = 192; % Length of estimated impulse response (FIR model order), Higher = more expressive system model, but risk overfitting/noise fitting
    
    cfg.n_iter = 8;% Number of alternating optimization iterations (x ↔ h updates), Controls convergence quality vs compute time
    
    cfg.lambda_h = 1e-4; % Ridge regularization on h estimation (stabilizes X'X inversion), Prevents unstable / spiky impulse responses
    
    cfg.smooth_mu = 5e-3;% Smoothness penalty on h (second-derivative regularization), Suppresses ringing and enforces physically plausible decay
    
    cfg.noise_floor = 1e-8;% Minimum noise estimate for Wiener filter stability, Prevents division-by-near-zero amplification
    
    cfg.maxInverseGain = 15;% Hard cap on frequency-domain inverse gain, Prevents extreme boosts at spectral nulls
    
    cfg.seg_seconds_list = [0.5 1.0 2.0];% Multi-resolution window sizes for segment selection, Captures both transient and steady-state structure
    
    cfg.K = 30; % Number of selected training segments (best windows)
    
    cfg.hop_fraction = 0.5;% Overlap between candidate windows (hop = L * hop_fraction), Controls density of scanning
    
    cfg.pool_mult = 4;% Pre-selection multiplier (top K * pool_mult candidates kept), Improves robustness before final filtering
    
    cfg.n_outer = 2;% Outer refinement passes (re-select segments after first model), Improves self-consistency of training data
    
    cfg.min_rms = 1e-10;% Silence rejection threshold (RMS floor), Removes invalid / silent windows from scoring

%% load
[y, Fs] = audioread(infile);
if size(y,2) > 1
    y = mean(y,2);
end
y = single(y(:));

fprintf('Loaded %.2f sec\n', length(y)/Fs);

%% iterative smart refinement
y_ref = y;
model = [];
for outer = 1:cfg.n_outer
    fprintf('\n=== Outer pass %d/%d ===\n', outer, cfg.n_outer);

    fprintf('Selecting best segments...\n');
    [segments, meta] = select_best_segments_smart(y_ref, Fs, cfg);
    fprintf('Selected %d segments.\n', numel(segments));

    fprintf('Training backend...\n');
    [model, diag] = smart_inverse_backend_multi(segments, meta, Fs, cfg);

    fprintf('Learned Lh=%d, NSR=%.3e | inliers=%d/%d\n', ...
        length(model.h), model.nsr, diag.nInliers, diag.nSeg);

    if outer < cfg.n_outer
        fprintf('Applying intermediate model for re-selection...\n');
        y_ref = model.apply(y);
    end
end

%% final apply
fprintf('\nApplying final model...\n');
x = model.apply(y);

audiowrite(outfile, single(x), Fs);
fprintf('Saved: %s\n', outfile);

%% ------------------------------------------------------------------------
%% Smart segment selector
function [segments, meta] = select_best_segments_smart(y, Fs, cfg)
% Multi-resolution scan + cheap score + conditioning-aware rerank.

N = length(y);
Lh = cfg.Lh;
seg_list = cfg.seg_seconds_list(:).';
hop_fraction = cfg.hop_fraction;
K = cfg.K;
pool_mult = cfg.pool_mult;
min_rms = cfg.min_rms;

cand = struct('start', {}, 'L', {}, 'score', {}, 'condProxy', {}, ...
              'energy', {}, 'entropy', {}, 'flatness', {}, 'cheapScore', {});

for s = 1:numel(seg_list)
    L = round(seg_list(s) * Fs);
    if L < max(64, Lh)
        continue;
    end
    hop = max(1, round(L * hop_fraction));

    for i = 1:hop:(N - L + 1)
        seg = y(i:i+L-1);

        % basic silence rejection
        pwr = mean(seg.^2);
        if pwr < min_rms
            continue;
        end

        w = hann(L, 'periodic');
        seg0 = seg - mean(seg);
        seg1 = seg0 .* w;

        % spectral richness
        S = abs(fft(seg1));
        S = S(1:floor(end/2));
        S = S + 1e-12;
        P = S / sum(S);
        entropy = -sum(P .* log(P));

        flatness = exp(mean(log(S))) / mean(S);

        % cheap first-pass score
        cheapScore = (pwr + eps) * (entropy + 1e-12) * (0.5 + 0.5 * flatness);

        cand(end+1) = struct( ...
            'start', i, ...
            'L', L, ...
            'score', 0, ...
            'condProxy', 0, ...
            'energy', pwr, ...
            'entropy', entropy, ...
            'flatness', flatness, ...
            'cheapScore', cheapScore); %#ok<AGROW>
    end
end

if isempty(cand)
    L = min(round(seg_list(1) * Fs), N);
    if L < 1
        error('Signal too short.');
    end
    segments = {y(1:L)};
    meta = struct('start', 1, 'L', L, 'score', 1, 'condProxy', 1, ...
                  'energy', mean(y(1:L).^2), 'entropy', 0, 'flatness', 1, ...
                  'cheapScore', 1);
    return;
end

% keep a pool of top cheap-scoring candidates
[~, order] = sort([cand.cheapScore], 'descend');
poolN = min(numel(cand), max(K * pool_mult, K + 10));
pool = cand(order(1:poolN));

% conditioning-aware rerank
for j = 1:numel(pool)
    seg = y(pool(j).start : pool(j).start + pool(j).L - 1);
    seg = seg - mean(seg);
    seg = seg / (rms(seg) + eps);

    % proxy conditioning from the segment autocorrelation structure
    lagMax = min(Lh - 1, length(seg) - 1);
    if lagMax < 2
        condProxy = 1e12;
    else
        r = xcorr(seg, lagMax, 'biased');
        rpos = r(lagMax+1:end);
        if numel(rpos) < Lh
            rpos(end+1:Lh) = 0;
        end
        Aproxy = toeplitz(rpos(1:Lh));
        condProxy = cond(Aproxy + 1e-10 * eye(Lh));
    end

    condProxy = min(condProxy, 1e12);
    score = pool(j).cheapScore / log1p(condProxy);

    pool(j).condProxy = condProxy;
    pool(j).score = score;
end

% final selection
[~, order2] = sort([pool.score], 'descend');
take = order2(1:min(K, numel(order2)));
sel = pool(take);

segments = cell(numel(sel), 1);
meta = sel;

for k = 1:numel(sel)
    i = sel(k).start;
    L = sel(k).L;
    segments{k} = y(i:i+L-1);
end
end

%% ------------------------------------------------------------------------
%% Multi-segment backend with robust consensus
function [model, diag] = smart_inverse_backend_multi(segments, meta, Fs, cfg)
nSeg = numel(segments);
if nSeg == 0
    error('No training segments provided.');
end

h_all = cell(nSeg,1);
nsr_all = zeros(nSeg,1);

for s = 1:nSeg
    [h_s, nsr_s] = smart_inverse_backend_single(segments{s}, Fs, cfg);
    h_all{s} = h_s;
    nsr_all(s) = nsr_s;
end

Lh = cfg.Lh;
Hstack = zeros(Lh, nSeg);
for s = 1:nSeg
    hs = h_all{s}(:);
    L = min(Lh, numel(hs));
    Hstack(1:L, s) = hs(1:L);
end

% first robust center
h0 = median(Hstack, 2);

% outlier rejection by distance to median kernel
d = vecnorm(Hstack - h0, 2, 1);
md = median(d);
madD = median(abs(d - md)) + eps;
inliers = d <= (md + 2.5 * madD);

if ~any(inliers)
    inliers = true(size(inliers));
end

% weighted consensus: high-score, low-condition windows matter more
scores = [meta.score].';
conds = [meta.condProxy].';
weights = scores ./ log1p(conds + 1);
weights(~inliers) = 0;

if sum(weights) <= 0
    weights = double(inliers(:));
end

weights = weights(:) / (sum(weights) + eps);

h = Hstack * weights;
h = h / (sum(h) + eps);

nsr = sum(nsr_all(:) .* weights);

model.h = h;
model.nsr = nsr;
model.apply = @(yin) apply_model(yin, h, nsr, cfg.maxInverseGain);

diag.nSeg = nSeg;
diag.nInliers = nnz(inliers);
diag.weights = weights;
diag.distances = d;
diag.meta = meta;
end

%% ------------------------------------------------------------------------
%% Single-segment alternating solver
function [h, nsr] = smart_inverse_backend_single(y, Fs, cfg) %#ok<INUSD>
y = double(y(:));
M = numel(y);

Lh = cfg.Lh;
n_iter = cfg.n_iter;

lambda_h = cfg.lambda_h;
smooth_mu = cfg.smooth_mu;
noise_floor = cfg.noise_floor;
maxGain = cfg.maxInverseGain;

% init h
t = (0:Lh-1).';
tau = max(4, round(Lh/12));
h = exp(-t/tau);
h = h / (sum(h) + eps);

x = y;
R = secondDiffPenalty(Lh);

for iter = 1:n_iter
    % x update (Wiener)
    Nfft = 2^nextpow2(M + Lh - 1);
    H = fft([h; zeros(Nfft-Lh,1)], Nfft);

    yhat = conv(x, h);
    yhat = yhat(1:M);
    resid = y - yhat;

    sig2 = max(noise_floor, var(resid));
    nsr_iter = sig2 / max(var(y), eps);

    W = conj(H) ./ (abs(H).^2 + nsr_iter);
    W = capSpectrum(W, maxGain);

    x = real(ifft(W .* fft([y; zeros(Nfft-M,1)], Nfft)));
    x = x(1:M);

    % h update (regularized LS)
    Xmat = zeros(M, Lh);
    for k = 1:Lh
        nrows = M - (k - 1);
        if nrows > 0
            Xmat(k-1 + (1:nrows), k) = x(1:nrows);
        end
    end

    A = Xmat' * Xmat + lambda_h * eye(Lh) + smooth_mu * R;
    b = Xmat' * y;

    h_new = A \ b;
    h_new = h_new / (sum(h_new) + eps);

    if norm(h_new - h) / (norm(h) + eps) < 1e-4
        h = h_new;
        break;
    end
    h = h_new;
end

% final NSR estimate
yhat = conv(x, h);
yhat = yhat(1:M);
sig2 = max(noise_floor, var(y - yhat));
nsr = sig2 / max(var(y), eps);
end

%% ------------------------------------------------------------------------
%% Apply model
function x = apply_model(y, h, nsr, maxGain)
y = double(y(:));
N = numel(y);
Lh = numel(h);

Nfft = 2^nextpow2(N + Lh - 1);
H = fft([h; zeros(Nfft-Lh,1)], Nfft);

W = conj(H) ./ (abs(H).^2 + nsr);
W = capSpectrum(W, maxGain);

x = real(ifft(W .* fft([y; zeros(Nfft-N,1)], Nfft)));
x = x(1:N);

% keep your current behavior
x = x / (max(abs(x)) + eps);
end

%% ------------------------------------------------------------------------
%% Helpers
function R = secondDiffPenalty(L)
if L < 3
    R = zeros(L);
    return;
end
e = ones(L,1);
D = spdiags([e -2*e e], 0:2, L-2, L);
R = D' * D;
end

function W = capSpectrum(W, maxGain)
mag = abs(W);
idx = mag > maxGain;
if any(idx)
    W(idx) = W(idx) .* (maxGain ./ mag(idx));
end
end