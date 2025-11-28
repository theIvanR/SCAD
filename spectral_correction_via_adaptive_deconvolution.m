%% Global W + Classic Overlap-Add (no windowing) + Verbose debug
% Replaces STFT-windowed OLA with true overlap-add FFT convolution using same W(f) per block.
% This removes block-period squelch while keeping the smart Lh/NSR/h pipeline.
clear; close all; clc;

%% ---------------- USER KNOBS ----------------
infile = 'distorted_audio.wav';
outprefix = 'recovered_ola_global';

seg_seconds = 1;            % seconds for estimation
deconv_num_iter = 50;
Lh_coarse = [64 128 256 512];
do_refine = true;
refine_frac = 0.5;
refine_steps = 7;

lambda_rel = 5e-6;
smoothness_mu = 5e-2;
NSR_mult_initial = 0.5;
nsr_grid = [0.05 0.1 0.2 0.5 1.0];
nsr_obj_beta = 0.6;
invGain_penalty = 0.6;
maxInverseGain = 12;       % cap on |W(f)|
NSR_floor = 1e-3;
max_iter = 3;

% OLA params (classic, zero-padded blocks)
block_len = 2^16;          % default 65536 (≈1.49 s). Make smaller for faster but more blocks.
% must pick block_len so Lfft = nextpow2(block_len + Lh -1) fits memory
apply_global_W = true;     % always true here; we use same W for each block

% Debug & output
debug_dir = 'debug_ola_global';
if ~exist(debug_dir,'dir'), mkdir(debug_dir); end
verbose = true;
save_debug_csv = true;
plot_results = true;
play_preview = true;
% ------------------------------------------------

%% 1) load audio
if verbose, fprintf('Loading: %s\n', infile); end
[y_full, Fs] = audioread(infile);
if size(y_full,2) > 1, y_full = y_full(:,1); end
y_full = single(y_full);
N_full = length(y_full);
if verbose, fprintf('Loaded %d samples (%.2f s) at %d Hz\n', N_full, N_full/Fs, Fs); end

%% 1b) short segment for estimation
Lseg = min(N_full, round(seg_seconds * Fs));
y_seg = y_full(1:Lseg);
if verbose, fprintf('Using first %d samples (%.3f s) for estimation.\n', Lseg, Lseg/Fs); end

%% 2) initial deconvblind seed
Lh_init = min(512, max(32, round(Lseg/8)));
t = (0:Lh_init-1)'; tau = max(4, Lh_init/60);
h_init = exp(-t/tau) .* (1 + 0.05*sin(2*pi*8000.*t./Fs).*exp(-t/(tau*2)));
h_init = h_init / sum(h_init);
try
    [x_est_seg, ~] = deconvblind(double(y_seg), double(h_init), deconv_num_iter);
catch
    warning('deconvblind failed; using y_seg as x seed.');
    x_est_seg = double(y_seg);
end
x = double(x_est_seg(:));
y = double(y_seg(:));
M = length(y);

%% 3) precompute autocorr for LS
Lh_candidates = unique(Lh_coarse(Lh_coarse <= Lseg));
if isempty(Lh_candidates), Lh_candidates = min(256, Lseg); end
maxLh_try = max(Lh_candidates);
maxlag = maxLh_try - 1;
rx_full = xcorr(x, x, maxlag);
rxy_full = xcorr(x, y, maxlag);
sigpow = var(y) + eps;

%% 4) coarse sweep + smoothness penalty
Lh_values = Lh_candidates;
nL = numel(Lh_values);
RSS = zeros(nL,1); NSRvals=zeros(nL,1); BIC=zeros(nL,1);
h_store = cell(nL,1);

for ii = 1:nL
    Lh = Lh_values(ii); ml=Lh-1;
    rxx_vec = rx_full(maxlag+1 : maxlag+1+ml);
    A = toeplitz(rxx_vec);
    rxy_vec = rxy_full(maxlag+1 : maxlag+1+ml);
    b = rxy_vec(:);
    if Lh>=3
        e = ones(Lh,1); D = spdiags([e -2*e e],0:2,Lh-2,Lh);
        smooth_pen = smoothness_mu*(D'*D);
    else
        smooth_pen = zeros(Lh);
    end
    lambda = lambda_rel * sigpow;
    Areg = A + lambda*eye(Lh) + smooth_pen;
    h_ls = Areg \ b;
    h_store{ii} = h_ls(:);
    conv_rec = conv(x, h_ls); conv_trunc = conv_rec(1:M);
    res = y - conv_trunc(:);
    RSS(ii) = sum(res.^2);
    NSRvals(ii) = max(1e-12, var(res)/max(sigpow,eps));
    n=M; k=Lh;
    BIC(ii) = k*log(n) + n*log(RSS(ii)/n + eps);
    if verbose, fprintf('Coarse Lh=%4d RSS=%.4e NSR=%.4e BIC=%.4e RMS(h)=%.4g\n', Lh, RSS(ii), NSRvals(ii), BIC(ii), rms(h_ls)); end
end

% bias toward longer Lh on tie
[minBIC, minIdx] = min(BIC);
rel_tol = 0.01;
cands = find( (BIC - minBIC) ./ max(abs(minBIC),1) <= rel_tol );
if numel(cands) > 1
    [~, idxmax] = max(Lh_values(cands));
    chosenIdx_coarse = cands(idxmax);
else
    chosenIdx_coarse = minIdx;
end
bestCoarseLh = Lh_values(chosenIdx_coarse);
if verbose, fprintf('Best coarse Lh initial = %d\n', bestCoarseLh); end

%% 5) refine around best
if do_refine
    low = max(8, round(bestCoarseLh*(1-refine_frac)));
    high = min(Lseg, round(bestCoarseLh*(1+refine_frac)));
    refine_grid = unique(round(linspace(low, high, refine_steps)));
    refine_grid = refine_grid(~ismember(refine_grid, Lh_values));
    if ~isempty(refine_grid)
        if verbose, fprintf('Refining Lh in [%d..%d]: %s\n', low, high, mat2str(refine_grid)); end
        for Lh = refine_grid
            if Lh > Lseg, continue; end
            ml=Lh-1;
            rxx_vec = rx_full(maxlag+1 : maxlag+1+ml);
            A = toeplitz(rxx_vec);
            rxy_vec = rxy_full(maxlag+1 : maxlag+1+ml);
            b = rxy_vec(:);
            if Lh>=3
                e = ones(Lh,1); D = spdiags([e -2*e e],0:2,Lh-2,Lh);
                smooth_pen = smoothness_mu*(D'*D);
            else
                smooth_pen = zeros(Lh);
            end
            lambda = lambda_rel * sigpow;
            Areg = A + lambda*eye(Lh) + smooth_pen;
            h_ls = Areg \ b;
            conv_rec = conv(x, h_ls); conv_trunc = conv_rec(1:M);
            res = y - conv_trunc(:);
            RSS_new = sum(res.^2);
            NSR_new = max(1e-12,var(res)/max(sigpow,eps));
            BIC_new = Lh*log(M) + M*log(RSS_new/M + eps);
            Lh_values(end+1)=Lh; RSS(end+1)=RSS_new; NSRvals(end+1)=NSR_new; BIC(end+1)=BIC_new;
            h_store{end+1}=h_ls(:);
            if verbose, fprintf('Refine Lh=%4d RSS=%.4e NSR=%.4e BIC=%.4e\n', Lh, RSS_new, NSR_new, BIC_new); end
        end
    end
end

[~,bestIdx] = min(BIC);
Lh_best = Lh_values(bestIdx);
h_best = h_store{bestIdx}(:);
NSR_base = NSRvals(bestIdx);
if verbose, fprintf('Selected Lh_best=%d base NSR=%.4e\n', Lh_best, NSR_base); end

%% 6) iterative NSR/h refinement (as before)
h_current = h_best;
NSR_current = max(NSR_floor, NSR_base * NSR_mult_initial);
Nfft_seg = 2^nextpow2(M);
sigpow = var(y) + eps;

for iter=1:max_iter
    if verbose, fprintf('\n-- Iter %d: NSR=%.4e --\n', iter, NSR_current); end
    Hseg_base = fft([h_current; zeros(Nfft_seg-length(h_current),1)]);
    nCand = numel(nsr_grid);
    residVar=zeros(nCand,1); specFlat=zeros(nCand,1); invG=zeros(nCand,1);
    for j=1:nCand
        nsr_try = max(NSR_floor, NSR_current * nsr_grid(j));
        Wseg = conj(Hseg_base) ./ (abs(Hseg_base).^2 + nsr_try);
        if max(abs(Wseg)) > maxInverseGain, Wseg = Wseg .* (min(1, maxInverseGain ./ abs(Wseg))); end
        xseg_rec = real(ifft(Wseg .* fft(double(y(1:M)), Nfft_seg))); xseg_rec = xseg_rec(1:M);
        conv_rec = conv(xseg_rec, h_current); conv_trunc = conv_rec(1:M);
        alpha = (conv_trunc' * double(y)) / (conv_trunc' * conv_trunc + eps);
        res = double(y) - alpha * conv_trunc;
        residVar(j) = var(res);
        wlen = min(M, round(0.1*Fs)); if wlen < 256, wlen = min(M,256); end
        segwin = xseg_rec(1:wlen) .* hann(wlen);
        S = abs(fft(segwin, 1024)); S = S(1:length(S)/2);
        specFlat(j) = exp(mean(log(S + 1e-12))) / mean(S + 1e-12);
        invG(j) = max(abs(Wseg));
    end
    rvn = (residVar - min(residVar)) / (max(residVar)-min(residVar)+eps);
    sfn = (specFlat - min(specFlat)) / (max(specFlat)-min(specFlat)+eps);
    ign = (invG - min(invG)) / (max(invG)-min(invG)+eps);
    beta = nsr_obj_beta; gamma = invGain_penalty;
    obj = rvn - beta * sfn + gamma * ign;
    [~,pick] = min(obj);
    chosen_ratio = nsr_grid(pick);
    NSR_current = max(NSR_floor, NSR_current * chosen_ratio);
    if verbose, fprintf('Chosen nsr ratio=%.3g => NSR now %.4e (invG=%.3g)\n', chosen_ratio, NSR_current, invG(pick)); end

    % recompute xseg_rec and re-estimate h via LS
    nsr_final = NSR_current;
    Wseg = conj(Hseg_base) ./ (abs(Hseg_base).^2 + nsr_final);
    if max(abs(Wseg)) > maxInverseGain, Wseg = Wseg .* (min(1, maxInverseGain ./ abs(Wseg))); end
    xseg_rec = real(ifft(Wseg .* fft(double(y(1:M)), Nfft_seg))); xseg_rec = xseg_rec(1:M);

    Lh = Lh_best;
    Xmat = zeros(M, Lh);
    for k=1:Lh
        nrows = M - (k-1);
        if nrows>0, Xmat(k-1 + (1:nrows), k) = xseg_rec(1:nrows); end
    end
    if Lh>=3
        e = ones(Lh,1); D = spdiags([e -2*e e],0:2,Lh-2,Lh);
        smooth_pen = smoothness_mu*(D'*D);
    else
        smooth_pen = zeros(Lh);
    end
    A = Xmat' * Xmat + (lambda_rel*sigpow)*eye(Lh) + smooth_pen;
    b = Xmat' * double(y(1:M));
    h_new = A \ b;
    change = norm(h_new - h_current) / (norm(h_current)+eps);
    h_current = h_new(:);
    if verbose, fprintf('Updated h RMS=%.6g max=%.6g change=%.6g\n', rms(h_current), max(abs(h_current)), change); end
    if change < 1e-4, break; end
end

h_final = h_current; NSR_final = NSR_current;
if verbose, fprintf('Final Lh=%d NSR=%.4e RMS(h)=%.6g\n', length(h_final), NSR_final, rms(h_final)); end

%% 7) Build global Wfreq for chosen Lfft and apply via classic OLA (zero-pad blocks)
Lh_use = length(h_final);
Lfft = 2^nextpow2(block_len + Lh_use - 1);
if verbose, fprintf('Applying classic OLA: block_len=%d Lfft=%d (Lh=%d)\n', block_len, Lfft, Lh_use); end

Hfft_global = fft([h_final; zeros(Lfft - Lh_use, 1)], Lfft);
Wfreq_global = conj(Hfft_global) ./ (abs(Hfft_global).^2 + NSR_final);

% smooth & cap global Wfreq magnitude
smooth_sigma_bins = 8;
halfk = round(3*smooth_sigma_bins);
kgrid = -halfk:halfk;
gk = exp(-0.5*(kgrid/smooth_sigma_bins).^2); gk = gk / sum(gk);
magW = abs(Wfreq_global);
mag_smooth = conv(magW, gk, 'same');
Wfreq_global = mag_smooth .* exp(1i*angle(Wfreq_global));
% cap magnitude
if max(abs(Wfreq_global)) > maxInverseGain
    Wfreq_global = Wfreq_global .* (min(1, maxInverseGain ./ abs(Wfreq_global)));
end

% debug W stats & save figure
faxis = (0:(Lfft/2))' * (Fs / Lfft);
figW = figure('Visible','off'); plot(faxis, 20*log10(abs(Wfreq_global(1:Lfft/2+1))+1e-12));
title('Global Wfreq magnitude (dB)'); xlabel('Hz'); ylabel('dB');
saveas(figW, fullfile(debug_dir,'globalW_dB.png'));

%% 8) Variants (safe/best/aggressive) but using CLASSIC OLA conv with SAME Wfreq per block
variants = {'safe','best','aggressive'};
variant_settings = {
    struct('NSR', max(NSR_floor, NSR_final*4), 'cap', min(8, maxInverseGain), 'sigma_bins', 16), ...
    struct('NSR', max(NSR_floor, NSR_final), 'cap', maxInverseGain, 'sigma_bins', 8), ...
    struct('NSR', max(NSR_floor, NSR_final*0.5), 'cap', min(maxInverseGain*1.5, 40), 'sigma_bins', 6)
};

outfiles = cell(numel(variants),1);
per_block_stats = [];

for v = 1:numel(variants)
    tag = variants{v};
    s = variant_settings{v};
    nsr_use = s.NSR;
    cap_use = s.cap;
    sigma_bins_v = s.sigma_bins;
    fprintf('\nVariant="%s": NSR=%.4e cap=%.2f sigma=%d\n', tag, nsr_use, cap_use, sigma_bins_v);

    % build Wfreq_v from h_final for this Lfft
    Hfft_v = fft([h_final; zeros(Lfft - Lh_use,1)], Lfft);
    Wfreq_v = conj(Hfft_v) ./ (abs(Hfft_v).^2 + nsr_use);
    % smooth & cap
    gk2 = exp(-0.5*(kgrid/sigma_bins_v).^2); gk2 = gk2 / sum(gk2);
    mag = abs(Wfreq_v);
    mag_smooth_v = conv(mag, gk2, 'same');
    Wfreq_v = mag_smooth_v .* exp(1i*angle(Wfreq_v));
    if max(abs(Wfreq_v)) > cap_use
        Wfreq_v = Wfreq_v .* (min(1, cap_use ./ abs(Wfreq_v)));
    end

    % CLASSIC OLA: process non-overlapping blocks of length block_len, zero-pad to Lfft,
    % iFFT result length Lfft, add to output starting at idx
    Nout = N_full + Lfft; % allow tail
    xout = zeros(Nout,1);
    idx = 1;
    block_idx = 0;
    while idx <= N_full
        block_idx = block_idx + 1;
        idx_end = min(N_full, idx + block_len - 1);
        block = double(y_full(idx:idx_end));
        curLen = numel(block);
        % pad to Lfft
        blockPad = [block; zeros(Lfft - curLen, 1)];
        Bfft = fft(blockPad);
        Xb = Wfreq_v .* Bfft;
        xblk = real(ifft(Xb));
        out_start = idx;
        out_end = idx + Lfft - 1;
        xout(out_start:out_end) = xout(out_start:out_end) + xblk;
        % debug stats
        Ein = sum(block.^2);
        % energy of output portion corresponding to input (first curLen)
        Eout = sum( xblk(1:curLen).^2 );
        per_block_stats = [per_block_stats; v, block_idx, Ein, Eout, max(abs(Wfreq_v)), median(abs(Wfreq_v))];
        if mod(block_idx, 50) == 0
            fprintf('Variant=%s block %d: Ein=%.4e Eout=%.4e maxW=%.3g medW=%.3g\n', tag, block_idx, Ein, Eout, max(abs(Wfreq_v)), median(abs(Wfreq_v)));
        end
        idx = idx + block_len;
    end

    % trim to N_full and normalize
    xout = xout(1:N_full);
    xout = xout / (max(abs(xout)) + eps);
    outfile = sprintf('%s_%s_Lh%d_NSR%.4g_ola.wav', outprefix, tag, Lh_use, nsr_use);
    audiowrite(outfile, single(xout), Fs);
    outfiles{v} = outfile;
    fprintf('Wrote: %s\n', outfile);
end

%% 9) Save debug CSV and figs
if save_debug_csv && ~isempty(per_block_stats)
    T = array2table(per_block_stats, 'VariableNames', {'variant','block','Ein','Eout','maxW','medW'});
    csvname = fullfile(debug_dir, sprintf('per_block_stats_Lh%d_NSR%.4g_ola.csv', Lh_use, NSR_final));
    writetable(T, csvname);
    fprintf('Saved per-block CSV: %s (rows=%d)\n', csvname, size(T,1));
end

if plot_results
    % plot Wfreq magnitude
    fig1 = figure('Visible','off'); plot(faxis, 20*log10(abs(Wfreq_v(1:Lfft/2+1))+1e-12));
    title('Variant last Wfreq mag (dB)'); xlabel('Hz'); ylabel('dB'); saveas(fig1, fullfile(debug_dir,'Wvariant_dB.png'));
    % energy scatter
    if ~isempty(per_block_stats)
        fig2 = figure('Visible','off'); scatter(per_block_stats(:,2), per_block_stats(:,3), 8, per_block_stats(:,1),'filled');
        xlabel('block'); ylabel('Ein'); title('Block input energy'); saveas(fig2, fullfile(debug_dir,'block_Ein.png'));
    end
end

%% 10) play previews
disp('Generated files:'); disp(outfiles');
if play_preview
    for v=1:numel(outfiles)
        try
            [xr,fr] = audioread(outfiles{v});
            durp = min(8, floor(length(xr)/fr));
            fprintf('Playing preview: %s (first %d s)\n', outfiles{v}, durp);
            sound(single(xr(1:durp*fr)), fr);
            pause(durp + 0.2);
            clear sound;
        catch
            warning('Unable to play preview (non-interactive).');
        end
    end
end

fprintf('Done. Key knobs to adjust: block_len (smaller -> less latency, more blocks), maxInverseGain (cap), NSR_floor, smoothness_mu.\n');
