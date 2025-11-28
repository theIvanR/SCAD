% batch_spectrum.m
% Stream WAVs from a folder, compute averaged PSD (Welch-style) using single precision
% and block reads, and plot full-file spectra (one-sided PSD in dB/Hz).
%
% Usage:
%   >> batch_spectrum()
% or
%   >> batch_spectrum('C:\path\to\folder', paramsStruct)

batch_spectrum('C:\path\to\folder');

function batch_spectrum(folderPath, params)

if nargin < 2, params = struct(); end

% ---------- Parameters (tweak if you want) ----------
segmentLen      = getOrDefault(params, 'segmentLen', 65536);  % samples per segment (power of 2 recommended)
overlapFraction = getOrDefault(params, 'overlapFraction', 0.5); % 50% overlap
blockSeconds    = getOrDefault(params, 'blockSeconds', 5);    % how many seconds to read per disk read
nfft            = getOrDefault(params, 'nfft', max(2^nextpow2(segmentLen), segmentLen));
plotInDba       = getOrDefault(params, 'plotInDba', true);    % if true, plot 10*log10(PSD)
showLegend      = getOrDefault(params, 'showLegend', true);
% ----------------------------------------------------

% Validate
if overlapFraction <= 0 || overlapFraction >= 1
    error('overlapFraction must be in (0,1).');
end
hop = floor(segmentLen * (1 - overlapFraction));
if hop < 1, hop = 1; end
window = single(hann(segmentLen, 'periodic')); % window as single

% Find WAV files
d = dir(fullfile(folderPath, '*.wav'));
if isempty(d)
    fprintf('No .wav files found in %s\n', folderPath);
    return;
end

figure; hold on;
colors = lines(numel(d));
legendEntries = cell(numel(d),1);

for fi = 1:numel(d)
    fname = fullfile(folderPath, d(fi).name);
    fprintf('Processing (%d/%d): %s\n', fi, numel(d), d(fi).name);

    % Read audio info
    info = audioinfo(fname);
    if isfield(info, 'TotalSamples')
        totalSamples = info.TotalSamples;
    else
        totalSamples = round(info.TotalSamples); % fallback (shouldn't happen)
    end
    fs = info.SampleRate;
    numChannels = info.NumChannels;

    % compute block size in samples (min: at least segmentLen)
    blockSamples = max(segmentLen, round(blockSeconds * fs));
    % Make sure blockSamples is a multiple of hop for easier framing:
    blockSamples = hop * ceil(blockSamples / hop);

    % pre-alloc PSD accumulator (single)
    halfBinCount = floor(nfft/2) + 1;
    psdAccum = single(zeros(halfBinCount,1));
    frames = 0;

    % streaming read
    pos = 1;
    leftover = single([]); % leftover overlap from previous block
    % Normalization factor for PSD: U = sum(window.^2)
    U = sum(window .* window, 'native');         % single
    scaleConst = single(1 / (fs * double(U)));  % convert to double then single for stable division

    while pos <= totalSamples
        readEnd = min(totalSamples, pos + blockSamples - 1);
        % audioread supports sample-range reading: [start end]
        % audioread returns double by default; convert to single immediately.
        x = single(audioread(fname, [pos readEnd]));

        % convert to mono if needed
        if numChannels > 1
            % average channels safely (convert to double for mean then back to single)
            x = single(mean(double(x), 2));
        end

        % prepend leftover overlap from previous block
        if ~isempty(leftover)
            x = [leftover; x];
        end

        % framing: process all complete frames in x
        totalX = size(x,1);
        lastFrameStart = totalX - segmentLen + 1;
        if lastFrameStart < 1
            % not enough data to form a single frame: save for next iteration
            leftover = x;
            pos = readEnd + 1;
            continue;
        end

        % iterate frames by hop
        for s = 1:hop:lastFrameStart
            frame = x(s : s + segmentLen - 1);
            % window
            frame = frame .* window;
            % FFT (single)
            F = fft(frame, nfft);
            P = (abs(F(1:halfBinCount)).^2) * scaleConst; % single
            % one-sided correction (multiply by 2 except DC and Nyquist if present)
            if mod(nfft,2) == 0
                % even nfft: Nyquist bin exists at end
                P(2:end-1) = P(2:end-1) * 2;
            else
                P(2:end) = P(2:end) * 2;
            end
            psdAccum = psdAccum + single(P);
            frames = frames + 1;
        end

        % save leftover: last (segmentLen - hop) samples of x to overlap with next block
        overlapSamples = segmentLen - hop;
        if overlapSamples < 0, overlapSamples = 0; end
        if overlapSamples > 0
            leftover = x(end - overlapSamples + 1 : end);
        else
            leftover = single([]);
        end

        % advance
        pos = readEnd + 1;
    end % end while blocks

    % If file shorter than a single segment (frames==0) handle by zero-pad and compute one periodogram
    if frames == 0
        % read entire file
        x = single(audioread(fname));
        if numChannels > 1
            x = single(mean(double(x),2));
        end
        if numel(x) < segmentLen
            x = [x; zeros(segmentLen - numel(x),1,'single')];
        end
        frame = x(1:segmentLen) .* window;
        F = fft(frame, nfft);
        P = (abs(F(1:halfBinCount)).^2) * scaleConst;
        if mod(nfft,2) == 0
            P(2:end-1) = P(2:end-1) * 2;
        else
            P(2:end) = P(2:end) * 2;
        end
        psdMean = single(P);
    else
        psdMean = psdAccum / single(frames);
    end

    % frequency axis
    f = (0:halfBinCount-1) * (fs / nfft);

    % plot
    if plotInDba
        plotVals = 10 * log10(double(psdMean)); % convert to double for plotting dB
        plot(f, plotVals, 'DisplayName', d(fi).name, 'Color', colors(mod(fi-1,size(colors,1))+1,:));
        ylabel('PSD (dB/Hz)');
    else
        plot(f, double(psdMean), 'DisplayName', d(fi).name, 'Color', colors(mod(fi-1,size(colors,1))+1,:));
        ylabel('PSD (units^2/Hz)');
    end
    xlabel('Frequency (Hz)');
    xlim([0 fs/2]);
    grid on;
    legendEntries{fi} = d(fi).name;

    fprintf('  Done. Frames averaged: %d\n', frames);
end % file loop

if showLegend
    legend(legendEntries, 'Interpreter', 'none', 'Location', 'best');
end
title('Averaged one-sided PSD (full file duration)');

end % function

% === helper ===
function v = getOrDefault(s, name, defaultVal)
    if isstruct(s) && isfield(s, name)
        v = s.(name);
    else
        v = defaultVal;
    end
end
