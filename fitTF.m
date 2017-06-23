% Fit transfer function to spike noise spectrum. 
% 
% origin: population output for which to fit noise
% term: where this funciton should apply inputs
% points: list of points at which to assess noise (should be multiple
%   points for better spectral estimate -- essentially Bartlett's method
%   except it's best to randomize a bit to de-emphasize spurious spike patterns)
% doPlot: true if spectrum is to be plotted
function [sys, RMS, spikePSD, modelPSD] = fitTF(noise, doPlot, varargin)
%     if doPlot
%         figure, hold on
%     end
    
    if ~isempty(varargin) 
        orders = varargin{1};
    else 
        orders = [2 2];
    end
    
    dt = .001;
    windowLen = 2^8;
    [freq, spikePSD, minPSD, maxPSD] = PMUtils.getPSD(noise, windowLen, .001, doPlot);

    Z = iddata(noise', [], .001);
    M = armax(Z, orders);
    sys = M.NoiseVariance^.5*idpoly(M.A, M.C, 1, 1, 1, 0, M.Ts);

    modelNoise = lsim(sys, randn(size(noise')));
    [freq, modelPSD, minPSD, maxPSD] = PMUtils.getPSD(modelNoise, windowLen, .001, 0);
        
    RMS = mean( (spikePSD - modelPSD).^2 )^.5;
    
    if doPlot
        [sysMag, ~] = bode(sys, 2*pi*freq); 
        sysPSD = 2*(squeeze(sysMag)).^2*dt;
        plot(freq, sysPSD, 'k', 'LineWidth', 3) 

        % generate model noise in same pattern as spike noise to plot together
        
        plot(freq, minPSD, 'k', 'LineWidth', 1)
        plot(freq, maxPSD, 'k', 'LineWidth', 1)
        
        xlabel('Frequency (Hz)', 'FontSize', 18)
        ylabel('PSD', 'FontSize', 18)
        set(gca, 'FontSize', 18)
        
        % check amplitude OK ... 
        sprintf('this ratio should be close to 1: %f', std(modelNoise(:)) / std(noise(:)))
    end
end
