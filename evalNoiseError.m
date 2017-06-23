% Finds error in noise spectra approximations across all test populations
% (1D; 100 neurons) with ramps around 0 and 1.5. 
% 
% orders: orders of autoregressive and moving average polynomials, i.e. [na nc]
% steps: number of steps to simulate
% doPlot: if >0, show plots of fits
function [RMSE, RMSP] = evalNoiseError(orders, steps, doPlot)

    sysPSC = tf(1, [.005 1]);
    
    evalSteps = 2^13; %2^13 %always use same # of steps to compare PSDs
    dt = .001;
    
    sg = getTestSpikeGenerators(100, 1);
    
    if doPlot
        figure, set(gcf, 'Position', [440   275   720   523]) 
    end
    
    rampRanges = [-.1 .1; 1.25 1.75];
    RMSE = zeros(length(sg), 2);
    RMSP = zeros(length(sg), 2);
    for i = 1:length(sg)
        cp = CosinePopulation(1, sg{i}, '1D');        
        fun = @(x) x; 
        origin = DecodedOrigin('X', fun, cp);
        findDecoders(origin, 1*randn(1,300));
        addOrigin(cp, 'X', fun, origin);
        term = addTermination(cp, 'input', .005, 1);
        
        for j = 1:2
            fitNoise = PMUtils.getRampNoise(origin, term, rampRanges(j,:), steps);
            fitNoise = lsim(sysPSC, fitNoise, (0:length(fitNoise)-1)*dt)';
            [freq, fitPSD] = PMUtils.getPSD(fitNoise, 2^8, dt, 0);
            sys = fitTF(fitNoise, 0, orders);
            
            evalNoise = PMUtils.getRampNoise(origin, term, rampRanges(j,:), evalSteps);   
            evalNoise = lsim(sysPSC, evalNoise, (0:length(evalNoise)-1)*dt)';
            modelNoise = lsim(sys, randn(size(evalNoise')));               
            [freq, spikePSD] = PMUtils.getPSD(evalNoise, 2^8, dt, 0);
            [freq, modelPSD] = PMUtils.getPSD(modelNoise, 2^8, dt, 0);
            [sysMag, ~] = bode(sys, 2*pi*freq);
            sysPSD = 2*(squeeze(sysMag)).^2*dt;
            RMSE(i,j) = mean( (spikePSD - sysPSD).^2 )^.5;
%             RMSE(i,j) = mean( (spikePSD - modelPSD).^2 )^.5;
            RMSP(i,j) = mean( spikePSD.^2 )^.5;
            
            if doPlot
                subplot(4, length(sg)/2, i+(j-1)*length(sg)), hold on
%                 hold on

                %                 plot(modelPSD, 'k', 'LineWidth', 1)
                plot(fitPSD, 'k', 'LineWidth', 2, 'Color', [.8 .8 .8])
                plot(spikePSD, 'k', 'LineWidth', 2, 'Color', [.5 .5 .5])
                plot(sysPSD, 'k', 'LineWidth', 1)
                set(gca, 'XTick', [])
                set(gca, 'YTick', [])
                set(gca, 'XLim', [0 length(freq)])
            end            
        end        
        
%         fitNoiseA = PMUtils.getRampNoise(origin, term, [-.1 .1], steps);
%         sysA = fitTF(fitNoiseA, 0, orders);
%         
%         fitNoiseB = PMUtils.getRampNoise(origin, term, [1.4 1.6], steps);
%         sysB = fitTF(fitNoiseB, 0, orders);   
%         
%         evalSpikeNoiseA = PMUtils.getRampNoise(origin, term, [-.1 .1], evalSteps);
%         evalModelNoiseA = lsim(sysA, randn(size(evalSpikeNoiseA')));        
%         [freq, spikePSDA] = PMUtils.getPSD(evalSpikeNoiseA, 2^8, .001, 0);
% 
%         evalNoiseB = PMUtils.getRampNoise(origin, term, [-.1 .1], evalSteps);
%         [freq, spikePSD] = PMUtils.getPSD(noise, windowLen, .001, doPlot);
    
%         if doPlot
%             subplot(2, length(sg), i), hold on
%             plot(spikePSDA, 'k', 'LineWidth', 2)
%             plot(modelPSDA, 'k', 'LineWidth', 1)
%             set(gca, 'XTick', [])
%             set(gca, 'YTick', [])
% 
%             subplot(2, length(sg), length(sg)+i), hold on
%             plot(spikePSDB, 'k', 'LineWidth', 2)
%             plot(modelPSDB, 'k', 'LineWidth', 1)
%             set(gca, 'XTick', [])
%             set(gca, 'YTick', [])
%         end
    end
end

