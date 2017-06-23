classdef PMUtils < handle
   
    methods (Static) 
        
        % points: points at which to approximate population output
        % fitRadius: radius of region over which to fit error (may be different than range of eval points)
        % pop: population that we are modelling output of
        % decoders: decoders of modelled origin
        % fun: ideal function calculated by origin
        % order: number of parameters in this model
        % outputApprox: model approximation of population output at points
        % biasApprox: model approximation of population bias error at points
        function [outputApprox, biasApprox] = interpolationError(points, fitRadius, pop, decoders, fun, order)
            gridX = -fitRadius:2*fitRadius/(order-1):fitRadius;
            gridY = PMUtils.getDecoded(gridX, decoders, pop);
            outputApprox = interp1(gridX, gridY, points, 'linear');
            biasApprox = interp1(gridX, gridY-fun(gridX), points, 'linear');
        end
        
        % points: a row vector of points along x at which to approximate
        %   population output; the same points along y are used to make a grid  
        % fitRadius: radius of region over which to fit error (may be different than range of eval points)
        % pop: population that we are modelling output of
        % decoders: decoders of modelled origin
        % fun: ideal function calculated by origin
        % order: number of parameters in this model
        % outputApprox: model approximation of population output at points
        % biasApprox: model approximation of population bias error at points
        function [outputApprox, biasApprox] = interpolationError2D(points, fitRadius, pop, decoders, fun, order)
            gridSteps = -fitRadius:2*fitRadius/(order-1):fitRadius;
            gridX = repmat(gridSteps, length(gridSteps), 1);
            gridY = gridX';
            gridPoints = [reshape(gridX, 1, numel(gridX)); reshape(gridY, 1, numel(gridY))];
            gridZ = reshape(PMUtils.getDecoded(gridPoints, decoders, pop), length(gridSteps), length(gridSteps));
            ideal = reshape(fun(gridPoints), length(gridSteps), length(gridSteps));
            outputApprox = interp2(gridX, gridY, gridZ, points, points', 'linear');
            biasApprox = interp2(gridX, gridY, gridZ-ideal, points, points', 'linear');
        end
        
        % points: a row vector of points along x at which to approximate
        %   population output; the same points along y and z are used to make a grid  
        % fitRadius: radius of region over which to fit error (may be different than range of eval points)
        % pop: population that we are modelling output of
        % decoders: decoders of modelled origin
        % fun: ideal function calculated by origin
        % order: number of parameters in this model
        % outputApprox: model approximation of population output at points
        % biasApprox: model approximation of population bias error at points
        function [outputApprox, biasApprox] = interpolationError3D(points, fitRadius, pop, decoders, fun, order)
            gridSteps = -fitRadius:2*fitRadius/(order-1):fitRadius;
            [gridX, gridY, gridZ] = meshgrid(gridSteps);
            gridPoints = [reshape(gridX, 1, numel(gridX)); reshape(gridY, 1, numel(gridY)); reshape(gridZ, 1, numel(gridZ))];
            gridV = reshape(PMUtils.getDecoded(gridPoints, decoders, pop), order, order, order);
            ideal = reshape(fun(gridPoints), order, order, order);
            outputApprox = interp3(gridX, gridY, gridZ, gridV, points, points', points, 'linear');
            biasApprox = interp3(gridX, gridY, gridZ, gridV-ideal, points, points', points, 'linear');
        end
        

        function [outputApprox, biasApprox] = polyfitError(points, fitRadius, pop, decoders, fun, order) 
            % note: doesn't make noticeable difference if we rescale x to [-1,1]
            gridX = -fitRadius:2*fitRadius/499:fitRadius;
            gridY = PMUtils.getDecoded(gridX, decoders, pop);            
            [POut, S] = polyfit(gridX, gridY, order-1);
            outputApprox = polyval(POut, points);
            PBias = polyfit(gridX, gridY-fun(gridX), order-1);
            biasApprox = polyval(PBias, points);
        end
        
        % points: points at which to approximate population output
        function [outputApprox, biasApprox] = fftfit(points, fitRadius, pop, decoders, fun, order)
            assert (mod(order,2) == 1)
            gridX = -fitRadius:2*fitRadius/499:fitRadius;
            gridY = PMUtils.getDecoded(gridX, decoders, pop); 
            FOut = fft(gridY)/length(gridY);
            outputApprox = PMUtils.ffteval(FOut, fitRadius, points, order);
            FBias = fft(gridY-fun(gridX))/length(gridY);
            biasApprox = PMUtils.ffteval(FBias, fitRadius, points, order);
            
%             decoded = PMUtils.getDecoded(points, decoders, pop);
%             ideal = fun(points);

%             toKeep = ceil(order/2);
%             
%             FY = fft(decoded);
%             FError = fft(decoded-ideal);
%             
%             FY = FY(1:toKeep)/length(decoded);
%             FError = FError(1:toKeep)/length(decoded);
        end
        
        % Invert FFT on unevenly spaced points. 
        % 
        % FY: fourier coefficients from fft, brickwall lowpass filtered, with zeros removed
        % radius: FFT calculated on [-width width]
        % points: points at which to evaluate
        % order: number of coefficients to use 
        % y: inverted FFT at given points 
        function y = ffteval(FY, radius, points, order)
            w = 2*pi/(2*radius);
            y = real(FY(1)) * ones(size(points));
%             figure, hold on, title(order)
%             plot(points, y)
            for i = 2:ceil(order/2)
                frequency = w*(i-1);
                even = 2*real(FY(i))*cos(frequency*(points+radius));
                odd = - 2*imag(FY(i))*sin(frequency*(points+radius));
                y = y + even + odd;
%                 plot(points, even)
%                 plot(points, odd)
            end 
        end
        
        function approx = getDecoded(points, decoders, pop)
            approx = decoders * getRates(pop, points, 0, 0);
        end

        function RMS = getRMSD(A, B)
            RMS = mean( (A-B).^2 )^.5;
        end
        
%         function gridX = getGridX(ng)
%             gridX = -3:(6/(ng-1)):3;
% %             gridX = -2:(4/(ng-1)):2; 
% %             c = 1.5;
% %             gridX(gridX<-c) = -c+(gridX(gridX<-c)+c)*(3-c)/(2-c); 
% %             gridX(gridX>c) = c+(gridX(gridX>c)-c)*(3-c)/(2-c); 
%         end
        
        % This is for multidimensional interpolation of bias. 
        % 
        % x: projection of points onto dim axis
        % y: distance of points from dim axis
        function [x, y] = getProjectionAndDistance(points, dim)
            x = points(dim,:);
            notDim = setdiff(1:size(points,1), dim);
            y = sum(points(notDim,:).^2, 1).^.5;
        end
        
        % Rescale points so they lie within interpolation range. 
        function points = scaleProjectionAndDistance(points, dim, maxProj, maxDist)
            [x, y] = PMUtils.getProjectionAndDistance(points, dim);
            relativeToBoundaries = max([-x/maxProj; x/maxProj; y/maxDist], [], 1);
            tooBig = find(relativeToBoundaries > 1);
            points(:,tooBig) = points(:,tooBig) ./ repmat(relativeToBoundaries(tooBig), size(points,1), 1);
        end

        % x: a vector
        % gridx: a vector of sample points for an interpolation grid
        % gridy: matrix of values on sample points along each axis (each
        %   row corresponds to an axis)
        function y = interpolateND(x, gridx, gridy)
            assert (length(x) == size(gridy,1))
            assert (length(gridx) == size(gridy,2))
            
            nx = norm(x);
            xrad = nx * sign(x);
            yOnAxes = zeros(size(x));
            for i = 1:length(x)
                yOnAxes(i) = interp1(gridx, gridy(i,:), xrad(i), 'linear', 'extrap'); 
            end
            
            y = 0;
            ax = abs(x);
            if nx>0
                y = (x.^2)' * yOnAxes / nx^2; 
%                 y = abs(x)' * yOnAxes / sum(ax); 
            end
        end
        
        % Note: this needs a smaller grid than interpolateND. 
        function y = interpolateND2(x, gridx, gridy)
            assert (length(x) == size(gridy,1))
            assert (length(gridx) == size(gridy,2))

            y = 0;
            for i = 1:length(x)
                y = y + abs(x(i))*interp1(gridx, gridy(i,:), x(i), 'linear', 'extrap'); 
            end            
        end
                
        function [gridx, gridYDecoded, gridYError] = getNDGrid(ng, rad, pop, decoders, fun)
            gridx = -rad:(2*rad/(ng-1)):rad;
            gridYDecoded = zeros(pop.dimension, ng);
            gridYError = zeros(pop.dimension, ng);
            for i = 1:pop.dimension
                points = zeros(pop.dimension, ng);
                points(i,:) = gridx;
                gridYDecoded(i,:) = PMUtils.getDecoded(points, decoders, pop);   
                gridYError(i,:) = gridYDecoded(i,:) - fun(points);
            end
        end
        
        % Inverse distance weighting
        % p: exponent on distance
        function y = interpolateID(x, p, gridx, gridy)
            distances = sum((gridx - repmat(x, 1, size(gridx,2))).^2, 1).^.5;
            if min(distances == 0)
                ind = find(distances == 0, 1, 'first');
                y = gridy(ind);
            else
                weights = 1./distances.^p;
%                 this doesn't work either
                sw = sort(weights); mw = sw(end-length(x)); 
                weights(weights <= mw) = 0;
                y = sum(weights .* gridy) / sum(weights);
            end
        end
        
        % np: number of points between which to interpolate
        % rad: radius; samples drawn from gaussian with this width
        function [gridx, gridYDecoded, gridYError] = getIDGrid(np, rad, pop, decoders, fun)
            gridx = rad * randn(pop.dimension, np);
            gridYDecoded = PMUtils.getDecoded(gridx, decoders, pop);
            gridYError = gridYDecoded - fun(gridx);
        end
        
        % smooth non-uniform data with Gaussian kernels
        % x, y, z: data vectors (as for scatter3)
        % sigma: width of smoothing kernel
        % xx: list of grid points in x at which to find smoothed values
        % yy: list of grid points in x at which to find smoothed values
        function zz = smooth(x, y, z, sigma, xx, yy)
            assert (length(x) == length(z));
            assert (length(y) == length(z));
            
            npoints = length(xx) * length(yy);
            points = [reshape(repmat(xx', 1, length(yy)), 1, npoints); reshape(repmat(yy, length(xx), 1), 1, npoints)];
            num = zeros(1, length(points));
            den = zeros(size(num));
            for i = 1:length(z)
                d = points-repmat([x(i); y(i)], 1, size(points,2));
                g = exp(-sum(d.^2,1)/2/sigma);
                num = num + z(i)*g;
                den = den + g;
            end   
            
            zz = reshape((num ./ den), length(xx), length(yy));
        end
        
        function noise = getNoise(origin, termination, points, steps)
            static = PMUtils.getDecoded(points, origin.decoders, origin.population);
            noise = zeros(size(points,2), steps);
            dt = .001;
            transientSteps = 100;
            for i = 1:size(points,2)
                input = FunctionInput(@(t) points(:,i));
                n = Network(.001);
                n.addNode(input);
                n.addNode(origin.population);
                n.addConnection(input.origins{1}, termination);
                outputProbe = n.addProbe(getOrigin(origin.population, 'X'), 'output');
                n.run(0,(transientSteps+steps)*dt);
                [~, history] = outputProbe.getHistory();
                noise(i,:) = history(transientSteps+1:transientSteps+steps) - static(:,i);
                n.reset();
            end            
        end
        
        % termination: termination to which ramp is applied (use this to
        %   project the ramp into the population's space)
        function noise = getRampNoise(origin, termination, points, steps)
            assert(length(steps) == 1)
            
            dt = .001;
            T = dt*steps;
            transientSteps = 100;
            input = FunctionInput(@(t) points(:,1) + (t>0)*t/T*(points(:,2)-points(:,1)));
            n = Network(.001);
            n.addNode(input);
            n.addNode(origin.population);
            n.addConnection(input.origins{1}, termination);
            outputProbe = n.addProbe(origin, 'output');
            
            n.setSimulationMode(ModeConfigurable.RATE_MODE)
            n.reset();
            n.run(-transientSteps*dt,steps*dt);
            [~, rateHistory] = outputProbe.getHistory();
            n.reset();

            n.setSimulationMode(ModeConfigurable.DEFAULT_MODE)
            n.run(-transientSteps*dt,steps*dt);
            [~, spikeHistory] = outputProbe.getHistory();
            n.reset();                       
            
            dim = 1;
            noise = spikeHistory(dim, transientSteps+1:transientSteps+steps) - rateHistory(dim, transientSteps+1:transientSteps+steps);
        end

        % Fit a linear system to noise from a ramp simulation. 
        % 
        % origin: origin from which to collect noise
        % termination: termination to drive during simulation
        % points: start and end points of ramp
        % steps: number of time steps to simulate
        % orders: ARMA model orders
        % tauPSC: synaptic time constant with which to filter the noise
        %   before fitting
        function sys = fitRampNoise(origin, termination, points, steps, orders, tauPSC, dt)
            sysPSC = tf(1, [tauPSC 1]);
            noise = PMUtils.getRampNoise(origin, termination, points, steps);
            noise = lsim(sysPSC, noise, (0:length(noise)-1)*dt);
            [sys, RMS, spikePSD, modelPSD] = fitTF(noise', 0, orders);
            
%             modelNoise = lsim(sys, randn(1,steps), (0:steps-1)*dt);
%             sprintf('real noise SD %f; model noise SD %f', std(noise,1), std(modelNoise,1)), pause 
        end
        
        % Tests whether samples have a normal distribution. 
        % 
        % samples: random samples to test
        % P: probability that samples come from a non-normal distribution
        function P = testNormality(samples)
            normalizedSamples = samples / std(samples);
            [~,P] = kstest(normalizedSamples);
        end
        
        % Run a population and get origin output. 
        % 
        % origin: the origin to get output from (must be attached to a
        %   population
        % term: where to apply input
        % f: function of time to drive population with 
        % T: end time or [start-time end-time]
        function [time, history] = driveOrigin(origin, term, f, T)
            input = FunctionInput(f);
            dt = .001;
            n = Network(dt);
            n.addNode(input);
            n.addNode(origin.population);
            n.addConnection(input.origins{1}, term);
            outputProbe = n.addProbe(origin, 'output');
            n.reset();
            if length(T) > 1
                n.run(T(1), T(2));
            else
                n.run(0, T);
            end
            [time, history] = outputProbe.getHistory();
        end
        
%         function plotBias(points, approx, ideal)
%             hold on
%             plot(points, ideal, 'k--')
%             plot(points, approx, 'k')
%             set(gca, 'FontSize', 18)
%             xlabel('State', 'FontSize', 18)  
% %             ylabel('OLE', 'FontSize', 18)
%             set(gca, 'XLim', [-2.5 2.5])
%             p = get(gca, 'Position');
%             ha2 = axes('XAxisLocation', 'top', 'YAxisLocation', 'right', 'Color', 'none', 'XTick', [], 'Position', get(gca, 'Position'));  
%             line(points, approx - ideal, 'Color', 'k', 'Parent', ha2)
% %             ylabel('Bias', 'FontSize', 18)
%             set(gca, 'FontSize', 18)
%             set(gca, 'XLim', [-2.5 2.5])
%             set(ha2, 'Position', p)
%         end

        % Estimates power spectral density. 
        % 
        % noise: noise to estimate PSD of 
        % wlen: window length
        % dt: step size
        % 
        % freq: list of frequencies (Hz) at which PSD is estimated
        % PSD: PSD estimates
        % minPSD: bottom of 95% confidence interval
        % maxPSD: top of 95% confidence interval
        function [freq, PSD, minPSD, maxPSD] = getPSD(noise, wlen, dt, varargin)
%             [PSD, freq, confidenceInterval] = pwelch(noise, wlen, wlen/2, wlen/2, 1/dt, 'ConfidenceLevel', .95);
            [PSD, freq, confidenceInterval] = pwelch(noise, hann(wlen), wlen/2, wlen/2, 1/dt, 'ConfidenceLevel', .95);
            minPSD = confidenceInterval(:,1);
            maxPSD = confidenceInterval(:,2); 
            
            if ~isempty(varargin) && varargin{1} 
                fill([freq; flipud(freq)], [minPSD; flipud(maxPSD)], 'k', 'FaceAlpha', .2, 'EdgeAlpha', .2)
            end
        end


    end
    
end
