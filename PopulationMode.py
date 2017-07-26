import nengo
import numpy as np
import random
import nengo.utils.numpy as npext
from numpy.polynomial.polynomial import polyfromroots
from scipy.interpolate import interp1d, interp2d, griddata, RectBivariateSpline, RegularGridInterpolator, interpn
from scipy.optimize import leastsq
from scipy.stats import chi2, truncnorm
from scipy import signal, linalg
import pandas as pd
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from control import matlab, balred, tf2ss

def compute_decoder(self, pre, post):
    """
    Compute decoder; function based from nengo
    """
    # # get the tuning curves
    # x_values, A = self.compute_tuning_curves(encoder, gain, bias)
    # # print "x_values: {}\nshape: {}".format(x_values, x_values.shape)
    # # get the desired decoded value for each sample point
    # temp = [[function(x)] for x in x_values.squeeze()]
    # value = np.array(temp)

    # # find the optimal linear decoder
    # A = A.T
    # Gamma = np.dot(A, A.T)
    # Upsilon = np.dot(A, value)
    # Ginv = np.linalg.pinv(Gamma)
    # decoder = np.dot(Ginv, Upsilon) / self.dt
    # return decoder.squeeze()

    tstart = time.time()
    sigma = self.reg * A.max()
    X, info = self.solver(A, Y, sigma, rng=rng)
    info['time'] = time.time() - tstart
    return self.mul_encoders(X, E), info

class PopulationModeModel(object):    
    def __init__(self, n_neuron, sim_time, dim, n_samples=100, tau_ref=0.002, 
                tau_RC=0.02, tau_pstc=0.1, radii=1, dt=0.001, params=None, seed=None):
        """
        within params...

        population: desired ensemble that would be represented in this mode
        rates: firing rates of the neurons within the population
        decoders: decoding vectors of the neurons within the population
        """
        # Conversion of parameters from the nengo population model to actual population mode model
 
        self.pop = params['population'] # default LIF Model Population
        self.origins = params['origins'] # source of the ideal output
        self.radii = self.pop.radius # max amplitude in each dimension
        self.dim = self.pop.dimensions # dimension of the population 
        self.x_vals = params['x_vals'] # x; points from which to interpolate bias
        self.rates = params['rates'] # activities
        self.decoders = params['decoders']
        self.dt = params['dt']
        # self.gain = params['gain']
        # self.bias = params['bias']

        # a vector that speicfies the origin to which each bias element belongs
        self.origin_indices = self.makeOriginIndices()
        self.trange = params['timerange'] # simulation time range
        self.dt = params['dt'] # simulation time step
        self.tau_syn = 0.005
        self.window_length = 2**8

        # Params for bias model
        self.r = [] # For cases higher than 3D
        self.bias_values_sim = []
        self.bias_values_est = [] # Estimated bias term
        self.ideal_values = params['outputs'][0]
        self.actual_values = params['outputs'][1]

        # Params for noise model
        self.noise_values_sim = []
        self.noise_values_est = []
        self.noise_originind = []
        self.noise_corr = []

        self.sds = [] # standard deviations of noise

        self.noiseTime = []
        self.noiseSamples = []

        self.df = 0
        self.ARMA_model = None
        self.RMSE = None
        self.RMSP = None

    ###############################
    # SIMULATION HELPER FUNCITONS #
    ###############################

    def input(t):
        """ 
        Cyclic ramp input
        """
        return 2*(t % 1) - 1

    ############################
    # GENERAL HELPER FUNCITONS #
    ############################
    def makeOriginIndices(self):
        """
        Generates and returns a list of indexes of origin numbers.

        Steps:
        1. Find the total number of dimensions
        2. For each origin, record the index of the origin n times, where n is 
            number of dimensions
        3. Return the indices

        """
        nTotalDim = 0
        for origin in self.origins:
            nTotalDim += origin.dimensions 

        origin_indices = np.zeros((1, nTotalDim))
        c = 0
        for i in range(len(self.origins)):
            for j in range(c, c+self.origins[i].dimensions):
                origin_indices[0:j] = i
            c += self.origins[i].dimensions

        return origin_indices.squeeze()

    def genRandomPoints(self, orig_points, distribution="uniform"):
        """
        generates a new set of points based on the original set of points; assume that you want same number of points as
        original points

        """
        if distribution == "uniform":
            if self.dim == 1:
                points = np.linspace(orig_points[0], orig_points[-1], 2*len(orig_points), endpoint=True)
                new_points = points[1::2]
            elif self.dim == 2:
                x_points = np.linspace(orig_points[0:,0][0], orig_points[0:,0][-1], 2*len(orig_points), endpoint=True)
                y_points = x_points
                x_points = x_points[1::2]
                y_points = y_points[1::2]
                new_points = np.asarray(np.meshgrid(x_points, y_points))
                new_points_copy = np.copy(new_points)
                new_points[0,:] = new_points[1,:]

        elif distribution == "gaussian":
            # TODO: Implement generating samples under gaussian distribution for higher dimensions
            dimensions = orig_points.shape
            np.random.seed()
            # new_points = sorted(np.random.randn(1, dimensions[0])*orig_points)
            mean_val = np.mean(orig_points)
            sd_val = np.std(orig_points)
            trunc_norm = truncnorm((orig_points[0]-mean_val)/sd_val, (orig_points[-1]-mean_val)/sd_val, loc=mean_val, scale=sd_val)
            new_points = sorted(trunc_norm.rvs(dimensions))

        # print "orig_points: {}".format(orig_points.shape)
        return new_points

    def calcNeuralActivities(self):
        """
        Calculates the neural activity by performing dot product in between
        the decoding vector and the rates.
        """
        eval_points = self.x_vals
        bias = np.zeros(eval_points.T.shape)
        decoders_pre = self.decoders[0]
        decoders_post = self.decoders[1]
        rates_pre = self.rates[0]
        rates_post = self.rates[1]

        print "eval_points: {}".format(eval_points.shape)
        print "decoders_post: {}".format(decoders_post.shape)
        print "rates_post: {}".format(rates_post.shape)

        # actual = np.tensordot(decoders_post,rates_post, axes=([-1],[0]))
        # ideal = np.tensordot(decoders_pre, rates_pre, axes=([-1],[0]))    
        # self.actual_values = actual
        # self.ideal_values = ideal
        # self.output = actual.sum()
        

    ########################
    # BIAS MODEL FUNCTIONS #
    ########################

    def createBiasModel(self, mode="lin_interp"):
        """
        Creates bias model for estimating the bias term of the surrogate model.
        mode: different mode of bias approximation (types: lin_interp, poly_reg, fourier_reg)
        
        Steps:
        1. Perform interpolation based on the dimension
        2. Find bias values of the x_intercepts (evalpoints)
        3. Estimate the bias based on the the bias found from Step 2
        3. Return the bias values

        """
        a = 3 # bias is modelled over a*[-radius radius], this case, [-3 3]

        if self.dim < 4:
            self.getBiasSamples(mode)
        
        # TODO: Finish implementing for multidimensional case
        else: # uses exploited symmetry in the bias error
            radius = a*self.radii[1]
            self.r = np.linspace(0,radius,(radius/200))
            self.getBiasSamples(np.array([self.r, 
                                np.zeros(self.dim-1, len(self.r))]))
    
    def getBiasSamples(self, mode):
        """
        Obtains samples of bias (distortion) error which can then be fit to a 
        model. Returns an array of bias errors at each eval points for each 
        origin and an ideal values (ideal). (Note: actual = ideal + bias)

        Steps:
        1. Create empty array for bias based on the number of evalPoints
        2. For every evaluation pointss, calculate bias based on the simulation
        3. Estimate bias using the simulated bias term; interpolate
        """
        eval_points = self.x_vals
        bias = np.zeros(eval_points.T.shape)
        # self.calcNeuralActivities()
        actual = self.actual_values
        ideal = self.ideal_values
        print "actual shape: {}, ideal shape: {}".format(actual.shape, ideal.shape)

        bias = (actual-ideal).squeeze()
        # print "eval_points: {}".format(eval_points.shape)
        # Looping through pre node
        # for i in range(len(self.origins)):
        #     ind = np.where(self.origin_indices == i)
        #     bias[ind,:] = actual[ind,:] - ideal[ind,:]
        self.bias_values_sim = bias

        if self.dim == 1:
            eval_points = eval_points.squeeze()
            bias = bias
            # print "bias shape: {}\n eval_points: {}".format(bias, eval_points)

            # TODO: Complete different interpolation methods
            if mode == "lin_interp":
                func = interp1d(eval_points, bias)
                new_points = self.genRandomPoints(eval_points)
                self.bias_values_est = func(new_points)
        
        # TODO: Complete method for bias interpolation for multidimensional case
        elif self.dim == 2:
            mode = "RBS" # Temporarily set it since other interp are not working

            if mode == "lin_interp":
                new_points = self.genRandomPoints(eval_points[0])
                print "new_points: {}".format(new_points.shape)
                bias_matrices = np.empty(new_points.shape)
                for d in range(len(bias.T)):
                    bias_matrices[d] = np.diag(bias.T[d])

                # print "bias_matrices: {}".format(bias_matrices[0].shape)              
                # print "orig domain: {}".format(eval_points.shape)

                # func_first = griddata(eval_points[0], bias_matrices[0], new_points[0], method='linear')
                # func_second = griddata(eval_points[1], bias_matrices[1], new_points[1], method='linear')

                # func_first = RectBivariateSpline(eval_points[0][0:,0], eval_points[0][0:,0], bias_matrices[0])
                # func_second = RectBivariateSpline(eval_points[1][0,0:], eval_points[1][0,0:], bias_matrices[1])
                points = np.vstack((eval_points[0][0:,0], eval_points[0][0:,0], bias_matrices[0]))

                func_first = interpn(eval_points[0][0:,0], bias_matrices[0], new_points, method='linear')
                func_second = interpn(eval_points[0], bias_matrices, new_points, method='linear')

                # func_first = RegularGridInterpolator(eval_points[0][0:,0].T, bias_matrices[0])
                # func_second = RegularGridInterpolator(eval_points[1][0,0:].T, bias_matrices[1])

                # print "x shape: {}".format(x.shape)
                # print "y shape: {}".format(y.shape)

                bias_est_first = func_first(new_points[0][0:,0], new_points[0][0:,0])
                bias_est_second = func_second(new_points[1][0,0:], new_points[0][0:,0])


                # print "x_est.shape: {}".format(np.diag(bias_est_first).shape)
                # print "y_est.shape: {}".format(bias_est_second.shape)

                self.bias_values_est = np.vstack((np.diag(bias_est_first), np.diag(bias_est_second)))

            if mode == "RBS":
                new_points = self.genRandomPoints(eval_points[0])
                bias_matrices = np.empty(new_points.shape)
                for d in range(len(bias.T)):
                    bias_matrices[d] = np.diag(bias.T[d])

                func_first = RectBivariateSpline(eval_points[0][0:,0], eval_points[0][0:,0], bias_matrices[0])
                func_second = RectBivariateSpline(eval_points[1][0,0:], eval_points[1][0,0:], bias_matrices[1])

                bias_est_first = func_first(new_points[0][0:,0], new_points[0][0:,0])
                bias_est_second = func_second(new_points[1][0,0:], new_points[1][0:,0])

                self.bias_values_est = np.vstack((np.diag(bias_est_first), np.diag(bias_est_second))).transpose()

            print "self.bias_values_sim: {}\nand its shape: {}".format(self.bias_values_sim, self.bias_values_sim.shape)
            print "self.bias_values_est: {}\nand its shape: {}".format(self.bias_values_est, self.bias_values_est.shape)

    #########################
    # NOISE MODEL FUNCTIONS #
    #########################

    def createNoiseModel(self, mode="lin_interp"):
        """
        Create noise model for estimating the noise term of the surrogate model
        mode: different mode of bias approximation (types: lin_interp, poly_reg, fourier_reg)

        Steps:
        1. Calculate the noise based on actual values and bias values (noise = (est - actual) - bias)
        2. Normalize the noise by passing it to the LPF
        3. 

        """
        n_totaldim = len(self.origin_indices) # total dim of all origins
        time = self.trange
        num_points = 3
        sim_time = time[-1] # simulated time
        eval_points = self.x_vals

        # Need bias to calculate noise
        if len(self.bias_values_est) == 0 or len(self.bias_values_sim) == 0:
            self.getBiasSamples(mode)

        actual = self.actual_values
        ideal = self.ideal_values
        noise = (ideal-actual)-self.bias_values_sim
        self.noise_values_sim = noise
        # print "noise: {}".format(noise)
        
        freq = 2*np.pi*np.linspace(0,(0.5/self.dt-1/sim_time),1/sim_time, endpoint=True)
        rho = np.zeros((n_totaldim, n_totaldim)) # correlations

        # Use LPF to normalize the noise
    	mags = np.zeros((n_totaldim, len(freq), self.dim)) # fourier magnitudes
    	points = np.matlib.repmat(np.zeros(eval_points.shape), 1, 2) # TODO: Figure out why these parameters are used
    	points[0,1] = 2*eval_points[0] # a point at 2*radius
    	self.sds = np.zeros((n_totaldim, 2)) # standard deviations of the noise; why are we using 2 here.... (# of dimensions perhaps?)

    	for i in range(self.dim):
    		self.sds[:,i] = np.std(noise,axis=1,ddof=1)
    		if i == 1:
    			rho = np.corrcoef(np.transpose(noise))

    		for j in range(n_totaldim):
    			fourier = np.fft.fft(noise[j,:]) / len(time) / (2*np.pi**0.5) # TODO: Confirm math here
    			mags[j,:,i] = np.abs(fourier[0:len(freq)])

    	self.noise_corr = rho

        RMSE, RMSP = self.evalNoiseError(steps=2**14, orders=[2,2])
        self.RMSE = RMSE
        self.RMSP = RMSP

    # def getNoiseSamples(self, dt, T):
    #     """
    #     Obtains samples of time-varying noise.
    #     Returns noise, a cell array of noise samples per origin 

    #     originInd: a vector with same length as the sum of dimensions of all
    #     origins. Specifies the origin to which each bias element belongs.
    #     evalPoint: a populaiton state at which the noise is to be sampled for 
    #     some period of time
    #     dt: time step of simulation for collecting samples (in seconds)
    #     T: duration of simulation period (in seconds)

    #     """
    #     p = self.pop

    #     if not evalPoint:
    #         evalPoint = np.zeros(self.x_vals)

    #     time = self.trange
    #     sim_time = time[-1] # simulated time
    #     eval_points = self.x_vals
    #     noise = np.zeros((len(self.originIndices), len(time)))

    #     drive = self.__drive
    #     for i in range(len(time)):
    #         activity = run(p.spikeGenerator, drive, time[i]-dt, time[i], 1) 
    #         for j in range(len(p.origins)):
    #             setActivity(p.origins[j], time[i], activity) #TODO
    #             noise[self.originIndices == j,i] = getOutput(p.origins[j])

    #     [bias, ideal] = self.getBiasSamples(evalPoint)
    #     noise = noise - np.matlib.repmat(ideal + bias, 1, len(time))


    def evalNoiseError(self, steps, orders=None):
        """
        Finds error in noise spectra approximations across all test populations with ramps.
        Returns: RMSE, RMSP

        Steps:
        1. Calculate the noise based on the simulated result
        2. Find the transfer function of the ARMA system model
        3. Recalculate the noise based on the new normally distributed noise samples
        4. Calculte the model noise by using the new sample points and the transfer function
        5. Calculte the system's power spectra density (PSD) from the simulation result
        6. Calculate the model's power spectra density (PSD) from the noise model
        7. Calculate the difference between the modeled PSD and original PSD

        orders: orders of autoregressive and moving average polynomials [na nc]
        steps: # of steps to simulate
        
        """

        sys_PSC = signal.TransferFunction(1, [self.tau_syn, 1])
        # sys_PSC = matlab.TransferFunction(1, [self.tau_syn, 1])
        ramp_ranges = np.array([[-0.1, 0.1], [1.25, 1.75]])
        RMSE = np.zeros((len(self.x_vals), 2))
        RMSP = np.zeros((len(self.x_vals), 2))
        
        raw_noise = self.noise_values_sim[0]
        fit_noise = np.linspace(raw_noise[0], raw_noise[-1], len(self.trange), endpoint=True)
        eval_noise = signal.lsim(sys_PSC, fit_noise, np.transpose(self.trange))
        freq_1, spike_PSD = self.getPSD(eval_noise, self.window_length)
        sys_ARMA_1 = self.fitARMAModel(eval_noise, orders)
        sys_state = sys_ARMA_1.to_ss()

        print "sys_state: {}".format(sys_state)
        # print "arma_model summary: {}".format(arma_model.summary())

        ran_points = np.array(self.genRandomPoints(eval_noise[0], distribution='gaussian'))
        trange_gaus = np.array(self.genRandomPoints(self.trange, distribution='gaussian'))
        # print "sys_psc: {}\nsys_ARMA: {}".format(sys_psc, sys_ARMA_1)
        # print "ran_points: {}".format(ran_points)
        # print "trange_gaus: {}".format(trange_gaus)
        # print "ran_points before lsim type: {}; shape: {}".format(type(ran_points), ran_points.shape)

        # TODO: State reduction for sys_ARMA for trying to make transfer funciton return 1 value not two
        # sys_ARMA_red = balred(sys_ARMA, orders=1)

        A = sys_state.A
        B = sys_state.B
        C = sys_state.C/sys_state.A
        D = 1/sys_state.A

        print "original sizes of a: {}, b: {}, c: {}, d: {}".format(sys_state.A.shape, sys_state.B.shape, sys_state.C.shape, sys_state.D.shape)

        C[C == np.inf] = 0
        D[D == np.inf] = 0
        C[C == -np.inf] = 0
        D[D == -np.inf] = 0

        print "a: {}".format(A)
        print "b: {}".format(B)
        print "c: {}".format(C)
        print "d: {}".format(D)

        # model_noise = signal.lsim([A, B, C, D], ran_points, self.trange)
        model_noise = signal.lsim(sys_ARMA_1, ran_points, self.trange)
        print "eval_noise len: {} model_noise len: {}".format(len(eval_noise), len(model_noise))
        # print "eval_noise: {}\n".format(eval_noise[-1])
        # print "model_noise: {}\n".format(model_noise[-1])
        for i in range(len(eval_noise)):
            print "fit_noise[{}] shape: {} model_noise[{}] shape: {}".format(i, eval_noise[i].shape, i, model_noise[i].shape)

        modeled_noise = np.empty([len(model_noise[0]), 3])
        print "modeled_noise: {}\n".format(modeled_noise[-1].shape)

        # TODO: Currently discarding one of the dimensions for the states; FIX THIS
        for i in range(len(eval_noise)):
            if i == (len(eval_noise)-1):
                print "model_noise[i]: {}".format(model_noise[i].transpose)
                print "shape: {}".format(model_noise[i].shape)
                modeled_noise[:,i] = model_noise[i].transpose()[-1]
            else:
                modeled_noise[:,i] = model_noise[i].squeeze()
        
        modeled_noise = modeled_noise.transpose()

        # print "modeled_noise[0]: {}".format(modeled_noise[0])
        # print "modeled_noise[1]: {}".format(modeled_noise[1])
        # print "modeled_noise[2]: {}".format(modeled_noise[2])

        freq_2, model_PSD = self.getPSD(modeled_noise, self.window_length)
        
        # print "freq_1: {}".format(freq_1)
        # print "freq_2: {}".format(freq_2)

        freq, sys_mag, phase = signal.bode(sys_ARMA_1, 2*np.pi*freq_2)
        sys_PSD = 2*(sys_mag.squeeze())**2*self.dt

        # RMSE[eval_point,ramp_range] = mean((spike_PSD - model_PSD)**2)**0.5

        # TODO: PSD values are off by magnitude of 2 and they have different dimensions. FIX THIS
        RMSE = np.mean((spike_PSD - sys_PSD)**2)**0.5 # more accurate
        print "sys_PSD: {}, model_PSD: {}".format(sys_PSD, model_PSD)
        print "RMSE: {}".format(RMSE)
        RMSP = np.mean(spike_PSD**2)**5

        return RMSE, RMSP

    def fitARMAModel(self, noise, orders):
        """
        Fit ARMA model to spike noise spectrum.

        orders: ARMA model orders 
        """
        if orders is None: orders = [2,2]
        freq, spike_PSD, min_PSD, max_PSD = self.getPSD(noise, self.window_length, return_minmax=True)
        # Before fitting the curve, assert the stabilty of the data
        noise_stable = self.assertStability(noise)
        time, data, x_out = noise_stable

        # TODO: Create a function that performs ARMA parameter optimization
        # Look into this link: http://machinelearningmastery.com/tune-arima-parameters-python/
        
        model = ARMA(data, orders);
        model_fit = model.fit()
        ma_roots = model_fit.maroots
        ar_roots = model_fit.arroots
        self.ARMA_model = model_fit

        ma_poly = (model_fit.sigma2**0.5) * polyfromroots(ma_roots)
        ar_poly = (model_fit.sigma2**0.5) * polyfromroots(ar_roots) 

        # print "ma_poly: {}".format(ma_poly)
        # print "ar_poly: {}".format(ar_poly)

        sys_1 = signal.TransferFunction(ma_poly, ar_poly)
        # print "sys_1: {}".format(sys_1)
        # sys_2 = matlab.TransferFunction(ma_poly, ar_poly)

        # sys = model_fit.NoiseVariance^.5*idpoly(M.A, M.C, 1, 1, 1, 0, M.Ts); # TODO: What does this line do?
        # A,B,C,D = signal.tf2ss(ma_poly, ar_poly)

        # eig_vals = linalg.eigvals(A)
        # print "eig_vals: {}".format(eig_vals)

        # return signal.tf2ss(ma_poly, ar_poly)
        # print "variance: {}".format(model_fit.sigma2)
        return sys_1

    def assertStability(self, noise, display_testresult=False):
        """
        Assert the stability of the ARMA data before performing model.fit using Dickey-Fuller test 
        if the model is found instable, a data transformation (first difference) would be performed
        until stabilization then would be returned

        Returns: stable data
        """
        time, data, x_out = noise
        table_dict = {'time':time, 'data':data}
        table = pd.DataFrame(table_dict)

        is_stable = False
        # data_vals = data.squeeze().values.tolist()
        ver_count = 0
        
        while(not is_stable):
            dftest = adfuller(table.data.values.squeeze().tolist())
            test_stat = abs(dftest[0])
            crit_vals = dftest[4]
            crit_val_5per = abs(crit_vals['5%'])

            if display_testresult:
                print 'Results of Dickey-Fuller Test:'
                dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
                for key,value in dftest[4].items():
                    dfoutput['Critical Value (%s)'%key] = value
                print dfoutput
                print "\n"

            if test_stat > crit_val_5per:
                is_stable = True
            else:
                table['first_difference'] = table.data - table.data.shift()
                table['first_difference'][0] = 0
                table['data ver_'+str(ver_count)] = table.data
                table['data'] = table['first_difference'].dropna(inplace=False)
                ver_count += 1

        noise_stable = (time, table.data.values.squeeze().tolist(), x_out)
        return noise_stable


    def getPSD(self, noise, wlen, return_minmax=False): # TODO: ask about the purpose of varagin if varagin is always 0
        """
        Calculates the power spectrum density of the noise based in the given parameters.
        Also returns the minima and maxima of the PSD if return_minmax is True

        """
        probability = 0.95
        PSD, freq = signal.welch(noise,fs=1/self.dt,window="hanning",nperseg=wlen/2,noverlap=wlen/4,nfft=wlen/2) #TODO: Investigate the behaviour of nperseg

        alfa = 1 - probability
        v = 2 * PSD
        self.df = v
        # c = chi2.ppf([(1 - alfa / 2), alfa / 2], v)
        # c = v / c
        conf_interval = chi2.interval(alfa, v)
        min_PSD = conf_interval[0]
        max_PSD = conf_interval[-1]
        # print "conf_interval: {}".format(conf_interval)

        if return_minmax:
            return freq, PSD, min_PSD, max_PSD

        else:
            return freq, PSD 

       