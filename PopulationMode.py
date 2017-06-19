import numpy as np
import nengo
import math
from scipy.interpolate import interp1d, interp2d
class PopulationModeModel(object):    
    def __init__(self, params):
        """
        within params...

        population: desired ensemble that would be represented in this mode
        rates: firing rates of the neurons within the population
        decoders: decoding vectors of the neurons within the population
        """
        self.pop = params['population'] # default LIF Model Population
        self.origins = params['origins'] # source of the ideal output
        self.radii = self.pop.radius # max amplitude in each dimension
        self.dim = self.pop.dimensions # dimension of the population 
        # a vector that speicfies the origin to which each bias element belongs
        self.origin_indices = self.makeOriginIndices()
        self.trange = params['timerange'] # simulation time range
        self.dt = params['dt'] # simulation time step
        
        self.__bias_points = params['evalpoints'] # x; points from which to interpolate bias
        self.__rates = params['firingrates'] # f_hat(x) of the population
        self.__decoders = params['decoders']
        self.__drive = params['drive']
        self.__voltage = params['voltage_post']

        # Params for bias model
        self.r = [] # For cases higher than 3D
        self.bias_values_sim = []
        self.bias_values_est = [] # Estimated bias term
        self.ideal_values = []
        self.actual_values = []

        # Params for noise model
        self.noise_values_sim = []
        self.noise_values_est = []
        self.noiseOriginInd = []
        self.noiseCorr = []

        self.nfX = [] # state of noise filter
        self.nfU = [] # input from prev time step 
        self.nfA = [] # dynamics matrix for model of noise frequency dependence
        self.nfB = [] # input matrix
        self.nfC = [] # output matrix
        self.nfD = [] # passtrhough matrix
        self.sds = [] # standard deviations of noise

        self.noiseTime = []
        self.noiseSamples = []

    #########################
    # MAKE HELPER FUNCITONS #
    #########################

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

        return origin_indices


    ###########################
    # CREATE HELPER FUNCTIONS #
    ###########################

    def createBiasModel(self):
        """
        Creates bias model for estimating the bias term of the surrogate model.
        
        Steps:
        1. Perform interpolation based on the dimension
        2. Find bias values of the x_intercepts (evalpoints)
        3. Estimate the bias based on the the bias found from Step 2
        3. Return the bias values

        """
        a = 3 # bias is modelled over a*[-radius radius], this case, [-3 3]

        if self.dim < 4:
            self.getBiasSamples()
        
        # TODO: Finish implementing for multidimensional case
        else: # uses exploited symmetry in the bias error
            radius = a*self.radii[1]
            self.r = np.linspace(0,radius,(radius/200))
            self.bias_values = self.getBiasSamples(np.array([self.r, 
                                np.zeros(self.dim-1, len(self.r))]))

    def createNoiseModel(self):
        """
        Create noise model for estimating the noise term of the surrogate model

        Steps:
        1. Come up with forier transform parameters
        2. Calculate the noise based on actual values and bias values (noise = (est - actual) - bias)
        3.

        """
        n_totaldim = len(self.origin_indices) # total dim of all origins
        time = self.trange
        num_points = 3
        sim_time = time[-1] # simulated time
        eval_points = self.__bias_points

        # Need bias to calculate noise
        if len(self.bias_values_est) == 0 or len(self.bias_values_sim) == 0:
            self.getBiasSamples()

        actual = self.actual_values
        ideal = self.ideal_values
        noise = (ideal-actual)-self.bias_values_sim
        self.noise_values_sim = noise
        print "noise: {}".format(noise)
        
        freq = 2*math.pi*np.linspace(0,(0.5/self.dt-1/sim_time),1/sim_time, endpoint=True)
        rho = np.zeros((n_totaldim, n_totaldim)) # correlations

        # Use LPF to normalize the noise
    	mags = np.zeros((n_totaldim, len(freq), self.dim)) # fourier magnitudes
    	points = np.matlib.repmat(np.zeros(eval_points.shape), 1, 2) # TODO: Figureout why these parameters are used
    	points[0,1] = 2*eval_points[0] # a point at 2*radius
    	self.sds = np.zeros((n_totaldim, 2)) # standard deviations of the noise; why are we using 2 here.... (# of dimensions perhaps?)

    	for i in range(self.dim):
    		self.sds[:,i] = np.std(noise, [], 2) # need to convert the params somehow....

    		if i == 1:
    			rho = np.corrcoef(np.transpose(noise))

    		for j in range(n_totaldim):
    			f = np.fft(noise[j,:]) / len(time) * 2 / math.pi**0.5
    			mags[j,:,i] = np.abs(f[1:len(freq)])

    	self.noiseCorr = rho

    # 	# Set up a linear system to filter random noise for noise with realistic spectrum
    # 	self.nfU = np.zeros(ndim, 1)
    # 	self.nfX = np.empty([1,2])
    # 	self.nfX[0] = np.zeros(2*ndim, 1)
    # 	self.nfA[0] = np.zeros(2*ndim, 2*ndim)
    # 	self.nfB[0] = np.zeros(2*ndim, ndim)
    # 	self.nfC[0] = np.zeros(ndim, 2*ndim)
    # 	self.nfD[0] = np.zeros(ndim, ndim)
    # 	self.nfX[1] = np.zeros(2*ndim, 1)
    # 	self.nfA[1] = np.zeros(2*ndim, 2*ndim)
    # 	self.nfB[1] = np.zeros(2*ndim, ndim)
    # 	self.nfC[1] = np.zeros(ndim, 2*ndim)
    # 	self.nfD[1] = np.zeros(ndim, ndim)

    # 	# Find filter parameter for each output
    # 	for i in range(ndim):
    # 		for filterType in range(2):
    # 			sys = self.fitTF2(freq, np.squeeze(mags[i,:,j]))
    # 			self.setNoiseFilter(filterType, sys, i, dt)

    # def setNoiseFilter(self, cp, sys, i, dt):
    # 	""" 
    # 	Pass the noise term in to low-pass filter with time constants of 
    # 	typical synaptic current dynamics. Such filter would make the noise term
    # 	to have gaussian distribution

    # 	c: set 1 for central filter (near 0),2 for peripheral filter (radius x2)
    # 	sys: analog transfer func of filter
    # 	i: index of output dimension across all origins
    # 	"""
    # 	assert cp in range(2), "Wrong filter type!"
    # 	[num, den, dt] = np.cont2discrete(sys, dt)
    # 	[a, b, c, d] = np.tf2ss(num, den)
    # 	ii = 2*(i-1)+[1,2] # What is this math....?
    # 	self.nfA[cp][ii,ii] = self.a
    # 	self.nfB[cp][ii,i] = self.b
    # 	self.nfC[cp][i,ii] = self.c		
    # 	self.nfD[cp][i,i] = self.d

    def getBiasSamples(self):
        """
        Obtains samples of bias (distortion) error which can then be fit to a 
        model. Returns an array of bias errors at each eval points for each 
        origin and an ideal values (ideal). (Note: actual = ideal + bias)

        Steps:
        1. Create empty array for bias based on the number of evalPoints
        2. For every evaluation pointss, calculate bias based on the simulation
        3. Estimate bias using the simulated bias term; interpolate
        """
        eval_points = self.__bias_points
        bias = np.zeros(eval_points.T.shape)
        self.calcNeuralActivities()
        actual = self.actual_values
        ideal = self.ideal_values

        for i in range(len(self.origins)):
            origin = self.origins[i]
            ind = np.where(self.origin_indices == i)
            bias[ind,:] = actual[ind,:] - ideal[ind,:]

        self.bias_values_sim = bias

        if self.dim == 1:
            points = np.squeeze(eval_points)
            bias_vals = np.squeeze(bias)
            func = interp1d(points, bias)

            new_points = self.genRandomPoints(points)
            # print "points: {}".format(points)
            # print"new_points: {}".format(new_points)
            self.bias_values_est = func(new_points)[0]
        # elif self.dim == 2:
        #     self.bias_values_est = interp2d()

    def getNoise(self, time):
        """
        returns noise, a random vairable with spatial and temporal correlations

        time: end of simulation time step
        noise: vector of noise values(a random variable with spatial and 
                temporal correlations)
        """
        noiseInd = []
        if self.noiseTime: # noiseTime is not empty
            noiseInd = round((time - self.noiseTime(1)) / self.dt)
            if noiseInd < 1 or noiseInd > len(self.noiseTime):
                noiseInd = []
        if not self.noiseInd: # noistInd is empty
            nSteps = 1000
            self.noiseTime = np.linspace(time,(time+nSteps*self.dt),self.dt)
            self.noiseSamples = self.generateNoise(nSteps)
            noiseInd = 1

        noise = np.zeros(size(self.noiseSamples, 1),2)
        noise[:,:] = self.noiseSamples[:,noiseInd,:]

        return noise

    def generateNoise(self, nSteps):
        """
        generate and returns noise, the correlated, filtered noise samples

        nSteps: # of noise samples to generate (each sample is a vector where 
                each element corresponds to a certain dimension of a certain 
                origin)
        """
        scale = (1/self.dt)**0.5
        # unfiltered = [self.nfU scale *  ] # originally [pmm.nfU scale * PoissonSpikeGenerator.randncov(nSteps, pmm.noiseCorr)]; but since poissonSpikeGenerator is not the concern, change this later

        noise = np.zeros(size(unfiltered, 1), nSteps, 2)

        for cp in range(2):
            x = self.nfX[cp]
            A = self.nfA[cp]
            B = self.nfB[cp]
            C = self.nfC[cp]
            D = self.nfD[cp]

            # TODO: Find a better filtering mechanism
            for i in range(nSteps):
                x = A*x + B*unfiltered[:,i]
                noise[:,i,cp] = C*x + D*unfiltered[:, i+1]
            self.nfX[cp] = x

            gsds = np.std(noise[:,:,cp], [], 2)
            gain = self.sds[:,cp] / gsds
            for i in range(1,size(noise[:,1])):
                noise[i,:,cp] = noise[i,:,cp] * gain[i]

        self.nfU = unfiltered[:,-1]

        return noise

    def getNoiseSamples(self, dt, T):
        """
        Obtains samples of time-varying noise.
        Returns noise, a cell array of noise samples per origin 

        originInd: a vector with same length as the sum of dimensions of all
        origins. Specifies the origin to which each bias element belongs.
        evalPoint: a populaiton state at which the noise is to be sampled for 
        some period of time
        dt: time step of simulation for collecting samples (in seconds)
        T: duration of simulation period (in seconds)

        """
        p = self.pop

        if not evalPoint:
            evalPoint = np.zeros(self.__bias_points)

        time = self.trange
        sim_time = time[-1] # simulated time
        eval_points = self.__bias_points
        noise = np.zeros((len(self.originIndices), len(time)))

        drive = self.__drive
        for i in range(len(time)):
            activity = run(p.spikeGenerator, drive, time[i]-dt, time[i], 1) 
            for j in range(len(p.origins)):
                setActivity(p.origins[j], time[i], activity) #TODO
                noise[self.originIndices == j,i] = getOutput(p.origins[j])

        [bias, ideal] = self.getBiasSamples(evalPoint)
        noise = noise - np.matlib.repmat(ideal + bias, 1, len(time))

    def genRandomPoints(self, orig_points):
        """
        generates a new set of points based on the original set of points; assume that you want same number of points as
        original points

        """
        # directions = np.random.randn(self.dim, len(orig_points))
        # directions = np.sort(directions)
        # print "directions: {}".format(directions)
        # mag_direct = (np.sum(directions**2, 1))**0.5
        # mag_orig = (np.sum(orig_points**2, 0))**0.5
        # print "mag_direct: {}".format(mag_direct)
        # print "mag_orig: {}".format(mag_orig)
        # unit_points = directions / mag_direct * orig_points
        # print "unit_points: {}".format(unit_points)
        # # points = unit_points * (np.ones(orig_points.shape) * np.random.uniform(1,
        # #             len(orig_points))**(1/self.dim))
        # points = unit_points * orig_points
        # print "points: {}".format(points)

        # offsets = np.zeros(orig_points.shape)
        # points = offsets * np.ones(orig_points.shape) + points
        # dt = abs(orig_points[0] - orig_points[-1]) / len(orig_points) 
        points = np.linspace(orig_points[0], orig_points[-1], 2*len(orig_points), endpoint=True)
        new_points = points[1::2]

        # print "orig_points: {}".format(orig_points)
        # print "new_points: {}".format(new_points)
        return new_points

    def calcNeuralActivities(self):
        """
        Calculates the neural activity by performing dot product in between
        the decoding vector and the rates.
        """
        eval_points = self.__bias_points
        bias = np.zeros(eval_points.T.shape)
        decoders_pre = self.__decoders[0]
        decoders_post = self.__decoders[1]
        rates_pre = self.__rates[0]
        rates_post = self.__rates[1]

        actual = np.tensordot(decoders_post,rates_post, axes=([-1],[0]))
        ideal = np.tensordot(decoders_pre, rates_pre, axes=([-1],[0]))

        self.actual_values = actual
        self.ideal_values = ideal
        