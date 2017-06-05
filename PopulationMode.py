import numpy as np
import matplotlib.pyplot as plot
import nengo
import math

class PopulationModeModel(object):    
    def __init__(self, population):
        self.pop = population # default LIF Model Population 
        self.radii = population.max_rates # max amplitude in each dimension
        self.dim = population.dimensions # dimension of the population 
        # a vector that speicfies the origin to which each bias element belongs
        self.originIndices = self.makeOriginIndices() 

        # bias grid
        self.x = []
        self.y = []
        self.z = []
        self.r = [] # For cases higher than 3D
        self.biasValues = [] # len(originIndices) * bias grid
        self.biasPoints = [] # points from which to interpolate bias

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
        generates and returns a list of origin numbers to which each noise value belongs


        """
        nTotalDim = self.dim
        originIndices = np.zeros((1, nTotalDim))
        print "originIndices.shape: {}".format(originIndices.shape)
        nOrigin = 1 # TODO: Discard the assumption that there to be only one origin
        print "len(originIndices): {}".format(len(originIndices))
        for i in len(2):
            originIndices[0,i] = self.dim
        return originIndices

    def makeVector(radius, nPoints):
        """
        Creates vector based on the number of points and the radius of 
        the neural population

        radius:
        nPoints:
        """
        result = np.arange(2*radius/(nPoints), radius)

    ###########################
    # CREATE HELPER FUNCTIONS #
    ###########################

    def createBiasModel(self):
        """
        Creates bias model for estimating the bias term of the surrogate model

        biasPoints:
        """
        a = 3 # bias is modelled over a*[-radius radius], this case, [-3 3]
        self.biasPoints = np.linspace(-3,3,6)
        
        if self.dim == 1: # linear inperpolation
            self.x = self.biasPoints
            self.biasValues = self.getBiasSamples()
        elif self.dim == 2: # bilinear interpolation
            nPoints = 101
            self.x = makeVector(a*self.radii[1], numPoints)
            self.y = makeVector(a*self.radii[2], numPoints)
            grid = np.meshgrid(self.x,self.y)

            bv = self.getBiasSamples(np.array(reshape(np.transpose(x),1,
                    nPoints**2), reshape(np.transpose(y), 1, np^2)))
            self.biasValues = np.reshape(bv, np.arange(len(self.originIndices), 
                    nPoints, nPoints))
        elif self.dim == 3: # trilinear interpolation
            nPoints = 41
            x = makeVector(a*self.radii[1], nPoints)
            y = makeVector(a*self.radii[2], nPoints)
            z = makeVector(a*self.radii[3], nPoints)
            grid = np.meshgrid(x, y, z)
            self.x = np.permute(x, [2,1,3])
            self.y = np.permute(y, [2,1,3])
            self.z = np.permute(z, [2,1,3])

            bv = self.getBiasSamples(np.array(np.reshape(x, 1, numPoints**3), 
                    np.reshape(y, 1, np**3), np.reshape(z, 1, np**3)))
            self.biasValues = np.reshape(bv, np.array(self.originIndices, 
                                nPoints, nPoints, nPoints))
        else: # uses exploited symmetry in the bias error
            radius = a*self.radii[1]
            self.r = np.linspace(0,radius,(radius/200))
            self.biasValues = self.getBiasSamples(np.array(self.r, np.zeros(self.dim-1, len(self.r))))

    def createNoiseModel(self):
        """
        Create noise model for estimating the noise term of the surrogate model;
        returns the noise generating node

        """
        T = 2
        dt = self.dt
        freqTh = max(2*math.pi*(np.linspace(0,(1/dt/2),1/T)))
        time = np.arange(dt, T+dt, dt)
        noiseInp = nengo.Node(WhiteSignal(T, )) 

        return 

    # 	ndim = len(self.originIndices) # total # of dimensions across origins
    # 	dt = self.dt;
    # 	T = 2;
    # 	time = np.arange(dt, T+dt, dt)

    # 	n = 3

    # 	rho = np.zeros(ndim, ndim) # correlations
    # 	mags = np.zeros(ndim, len(freq), 2) # fourier magnitudes
    # 	points = np.repmat(np.zeros(size(self.radii)), 1, 2) 
    # 	points[1,2] = 2*self.radii(1) # a point at 2*radius
    # 	self.sds = np.zeros(ndim, 2) # standard deviations of the noise

    # 	for i in range(length(self.radii)):
    # 		noiseMatrix = self.getNoiseSamples(points[:,i], dt, T)
    # 		self.sds[:,i] = np.std(noiseMatrix, [], 2) # need to convert the params somehow....

    # 		if i == 1:
    # 			rho = np.corrcoef(noiseMatrix.transpose())

    # 		for j in range(ndim):
    # 			f = np.fft(noiseMatrix[j,:] / len(time)*2 / math.pi**0.5)
    # 			mags[j,:,i] = np.abs(f[1:len(freq)])

    # 	self.noiseCorr = rho

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

    def getBias(self, state):
        """
        returns bias based on the population model's state

        state: Population state vector
        bias: a vector of bias values (a static func of the state that is 
                encoded by the population)
        """
        bias = np.zeros(length(self.originIndices))
        dim = len(state)
        if dim == 1:
            xInd = np.where(self.x, state[0])
        elif dim == 2:
            xInd = np.where(self.x, state[0])
            yInd = np.where(self.y, state[1])
            bias = self.biasValues[:,xInd,yInd,zInd]
        elif dim == 3:
            xInd = np.where(self.x, state[0])
            yInd = np.where(self.y, state[1])
            zInd = np.where(self.z, state[2])
            bias = self.biasValues[:,xInd,yInd,zInd]
        else:
            rInd = np.where(self.r, np.norm(state))
            bias = self.biasValues[:,rInd]

        return bias

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

    def getBiasSamples(self):
        """
        Obtains samples of bias (distortion) error which can then be fit to a 
        model. Returns an array of bias errors at each eval points for each 
        origin (bias) and an ideal values (ideal). Note that actual=ideal+bias

        originInd: a vector the same length as the sum of dimensions of all 
        origins, that speicfies the origin to which each bias element belongs.
        evalPoints: population states at which bias is sampled(dim x #points)
        """
        evalPoints = self.x
        rates = self.getRates(self.pop, evalPoints, 0,0)
        bias = np.zeros(len(self.originIndices), size(evalPoints,2))
        ideal = np.zeros(size(bias))
        for i in range(1,len(self.pop.origins)):
            origin = self.pop.origins[i]
            ind = np.where(self.originIndices == i)
            # ideal[ind,:] = # origin.f(evalPoints)
            actual = origin.decoder * rates
            bias[ind,:] = actual - ideal[ind,:]

    def getNoiseSamples(self, evalPoint, dt, T):
        """
        Obtains samples of time-varying noise that specifies the origin to which
        each bias element belongs. Each element is the index of an origin. 
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
            evalPoint = np.zeros(size(p.radii))

        time= np.arange(dt,T+dt, dt)
        noise = np.zeros(len(self.originIndices), len(time))

        drive = getDrive(p, evalPoint)
        reset(p) #TODO: fix this
        for i in range(len(time)):
            activity = run(p.spikeGenerator, drive, time[i]-dt, time[i], 1) #TODO: FIX THIS CUZ NEF HAS ITS OWN ENSEMBLE
            for j in range(len(p.origins)):
                setActivity(p.origins[j], time[i], activity) #TODO
                noise[self.originIndices == j,i] = getOutput(p.origins[j])

        [bias, ideal] = self.getBiasSamples(evalPoint)
        noise = noise - np.repmat(ideal + bias, 1, len(time))



































