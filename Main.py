#!/usr/bin/python3

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# covariance can not be reassigned for stats.multivariate object


class GaussianMixture:
    def __init__(self, data, n_gaus):
        self.data = data

        # Nu of gaussians to be fitted
        self.n_gauss = n_gaus

        # Coffiecients of gaussians
        self.mix = np.repeat(1/n_gaus, n_gaus)

        # Number and dimensions of samples
        self.N, self.D = data.shape

        # List of gaussians
        self.gaussian = []

        # Responsibilties of each gaussian
        self.resp = np.zeros((self.N, self.n_gauss))

        # Initializing gaussians with different mean values 
        while True:
            start = np.random.randint(self.N, size=self.n_gauss)
            start = np.unique(start)
            if start.shape[0] == self.n_gauss:
                break
            else:
                pass

        for i in range(n_gaus):
            self.gaussian.append(stats.multivariate_normal(data[start[i]], np.eye(self.D)))

        print('Started with')
        print('pi values : {}'.format(self.mix))

    # Expectation function
    def eStep(self):
        # Finding out responsibilities for each gaussian
        for i in range(self.n_gauss):
            self.resp[:, i] = self.mix[i]*self.gaussian[i].pdf(self.data)

        # Normalizing responsibilities
        div = np.sum(self.resp, axis=1)
        self.resp = np.divide(self.resp, div[:, None])
        where_are_NaNs = np.isnan(self.resp)
        self.resp[where_are_NaNs] = 1/self.n_gauss

    # Maximization function
    def mStep(self):
        # Re-estimating gaussian parameters based on current responsibilities
        for i in range(self.n_gauss):
            Nk = np.sum(self.resp[:, i])
            mu = 1/Nk*np.dot(self.resp[:, i], self.data)
            x_sub_mu = self.data - np.tile(mu, (self.N, 1))
            cov = 1/Nk*sum([resp*np.dot(samp[:, None], samp[None, :]) for
                (resp, samp) in zip(self.resp[:, i], x_sub_mu)])
            self.gaussian[i] = stats.multivariate_normal(mean=mu, cov=cov)
            self.mix[i] = Nk/self.N

    # Iteration function
    def iteration(self):
        self.eStep()
        self.mStep()

    # Combined probability
    def pdf(self, num):
        res = 0
        for i in range(self.n_gauss):
            res += self.mix[i]*self.gaussian[i].pdf(num)
        return res

    def result(self):
        print('Finished With')
        print('pi values : {}'.format(self.mix))


# Loading old faithful geyser dataset for fitting GMM 
# http://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat
Data = np.loadtxt('faithful.txt')

# Intializing GMM and asking to fit two gaussians
a = GaussianMixture(Data, 2)

# Performing EM iterations 
for vin in range(25):
    a.iteration()

a.result()

# Plotting the result for this specific simulation
# The plots will not work for data with dimensions
# other than two

N = 100
X = np.linspace(1, 5.5, N)
Y = np.linspace(40, 100, N)
X, Y = np.meshgrid(X, Y)

pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

Z = a.pdf(pos)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
plt.plot(Data[:, 0], Data[:, 1], 'go')
plt.xlabel('Eruption time')
plt.ylabel('Waiting time')

plt.show()
