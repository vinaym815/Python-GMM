from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# stats.multivariate gaussian covariance can not be
# reassigned

class GaussianMixture:
    def __init__(self, data, n_gaus):
        self.data = data
        self.n_gauss = n_gaus                                          # Nu of gaussians to be fitted
        self.mix = np.repeat(1/n_gaus, n_gaus)                           # Linear coffiecients of gaussians
        self.N, self.D = data.shape                                     # Number and dimensions of samples
        self.gaussian = []                                              # List of gaussians
        self.resp = np.zeros((self.N, self.n_gauss))                    # Responsibilties of each gaussian

        # making sure that the gaussians are initialized with different means

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

    def eStep(self):
        for i in range(self.n_gauss):
            self.resp[:, i] = self.mix[i]*self.gaussian[i].pdf(self.data)

        div = np.sum(self.resp, axis=1)
        self.resp = np.divide(self.resp, div[:, None])
        where_are_NaNs = np.isnan(self.resp)
        self.resp[where_are_NaNs] = 1/self.n_gauss

    def mStep(self):
        for i in range(self.n_gauss):
            Nk = np.sum(self.resp[:, i])
            mu = 1/Nk*np.dot(self.resp[:, i], self.data)
            x_sub_mu = self.data - np.tile(mu, (self.N, 1))
            cov = 1/Nk*sum([resp*np.dot(samp[:, None], samp[None, :]) for
                (resp, samp) in zip(self.resp[:, i], x_sub_mu)])
            self.gaussian[i] = stats.multivariate_normal(mean=mu, cov=cov)
            self.mix[i] = Nk/self.N

    def iteration(self):
        self.eStep()
        self.mStep()

    def pdf(self, num):
        res = 0
        for i in range(self.n_gauss):
            res += self.mix[i]*self.gaussian[i].pdf(num)
        return res

    def result(self):
        print('Finished With')
        print('pi values : {}'.format(self.mix))
#        print(self.gaussian[0].mean,self.gaussian[1].mean)
#        print(self.gaussian[0].cov,self.gaussian[1].cov)


Data = np.loadtxt('faithful1.txt')

a = GaussianMixture(Data, 2)
for vin in range(25):
    a.iteration()

a.result()

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
plt.show()
