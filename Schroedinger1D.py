import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import scipy
class Schroedinger1D:
      def __init__(self, N: int, nu_: float, noisestd: float):
          self.N=N
          self.xx=np.linspace(0,1,self.N)[:, None]  #discretization points
          #here the prior is a GP with mean 0 and Matern covariance kernel
          if(nu_=='inf'):
            self.kernel=RBF()
          else:
            self.kernel=Matern(nu=nu_) 
          self.f_ref=self.set_fref()
          self.u_ref=self.solver(self.f_ref)
          self.noise_std=noisestd
          self.obs=self.generate_noisy_observations(self.f_ref)
          
      def set_fref(self)-> np.array:
          return np.exp(self.sample_from_prior(gen_reference=True))
      def solver(self, f:np.array, b1: float=1, bN: float=1)-> list:
        """use finite difference method to solve the second order ODE with boundary values u(0)=1, u(1)=1"""
        delta_t=1/self.N
        #U=np.zeros((self.N-2,self.N-2))
        U = -2*np.eye(self.N-2) *(1+f[1:self.N-1]*delta_t**2)
        U += np.diag(np.ones(self.N-3), -1)
        U += np.diag(np.ones(self.N-3), 1)
        b=np.zeros(self.N-2)
        b[0]=-b1
        b[-1]=-bN
        u=scipy.linalg.solve(U,b)
        #add boundary values
        u=list(u)
        u.insert(0,b1)
        u.insert(len(u),bN)
        return u 
    
      def generate_noisy_observations(self, f: np.array):
        u=self.solver(f)
        y= u+ np.random.normal(0,self.noise_std, size=len(u))
        return y
    
      def sample_from_prior(self, gen_reference=False)-> np.array:
        if(gen_reference):
          np.random.seed(1)
        cov_N=self.kernel(self.xx)
        mean=np.zeros(self.xx.shape[0])
        theta=np.random.multivariate_normal(mean, cov_N, size=1)[0]
        norm_const=(self.N**(1/(12+8+2)))**(-1) #Given nu the parameter of the Matern covariance, the corresponding RKHS is the sobolev space nu+d/2, where here d=1. Thus alpha=3, while k=2. 
        return norm_const*theta
    
      def likelihood(self, theta: np.array)->float:
          """function that computes the likelihood"""
          return (1/(2*np.pi*self.noise_std))*np.exp(self.phi(theta))
      
      def phi(self,theta: np.array)-> float:
        """function that computes the log likelihood"""
        f_theta=np.exp(theta)
        u_theta=self.solver(f_theta)
        phi=-np.sum((self.obs-u_theta)**2)*1/(2*self.noise_std**2)
        return phi
    
    