# Non-linear Bayesian Inverse Problems- Schrödinger model
Non-linear Bayesian Inference application to  the Schrödinger model in 1D.
## Problem description
Consider the  Schrödinger equation defined on $\mathcal{X}=[0,1]$ with boundary values $u(0)=u(1)=1$: <br>
$$\begin{cases}
\mathcal{L}_{f}u=\frac{\Delta u(x)}{2}-fu(x)=\frac{1}{2}\frac{\partial^2 u(x)}{\partial x^2}-fu(x)=0 &\text{ on } \mathcal{X} \\
u(x)=1 &\text{ on } \partial\mathcal{X} 
\end{cases},$$
that in the  one dimensional case becomes a second order ODE. <br>
The aim of the project is that of deriving through the pCN algorithm (MCMC) a Bayesian estimate of the parameter $f:\mathcal{X}\to\mathbb{R}$. The parameter is defined such that $f(x)=f(\theta(x))=e^{\theta(x)}$, where the function $\theta$ is then the actual parameter. We assume on it a GP prior with Matern covariance kernel and the observations of the solution $u$, generated by perturbing a reference solution, are assumed $i.i.d$. <br>
The reference solution, considering a discretization of $[0,1]$ of $N=100$ points, is obtained by solving the ODE using the finite difference method by which: $u^{''}(t_i)=\frac{u(t_{i+1})-2u(t_i)+u(t_{i-1})}{\Delta t^2} \ i\in\{1,\dots, N\}$. <br>
The same discretization scheme ($N=100$) is used to discretize the Gaussian Process prior. <br>
In the numerical experiments different assumptions on the smoothness of $f_{\theta_0}$ have been considered, thus different values for the parameter $\nu$ of the Matern covariance kernel. The considered values are: 2.5,3.5. Finally, the case $\nu\to\infty$, is considered as well.
## Usage
The file "numexpsnonlinearBI.ipynb" contains the numerical experiments, while "Schroedinger1D" the implementation of the model.
