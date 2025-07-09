#################################################
# small program to solve in a periodic domain
# for wave equation or  inviscid burgers equation to demonstrate
# iterative implicit scheme
#
# u_t + a*u_x = 0 or
# u_t + u*u_x = 0
#
# in conservative form this is 
#
# du/dt + d/dx(0.5*u**2) = 0
#
# x=[0,1), u(0,x)=sin(2*pi*x)
# 
# BDF2 in time
#
#################################################
import numpy as np
import matplotlib.pyplot as plt
from burgersEquation import burgersEquation
from waveEquation import waveEquation
#======================================================
# spatial residual
#======================================================
def residual(eqn,u,i,ip1,im1,dx):
    iflux=eqn.halfPointFlux(u,i,ip1)
    res=(iflux[i]-iflux[im1])/dx
    return res
#======================================================
# Jacobian of the spatial residual
#======================================================
def jacobian(eqn,u,i,ip1,im1,dx,dtau,dt):
    #
    ljac,rjac = eqn.halfPointFluxDerivative(u,i,ip1)
    #
    # arrange to create L,D,U 
    # Jacobian = L+D+U 
    # sparse matrix form with only the first order terms
    # so only the diagonal and the first term to the left and
    # right of the diagonal in each row is non-zero
    #
    # generally the linearization should be such that
    #
    # residual(u+du) = residual(u) + L*du[im1]+D*du[i]+U*du[ip1]
    #
    # To derive this, consider the following equations
    #
    # half-point flux = flux(u_k, u_{k+1})
    # half-point flux derivative is such that
    #
    # flux(u_k+du_k,u_{k+1}+du_{k+1})-flux(u_k,u_{k+1}) = ljac*du_k + rjac*du_{k+1}
    #
    # residual = -(flux(u_k,u_{k+1})-flux(u_{k-1},u_k))/dx 
    # jacobian = d(residual)/d(u) = lim_{du->0} (residual(u+du)-residual(u))/du
    #
    # substituting in the above equation
    # and performing some algebra using the residual and flux eqn one can obtain
    #
    # kth row of Jacobian J_k = [L_k,D_k,U_k]
    # where
    #
    # D_k = (ljac_k - rjac_{k-1})/dx + 1.0/dt + 1.0/dtau
    # L_k = -ljac_{k-1}
    # U_k = rjac_{k}
    #
    N=u.shape[0]
    L=np.empty((N,))
    D=np.empty((N,))
    U=np.empty((N,))
    for k in range(N):
        kp1=(k+1)%N
        km1=(k-1)%N
        L[k]=-ljac[km1]/dx
        D[k]=(ljac[k]-rjac[km1])/dx+1.0/dt+1.0/dtau
        U[k]=rjac[k]/dx
    return L,D,U
#=================================================
# perturbation jacobian of the residual
# i.e. derivative of the spatial residual below
# w.r.t u
# this is just for testing the analytic jacobian
# but could be used for any equation if analytic
# derivative is not possible
#=================================================
def perturbJacobian(eqn,u,i,ip1,im1,dx):
    u1=u.copy()
    res0=residual(eqn,u1,i,ip1,im1,dx)
    eps=1e-12
    N=u.shape[0]
    L=np.empty((N,))
    D=np.empty((N,))
    U=np.empty((N,))
    for k in range(N):
        kp1=(k+1)%N
        km1=(k-1)%N
        u1[k]=u[k]+eps
        res1=residual(eqn,u1,i,ip1,im1,dx)
        D[k]=(res1[k]-res0[k])/eps
        L[kp1]=(res1[kp1]-res0[kp1])/eps
        U[km1]=(res1[km1]-res0[km1])/eps
        u1[k]-=eps
    return L,D,U
#
# main program
#
#eqn=burgersEquation()
# uncomment below to do wave equation
eqn=waveEquation()
N=100
dx=1./N
x=np.arange(0,1,dx)
# max CFL of 10 now
dt=10*dx
# initialize function
u=np.sin(2*np.pi*x)
# set some index space
i=np.arange(0,N)
ip1=(i+1)%N
im1=(i-1)%N
#
nsteps=10
nsubiter=5
# increase linear iterations
# to send the solution to be exact every step
# Note: this is not feasible in a real CFD code
nlineariter=10
#
# time levels n and n-1
#
un=u.copy()
unn=un.copy()
#
# discrete equation
#
# (u^{s+1}-u^s)/dtau + (3*u^{s+1}-4*u^{n}+u^{n-1})/(2*dt)+ spatialResidual(u^s)=0
#
# so in an iterative form
# when the term with dtau goes to zero, we have satisfied the PDE
# In iterative form this becomes at every time step
#
# for s=1:nsubiter 
#   solve A*du = B
#   where A = (1/dtau + 3/(2*dt) + spatialJacobian(u^s))
#    and  B = -spatialResidual(u^s)-(3*u^{s}-4u^{n}+u^{n-1})/(2*dt)
#   let A = L+D+U
#   the D term will then absorb the 1/dtau and 3/(2*dt) terms
#
# dtau is the pseudo time term, pseudo time CFL often determines
# what dtau is. At the moment the pseudo time (or dual time) CFL
# is 100, i.e. abs(u)*dtau/dx <= 100 
dtau=dt*100
#
# perform time stepping
#
du0=np.zeros((N,))
for t in range(nsteps):
    print("timestep:%d"%t)
    # non-linear sub-iterations
    for s in range(nsubiter):
        if t > 0:
          # BDF2
          # unsteady residual = -d/dx(0.5*u**2)-(3*u^{n+1}-4*u^{n}+u^{n-1})/(2*dt)
          B=-(residual(eqn,u,i,ip1,im1,dx)+0.5*(3*u-4*un+unn)/dt)
          fac=1.5
        else:
          # BDF1
          # unsteady residual = -d/dx(0.5*u**2)-(u^{n}-u^{n-1})/dt)
          B=-(residual(eqn,u,i,ip1,im1,dx)+(u-un)/dt)
          fac=1.0
        print("\tsubitertion %d residual:%e"%(s,np.linalg.norm(B)))
        L,D,U=jacobian(eqn,u,i,ip1,im1,dx,dtau,dt/fac)
        #
        # linear iterations
        # maybe a while loop here to a certain convergence tolerance
        #
        if s==0:
          du=du0
        else:
          du[:]=0
        #print("\t\tlinear-iter %d linres:%e"%(0,np.linalg.norm(B-L*du[im1]-D*du[i]-U*du[ip1])))
        for l in range(nlineariter):
            # Gauss-Seidel iterations
            for m in range(N):
                mp1=(m+1)%N
                mm1=(m-1)%N
                du[m]=(B[m]-L[m]*du[mm1]-U[m]*du[mp1])/D[m]
            print("\t\tlinear-iter %d linres:%e"%(l,np.linalg.norm(B-L*du[im1]-D*du[i]-U*du[ip1])))
        print("\t\tnorm(du)=%e"%np.linalg.norm(du))
        u+=du
        #
        # TODO : train input(B) => output (du)
        #
    # update in time unn=>un, un=> u to prepare
    # for the next step
    du0=du.copy()
    unn[:]=un
    un[:]=u
    plt.plot(x,u)
plt.show()
