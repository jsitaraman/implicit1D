import numpy as np
#
# discrete spatial flux for Burger's equation
# i.e. the 
# d(0.5*u**2)/dx term
#
class burgersEquation:
    def __init__(self):
        self.diss=0.5
        pass
    #=================================================
    # flux(u) =0.5*u**2
    #=================================================
    def _flux(self,u):
       return 0.5*(u**2)
    def _dflux(self,u):
       return u
    #=================================================
    # half point conservative flux (Lax-Friedrichs)
    # (flux_l + flux_r)*0.5 - abs(u_half)*(u_r - u_l)
    #=================================================
    def halfPointFlux(self,u,i,ip1):
       iflux = 0.5*((self._flux(u[i])+self._flux(u[ip1]))-
            self.diss*np.abs(u[i]+u[ip1])*(u[ip1]-u[i]))
       return iflux
    #=================================================
    # derivative of inviscid Flux
    #=================================================
    def halfPointFluxDerivative(self,u,i,ip1):
       uhalf=(u[i]+u[ip1])
       udiff=(u[ip1]-u[i])
       sgn = ((uhalf >=0)*2-1)*self.diss
       ljac = 0.5*(self._dflux(u[i]) +self.diss*np.abs(uhalf)-sgn*udiff)
       rjac = 0.5*(self._dflux(u[ip1])-self.diss*np.abs(uhalf)-sgn*udiff)
       return ljac,rjac
    #=================================================
    # derivative of inviscid Flux
    #=================================================
    def checkDerivative(self):
        u=[0.5,1.5]
        i=0
        ip1=1
        r=self.halfPointFlux(u,i,ip1)
        du=np.random.rand(2)*1e-8
        r1=self.halfPointFlux(u+du,i,ip1)
        print(r1-r)
        ljac,rjac=self.halfPointFluxDerivative(u,i,ip1)
        print(ljac*du[0]+rjac*du[1])


#bg=burgersEquation()
#bg.checkDerivative()
