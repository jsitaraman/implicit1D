import numpy as np
#
# discrete spatial flux for the wave equation equation
# i.e. the 
# d(a*u)/dx term
#
class waveEquation:
    def __init__(self):
        self.a=1
        pass
    #=================================================
    # flux(u) =0.5*u**2
    #=================================================
    def _flux(self,u):
       return self.a*u
    def _dflux(self,u):
       return self.a*np.ones_like(u)
    #=================================================
    # half point conservative flux (Lax-Friedrichs)
    # (flux_l + flux_r)*0.5 - abs(u_half)*(u_r - u_l)
    #=================================================
    def halfPointFlux(self,u,i,ip1):
       iflux = 0.5*((self._flux(u[i])+self._flux(u[ip1]))-
            np.abs(self.a)*(u[ip1]-u[i]))
       return iflux
    #=================================================
    # derivative of inviscid Flux
    #=================================================
    def halfPointFluxDerivative(self,u,i,ip1):
       ljac = 0.5*(self._dflux(u[i]) +np.abs(self.a))
       rjac = 0.5*(self._dflux(u[ip1])-np.abs(self.a))
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


#bg=waveEquation()
#bg.checkDerivative()
