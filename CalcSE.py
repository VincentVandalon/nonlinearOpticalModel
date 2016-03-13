#Copyright 2012 Vincent Vandalon
#
#This file is part of <++>. <++> is free software: you can redistribute
#it and/or modify it under the terms of the GNU General Public License
#as published by the Free Software Foundation, either version 3 of the
#License, or (at your option) any later version.
#
#<++> is distributed in the hope that it will be useful, but WITHOUT
#ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
#License for more details. You should have received a copy of
#the GNU General Public License along with <++>. If not, see
#<http://www.gnu.org/licenses/>. "

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from OpticalModels.Stack import *
from OpticalModels.Layer import *
from OpticalModels.cachingMethods import *
from OpticalModels.opticalModels import *


airLayer=Layer(n_vac,10E-3)
siLayer=Layer(SiliconJellison.N,5E-9)
al2o3Layer=Layer(Al2O3.N,125E-9)
SESIO2=Layer(SiOSell.N,125E-9)

########### Build structure ###############
airToSubstrateSystem=Stack(airLayer,siLayer)
#airToSubstrateSystem.addLayer(al2o3Layer) #Top layer
#airToSubstrateSystem.addLayer(ALDSiO) #Top layer
airToSubstrateSystem.addLayer(SESIO2) #Top layer
def constrainBetween(x,disc=2):
   return x%disc


airToSubstrateSystem.updateSystemForAngle=np.vectorize(airToSubstrateSystem.updateSystemForAngle)


for angl in [70]:
   psis=[]
   deltas=[]
   for wl in np.linspace(300,900,100)*1E-9:
      airToSubstrateSystem.updateSystemForAngle(wl,np.sin(angl/180.*np.pi))
      psi,delta=airToSubstrateSystem.getEpsilonPsi()
      psis.append( [wl,psi] )
      deltas.append( [wl,delta] )
   psis=np.array(psis)
   deltas=np.array(deltas)

   plt.subplots_adjust(hspace=0.001,wspace=0.001)

   plt.subplot(211)
   plt.plot(psis[:,0]*1E9,psis[:,1]/np.pi)
   plt.ylim(-.09,.6)
   plt.xticks([])
   plt.xlim(300,900)
   plt.ylabel('$\Psi$ ($\pi$ Rad)')

   plt.subplot(212)
   plt.plot(deltas[:,0]*1E9,constrainBetween(deltas[:,1]/np.pi),marker='')
   plt.ylabel('$\Delta$ ($\pi$ Rad)')
   plt.ylim(-.1,2.1)
   plt.xlabel('Wavelength (nm)')
   plt.xlim(300,900)

print SiOSell.N(632.8E-9)
compuSiO2=np.loadtxt('./Computease120nmSiO.txt')

plt.subplot(211)
plt.plot(compuSiO2[:,0],compuSiO2[:,1]/180.,color='black',linestyle='dashed')

plt.subplot(212)
plt.plot(compuSiO2[:,0],constrainBetween(compuSiO2[:,2]/180.),color='black',linestyle='dashed')

plt.savefig('PsiDelta.pdf')



def errFun(p):
   fitSystem=Stack(airLayer,siLayer)
   SiO=Sellmeier(1.121*p[1],0.0914,0.01,1.)
   SESIO2=Layer(SiO.N,p[0]*1.E-9)
   fitSystem.addLayer(SESIO2) #Top layer
   psiErr=np.array([])
   deltaErr=np.array([])
   i=0
   for wl in compuSiO2[:,0]:
      fitSystem.updateSystemForAngle(wl*1E-9,np.sin(70./180.*np.pi))
      psi,delta=fitSystem.getEpsilonPsi()

      psiErr=np.r_[psiErr,compuSiO2[i,1]/180.-psi/np.pi]
      deltaErr=np.r_[deltaErr,constrainBetween(compuSiO2[i,2]/180.)-constrainBetween(delta/np.pi)]
      i+=1

   return psiErr,deltaErr

pInit=[80,1]
pFit,bla=optimize.leastsq(lambda p:np.r_[errFun(p)],pInit)
print pFit

plt.clf()

plt.subplot(211)
plt.plot(compuSiO2[:,0],errFun(pFit)[0])
plt.subplot(212)
plt.plot(compuSiO2[:,0],errFun(pFit)[1])

plt.savefig('error.pdf')

