import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os,sys
from scipy import optimize
from scipy import constants

from OpticalModels.Stack import *
from OpticalModels.Layer import *
from OpticalModels.cachingMethods import *
from OpticalModels.FitStructure import *
from OpticalModels.MultiFit import *
from OpticalModels.opticalModels import *
import matplotlib.gridspec as gridspec

X=np.linspace(wl2ev/2.6,wl2ev/3.6,100)

##################################SAMPLE 569
specIData=np.loadtxt('../phdRepos/data/modeling/20120111_ALDstack_5nm_S569.txt')
airLayer=Layer(n_vac,10E-3)
siLayer=Layer(SiliconJellison.N,5E-9)
al2o3Layer=Layer(Al2O3.N,31.5E-9)
ALDSiO=Layer(ALD_SiO.N,4.72E-9)

########### Build structure ###############
airToSubstrateSystem=Stack(airLayer,siLayer)
airToSubstrateSystem.addLayer(al2o3Layer) #Top layer
airToSubstrateSystem.addLayer(ALDSiO) #Top layer

specPData=np.loadtxt('../phdRepos/data/Exp2/1201-02016_results.txt')
#this should be in m for SHG
specPData[:,0]=.5E-9*specPData[:,0]
#data08048=data08048[1:]

iErr=np.ones_like(specIData[:,0])*specIData[:,1].max()*0.05
pErr=0.10*np.ones_like(specPData[:,0])
pErr[0]=.3

########Create Object##########
modelSample02016=FittingStructure(airToSubstrateSystem,np.c_[.5*wl2ev/specIData[:,0],specIData[:,1]],iErr,\
      np.c_[specPData[:,0],specPData[:,2]],pErr)
modelSample02016.fillCorrections(X)
modelSample02016.fillCorrections(.5*wl2ev/specIData[:,0])
modelSample02016.fillCorrections(specPData[:,0])


def paramsExcitonic(p):
   return [
         (p[0]*2.0E-19   ,p[5] ,3.32 ,.12),
         (p[1]*1E-19    ,p[6]*.9*np.pi , 3.4 ,.115),
         (p[2]*7E-19  ,p[7]*.5*np.pi , 3.7 ,.3),
          ]
modelSample02016.paramsExcitonic=paramsExcitonic
pFit,fitText=modelSample02016.runFit()
print fitText

plt.subplot(211)
print wl2ev/specPData[:,0]
plt.plot(wl2ev/X,modelSample02016.calcPhase(X),color='red')
plt.errorbar(wl2ev/specPData[:,0],specPData[:,2],pErr,marker='o',color='black',linestyle='')
plt.ylabel('SH Phase ($\pi$ rad)')
plt.gca().set_xticklabels([])
plt.ylim(-1.1,1.1)

plt.subplot(212)
plt.plot(wl2ev/X,modelSample02016.calcInt(X),color='red')
plt.errorbar(2*specIData[:,0],specIData[:,1],iErr,marker='o',color='black',linestyle='')
plt.ylabel('SH Intensity (a.u.)')
plt.xlabel('Two-photon energy (eV)')
plt.ylim(-.5,5.5)

plt.tight_layout(h_pad=0)

plt.savefig('singleFit.pdf')

