import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os,sys
from scipy import optimize
from scipy import constants

from OpticalModels.Layer import *
from OpticalModels.Stack import *
from OpticalModels.cachingMethods import *
from OpticalModels.FitStructure import *
from OpticalModels.MultiFit import *
from OpticalModels.opticalModels import *

Layer=Layer.Layer

def reducePhase(x):
   rPhase=[]
   for xi in x:
      rPhase.append(xi%2.)
   return rPhase

X=np.linspace(wl2ev/2.6,wl2ev/3.6,100)

##################################SAMPLE 569
specIData=np.loadtxt('../phdRepos/data/modeling/20120111_ALDstack_5nm_S569.txt')
airLayer=Layer(n_vac,10E-3)
siLayer=Layer(SiliconJellison.N,5E-9)
al2o3Layer=Layer(Al2O3.N,30E-9)
ALDSiO=Layer(ALD_SiO.N,5.E-9)

########### Build structure ###############
airToSubstrateSystem=Stack(airLayer,siLayer)
airToSubstrateSystem.addLayer(al2o3Layer) #Top layer
airToSubstrateSystem.addLayer(ALDSiO) #Top layer

specPData=np.loadtxt('../phdRepos/data/Exp2/1201-02016_results.txt')
#this should be in m for SHG
specPData[:,0]=.5E-9*specPData[:,0]
#data08048=data08048[1:]

iErr=np.ones_like(specIData[:,0])*specIData[:,1].max()*0.1
pErr=0.15*np.ones_like(specPData[:,0])
pErr[0]=.3

########Create Object##########
modelSample02016=FittingStructure(airToSubstrateSystem,np.c_[.5*wl2ev/specIData[:,0],specIData[:,1]],iErr,\
      np.c_[specPData[:,0],specPData[:,2]],pErr)
modelSample02016.fillCorrections(X)
modelSample02016.fillCorrections(.5*wl2ev/specIData[:,0])
modelSample02016.fillCorrections(specPData[:,0])


multiFit=MultipleFittingStructure([modelSample02016])

def paramsExcitonic(p):
      #Returns N tables of fitvars
      globVars=p[:5]
      indVars=p[5:]
      params=[]
      for i in range(0,len(multiFit.structures)):
         i=8*i
         params.append(np.array((
         (globVars[0]*1E-18     ,indVars[i+4]
            ,3.31*globVars[1]    ,.08*globVars[2]),

         (indVars[i+1]*1E-18     ,.9*indVars[i+5]*np.pi ,
            3.41                ,.1*globVars[3]),

         (indVars[i+2]*1E-18     ,indVars[i+6]*.4*np.pi,
            3.6               ,.30),

         )))
      return params
multiFit.paramsExcitonic=paramsExcitonic
pFit=multiFit.runFit()

colors=['red','orange','blue','green','purple']
markers=['o','*','d','s','^']

fitParams=''
EfishQPhase=''
for i in range(0,len(multiFit.structures)):

   structure=multiFit.structures[i]
   structure.ChiExciton=lambda om,p:1
   structure.paramsExcitonic=lambda l:pFit[i]
   fitParams+='Fit %i\n'%i
   fitParams+=ExiParams(pFit[i])

   EfishQPhase+='%e %e\n'%(pFit[i][1][0],\
         (pFit[i][1][1]-pFit[i][0][1])/np.pi )

   #plt.plot(wl2ev/structure.dataI[:,0],structure.dataI[:,1],linestyle='-',marker='',color=colors[i])
   #plt.plot(wl2ev/structure.dataI[:,0],.0004*structure.errFun(None,False),color='pink')
   plt.plot(wl2ev/X,structure.calcInt(X)/structure.calcInt(X).max(),color='black')

   def calcInt(self,wl):
      p=self.paramsExcitonic(self.pCurrent)
      return np.abs(structure.ChiExciton(wl2ev/wl,p)\
            *structure.getElements(wl,structure.correctionsSHGSample)\
            )**2
   structure.calcInt=calcInt
   plt.ylabel('SH Intenstiy (arb. unit)')


plt.xlabel('Two Photon Energy (eV)')
plt.savefig('influenceLinearOptics.pdf')

