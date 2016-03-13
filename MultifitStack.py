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
import matplotlib.gridspec as gridspec

Layer=Layer.Layer
def reducePhase(x):
   rPhase=[]
   for xi in [x]:
      rPhase.append(xi%2)
   return rPhase

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

iErr=np.ones_like(specIData[:,0])*specIData[:,1].max()*0.03
pErr=0.15*np.ones_like(specPData[:,0])
pErr[0]=.3

########Create Object##########
modelSample02016=FittingStructure(airToSubstrateSystem,np.c_[.5*wl2ev/specIData[:,0],specIData[:,1]],iErr,\
      np.c_[specPData[:,0],specPData[:,2]],pErr)
modelSample02016.fillCorrections(X)
modelSample02016.fillCorrections(.5*wl2ev/specIData[:,0])
modelSample02016.fillCorrections(specPData[:,0])

###########################SAMPLE PA Al2O3 (S560)
airToSubstrateSystem=None
specIData=np.loadtxt('../phdRepos/data/modeling/20120110_plasmaAl2O3_anneal_S560.txt')
airLayer=Layer(n_vac,10E-3)
siLayer=Layer(SiliconJellison.N,5E-6)
al2o3Layer=Layer(Al2O3.N,31.5E-9)

########### Build structure ###############
airToSubstrateSystem=Stack(airLayer,siLayer)
airToSubstrateSystem.addLayer(al2o3Layer) #Top layer

specPData=np.loadtxt('../phdRepos/data/Exp2/1201-02015_results.txt')
specPData[:,0]=.5E-9*specPData[:,0]
specPData[-2,2]-=1


iErr=np.ones_like(specIData[:,0])*specIData[:,1].max()*0.03
pErr=0.15*np.ones_like(specPData[:,0])
pErr[0]=.3

########Create Object##########
modelSample02015=FittingStructure(airToSubstrateSystem,\
      np.c_[.5*wl2ev/specIData[:,0],specIData[:,1]],iErr,\
      np.c_[specPData[:,0],specPData[:,2]],pErr)
modelSample02015.fillCorrections(X)
modelSample02015.fillCorrections(.5*wl2ev/specIData[:,0])
modelSample02015.fillCorrections(specPData[:,0])

##########################SAMPLE PA Al2O3 (5493-5)
airToSubstrateSystem=None
specIData=np.loadtxt('../phdRepos/data/modeling/20100930_plasmaAl2O3_anneal_S4935.txt')
airLayer=Layer(n_vac,10E-3)
siLayer=Layer(SiliconJellison.N,5E-6 )
al2o3Layer=Layer(Al2O3.N,31.5E-9)

########### Build structure ###############
airToSubstrateSystem=Stack(airLayer,siLayer)
airToSubstrateSystem.addLayer(al2o3Layer) #Top layer

specPData=np.loadtxt('../phdRepos/data/Exp2/1201-02013_results.txt')
#this should be in m for SHG
specPData[:,0]=.5E-9*specPData[:,0]
specPData[-1,2]-=2

iErr=np.ones_like(specIData[:,0])*specIData[:,1].max()*0.03
pErr=0.15*np.ones_like(specPData[:,0])
pErr[0]=.3
########Create Object##########
modelSample02013=FittingStructure(airToSubstrateSystem,np.c_[.5*wl2ev/specIData[:,0],specIData[:,1]],iErr,\
      np.c_[specPData[:,0],specPData[:,2]],pErr)
modelSample02013.fillCorrections(X)
modelSample02013.fillCorrections(.5*wl2ev/specIData[:,0])
modelSample02013.fillCorrections(specPData[:,0])

######################3#SAMPLE TA SiO
airToSubstrateSystem=None
specIData=np.loadtxt('../phdRepos/data/modeling/20101029_SiO2_FGA_45nm_single.txt')

airLayer=Layer(n_vac,10E-3)
siLayer=Layer(SiliconJellison.N,5E-9)
TSiO=Layer(SiOSell.N,47.8E-9)
########### Build structure ###############
airToSubstrateSystem=Stack(airLayer,siLayer)
airToSubstrateSystem.addLayer(TSiO) #Top layer

###########PLOTTING FROM HERE ON OUT################

specPData=np.loadtxt('../phdRepos/data/Exp2/1201-02014_results.txt')
#this should be in m for SHG
specPData[:,0]=.5E-9*specPData[:,0]

iErr=np.ones_like(specIData[:,0])*specIData[:,1].max()*0.03
pErr=0.15*np.ones_like(specPData[:,0])
pErr[0]=.3

modelSample02014=FittingStructure(airToSubstrateSystem,np.c_[.5*wl2ev/specIData[:,0],specIData[:,1]],iErr,\
      np.c_[specPData[:,0],specPData[:,2]],pErr)
modelSample02014.fillCorrections(X)
modelSample02014.fillCorrections(.5*wl2ev/specIData[:,0])
modelSample02014.fillCorrections(specPData[:,0])

#SAMPLE S570
specIData=np.loadtxt('../phdRepos/data/modeling/20120111_ALDstack_8nm_S570.txt')
airLayer=Layer(n_vac,10E-3)
siLayer=Layer(SiliconJellison.N,5E-9)
ALDSiO=Layer(ALD_SiO.N,10.82E-9)
Al2O3Layer=Layer(Al2O3.N,31.5E-9)


specPData=np.loadtxt('../phdRepos/data/Exp2/1201-02012_results.txt')
#this should be in m for SHG
specPData[:,0]=.5E-9*specPData[:,0]
specPData[-1,2]-=2
specPData[-2,2]+=2
#data08048=data08048[1:]

iErr=np.ones_like(specIData[:,0])*specIData[:,1].max()*0.03
pErr=0.05*np.ones_like(specPData[:,0])
pErr[0]=.2

########### Build structure ###############
airToSubstrateSystem=Stack(airLayer,siLayer)
airToSubstrateSystem.addLayer(Al2O3Layer) #Top layer
airToSubstrateSystem.addLayer(ALDSiO) #Second layer

########Create Object##########
modelSample02012=FittingStructure(airToSubstrateSystem,np.c_[.5*wl2ev/specIData[:,0],specIData[:,1]],iErr,\
      np.c_[specPData[:,0],specPData[:,2]],pErr)
modelSample02012.fillCorrections(X)
modelSample02012.fillCorrections(.5*wl2ev/specIData[:,0])
modelSample02012.fillCorrections(specPData[:,0])
####################MULTIFIT PART########################

multiFit=MultipleFittingStructure([modelSample02014,modelSample02012,modelSample02016,modelSample02015])

#multiFit=MultipleFittingStructure([modelSample02014,modelSample02015])

globVarNr=4
indVarNr=4

def getPInit():
   pinit=np.ones(globVarNr+4*indVarNr)
   pinit[globVarNr]=.1
   pinit[globVarNr+indVarNr]=7
   pinit[globVarNr+indVarNr*2]=12
   pinit[globVarNr+indVarNr*3]=20
   return pinit

multiFit.getPInit=getPInit

def paramsExcitonic(p):
      #Returns N tables of fitvars
      globVars=p[:globVarNr]
      indVars=p[globVarNr:]
      params=[]
      for i in range(0,len(multiFit.structures)):
         i=indVarNr*i
         params.append(np.array((

         (globVars[1]*2.0E-19   ,indVars[i+1] ,3.32*globVars[2] ,.12*globVars[3]),

         (indVars[i]*1E-19    ,indVars[i+2]*.9*np.pi , 3.4 ,.115),

         (globVars[0]*7E-19  ,indVars[i+3]*.5*np.pi , 3.7 ,.3),

         #(True,0,indVars[i+1]*1E-20,indVars[i+5]),

         )))
      return params
multiFit.paramsExcitonic=paramsExcitonic
pFit=multiFit.runFit()
pError=multiFit.calcError()
#Print result to user
for (p,pE) in zip(paramsExcitonic(pFit),paramsExcitonic(pError)):
   print ExiParams(p,pE)

colors=['red','orange','blue','green']
markers=['o','*','d','s']

fitParams=''
EfishQPhase=''
#plt.subplots_adjust(hspace=0.001)

plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(3,1)
gs.update(hspace=0.001,wspace=0.001)
lowerPanel=plt.subplot(gs[1:,0])
upperPanel=plt.subplot(gs[0,0])


for i in range(0,len(multiFit.structures)):

   structure=multiFit.structures[i]
   structure.paramsExcitonic=lambda l:paramsExcitonic(pFit)[i]
   fitParams+='Fit %i\n'%i
   fitParams+=ExiParams(paramsExcitonic(pFit)[i],paramsExcitonic(pError)[i])

   EfishQPhase+='%e %e %e\n'%(paramsExcitonic(pFit)[i][1][0],\
            paramsExcitonic(pError)[i][1][1]/np.pi,
         (paramsExcitonic(pFit)[i][1][1]-paramsExcitonic(pFit)[i][0][1])/np.pi )


   lowerPanel.errorbar(wl2ev/structure.dataI[:,0],structure.dataI[:,1],structure.errI,linestyle='',marker=markers[i],color=colors[i])
   #plt.plot(wl2ev/structure.dataI[:,0],.0004*structure.errFun(None,False),color='pink')
   lowerPanel.plot(wl2ev/X,structure.calcInt(X),color='black')
   lowerPanel.set_ylabel('SH Intensity (arb. unit)',fontsize=18)
   #plt.xticks( [] )
   lowerPanel.set_ylim(-2,22)

   shiftPhase=1.
   upperPanel.errorbar(wl2ev/structure.dataP[:,0],shiftPhase*i+structure.dataP[:,1],structure.errP,linestyle='',marker=markers[i],color=colors[i])
   
   #if i == 0:
   upperPanel.plot( [2.6,3.6], [shiftPhase*i,shiftPhase*i],color=colors[i],linewidth=1,linestyle='dashed')

   upperPanel.plot(wl2ev/X,shiftPhase*i+structure.calcPhase(X),color='black')
   upperPanel.set_ylabel('SH Phase ($\pi$ Rad)',fontsize=18)
   upperPanel.set_ylim(-1.2,4.2)
   upperPanel.set_xticklabels([])
   #plt.yticks( np.arange(0,5),['0','0','0','0'])

   #upperPanel.arrow(2.62+0.02*i,0,0,shiftPhase*i,length_includes_head=True,\
   #                 shape='full',width=.001,head_length=0.05,overhang=0,\
   #                 head_width=.01,color=colors[i])
   #upperPanel.annotate(1,xy=(2.64,np.floor(plt.ylim()[1])-1./2),xycoords='data',va='center')

a = plt.axes([.23, .35, .3, .245])
i=0
structure=multiFit.structures[i]
structure.paramsExcitonic=lambda l:paramsExcitonic(pFit)[i]
a.errorbar(wl2ev/structure.dataI[:,0],structure.dataI[:,1],structure.errI,linestyle='',marker=markers[i],color=colors[i])
a.plot(wl2ev/X,structure.calcInt(X),color='black')

i=1
structure=multiFit.structures[i]
a.errorbar(wl2ev/structure.dataI[:,0],structure.dataI[:,1],structure.errI,linestyle='',marker=markers[i],color=colors[i])
a.plot(wl2ev/X,structure.calcInt(X),color='black')

a.set_ylim(-.04,1.6)
#a.set_yticks(np.arange(0,.6,.25))

fil=open('./fitParams.txt','w+')
fil.write(fitParams)
fil.close()

fil=open('FitEfishParams.txt','w+')
fil.write(EfishQPhase)
fil.close()

lowerPanel.set_xlabel('Two-photon Energy (eV)',fontsize=18)
plt.savefig('spectra.pdf')
exit()

plt.clf()
lowerPanel=plt.subplot(gs[2,0])
upperPanel=plt.subplot(gs[0:2,0])
for i in range(0,len(multiFit.structures)):

   Qval=np.abs(paramsExcitonic(pFit)[i][1][0]/paramsExcitonic(pFit)[2][1][0]*5E12)
   upperPanel.semilogx(10E11, 0,marker='',color='red')
   upperPanel.errorbar(Qval,
                       reducePhase((paramsExcitonic(pFit)[i][1][1]-paramsExcitonic(pFit)[i][0][1])/np.pi),paramsExcitonic(pError)[i][1][1]/np.pi,marker='d',color='red')
   upperPanel.set_ylabel("EFISH $\phi$ ($\pi$ rad)")
   upperPanel.set_xticks([])
   upperPanel.set_ylim(0,1)

   lowerPanel.semilogx(Qval, np.abs(paramsExcitonic(pFit)[i][0][0]),marker='d',color='red')
   lowerPanel.set_ylabel('SiOx Ampl')


lowerPanel.set_xlabel("Estimated Charge Density (cm$^{-2}$)")

plt.savefig('EfishQ.pdf')
