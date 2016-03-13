#Copyright 2012 Vincent Vandalon
#
#This file is part of the NonlinearModel. NonlinearModel is free software: you can
#redistribute it and/or modify it under the terms of the GNU
#General Public License as published by the Free Software
#Foundation, either version 3 of the License, or (at your
# option) any later version.
#NonlinearModel is distributed in the hope that it will be useful, but
#WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See
#the GNU General Public License for more details.
#You should have received a copy of the GNU General Public
#License along with NonlinearModel.  If not, see
#<http://www.gnu.org/licenses/>.  
from cachingMethods import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import constants
from scipy import optimize

class FittingStructure:
   pFit=[] #ALWAYS last fit result
   pCurrent=[] #Last fit result if not modified by errFun

   def __init__(self,stack,dataI,errI,dataP,errP):
      self.stack=stack
      self.dataI=dataI
      self.errI=errI
      self.dataP=dataP
      self.errP=errP
      self.correctionsSHGSample={}
      self.correctionsSHGReflection={}

   def paramsExcitonic(self,fitVars):
      raise Exception('Overwrite paramsExcitonic method in FittingStructure')

   def errFun(self,pFit,withPhase=True):
      pEx=self.paramsExcitonic(pFit)
      #We have to update the fit coefficients of the class so that
      #we do not have to supply pFit to calcP calcI
      self.pCurrent=pFit
      if withPhase == True:
         return np.r_[( (self.dataP[:,1]-self.calcPhase(self.dataP[:,0]))/self.errP )**2,\
            ( (self.dataI[:,1]-self.calcInt(self.dataI[:,0]))/self.errI )**2]
      else:
         return ( (self.dataI[:,1]-self.calcInt(self.dataI[:,0]))/self.errI )**2

   #Could be overwritten by user if other starting conditions are desired. Equivalent to 
   #defining different paramsExcitonic.
   def getInit(self):
      pInit=np.ones(15)
      pInit[1]=0
      pInit[5]=0
      pInit[9]=0
      pInit[13]=0
      return pInit

   def runFit(self,withPhase=False,pInit=True):
      if pInit == True:
         pInit=self.getInit()
      
      pFit,bla=optimize.leastsq(lambda p:self.errFun(p,withPhase),pInit)

      self.pFit=pFit
      self.pCurrent=pFit
      params=self.paramsExcitonic(pFit)
      N=len(self.dataI)
      if withPhase==True:
         N=len(self.dataI)+len(self.dataP)
      s=ExiParams(self.paramsExcitonic(pFit),self.paramsExcitonic(self.fitVariance(withPhase)))
      s+="Total reduced chi2: %s"%(self.errFun(pFit,withPhase).sum()/(N-len(pFit)-1) )

      return pFit,s


   # Vary over full parameter space and determine the 
   # error in each fitting parameters
   def fitVariance(self,withPhase=True):
      def detectZeroRow(m):
         zeroRow=[]
         for i in range(0,len(m)):
            if sumRow(np.abs(m[i,:]))== 0 or sp.isnan(m[i,0]):
               zeroRow.append(i)
         return np.array(zeroRow)

      def sumRow(row):
         for element in row:
            tot=0
            if sp.isnan(element) == False:
               tot+=element
            return tot

      def laplacian():
         deltat=0.05
         grads=np.zeros((len(self.pFit),len(self.pFit)))
         nrVars=len(self.pFit)
         print self.pFit
         dataIndex=0
         for wl in self.dataI[:,0]:
            wl=np.array([wl])
            for i in range(0,nrVars):
               for j in range(i,nrVars):
                  pi=np.copy(self.pFit)
                  pj=np.copy(self.pFit)
                  pi[i]=(1-deltat)*self.pFit[i]
                  pj[j]=(1-deltat)*self.pFit[j]

                  self.pCurrent=pi
                  dEdi=self.calcInt(wl)
                  self.pCurrent=self.pFit
                  dEdi-=self.calcInt(wl)

                  dEdi/=pi[i]-self.pFit[i]

                  self.pCurrent=pj
                  dEdj=self.calcInt(wl)
                  self.pCurrent=self.pFit
                  dEdj-=self.calcInt(wl)
                  dEdj/=pj[j]-self.pFit[j]
                  #TODO: dEdj=np.misc.derivative(self.calcInt(wl))
                  grads[i,j]+=(dEdi*dEdj)/self.errI[dataIndex]**2
                  if i!=j:
                     grads[j,i]+=(dEdi*dEdj)/self.errI[dataIndex]**2
            dataIndex+=1
         dataIndex=0
         if withPhase==True:
            for wl in self.dataP[:,0]:
               wl=np.array([wl])
               for i in range(0,nrVars):
                  for j in range(i,nrVars):
                     pi=np.copy(self.pFit)
                     pj=np.copy(self.pFit)
                     pi[i]=(1-deltat)*self.pFit[i]
                     pj[j]=(1-deltat)*self.pFit[j]

                     self.pCurrent=pi
                     dEdi=self.calcPhase(wl)
                     self.pCurrent=self.pFit
                     dEdi-=self.calcPhase(wl)
                     dEdi/=pi[i]-self.pFit[i]

                     self.pCurrent=pj
                     dEdj=self.calcPhase(wl)
                     self.pCurrent=self.pFit
                     dEdj-=self.calcPhase(wl)
                     dEdj/=pj[j]-self.pFit[j]
                     grads[i,j]+=(dEdi*dEdj)/self.errP[dataIndex]**2
                     if i!=j:
                        grads[j,i]+=(dEdi*dEdj)/self.errP[dataIndex]**2
               dataIndex+=1
         return np.array(grads)

      alph=laplacian()

      #Remove unused fitting params
      zeroRows=detectZeroRow(alph)
      alph=np.delete(alph,zeroRows,0)
      alph=np.delete(alph,zeroRows,1)
      Cred=sp.linalg.inv(alph)

      #Now pad C matrix with zeros to original size
      nrFitParams=len(self.pFit)
      C=np.zeros((nrFitParams,nrFitParams))
      ired=0
      for i in np.delete(range(0,nrFitParams),zeroRows,0):
         jred=0
         for j in np.delete(range(0,nrFitParams),zeroRows,0):
            C[i,j]=Cred[ired,jred]
            jred+=1
         ired+=1

      N=len(self.dataI)
      if withPhase==True:
         N=len(self.dataI)+len(self.dataP)

      chiTot=self.errFun(self.pFit,withPhase).sum()/(N-nrFitParams-1)
      if np.any(chiTot>3):
         print "Chi is rather large [method: fitVariance()]"

      covMat=np.zeros( (nrFitParams,nrFitParams))
      for i in range(0,nrFitParams):
         s="%i"%i
         for j in range(0,nrFitParams):
            covMat[i][j]=1.65*np.sqrt(np.abs(chiTot*C[i,j]))
            if C[i,i]!=0 and C[j,j]!=0:
               s+="\t%f"%(C[i,j]/np.sqrt(C[i,i]*C[j,j]))
         print s
      varFit=covMat[range(0,nrFitParams),range(0,nrFitParams)]
      return np.array(varFit)

   """
   Calculate the complex excitonic response
   """
   def ChiExciton(self,om,p):
      I=np.zeros(len(om))
      for pex in p:
         if pex[0] is True:
            I+=pex[2]*np.exp(1j*pex[3])
         else:
            h  =np.abs(pex[0])
            ph =pex[1]
            omq=np.abs(pex[2])
            gamma=np.abs(pex[3])
            I=I+h*np.exp(1.j*ph)/(om-omq+1.j*gamma)
      return I

   #Definition of what we consider phase
   def calcPhase(self,wl):
      p=self.paramsExcitonic(self.pCurrent)
      totPhase=np.angle(1\
            *self.ChiExciton(wl2ev/wl,p)\
            *self.getElements(wl,self.correctionsSHGSample)\
            /self.getElements(wl,self.correctionsSHGReflection)\
            /np.exp(-0.2j*np.pi/.5*(3.-wl2ev/wl))
            )
      return np.unwrap(totPhase)/np.pi

   #Definition of what we consider intensity
   def calcInt(self,wl):
      p=self.paramsExcitonic(self.pCurrent)
      return np.abs(self.ChiExciton(wl2ev/wl,p)\
            *self.getElements(wl,self.correctionsSHGSample)\
            )**2
   
   ##################CACHING METHODS###################
   #We neeeeed performance.......

   def fillCorrections(self,wls):
      print 'WARNING: LINEAR CORRECTIONS OFF'
      for wl in wls:
         #Get the source strength for all layers, select bottom layer
         self.stack.updateSystemForAngle(wl,n_vac(wl)*np.sin(angleInci))
         self.correctionsSHGSample[wl]=1#self.stack.getSources()[-1]

         #Perform calculation of reflection SHG
         self.stack.updateSystemForAngle(wl,n_vac(wl)*np.sin(angleInci))
         self.correctionsSHGReflection[wl]=1#self.stack.calcInterfaceFields()[0][0] #Select field in ambient and upwards


   def getElements(self,wls,fromList):
      ret=[]
      for wl in wls:
         ret.append(fromList[wl])
      return np.array(ret)
