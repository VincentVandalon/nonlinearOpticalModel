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
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import FitStructure
from scipy import constants
from scipy import optimize
from cachingMethods import *

class MultipleFittingStructure:
   structures=None
   pInit=None
   pFit=None

   def __init__(self,structures):
      if len(structures)<=0:
         raise Exception('At least 1 structure needed')
      self.structures=structures
      self.pInit=np.ones(12*len(self.structures))

   def paramsExcitonic(self,p):
      #Returns N tables of fitvars
      raise Exception('Overwrite paramsExcitonic method in FittingStructure')

   def combinedErrFun(self,p,withPhase):
      i=0
      errors=[]
      params=self.paramsExcitonic(p)
      for struc in self.structures:
         #This is a hack: the struc.paramsExcitonic is disabled
         struc.paramsExcitonic=lambda l:params[i]
         errors=np.r_[errors,struc.errFun(None,withPhase)]
         i+=1
      return errors

   def getPInit(self):
      return np.ones(12*len(self.structures))

   def runFit(self,withPhase=True):
      self.pInit=self.getPInit()

      pFit,bla=optimize.leastsq(lambda p:self.combinedErrFun(p,withPhase),self.pInit,xtol=.0001)
      self.pFit=pFit
      return pFit

   def calcError(self,withPhase=True):
      def detectZeroRow(m):
         zeroRow=[]
         for i in range(0,len(m)):
            if sumRow(np.abs(m[i,:]))== 0 or sp.isnan(m[i,0]):
               zeroRow.append(i)
         return np.array(zeroRow)

      def sumRow(row):
         tot=0
         for element in row:
            if sp.isnan(element) == False:
               tot+=element
         return tot

      def laplacian():
         deltat=0.05
         grads=np.zeros((len(self.pFit),len(self.pFit)))
         nrVars=len(self.pFit)
         for i in range(0,nrVars):
            for j in range(i,nrVars):
               pi=np.copy(self.pFit)
               pj=np.copy(self.pFit)
               pi[i]=(1-deltat)*self.pFit[i]
               pj[j]=(1-deltat)*self.pFit[j]

               #dE/di
               dEdi=calcResonse(pi,withPhase)
               dEdi-=calcResonse(self.pFit,withPhase)
               dEdi/=pi[i]-self.pFit[i]

               #dE/dj
               dEdj=calcResonse(pj,withPhase)
               dEdj-=calcResonse(self.pFit,withPhase)
               dEdj/=pj[j]-self.pFit[j]

               #This is included in calcResponse/self.errI[dataIndex]**2
               grads[i,j]+=(dEdi*dEdj).sum()

               #Complementary need not be calculated
               if i!=j:
                  grads[j,i]+=(dEdi*dEdj).sum()

         return np.array(grads)

      #Calculate the response for the excitonic model as a
      #row of numbers
      def calcResonse(pFit,withPhase):
         result=np.array([])
         fitStrucNumber=0
         for s in self.structures:
            wls=s.dataI[:,0]
            #TODO:Understand this
            s.paramsExcitonic=lambda p:(self.paramsExcitonic(pFit))[fitStrucNumber]
            #s.pCurrent=pFit[fitStrucNumber]
            result=np.r_[result,s.calcInt(wls)/s.errI]
            if withPhase==True:
               wls=s.dataP[:,0]
               result=np.r_[result,s.calcPhase(wls)/s.errP]

            fitStrucNumber+=1
         return result

      def getNrDatapoints(withPhase):
         nrPoints=0
         for s in self.structures:
            nrPoints+=len(s.dataI)
            if withPhase==True:
               nrPoints+=len(s.dataP)
         
         return nrPoints

      
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

      N=getNrDatapoints(withPhase)

      chiTot=self.combinedErrFun(self.pFit,withPhase).sum()/(N-nrFitParams-1)
      if np.any(chiTot>3):
         print "Chi is rather large [method: fitVariance()]"

      covMat=np.zeros( (nrFitParams,nrFitParams))
      for i in range(0,nrFitParams):
         s="%i"%i
         for j in range(0,nrFitParams):
            covMat[i][j]=1.65*np.sqrt(np.abs(chiTot*C[i,j]))
            if C[i,i]!=0 and C[j,j]!=0:
               s+="\t%f"%(C[i,j]/np.sqrt(C[i,i]*C[j,j]))
         #print s
      #TODO: should be equal to covMat.diagonal()
      varFit=covMat[range(0,nrFitParams),range(0,nrFitParams)]
      return np.array(varFit)
