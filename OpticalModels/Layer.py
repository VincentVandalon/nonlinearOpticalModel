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
import scipy.constants

print "Take a look at angle of fundamental E field with respect to p"
class Layer:
   c=scipy.constants.c

   def __init__(self,n,d):
      self.flagFresnel=False
      self.n_f=n #Set the n_function
      self.n=0
      self.lambd=0
      self.opticInvar=0
      self.thickness=d

   #wl in meters, theta in radians
   def updateWL(self,wl,opticInvar):
      if self.lambd != wl or self.opticInvar != opticInvar:
          self.lambd=wl
          self.n=self.n_f(wl)
          self.opticInvar=opticInvar

          #Solve weird array problem with UnivariateSpline
          if type(self.n) is type(np.array([])):
             self.n=self.n
          thet=np.arcsin(opticInvar/self.n)

          self.omeg=2.*np.pi*scipy.constants.c/self.lambd
          self.omegRed=2.*np.pi/self.lambd
          self.nu2=self.n**2*self.omegRed**2       #What is normally called the k^2 vector
          self.kappa=self.n*np.sin(thet)*self.omegRed  #Component parallel to the interface
          self.w=self.n*np.cos(thet)*self.omegRed #Component perpendicular to the interface

          if np.imag(self.w)<0:
             self.w=np.conj(self.w)#np.real(self.w)+1.j*np.abs(np.imag(self.w))
          if np.abs(np.imag(self.w))<1E-3:
             self.w=np.real(self.w)

          self.vkappa=np.array((1,0,0))  #No y component required
          self.vz=np.array((0,0,1))
          self.wplus=(self.kappa*self.vkappa+self.w*self.vz)/np.sqrt(self.nu2)
          self.wmin=(self.kappa*self.vkappa-self.w*self.vz)/np.sqrt(self.nu2)

          #Note that signs are inverted
          self.pplus=(self.kappa*self.vz-self.w*self.vkappa)/np.sqrt(self.nu2)
          self.pmin=(self.kappa*self.vz+self.w*self.vkappa)/np.sqrt(self.nu2)
          
          #the s vector
          self.vs=np.cross(self.vkappa,self.vz)


   def fresnellCoeff(self,to):
      fracP=(self.w*to.n**2+to.w*self.n**2)
      fracS=(self.w+to.w)
      rs=(self.w-to.w)/fracS
      rp=(self.w*to.n**2-to.w*self.n**2)/fracP
      ts=2.*self.w/fracS
      tp=2.*self.n*to.n*self.w/fracP

      #TODO:Do some sanity checking as this is a 

      return (rs,rp,ts,tp)

   def transferMatrixP(self,to,fro):
      rs,rp,ts,tp=fro.fresnellCoeff(to)

      return 1./tp*np.array([[1.,rp],[rp,1.]])

   def transferMatrixS(self,to,fro):
      rs,rp,ts,tp=fro.fresnellCoeff(to)

      return 1./ts*np.array([[1.,rs],[rs,1.]])

   def propMatrix(self,z):
      return [[np.exp(1.j*(np.real(self.w)-1j*np.abs(np.imag(self.w)))*z),0],
               [0,np.exp(-1.j*(np.real(self.w)-1j*np.abs(np.imag(self.w)))*z)]]

   def setIncoherent(self,bool,cohDeg):
      if bool == True:
         phi=np.pi*.125j
         self.propMatrix=lambda z:[
            [np.exp(phi),0],[0,np.exp(-phi)]
         ]
      else:
         self.propMatrix=lambda z:[[np.exp(1.j*(np.real(self.w)-1j*np.abs(np.imag(self.w)))*z),0],[0,np.exp(-1.j*(np.real(self.w)-1j*np.abs(np.imag(self.w)))*z)]]

   def errorOccurred(setlf):
      if self.flagFresnel==True:
         print "Fresnel coefficients do not match. Abort"

   #Calculate with a source term unity (1)
   def sourceTerm(self):
      self.qplus=self.pplus
      self.qmin=self.pmin
      #Pick the direction of the polarization 
      #see also Paul's comments
      self.P=self.vz  #NOTE This is actually Chi:E:E

      #Corrected for a excess of 4pi
      #This source has dimensions of /length!!!! so it depends on the units chosen
      sourceP=2.j*np.pi/(4*np.pi*scipy.constants.epsilon_0)\
            *self.omegRed**2/self.w*np.array(( \
            #1,-1))
         -np.dot(self.qplus,self.P), \
         np.dot(self.qmin,self.P)))
      return sourceP
