#   Copyright 2011 Vincent Vandalon
#   For the most recent version of this file see
#   http://www.nijf.nl/
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

import scipy as sp
import numpy as np
from scipy.constants import *
from scipy.interpolate import *
import matplotlib.pyplot as plt
import os
import Tkinter, tkFileDialog

root=Tkinter.Tk()

# This class describes exposes the interface that all optical models
# of materials share such as n(lambda) and k(lambda)
class OpticalResponseMaterial:
   def __init__(self):
      raise Exception('asdf')

   #return complex refractive index
   def N(self,lamb):
      return np.sqrt(0j+self.eps(lamb))

   #Return the complex epsilon
   def eps(self,lamb):
      return self.N(lamb)**2

class CauchyModel(OpticalResponseMaterial):

   def __init__(self,A,B,C):
      self.A=A
      self.B=B
      self.C=C

   def N(self,lamb):
      lamb=1E6*lamb #Convert to um
      return self.A+self.B/lamb**2+self.C/lamb**4

class Sellmeier(OpticalResponseMaterial):

   def __init__(self,A,lamb0,ampIR,epsInf):
      self.A=A
      self.lamb0=lamb0
      self.ampIR=ampIR
      self.epsInf=epsInf

   def eps(self,lamb):
      lamb=1E+6*lamb #Convert m to um
      return self.epsInf+ 0j+self.A*lamb**2/(lamb**2-self.lamb0**2)-self.ampIR*lamb**2

class SellmeierCompleteEase(OpticalResponseMaterial):

   def __init__(self,A,B,ampIR,epsInf):
      self.A=A    #Amp
      self.B=B    #Center En.
      self.ampIR=ampIR     #IR Pole Amp
      self.epsInf=epsInf   #Einf

   def N(self,lamb):
      lamb=1E+6*lamb #Convert m to um
      return np.sqrt( self.epsInf + self.A*lamb**2/(lamb**2-self.B**2) - self.ampIR*lamb**2)

class Lorenz(OpticalResponseMaterial):
   def __init__(self,A,Ec,B):
      self.A=A
      self.Ec=Ec
      self.B=B

   def eps(self,lamb):
      #Convert m to eV
      E=1/sp.constants.value('electron volt-inverse meter relationship')/lamb
      return self.A*self.Ec*self.B/(self.Ec**2-E**2-1.j*self.B*E)

class TaucLorentz(OpticalResponseMaterial):
   def __init__(self,E0,A,Eg,C):
      self.Eg=Eg
      self.A=A
      self.E0=E0
      self.C=C

   # From:
   # Appl. Phys. Lett. 69, 371 (1996); doi:10.1063/1.118064 (3 pages)
   # Parameterization of the optical functions of amorphous materials in the interband region

   def eps(self,lamb):
      E=1/sp.constants.value('electron volt-inverse meter relationship')/lamb
      Eg=self.Eg
      A=self.A
      E0=self.E0
      C=self.C

      aln=(Eg**2-E0**2)*E**2+Eg**2*C**2-E0**2*(E0**2+3.*Eg**2)
      aatan=(E**2-E0**2)*(E0**2+Eg**2)+Eg**2*C**2
      gamma=np.sqrt(E0**2-C**2/2.)
      alpha=np.sqrt(4*E0**2-C**2)
      zeta=(E**2-gamma**2)**2+alpha**2*C**2/4.

      e1tl=A*C*aln/(2*np.pi*zeta*alpha*E0)*np.log( (E0**2+Eg**2+alpha*Eg) / (E0**2+Eg**2-alpha*Eg) )
      e1tl-= A*aatan/(np.pi*zeta*E0)*( np.pi - np.arctan( (2*Eg+alpha)/C) + np.arctan( (-2*Eg+alpha)/C) )
      e1tl+=2*A*E0*Eg*(E**2-gamma**2)/(np.pi*zeta*alpha)*(np.pi+2.*np.arctan(2*(gamma**2-Eg**2)/(alpha*C)))
      #See Erratum*C/(np.pi*zeta)*(  Eg*(E**2-gamma**2)*(np.pi+2*np.arctan( (gamma**2-Eg**2)/alpha/C)) )
      e1tl-=A*E0*C*(E**2+Eg**2)/(np.pi*zeta*E)*np.log( np.abs(E-Eg)/(E+Eg))
      e1tl+=2*A*E0*C*Eg/(np.pi*zeta)*np.log( np.abs(E-Eg)*(E+Eg)/np.sqrt( (E0**2-Eg**2)**2+Eg**2*C**2))

      #Calculate e2 part
      if E>Eg:
         e2tl=A*E0*C*(E-Eg)**2/( (E**2-E0**2)**2+C**2*E**2)/E
      else:
         e2tl=0
      return 1.*e1tl+1.j*e2tl

class CodyLorentz(OpticalResponseMaterial):
   def a():
      return 0

class Material:
   def N(self,lamb):
      raise Exception('asdf')
   def eps(self,lamb):
      raise Exception('asdf')

class SiliconJellison(Material):
   #Characterization of Si
   #E0,A,Eg,C,etlInf
   #TaucLorentz[eV, (An) 535.3, (E0) 3.376, (Cn) 0.27, (Eg) 3.000001] +
   Si1=TaucLorentz( 3.376,535.3, 3.000001,0.27)
   #Lorenz input is OK
   Si2=Lorenz( 27.86,  4.246,0.43)
   #E0,A,Eg,C,etlInf
   #TaucLorentz[eV, (An) 232.7, (E0) 3.746,(Cn) 0.936, (Eg) 2.714] +
   Si3=TaucLorentz( 3.746,232.7,2.714,0.936)
   #Lorenz input is OK
   Si4=Lorenz( 3.163, 5.488, 0.9)

   @classmethod
   def N(self,lamb):
      return np.sqrt(0j+self.eps(lamb)) 

   @classmethod
   def eps(self,lamb):
      return 1. +\
            1*np.vectorize(self.Si1.eps)(lamb)+\
            1*np.vectorize(self.Si2.eps)(lamb)+\
            1*np.vectorize(self.Si3.eps)(lamb)+\
            1*np.vectorize(self.Si4.eps)(lamb)


class SiOSell(OpticalResponseMaterial):
   @classmethod
   def N(self,lamb):
      #Sellmeier[eV_, A_, B_, ampIR_, \[Epsilon]inf_] = 
      #Sellmeier[eV, 1.121, 0.09140, 0.01, 1.0] ;(* thermal SiO2 *)
      
      #From Nick for fits 
      SiO=Sellmeier(1.121,0.0914,0.01,1.)
      #SiO=SellmeierCompleteEase(.1,.28675,0.01338,2.018)
      return np.vectorize(SiO.N)(lamb)



class Si_Palik(OpticalResponseMaterial):

   @classmethod
   def N(self, lamb):

      ndata=np.loadtxt('OpticalModels/CRYSTALS_Si_Palik.txt')
      cutoff=475
      #cutoff=300
      n=InterpolatedUnivariateSpline(1e-6*ndata[cutoff:,0],ndata[cutoff:,1])
      k=InterpolatedUnivariateSpline(1e-6*ndata[cutoff:,0],ndata[cutoff:,2])
      return n(lamb)+1.j*k(lamb)


class ALD_SiO(OpticalResponseMaterial):
   @classmethod
   def N(self,lamb):
      SiO=Sellmeier(1.077,0.09124,0.01,1.0)
      return np.vectorize(SiO.N)(lamb)

class Al2O3(OpticalResponseMaterial):
   @classmethod
   def N(self,lamb):
      Al2O3=Sellmeier(1.611,0.10871,0.01,1.0)
      return np.vectorize(Al2O3.N)(lamb)

class Tabulated(OpticalResponseMaterial):
   def __init__(self,fil=''):  
      if fil =='':
         ndata=tkFileDialog.askopenfilename(parent=root,title='Please select the file with the tabulated n and k values')      
         self.ndata=np.loadtxt(ndata)
      else:
         self.ndata=np.loadtxt(fil)
   
   def N(self,lamb):
      ndata=self.ndata
      n=InterpolatedUnivariateSpline(ndata[:,0],ndata[:,1])
      k=InterpolatedUnivariateSpline(ndata[:,0],ndata[:,2])
      return n(lamb*1E6)+1.j*k(lamb*1E6)

      
class ArGas(OpticalResponseMaterial):
   @classmethod
   def N(self,lamb):
      x=lamb*1E6
      n = 1 + 0.012055*(0.2075*x**2/(91.012*x**2-1) + 0.0415*x**2/(87.892*x**2-1) + 4.3330*x**2/(214.02*x**2-1))
      return n

class N2Gas(OpticalResponseMaterial):
   @classmethod
   def N(self,lamb):
      x=lamb*1E6
      n = 1 + 68.5520E-6 + 32431.57E-6*x**2/(144*x**2-1)
      return n

class CO2Gas(OpticalResponseMaterial):
   @classmethod
   def N(self,lamb):
      x=lamb*1E6
      n = 1 + 0.012055*(5.79925*x**2/(166.175*x**2-1) + 0.12005*x**2/(79.609*x**2-1) + 0.0053334*x**2/(56.3064*x**2-1) + 0.0043244*x**2/(46.0196*x**2-1) + 0.0001218145*x**2/(0.0584738*x**2-1))
      return n

#Plot the n and k
if False:
   wls=np.linspace(200,1000000,1000)*1E-9
   plt.subplot(211)
   ndata=np.loadtxt('OpticalModels/CRYSTALS_Si_Palik.txt')
   plt.semilogx(1e-6*ndata[:,0],ndata[:,1])
   plt.semilogx(wls,np.real(Si_Palik.N(wls)),linestyle='-',color='red')
   plt.ylim(3.3,3.6)
   plt.ylabel("n (1)")

   plt.subplot(212)
   plt.semilogx(1e-6*ndata[:,0],ndata[:,2])
   plt.semilogx(wls,np.imag(Si_Palik.N(wls)),linestyle='-',color='orange')
   plt.ylim(0,0.03)
   plt.xlabel("Wavelength (m)")
   plt.ylabel("k (1)")

   plt.savefig(os.getcwdu()+'/si.pdf')
