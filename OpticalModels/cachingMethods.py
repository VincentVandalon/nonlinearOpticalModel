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
from scipy import constants
from scipy import optimize
################ CONSTANTS ###########

n_vac = lambda l:1.00
wl2ev=sp.constants.c*sp.constants.h/sp.constants.e
angleInci=35*sp.constants.degree

np.seterr(invalid='ignore')

#NOTE all functions work with SI units, therefore, they 
#have meter (m) as input for wavelength. 

def ExiParams(p,pErr=None):
   p=np.copy(p)
   s=''
   phiOffset=p[0,1]
   if pErr==None:
      for contrib in p:
         if contrib[0]==True:
            s+='Amplitude %s and phase %s'%(contrib[2],contrib[3])
         else:
            contrib[1]-=phiOffset
            contrib[1]=contrib[1]/np.pi
            s+= "Amplitude %s, phase %.2f, resonant energy %.2f, and broadening %.2f\n"%(contrib[0],contrib[1],contrib[2],contrib[3],)
   else:
      pErr=np.copy(pErr)
      i=0
      for contrib in p:
         if contrib[0]==True:
            s+='Amplitude %f and phase %f'%(contrib[2],contrib[3])
         else:
            contrib[1]-=phiOffset
            contrib[1]=contrib[1]/np.pi
            #TODO: When using functions to limit range
            # this is wrong. Need to take into account that function 
            # dVarReal= f(dVar)
            contribErr=pErr[i]
            contribErr[1]=(pErr[i][1])/np.pi
            s+= "Amplitude %.2e+-%.2e, phase %.2f+-%.2f, resonant energy %.3f+-%.3f, and broadening %.3f+-%.3f \n"%(contrib[0],contribErr[0],contrib[1],contribErr[1],contrib[2],contribErr[2],contrib[3],contribErr[3],)
         i+=1
   return s
