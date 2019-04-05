#Copyright 2012 Vincent Vandalon
#
#This file is part of the NonlinearModel. NonlinearModel is free software: you can
#redistribute it and/or modify it under the terms of the GNU
#General Public License as published by the Free Software
#Foundation, either version 3 of the License, or (at your
# option) any later version.
#NonlinearModel is distributed in the hope that it will be useful, but
#WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See #the GNU General Public License for more details.  #You should have received a copy of the GNU General Public
#License along with NonlinearModel.  If not, see
#<http://www.gnu.org/licenses/>.
import Layer
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.linalg
import numpy.linalg


class Stack:

   def __init__(self,ambientLayer,substrate):
      self.substrate=substrate
      self.layers=[]
      self.ambientLayer=ambientLayer

   #NOTE layer[0] is next to ambient
   # and layer[n] is next to substrate
   def addLayer(self,lay):
      self.layers.append(lay)

   def clearLayers(self):
      self.layers=[]

   def getLayer(self,i):
      return self.layers[i]

   def updateSystemForAngle(self,wl,opticInvar):
      self.ambientLayer.updateWL(wl,opticInvar)
      self.substrate.updateWL(wl,opticInvar)
      self.wl=wl
      self.opticInvar=opticInvar
      for lay in self.layers:
         lay.updateWL(wl,opticInvar)

   def __getTransferMatrix(self,layers,directionUp):
      Ms=np.eye(2,2)

      Mp=np.eye(2,2)
      if len(layers)<2:
         print "Number of layers to low"
      elif len(layers)==2:
         Mp=np.dot(self.ambientLayer.transferMatrixP(layers[0],layers[1]),Mp)
         Ms=np.dot(self.ambientLayer.transferMatrixS(layers[0],layers[1]),Ms)
      else:
         prevLayer=layers[0]
         #Skip the first one since this is already
         #previous layer and skip last
         for layer in layers[1:-1]:
            Mp=np.dot(layer.transferMatrixP(prevLayer,layer),Mp)
            Ms=np.dot(layer.transferMatrixS(prevLayer,layer),Ms)
            #In both cases the thickness to the end of the stack needs to be known
            if directionUp==True:
               #Mp=np.dot(layer.propMatrix(self.__cummulativeThickness(layers,layer)),Mp)
               Mp=np.dot(layer.propMatrix(layer.thickness),Mp)
               Ms=np.dot(layer.propMatrix(layer.thickness),Ms)
            else:
               #Mp=np.dot(layer.propMatrix(-self.__cummulativeThickness(layers,layer)),Mp)
               Mp=np.dot(layer.propMatrix(-layer.thickness),Mp)
               Ms=np.dot(layer.propMatrix(-layer.thickness),Ms)
            prevLayer=layer
         Mp=np.dot(self.substrate.transferMatrixP(prevLayer,layers[-1]),Mp)
         Ms=np.dot(self.substrate.transferMatrixS(prevLayer,layers[-1]),Ms)
      return Ms,Mp

   #NOTE This ignores thickness of first layer and last layer
   def __cummulativeThickness(self,layers,layer):
      d=0
      for l in layers[1:-1]:
         d+=l.thickness
         if l==layer:
            break
      return d


   #Get the transmission for substrate towards ambient in the case
   #that there are no source terms. The system is lit by a source far
   #inside the semi infinite substrate
   def __getTransRef(self,pol=1):
      M=self.__getTransferMatrix(np.r_[[self.ambientLayer],self.layers,[self.substrate]]\
            ,directionUp=False)[pol] #here p pol is selected
      #E_substrate=(0,?)=M E_ambient=M(?,1)
      return np.array([-M[0,1]/M[0,0],-M[1,0]*M[0,1]/M[0,0]+M[1,1]])

   #Calculate the electric field in Cartesian coordinates at an interface
   #Always at the bottom of the layer
   def calcInterfaceFields(self,pol=1):

      [E1plus,E2min]=self.__getTransRef(pol)
      #Eamb=self.ambientLayer.EField(E1plus,1,0)
      #Esub=self.substrate.EField(0,E2min,-self.__cummulativeThickness(self.layers,all))
      Eamb=(E1plus,1)
      Esub=(0,E2min)

      #NOTE: we would like to work from ambient down
      # since substrate could be (0,0) electric field
      # Solve Eambient = M Elayer
      i=0
      fields=[]
      for layer in self.layers:

         #Get the layers 0..i and invert their sequence
         M=self.__getTransferMatrix(np.r_[self.layers[0:i+1][::-1],[self.ambientLayer]]\
               ,directionUp=True)[pol]
         #Solution is the field at the lower interface
         M=np.dot(M,layer.propMatrix(layer.thickness))
         fields.append(np.dot(scipy.linalg.inv(M),Eamb))
         i+=1

      return np.r_[[Eamb],fields,[Esub]]

   #Get the emission of the SHG light generated at the substrate-first
   #layer interface
   def __getSourceEmission(self,interface=0,pol=1):
      #Basically solve Eq. 4.16 in Sipe

      #Need to put source term in ambient
      v=self.ambientLayer.sourceTerm()
      if len(self.layers)>0:
         mv=np.dot(self.__getTransferMatrix(np.r_[ self.layers[interface:]\
               ,[self.substrate]],directionUp=False)[pol], v)
      else:
         print 'Warning: evaluating without layers'
         mv=np.dot(self.__getTransferMatrix(np.r_[ [self.ambientLayer],[self.substrate]]\
               ,directionUp=False)[pol], v)

      M=self.__getTransferMatrix(np.r_[[self.ambientLayer],self.layers,[self.substrate]]\
            ,directionUp=False)[pol] #here p pol is selected

      e1=np.array([mv[0]/M[0,0],0])

      e2=[0,-mv[1]+np.dot(M,e1)[1]]
      return e2,e1

   #Get the expected intensity for a source at the
   #bottom of each layer
   def getSources(self,pol=1):
      sources=[]
      layerNr=0
      for layer in self.layers:
         source=self.__getSourceEmission(layerNr,pol)
         #Calculate field just above interface
         #for FUNDAMENTAL energy
         self.updateSystemForAngle(self.wl*2,self.opticInvar)
         incidentField=self.calcInterfaceFields()[layerNr+1]
         self.updateSystemForAngle(self.wl*.5,self.opticInvar)

         incidentField=(np.dot(layer.pplus,layer.vz)* incidentField[0]
            + np.dot(layer.pmin,layer.vz)* incidentField[1])**2

         sources.append(source[1][0]*incidentField)
         layerNr+=1

      return sources

   def getErrors(self):
      for lay in self.layers:
         lay.errorOccurred()

   def getEpsilonPsi(self):
      Ms,Mp=self.__getTransferMatrix(np.r_[[self.substrate],
         self.layers,[self.ambientLayer]]
                ,directionUp=True)
      rs=Ms[0,1]/Ms[1,1]
      rp=Mp[0,1]/Mp[1,1]
      rho=rp/rs
      psi=np.arctan(np.abs(rho))
      delta=np.angle(rho)
      return psi,delta

   def getTransmission(self):
      Ms,Mp=self.__getTransferMatrix(np.r_[[self.substrate],
         self.layers,[self.ambientLayer]]
                ,directionUp=True)
      ts=1./Ms[1,1]
      tp=1./Mp[1,1]

      return ts,tp

print "Only p pol enabled at the moment and source at substrate layer[0]"
print "This can however easily be modified to be more flexible"
