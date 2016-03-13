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

wls=[]
data=[]
iData=[]
idataErr=[]
iwls=[]
dataErr=[]
iFit=[]
phFitArr=[]
for i in range(12,17):
   #if i!=13:
   # 12= 10nm SiO2 13= Al2O3 14=TSIO2 15=Al2O3 16=5 nm SiO2ALD
   if i!=14:
      correctionsSHGSample={}
      correctionsSHGReflection={}
      pInit=[]
      pFit=[]
      plt.clf()
      execfile('./Model201201%i.py'%(i))
      wls.append(data08048[:,0])
      data.append(data08048[:,2])
      dataErr.append(pErr)

      iwls.append(specData[:,0])
      iData.append(specData[:,1])
      idataErr.append(iErr)

      iFit.append(Ieph)
      phFitArr.append(phFit)


plt.clf()

NicksFormat=False
if NicksFormat==True:
   plt.figure(figsize=(8,12))
else:
   plt.figure(figsize=(8,6))

markers=['d','s','o','^','o','o']
#offsets=[1.2,0,.8,.4,5]
#Phase labels=['(c)','(d)','(a)','(b)']
# 12= 10nm SiO2 13= Al2O3 14=TSIO2 15=Al2O3 16=5 nm SiO2ALD
#Phase Shift
labels=['(C)','(A1)','(A2)','(B)']
labels=['','','','']
offsets=[1.2,0,.4,.8,5]
colors=['red','green','purple','orange']
for i in range(0,len(wls)):
   dat=np.array(data[i])
   offs=np.array(offsets[i])
   X=np.linspace(wl2ev/2.6,wl2ev/3.6,100)

   upperPanel=plt.subplot(211)
   plt.subplots_adjust(hspace=0.001)

   plt.plot(wl2ev/X[:-7],offs+phFitArr[i][:-7],marker="",linestyle="-",color='red',label="Model")
   plt.errorbar(wl2ev/wls[i],offs+dat,dataErr[i],marker=markers[i],color=colors[i],linestyle='')
   plt.annotate(labels[i],(3.51,offs+dat[-1]))
   plt.ylabel('SH Phase ($\pi$ rad)')
   plt.xlim(2.78,3.62)
   plt.ylim(-.7,2.5)
   #plt.xticks([])

   lowerPanel=plt.subplot(212)
   plt.plot(wl2ev/X[:-7],iFit[i][:-7],marker="",linestyle="-",color='red',label="Model")
   plt.errorbar(2.*iwls[i],iData[i],idataErr[i],marker=markers[i],color=colors[i],linestyle='')
   if i==1:
      plt.annotate(labels[i],(3.51,iData[i][0]-.4))
   else:
      plt.annotate(labels[i],(3.51,iData[i][0]))
   plt.ylabel('SH Intensity (arb. units)')
   plt.yticks(np.arange(0,25,5))
   plt.xlim(2.78,3.62)
   plt.ylim(-.5,26)
if False:
   a = plt.axes([.2, .25, .3, .2])
   plt.plot(wl2ev/X,iFit[1],marker="",linestyle="-",color='red',label="Model")
   plt.errorbar(2.*iwls[1],iData[1],idataErr[1],color=colors[1],marker=markers[1],linestyle='')
   plt.annotate(labels[1],(3.51,iData[1][3]))
   plt.yticks(np.arange(0,.5,.1))
   plt.xlim(2.8,3.6)
   plt.xticks(np.arange(2.8,3.6,0.2))
   plt.ylim(-.05,0.5)

upperPanel.text(.05,.9,'(III)',transform = upperPanel.transAxes)
lowerPanel.text(.05,.9,'(IV)',transform = lowerPanel.transAxes)
upperPanel.xticks([])

plt.xlabel('Two-Photon Energy (eV)')
plt.savefig('./combinedModel.pdf')
plt.savefig('./combinedModel.png')
