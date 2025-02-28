'''
Created on Apr 26, 2024

@author: keith
'''

from ..quant.quants import Voltage, Angle, Frequency, Vector  # @UnresolvedImport
from ..fields.polarization import Polarization  # @UnresolvedImport
from ..fields.emField import EMField, ArrayFactor
import numpy as np


class AntArray(Vector):
    """
    The array class is a subclass of an xyz Quant.Quants.Vector class
    This allows for the direct manipulation of the array element coordinates
    quaternion rotation and xyz offsets are directly supported,
    
    An Array holds Antenna objects or huygen's sources.
    If Antenna objects then the EMField is the total field
    If huygen's sources then the EMfield is the array factor 
    and must be multiplied times the embedded element pattern o get the total field.

    The Array constructor takes the xyz coodinates of each element and an optional
    list of Antenna objects. 
    
    static class methods are provided to ease the construction of some common array configurations:
    
    """
    
    def __new__(cls, xyz, units=None, antennas=None):

        obj = super().__new__(cls, xyz, units=units)
        if antennas is None:
            obj.isHuygens = True 
        else:
            obj.isHuygens = False
            if xyz.size != antennas.size:
                raise Exception("the size of vector positions must be the same as the number of antennas") 

        
        obj.antennas = antennas
        
        return obj
    
    @property 
    def origin(self):
        """
        Calculates and returns origin (center of mass)
        """
        return np.average(self, axis=0)
 
    def transformToEllipsoid(self,edef):
        """
        moves array elements to surface of ellipsoid either by wrapping or projection
        """
        pass
              
    
    def getEMField(self, driveVoltages, freq, obsPts):
        """
        Returns an EM pattern.
    
            if antennas==none then its marked as an array factor
        """
        w = 2*np.pi/freq.wavelength
        fieldAmp = Voltage(np.zeros(obsPts.shape[0]))
        
        if self.isHuygens:
            for oindx,obsxyz in enumerate(obsPts):
                for aindx,arxyz in enumerate(self):
                    if obsPts.units == "unit":
                        fieldAmp[oindx] += driveVoltages[aindx]*np.exp(1J*w*np.dot(arxyz, obsxyz))
                    else:
                        fieldAmp[oindx] += driveVoltages[aindx]*np.exp(1J*w*np.abs((arxyz - obsxyz).rv))
            res = ArrayFactor(Voltage(fieldAmp / self.shape[0]) , freq,  obsPts)
        else:
            tau = np.zeros(obsPts.shape[0])
            lamda = np.zeros(obsPts.shape[0])
            for oindx, obsxyz in enumerate(obsPts):
                totalEMF = Voltage([0, 0])
                for aindx,arxyz in enumerate(self):
                    antEMF = self.antennas[aindx].getEMField(driveVoltages[aindx], freq, obsxyz)
                    totalEMF += antEMF.volta*antEMF.pol.cirPolVec()
                    if obsPts.units == "unit":
                        fieldAmp += antEMF.volts*np.e(1J*w*np.dot(arxyz, obsxyz))
                    else:
                        fieldAmp += antEMF.volts*np.e(1J*w*np.mag(arxyz - obsxyz)) 
                    apol = Polarization.fromCirPolVec(totalEMF)
                    tau[oindx] = apol.tau
                    lamda[oindx] = apol.lamda                  
            res = EMField(fieldAmp, freq, Polarization(tau, lamda), obsPts)
        
        return res
    
    @classmethod
    def createPlanarArray(cls, xNum, xInc, xOffset=0, yNum=1, yInc=0, yOffset=0, units=None, antennas=None):
        """
        This class method is a helper function to aid in the creation of
        planar and linear arrays.  The array is defined in the x-y plane and
        must be rotated and offset as desired. Optional alternating offsets in both x and/or
        y along for triangular type grids.
        
        """
        xExtent = (xNum-1)*xInc + xOffset
        yExtent = (yNum-1)*yInc + yOffset
        
        xyz = []
        for yIndx in range(yNum):
            for xIndx in range(xNum):
                x = xIndx * xInc + ((yIndx%2) - 1)*xOffset/2 - xExtent/2
                y = yIndx * yInc + ((xIndx%2) - 1)*yOffset/2 - yExtent/2
                z = 0
                xyz.append([x, y, z])
        return Array(xyz, units=units, antennas=antennas)
    
       