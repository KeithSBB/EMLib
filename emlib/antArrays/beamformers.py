'''
Created on May 31, 2024

@author: keith
'''
from ..quant.quants import Voltage, Angle, Frequency, Vector  # @UnresolvedImport
from ..fields.polarization import Polarization  # @UnresolvedImport
from ..fields.emField import EMField, ArrayFactor
import numpy as np
from abc import ABC, abstractmethod
from  .antArray import AntArray


class Beamformer(ABC):
    '''
    this class takes an Array object and provides the base class for all Beamformers    '''


    def __init__(self, arrayObj):
        '''
        Constructor
        
        '''
        assert(isinstance(arrayObj, AntArray))
        self.array = arrayObj
    
    @abstractmethod 
    def generateExcitations(self ):
        pass
    
    
class SimpleBF(Beamformer):
    
    def generateExcitations(self, freq, theta, phi ):
        '''
        Generates phasing so that a beam is formed in the theta, phi direction
        '''
        w = 2*np.pi/freq.wavelength
        dirxyz = Vector([np.sin(theta)*np.cos(phi),
                        np.sin(theta)*np.sin(phi),
                        np.cos(theta)], units='unit')
        
        driveVoltages = Voltage(np.zeros(self.array.shape[0]))
        for  aindx,arxyz in enumerate(self.array):
            driveVoltages[aindx] = np.exp(-1J*w*np.dot(arxyz, dirxyz[0]))
        return driveVoltages
    
    
    
        