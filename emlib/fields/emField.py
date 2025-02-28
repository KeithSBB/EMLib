'''
Created on Mar 30, 2024

@author: keith
'''

from ..quant.quants import Voltage, Angle, Frequency, Vector  # @UnresolvedImport
from ..fields.polarization import Polarization  # @UnresolvedImport
import numpy as np
from enum import property

class ArrayFactor(object):
    def __init__(self, volts, freq, obsPts, origin=Vector([0,0,0])):
        '''
        Constructor
        '''
        assert isinstance(volts, Voltage)
        assert isinstance(freq, Frequency)
        assert isinstance(obsPts, Vector)
        self.volts = volts
        self.freq = freq
        self._obspts = obsPts 
        self.origin = origin
        
    @property
    def obsPts(self):
        return self._obspts 
    
    
    def __mul__(self, other):
        assert(isinstance(other, ArrayFactor))
        assert(np.all(self.obsPts == other.obsPts))
        assert(self.freq == other.freq)
        
        if isinstance(other, ArrayFactor) and isinstance(self, ArrayFactor):
            res = ArrayFactor(self.volts*other.volts, self.freq, self.obsPts, origin = (self.origin + other.origin)/2)
        else:
            res = EMField(self.volts*other.volts, self.freq, self.obsPts, origin = (self.origin + other.origin)/2)
            
        return res  

class EMField(ArrayFactor):
    '''
    classdocs
    '''


    def __init__(self, volts, freq, pol, obsPts, origin=Vector([0,0,0])):
        '''
        Constructor
        '''
        assert isinstance(volts, Voltage)
        assert isinstance(freq, Frequency)
        assert isinstance(pol, Polarization)
        assert isinstance(obsPts, Vector)
        super().__init__(volts, freq, obsPts, origin)

        self.pol = pol

        
    def responseTo(self, matchPol=None):
        if matchPol is None:
            return np.abs(self.volts)  #* matchPol * self.pol
        else:
            return np.abs(self.volts*(matchPol * self.pol))
