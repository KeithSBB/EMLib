'''
Created on Apr 1, 2024

@author: keith
'''

from ..quant.quants import Vector, Frequency, Voltage  # @UnresolvedImport
from ..fields.emField import EMField  # @UnresolvedImport
 
class  AntBase(object):
    '''
    classdocs
    '''
    refOrient =  None #Model definition orientation in Local Coordinates

    def __init__(self, origin, worientQ):
        '''
        Constructor
        '''
        self.origin = origin       # origin = 0,0,0 for local otherwise World offset
        self.worientQ = worientQ      #quaternion to orient in World Coordinates 
        self.kwparams = {}   #Each model may have a unique set of model key:value pairs
        
        
    def getEMField(self, driveVoltage, freq, obsPts):
        '''
        This is where the actual model code is placed
        '''
        return EMField()
    
    def get3DModel(self):
        return None
    
    def vectorTo(self, otherant):
        return otherant.origin - self.origin
    
    def world2local(self, v):
        return self.worientQ.rotate(v)