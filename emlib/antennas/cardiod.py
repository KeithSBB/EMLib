'''
Created on Apr 1, 2024

@author: keith
'''
import numpy as np
from .antBaseClass import AntBase  # @UnresolvedImport

from ..quant.quants import Vector, Angle, Voltage # @UnresolvedImport
from ..fields.emField import EMField
from ..fields.polarization import  Polarization # @UnresolvedImport

class Cardiod(AntBase):
    '''
    classdocs
    '''
    refOrient = Vector([1, 0, 0])

    def __init__(self, origin=Vector([0,0,0]), worientQ=None, EFactor=1, HFactor=1):
        '''
        Constructor
        '''
        super().__init__(origin, worientQ)
        self.kwparams['EFactor'] = EFactor
        self.kwparams['HFactor'] = HFactor
        
    def getEMField(self, driveVoltage, freq, obsPts):
        '''
        EFactor and HFactor are used to narrow up pattern
        in the E or H planes
        '''
        
        # Transform obsPts back to /antenna's local coordinate system
        if self.worientQ is None:
            antLocalObsPts = obsPts
        else:
            antLocalObsPts = self.worientQ.forwardRotate(obsPts)
            
        theta = antLocalObsPts.theta
        phi = antLocalObsPts.phi
        
        volts = driveVoltage * Voltage(((1 - np.cos(theta))/2)**(np.cos(phi)**self.kwparams['EFactor'] + 
                                                                np.sin(phi)**self.kwparams['HFactor']))
        
        # for this model the polarization is vertical linear
        # in the x-y plane the E field aligns with the y-axis so the 
        # tilt angle wrt utheta must constantly change to maintain this alinment
        # and xi - 0 everywhere
        
        # y = ur*sin(theta)sin(phi) = 1
        # x = ur*sin(theta)cos(phi) = 0
        # z = ur*cos(theta) = 0   thus sin(theta = 1)
        
        tau = Angle(np.pi/2) + phi * (2*(theta >= np.pi/2) - 1)
        
        pol = Polarization(tau, Angle(np.zeros(tau.size)))
        return EMField(volts, freq, pol, obsPts)
    
    
    def get3DModel(self):
        verts = Vector([[-10, 10, 5], #0
                 [10, 10, 5], #1
                 [10, -10, 5], #2
                 [-10, -10, 5], #3
                 [-8, 8, 5],   #4
                 [8, 8, 5],  #5
                 [8, -8, 5], #6
                 [-8, -8, 5], #7
                 [-5, 5, -5], #8
                 [5, 5, -5], #9
                 [5, -5, -5], #10
                 [-5, -5, -5], #11
                 [-3, 3, -5], #12
                 [3, 3, -5], #13
                 [3, -3, -5], #14
                 [-3, -3, -5],#15
                 [-2, 10, 5],
                 [2, 10, 5],
                 [0, 14, 5]] )
        

        verts = self.worientQ.activeRotate(verts) + self.origin 
        
        tri = [[0, 1, 4],
               [1, 4, 5],
               [1, 2, 5],
               [2, 6, 5],
               [2, 3, 6],
               [6, 3, 7],
               [3, 0, 4],
               [4, 7, 3],
               [0, 8, 9],
               [0, 1, 9],
               [1, 9, 10],
               [1, 2, 10],
               [2, 3, 11 ],
               [2, 11, 10],
               [3, 0, 8],
               [8, 3, 11],
               [16, 17, 18]]
        return (np.array(verts), np.array(tri))
    
    
    
    
        