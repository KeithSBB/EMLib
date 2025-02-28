'''
Quantities Module 
=================

contains classes for created values with units


**Quatities** are subclasses of numpy arrays that represent fundemental quatities.
Their constructors take a values and an optional string units.
The default units are the standard MKS system.  Any other units convert to the MKS 
system before assigning the value to the super class (numpy array)
quatities includes the following:

    1. Base quantities class 
    
    2. length
    
    3. Frequency
    
    4. Voltage
    
    5. current
    
    6. Power 
    
    7. Time 
    
    8. Angle
    
    9. Mass

'''

__all__ = ['Length','Angle','Voltage']
   
        
        
        