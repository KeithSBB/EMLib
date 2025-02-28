

import numpy as np
from enum import property
from scipy.constants import c
from pip._vendor.pygments.formatters import other
from statsmodels.sandbox.distributions import otherdist


class RQuantBase(np.ndarray):
    '''
    RQuantBase
  
    
    This is the abstract base class for all quantities
    
    '''
    
    MKSunit = ''
    unit2MKS ={}

    def __new__(cls, input_array, units=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        if units is None:
            units = cls.MKSunit
            obj = np.asarray(input_array).view(cls)
        else:
            obj = cls.unit2MKS[units] * np.asarray(input_array).view(cls)
           
        # add the new attribute to the created instance
        obj.units = units
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.units = getattr(obj, 'units', None)
        
    def u(self, units=None):
        if units is None:
            units = self.MKSunit
        return np.asarray(self / self.unit2MKS[units])
     
    @property
    def rv(self):
        return np.ravel(self)  
        
    def __str__(self):
        return f'{self.u(self.units)} {self.units}'
        
        
class Length(RQuantBase):
    """
    Length(value, units=None)
    
    
    A  numpy array type class for values representing length.
    The default units are meters.
    
    :param value:  Any real number, list of numbers or numpy array
    :param units:  default=None.  A string defining the units of value
    
    units may optional be set to:
        'm' 
        'cm' 
        'mm' 
        'um' 
        'µm' 
        'nm'
        'pm'
        'in' 
        'ft' 
        'yd'
        'mi'    
        'nmi'
        'mils'
    
    
    """
    MKSunit = 'm'
    unit2MKS = {'m':1, 
                'cm':1e-2, 
                'mm':1e-3, 
                'um':1e-6, 
                'µm':1e-6, 
                'nm':1e-9, 
                'pm':1e-12,
                'in':0.0254, 
                'ft':0.3048, 
                'yd':0.9144,
                'mi':1609.344,     #miles
                'nmi':1852,        #nautical miles
                'mils':0.0254e-3}        
        
class Time(RQuantBase):  
    """
    Time(value, units=None)
    
    A  numpy array type class for values representing time.
    The default units are seconds.
    
    :param value:  Any real number, list of numbers or numpy array
    :param units:  default=None.  A string defining the units of value
    
    units may optional be set to:
        'sec'
        's'
        'ms'
        'us'
        'µs'
        'ns'
        'ps'
        'min'
        'hr'
        'day'
        'wk'
        'yr'
    """      
    MKSunit = 'sec'
    unit2MKS = {'sec':1, 
                 's':1,
                'ms':1e-3, 
                'us':1e-6, 
                'µs':1e-6, 
                'ns':1e-9, 
                'ps':1e-12,
                'min':60 , 
                'hr':3600, 
                'day':86400, 
                'wk':604800, 
                'yr':31557600} #julian year       
        
class Mass(RQuantBase): 
    """
    Mass(value, units=None)
    
    A  numpy array type class for values representing mass.
    The default units are kg.
    
    :param value:  Any real number, list of numbers or numpy array
    :param units:  default=None.  A string defining the units of value
    
    units may optional be set to:
        'kg
        'g'
        'mg
        'ug
        'µg
        'ng
        'oz
        'lb
        't'
    """       
    MKSunit = 'kg'
    unit2MKS = {'kg':1, 
                'g':1e-3,
                'mg':1e-6,
                'ug':1e-9,
                'µg':1e-9,
                'ng':1e-12,
                'oz':35.27396,
                'lb':2.204623,
                't':0.001102311}    #ton    
        
class Voltage(RQuantBase):        
    """
    Voltage(value, units=None)
    
    A  numpy array type class for values representing voltage.
    The default units are volts.
    
    :param value:  Any real number, list of numbers or numpy array
    :param units:  default=None.  A string defining the units of value
    
    units may optional be set to:
        'v':
        'mV'
        'kV'
        'eV'
    """ 
    MKSunit = 'V'
    unit2MKS = {'V':1, 
                'mV':1e-3, 
                'kV':1e3, 
                'eV':116000} 
    
    def __new__(cls, input_array, units=None):
        return super().__new__(cls, np.asarray(input_array).astype(complex) , units=None) 
    
    def u(self, units=None):
        if units is not None and units == 'dB': 
            obj = 20 * np.log10(np.abs(super().u('V')))  
        else:
            obj = super().u(units)
        return obj 
    
    def __mul__(self, other):
        """
        Multiplication of a voltage obj times a Current obj
        returns a Power obj
        """ 
        if isinstance(other, Current):
            res = Power(np.asarray(self)*other) 
        else:
            res = np.asarray(self) * other
        return res
    
    def __div__(self, denom):
        """
        Division of a voltage obj by a Current obj returns
        a Resistance obj
        """
        if isinstance(other, Current):
            res = Resistance(self/other)
        else:
            res = self/other 
        return res
        
class Resistance(RQuantBase):        
    """
    Resistance(value, units=None)
    
    A  numpy array type class for values representing resistance.
    The default units are ohms.
    
    :param value:  Any real number, list of numbers or numpy array
    :param units:  default=None.  A string defining the units of value
    
    units may optional be set to:
        'ohm'
        'Ω'
        'mohm'
        'mΩ'
        'kohm'
        'kΩ'
    
    """ 
    MKSunit = 'ohm'
    unit2MKS = {'ohm':1,
                'Ω':1,
                'mohm':1e-3,
                'mΩ':1e-3,
                'kohm':1e3,        
                'kΩ':1e3}        
        
class Current(RQuantBase):  
    """
    Current(value, units=None)
    
    A  numpy array type class for values representing electrical current.
    The default units are amperes.
    
    :param value:  Any real number, list of numbers or numpy array
    :param units:  default=None.  A string defining the units of value
    
    units may optional be set to:
        'A'
        'mA'
        'uA'
        'µA'
        'kA'
    """       
    MKSunit = 'A'  #amperes
    unit2MKS = {'A':1, 
                'mA':1e-3, 
                'uA':1e-6,
                'µA':1e-6,
                'kA':1e3
                 } 
    
    def __mul__(self, other): 
        if isinstance(other, Voltage):
            res = Power(self*other) 
        else:
            res = self*other
        return res
       
        
class Angle(RQuantBase): 
    """
    Angle(value, units=None)
    
    A  numpy array type class for values representing angles
    The default units are radians.
    
    :param value:  Any real number, list of numbers or numpy array
    :param units:  default=None.  A string defining the units of value
    
    units may optional be set to:
        'rad'
        'deg'
    """        
    MKSunit = 'rad'
    unit2MKS = {'rad':1, 'deg':(np.pi / 180.0) } 
    
    def normPMpi(self):
        #return (( -self + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0 
        return np.arctan2(np.sin(self), np.cos(self))      
        
class Frequency(RQuantBase): 
    """
    Frequency(value, units=None)
    
    A  numpy array type class for values representing frequency.
    The default units are meters.
    
    :param value:  Any real number, list of numbers or numpy array
    :param units:  default=None.  A string defining the units of value
    
    units may optional be set to:
        'Hz'
        'kHz'
        'MHz'
        'GHz'
    """        
    MKSunit = 'Hz'
    unit2MKS = {'Hz':1, 
                'kHz':1e3, 
                'MHz':1e6, 
                'GHz':1e9} 
    
    @property
    def wavelength(self):
        """
        Returns the freespace wavelength as a Length object
        """
        return Length(c / self)
    
class Power(RQuantBase): 
    """
    Power(value, units=None)
    
    A  numpy array type class for values representing power.
    The default units are watts.
    
    :param value:  Any real number, list of numbers or numpy array
    :param units:  default=None.  A string defining the units of value
    
    units may optional be set to:
        'W'
        'mW'
        'kW'
        'MW'
    """        
    MKSunit = 'W'
    unit2MKS = {'W':1, 
                'mW':1e-3, 
                'kW':1e3, 
                'MW':1e6}        
        
class Vector(Length):
    """
    Vector(value, units=None)
    
    A  numpy array type class for values representing x-y-z vectors.
    The default units are meters.
    
    :param value: Must be divisable by 3, list of numbers or numpy array
    :param units:  default=None.  A string defining the units of value
    
    units may optional be set to:
        'm' 
        'cm' 
        'mm' 
        'um' 
        'µm' 
        'nm'
        'pm'
        'in' 
        'ft' 
        'yd'
        'mi'    
        'nmi'
        'mils'
        'unit'       -This has special meaning and the vectors will be unit
    """ 
    
    unit2MKS = dict(Length.unit2MKS, **{"unit":1})
    
    def __new__(cls, input_array, units=None):
        # force shape and reject wrong size
        if units != "unit":
            obj = super().__new__(cls, np.reshape(input_array, (-1,3)), units)
        else:
            v = np.reshape(input_array, (-1,3))
            obj = super().__new__(cls, v/np.linalg.norm(v, axis=1, keepdims=True), units)
        return obj
    

    @property
    def theta(self):
        """
        returns the spherical theta coordinate as an Angle obj
        """
        return Angle(np.arccos(self[:,2]/np.linalg.norm(self, axis=1)))
                     
    @property
    def phi(self):
        """
        returns the spherical phi coordinate as an Angle obj
        """
        return Angle(np.arctan2(self[:,1], self[:,0]))
    
    @property
    def r(self):
        """
        returns the spherical r coordinate as a Length obj
        """
        return Length(np.linalg.norm(self, axis=1), self.units) 
    
    @property
    def norm(self):
        """
        Normalizes the vectors to all be unit vectors
        """
        return self / np.linalg.norm(self, axis=1)
    
    def rotateBy(self, q):
        """
        Performs a quaternion rotation on this vector obj
        :param q:  a Quaternion obj
        """
        for row in range(len(self)):
            self[row,:] = np.array(q.rotate(self[row, :]))
    
 
class ObsPoints(Vector):  
    """
    ObsPoints(value, units=None)
    
    A  numpy array type class for values representing observation points.
    This class is similar to Vector, but includes class methods for creating
    vectors that support 3D and linear pattern cuts.
    The default units are meters.
    
    :param value:  Any real number, list of numbers or numpy array
    :param units:  default=None.  A string defining the units of value
    
    units may optional be set to:
        'm' 
        'cm' 
        'mm' 
        'um' 
        'µm' 
        'nm'
        'pm'
        'in' 
        'ft' 
        'yd'
        'mi'    
        'nmi'
        'mils'
        'unit'       -This has special meaning and the vectors will be unit
    """    
       
    def __new__(cls, input_array, units=None, pattype=None, patinfo={}):
        obj = super().__new__(cls, input_array, units=units)
        obj.pattype = pattype
        obj.patinfo=patinfo
        return obj
         
         
         
            
    @classmethod
    def fromSpherical(cls, phi, theta, r, makeUnit=False):
        '''
        Creates an ObsPoints obj from spherical coordinates
        '''
        assert isinstance(phi, Angle), "phi must be an Angle object"
        assert isinstance(theta, Angle), "theta must be an Angle object"
        assert isinstance(r, Length), "r must be a Length object"
        
        x = Length(r * np.cos(phi) * np.sin(theta), units=r.units)
        y = Length(r * np.sin(phi) * np.sin(theta), units=r.units)
        z = Length(r * np.cos(theta), units=r.units)
        return cls.fromXYZ(x, y, z, makeUnit=makeUnit, pattype="Spherical")
    
    @classmethod    
    def fromXYZ(cls, x, y, z, makeUnit=False, pattype=None, patinfo={}):
        """
        Creates an ObsPoints obj from x, y and z values
        """
        # add assert for Length
        # deal with units ?
        assert isinstance(x, Length)
        assert isinstance(y, Length)
        assert isinstance(z, Length)
        if pattype is None:
            pattype = "XYZ"
        if makeUnit:
            units = 'unit'
        else:
            units = x.units
        if x.shape == ():
            xyz = np.array(([x], [y], [z]), dtype=float)
        else:
            xyz = np.array((x, y, z), dtype=float)
            
        return ObsPoints(np.swapaxes(xyz, 0, 1), units=units, pattype=pattype, patinfo=patinfo)
    
    @classmethod
    def makeUniformSphericalGrid(cls, numPts=1000, radius=1.0, units=None):
        """
        cretes an obsPoints obj that has points that are evenly distributed
        over a  sphere
        """
        points = []
        phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

        for i in range(numPts):
            y = (1 - (i / float(numPts - 1)) * 2 ) # y goes from 1 to -1
            radius2d = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius2d
            z = np.sin(theta) * radius2d

            points.append((radius*x, radius*y, radius*z))
        return ObsPoints(points, units=units, pattype="UniformSphericalGrid", patinfo={"numPts":numPts})
    
    @classmethod
    def makePlanarCut(cls, cutType="Arbitrary", params=[Length([0,-1,0]), 
                                                        Angle(np.pi/2), 
                                                        Angle(0.1), 
                                                        Length([0,0,1])], units=None):
        '''

        
        Three points define a plane. The origin could be used as one point.
        some common planar cuts cutType:
        
        1. Azimuth: varies in az at a fixed el: [azStart, azStop, azInc, fixedEl] 

        2. Phi: varies in phi at a fixed theta: [phiStart, phiStop, phiInc, fixedtheta]

        3. Elevation: varies in el at fixed az: [elStart, elStop, elInc, fixedaz]

        4. Theta: varies in theta at fixed phi: [thetaStart, thetaStop, thetaInc, fixedPhi]

        5. Arbitrary: rotation about a vector (quaternion) [startPointxyz, angleStop, angleInc, rotaxis]
        Note # 5 includes 1 - 4

        az = -phi and el = 90 - theta
        '''
        if cutType == "Azimuth":
            startVec = ObsPoints.fromSpherical(-params[0], Angle(np.pi/2) - params[3], Length(1.0 ), makeUnit=True)
            angleStart = -params[0].normPMpi()
            angleStop = -params[1].normPMpi()
            angInc = -params[2].normPMpi()
            qinc = Quaternion(angInc, Vector([0,0,1], units='unit'))
            patinfo = {"start":angleStart, 'stop':angleStop,'inc':angInc, 'Fixed': params[3]}
        elif cutType == "Phi":
            startVec = ObsPoints.fromSpherical(params[0], params[3], Length(1.0) , makeUnit=True)
            angleStart = params[0].normPMpi()
            angleStop = params[1].normPMpi()
            angInc = params[2].normPMpi()
            qinc = Quaternion(angInc, Vector([0,0,1], units='unit'))
            patinfo = {"start":angleStart, 'stop':angleStop,'inc':angInc, 'Fixed': params[3]}

        elif cutType == "Elevation":
            #TODO: fix negative elevation
            startVec = ObsPoints.fromSpherical(-params[3], (Angle(np.pi/2) - params[0]), Length(1.0) , makeUnit=True)
            angleStart = (Angle(np.pi/2) - params[0]).normPMpi()
            angleStop = (Angle(np.pi/2) - params[1]).normPMpi()
            angInc = -params[2].normPMpi()
            rotVec = ObsPoints.fromSpherical(Angle(np.pi/2)-params[3], Angle(np.pi/2), Length(1.0) , makeUnit=True)
            qinc = Quaternion(angInc, rotVec)
            patinfo = {"start":angleStart, 'stop':angleStop,'inc':angInc, 'Fixed': params[3]}
        elif cutType == "Theta":
            startVec = ObsPoints.fromSpherical(params[3], params[0], Length(1.0), makeUnit=True)
            angleStart =  params[0].normPMpi()
            angleStop = params[1].normPMpi()
            angInc = params[2].normPMpi()
            rotVec = ObsPoints.fromSpherical(Angle(np.pi/2)-params[3], Angle(np.pi/2), Length(1.0) , makeUnit=True)
            qinc = Quaternion(angInc, rotVec)
            patinfo = {"start":angleStart, 'stop':angleStop,'inc':angInc, 'Fixed': params[3]}
        elif cutType == "Arbitrary":
            startVec = params[0]
            angleStart = Angle(0)
            angleStop = params[1].normPMpi()
            angInc = params[2].normPMpi()
            qinc = Quaternion(angInc, params[3])
            patinfo = {"start":angleStart, 'stop':angleStop,'inc':angInc, 'rotaxis': params[3], 'startVec':params[0]}
        else:
            raise Exception("Unknown cutType")            
         
        if angInc < 0:
            if angleStart < angleStop:
                angleStart = Angle(2*np.pi) + angleStart
        else:
            if angleStop < angleStart: 
                angleStop = Angle(2*np.pi) + angleStop    
            
        def cumRotate(qinc, startvec, angleStart, angleStop, angInc):
            #print(startvec)
            aVec = startVec
            #print(angleStart.u('deg'))
            #print(angInc)
            ang = angleStart
            minAngDiff = Angle(3*np.pi)
            curAngDiff = Angle(2*np.pi)
            while curAngDiff < minAngDiff:
                minAngDiff = curAngDiff
                yield aVec
                ang += angInc
                
                curAngDiff = Angle(np.abs(angleStop - ang)) #.normPMpi()
                
                #print(f"{ang.u('deg')} d = {curAngDiff.u('deg')}")
                aVec = Vector(qinc.forwardRotate(aVec))
            #print("ignore last")
        return ObsPoints([v for v in cumRotate(qinc, startVec, angleStart, angleStop, angInc)], units=units, pattype=cutType, patinfo=patinfo)
            
            

            
        
        
class Quaternion(object): 
    """
    This is a limited quaternion class that performs 3D rotations on Vector arrays.
    It has two constructor:
    1. Quaternion(angle, axis)
    :param angle:  An Angle object with a single value that is the CCW rotation about axis
    :param axis:  A single row Vector defining the xyz axis of rotation
    
    2. Quaternion(q0, q1, q2, q3)
    :param q0,q1, q2, q3:   the four elements of a quaternion
    
    """
    
    def __init__(self, *args):
        if len(args) == 2:
            # axis, angle
            assert(isinstance(args[0], Angle))
            assert(isinstance(args[1], Vector))    
            self.q0 = np.cos(args[0].u()/2)
            self.q1 = args[1][0,0]*np.sin(args[0].u()/2)
            self.q2 = args[1][0,1]*np.sin(args[0].u()/2)
            self.q3 = args[1][0,2]*np.sin(args[0].u()/2)
        elif len(args) == 4:
            # q0, q1, q2, q3 
            self.q0 = args[0]
            self.q1 = args[1]
            self.q2 = args[2]
            self.q3 = args[3]
        else:
            raise Exception("unknown number of arguments")
        
    def __str__(self): 
        return f"Quaternion: q0={self.q0}, q1={self.q1}, q2={self.q2}, q3={self.q3} "                  
    
    def __mul__(self, other): 
        assert(isinstance(other, Quaternion)) 
        q0 =  self.q0*other.q0 - self.q1*other.q1 - self.q2*other.q2 - self.q3*other.q3
        q1 =  self.q0*other.q1 + self.q1*other.q0 + self.q2*other.q3 - self.q3*other.q2
        q2 =  self.q0*other.q2 - self.q1*other.q3 + self.q2*other.q0 + self.q3*other.q1
        q3 =  self.q0*other.q3 + self.q1*other.q2 - self.q2*other.q1 + self.q3*other.q0
        return Quaternion( q0, q1, q2, q3)
     
    @property    
    def inverse(self):
        return Quaternion( self.q0, -self.q1, -self.q2, -self.q3 )

    
    def reverseRotate(self, xyzs):
        res = 0.0 * xyzs
        qinv = self.inverse
        for indx,xyz in enumerate(xyzs):
            p = Quaternion(0.0, xyz[0], xyz[1],xyz[2])
            pp = qinv*p*self
            res[indx,:] = Vector([pp.q1, pp.q2, pp.q3])
            
        return res
            
    def forwardRotate(self, xyzs):
        res = 0.0 * xyzs
        qinv = self.inverse
        for indx,xyz in enumerate(xyzs):
            p = Quaternion(0.0, xyz[0], xyz[1],xyz[2])
            pp = self*p*qinv
            res[indx,:] = Vector([pp.q1, pp.q2, pp.q3])
            
        return res
            
        
        
                    
   
        
if __name__ == "__main__":

    #print(Vector.fromSpherical(Angle([10,20], 'deg'), Angle([45,67], 'deg'), Length([2, 100])))  
    # mtobspts = ObsPoints.makePlanarCut(cutType='Elevation', 
    #                                    params=[Angle(-180, units='deg'), 
    #                                            Angle(180, units='deg'), 
    #                                            Angle(10, units='deg'), 
    #                                            Angle(0, units='deg')])   
    # print(["%.0f"% num for num in (Angle(np.pi/2) - mtobspts.theta).u('deg')])
    # print(["%.0f"% num for num in ( - mtobspts.phi).u('deg')])
    #
    p = Vector([0,0,1])
    print(p)
    q = Quaternion(Angle(90, 'deg'), Vector([0, 1, 0]))
    print(q)
    print(q.passiveRotate(p))
    
    