# EMLib
EMLib is a library of python classes and functions for the system level modeling of Antennas, Arrays and electromagnetic fields. This library is composed of the following submodules:
## Quant
A set of classes that represent quantities that may be created with arbitrary units.  These are subclasses of numpy arrays and internally the data values are stored in the MKS system. The following classes are defined:
- **RQuantBase(input_array, units=None)**:  this is the base class that all other quant classes are built from.
    - *input_array:* maybe a scalar, list or numpy array of values
    - *units:* A text string defining the units of the values, if left as None then the default units is assumed.
    **Methods:**
    - self.u(unit) where unit is string for values in desired units to be returned
- **Length**:  internally as meters m accepts: mm, km, in, mil, yd, ft, mi
- **Vector**: Based on Length and forced to be three columns for x-y-z
- **Angle:**  Internally in radians rad accepts: deg
- **Frequency**: internally in hertz Hz accepts: KHz, MHz, GHz
- **Mass**: internally in kilograms kg accepts: g, lbs
- **Time**: internally in seconds sec accepts: ms, min, hrs, days, wks, mn, yr
- **Voltage**: internally in volts v accepts:  dB *Note: need to change to dBm, or dBv ?*
- **Resistance**: internally in ohms accepts: mohm
- **Current**: internally in amperes amp accepts: ma
- **Power**: internally in watts w, accepts: Kw Mw

*todo*
- [ ] Complete entering correct conversion factors for all quantity classes
- [ ] correctly implement dB for voltage, current and Power classes
- [ ] implement __mul__ and __div__ for creation of Power from Voltage, Current and Resistance and other relations

## Fields
A set of classes used for EM fields.  these include:
- **EMField**: This class takes arrays of Voltage, Polarization and Vector of observation points as well as a scalar Frequency.  The number of observation point rows must equal the number of Voltage and Polarization items.
- **Polarization**: This class takes two arrays (which may only have one element) one for the tilt angle &tau; and one for &xi; the ellipticity (+45 deg is RHCP, 0  deg is linear and -45 deg is LHCP)

*todo*
- [ ] figure out how to compute polarization mismatch loss when one is receding ad the other is approaching

## Antennas
A set of classes that represent single antennas.  Antenna class may have arbitrary keyword parameters specific to the particular class.   They Have a world coordinate origin (defaults to 0,0,0) and a worientQ (a Quaternion to transform vis rotation to world coordinates.  Antenna objects provide a physical mesh of the antenna, create EMFields, calculate vectors to other antenna objects and tranforms between Local and world Coordinates
- **Cardiod**: Generates a linear polarized cardiod pattern.  Local coordinates are in the X-Y plane looking towards the +Z direction with the linear polarization aligned with the +Y axis.

*todo*
- [ ] fix E and H plane factor problem in cardiod

## Arrays
A class that represent antenna arrays.  
- **AntArray**: Provides several different ways to generate arrays including linear, planar with optional stagger, curved from radius, on a sphere with radius, arbitary x-y-z.  antarrays can either be Huygen's source that generate an array factor that can be multiplied times and embedded Antenna pattern, or a collection of arbitrary types and orientations of Antenna objects which produce the total field pattern.

*todo*
- [ ] write basic array class

## RotmanLens
A Class that represents a rotman Lens.
- **RotmanLens**: Takes several parameters that define the Rotman Lens and is used to calculated the drive voltages of an antenna array.

*todo*
- [ ] lacate rotman Lens theory document

## Plot
A set of functions which provide 2D rectangular and polar plots of EMField patterns, and 3D plots of EMField patterns, Antenna layout, and Poincare sphere.  There are also two specialized interactive applications for Polarization and Rotman Lens design.
- **polarPlot(emfield, raxis='auto', aaxis='auto')**: Takes a EMField calculated on a planar cut though an antenna's 3D field.
    - *emfield:* an EMFiled object that is planar. 	Typically either &theta; or &phi; is fixed while the other varies.
    - *raxis:* radial axis may be ['auto' | [min, max ,inc] ]
    - *taxis:* angular may be ['auto' | [min, max, inc] ]
- **Plot3Dpat{emfield, rais-'auto')**: Creates a 3D colored mesh of a 4pi steradian EMField
    - *emfield:* an EMFiled object that sampled evenly over a sphere. 	See Vector
    - *raxis:* radial axis may be ['auto' | [min, max ,inc] ]
- **Plot3D(objs)**: takes antenna and other physical objects and plots them in 3D world coordinates
    - *objs:*  a list of one or more objects which have a get3DModel() method.

*TODO*
- [ ] Write polar plot
- [ ]add polarization plane mesh for Plot3D