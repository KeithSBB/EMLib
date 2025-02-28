'''
The EMLib Library
=================

EMLib is a package which provides a set of classes and functions
used for *System* level modeling of antennas, antenna arrays and electromagnetic fields.

With this library you can model single or a 3D system of antennas and/or antenna arrays.
Calculate their field patterns, analize polarization, coupling between antennas and array
feed networks.  It provides web based plotting functions for 3D layout, polar plots and 
3D field patterns.  It includes a polarization application that supports analysis of
polarization diverse antennas.  Antenna, Array and EM field objects may be arbitrarily 
rotated and/or offset in world coordinates.  

This library is not appropriate for detailed design as it does not peform MoM or finite element
modeling.
'''
__version__ = 'dev'
from .quant.quants import Length, Vector, Time, Mass, Angle, Voltage, ObsPoints, Resistance, Current, Power, Frequency, Quaternion
import emlib.antennas as ant
from .antArrays.antArray import AntArray
from .antArrays.beamformers import SimpleBF
from .fields.emField import EMField, ArrayFactor
from .fields.polarization import Polarization
from .plot.plotFuncs import polarPlot, Plot3D, Plot3Dpat, Plot3Dpol
from .plot.poincarePlot import PoincarePlot, PolarizationApp



