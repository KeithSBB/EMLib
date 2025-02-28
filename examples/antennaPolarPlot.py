'''
Simple cardiod Antenna pattern
===============================

Shows use of quaternion rotations and 
Plots polar and 3D patterns of cardiod antenna
'''
import emlib as em



# Define the phi planar pattern observation points
phicut = em.ObsPoints.makePlanarCut(cutType='Phi', 
                                    params=[ em.Angle(-180, units='deg'),
                                             em.Angle(180, units='deg'), 
                                             em.Angle(1, units='deg'), 
                                             em.Angle(90, units='deg')])

# The cardiod antenna model is defined in the x-y plane with
# boresight looking up (+z-axis)
# We wan to rotate it about the y-axis so boresight is looking out the +x-axis
# and to make it more interesting rotate it again by 10 degs about the z-axis
q1 = em.Quaternion(em.Angle(90, 'deg'), em.Vector([0, 1, 0]))
q2 = em.Quaternion(em.Angle(-10, 'deg'), em.Vector([0, 0, 1]))
#Note: multiplying quaternions creates the equivalent of applying rotations
#      in right-to-left order
qt = q2 * q1

origin = em.Vector([0,0,0])

# Create the cardiod antenna
ant = em.ant.Cardiod( origin=origin, worientQ=qt, EFactor=0.0, HFactor=0.0)
# Generate the phi-cut EM field pattern
pat = ant.getEMField( em.Voltage(1), em.Frequency(1000), phicut)
# Plot in polar coordinates
em.polarPlot(pat, None, range_r=-40)

# Create observation points uniformly over a sphere for 3D plot
obsPts = em.ObsPoints.makeUniformSphericalGrid(numPts=1000)
# Generate the 3D EM field pattern
pat3D = ant.getEMField( em.Voltage(1), em.Frequency(1000), obsPts)
# Plot in 3D
em.Plot3Dpat(pat3D, rangeMin_r=-40)   



    
    
    
    
    
    
# Timer(1, open_browser).start()
# app.run(debug=True, port=1222) 