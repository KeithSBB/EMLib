'''
Polarization Example
=====================

'''
import emlib as em

# Create a polarization object.   Tauis the tilt angle and lamda is the elliticity

tau1 = em.Angle(90, 'deg')
lamda1 = em.Angle(0, 'deg')
p1 = em.Polarization(tau1, lamda1)
print(p1)
print(p1.poincareLatLong)
print(p1.poincareXyz)
print(p1.gamma)
print(p1.rho_c)
print(p1.cirPolVec)