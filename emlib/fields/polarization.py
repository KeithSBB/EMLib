'''
Created on Mar 28, 2024

@author: keith
'''
import numpy as np
from scipy.optimize import fsolve, least_squares
from emlib.quant.quants import Angle, Length, Vector, Voltage, ObsPoints, Quaternion  # @UnresolvedImport
from enum import property


class Polarization(object):
    '''
    This class creates polarization objects
    Polarization(tau:Angle, lamda:Angle)
    wrt the Poincare sphere, 2*tau is the longitude measure counterclockwise from the H-pl alamdas
    and 2*lamda is the longitude wrt the linear equator'''

    def __init__(self, tau, lamda):
        '''
        Constructor
        tau = dc / 2    where dc is the counterclockwise angle from the Poincare sphere H-alamdas
        lamda = lamda / 2
        '''
        #assert (-np.pi/4 <= lamda <= np.pi/4).all(), "lamda must range from -45 deg to +45 deg"
        assert (isinstance(tau, Angle) and 
                isinstance(lamda, Angle) and 
                (tau.shape == lamda.shape)),'tau and lamda must be Angle objects of the same size'

        self.tau = tau
        self.lamda = lamda
    
        
    def __str__(self):
        return f'Polarization: tau = {self.tau.u("deg")} deg, lamda = {self.lamda.u("deg")} deg'        
    
    @property    
    def poincareLatLong(self):
        return (2 * self.tau, 2 * self.lamda)
    
    @property
    def poincareXyz(self):
        return ObsPoints.fromSpherical( phi=2 * self.tau, theta=Angle(np.pi/2 - 2 * self.lamda), r=Length(1), makeUnit=True)
    
    def __getitem__(self, index):
        #print(self.tau[index])
        #print(self.lamda[index])
        return Polarization(Angle(self.tau[index]), Angle(self.lamda[index]))

    def __setitem__(self, index, pol):
        self.tau[index] = pol.tau 
        self.lamda[index] = pol.lamda 
    
    
    
    def __mul__(self, other):
        '''
        multiplication of two polarization objects returns the mismatch factor.
        '''
        vec1 = self.poincareXyz()
        vec2 = other.poincareXyz()
        aw = np.arccos(np.dot(vec1[0,:], vec2[0,:]))  # arccos of the dot product to get angle between vec1 and vec2
        return np.asarray(np.cos(0.5 * aw)**2)
       
    def rotated(self, polPlaneNormal, q, approaching=False): 
        '''
        polPlaneNormal is a vector normal the the polarization plane in the direction of propagation
        q is the quaternion to be applied.
        We can express q = q-taux * q-phi * q-theta   where q-taux is the rotation that 
        is applied to the tilt angle tau.
        
        1. apply q to polplaneNormal to get the new polPlaneNormalx
        
        2. Compute the unit utheta and uPhi coordinates of the plane tangent to the original normal

        3. Compute the unit uthetax and uPhix coordinates of the plane tangent to the transformed normal

        4. use q to transform the original uTheta to the Transformed uthetaxf at the new normal

        5  Take the dot product between uthetaxf and both uThetax and uPhix to determine the tilt angle rotation
           Which is the same as deltauTheta
           planeNormal and polPlanenormalx, if its negative the flip the sign of lamda
 
        6  The new rotate polariztion is olarization(self.tau + deltautheta.rv, self.lamda)
  
        '''

        polPlaneNormalx = Vector(q.activeRotate(polPlaneNormal), units='unit')
        #print(f"New transformed ur is {polPlaneNormalx}")
        
        ur, utheta, uphi = self.unitVecsOnSphere(polPlaneNormal.theta, polPlaneNormal.phi)
        #print(f'theta = {utheta} , Phi = {uphi}')
        
        urx, uthetax, uphix = self.unitVecsOnSphere(polPlaneNormalx.theta, polPlaneNormalx.phi)
        #print(f'thetax = {uthetax} , Phix = {uphix}')
        
        uthetaxf = Vector(q.activeRotate(utheta), units='unit')
        
        deltautheta = Angle(np.sign(np.inner(uthetaxf, uphix))*np.arccos(np.inner(uthetaxf, uthetax)))
        #print(f"The change in utheta is {deltatheta.u('deg')} degs")
      
        #print(self.tau, deltautheta, self.lamda)
        
        if approaching:
            pass   # need to figure out both lamda sign change AND impact on tau
        
        return Polarization(self.tau + deltautheta.rv, self.lamda)
        
    @property
    def gamma(self):
        return  Angle((np.pi/2.0 - 2.0*self.lamda.u())/2.0)   
        
    @property
    def axialRatio(self):
        return (np.abs(self.rho_c) + 1) / (np.abs(self.rho_c) -1)   
        
        
    def unitVecsOnSphere(self, theta, phi): 
        ur = Vector([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], units='unit')  
        utheta = Vector([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)], units='unit') 
        #print([-np.sin(phi),  np.cos(phi), Angle(0)])
        uphi = Vector([-np.sin(phi),  np.cos(phi), [0]], units='unit') 
        return (ur, utheta, uphi)
    
    @property
    def rho_c(self):
        '''
        Trho_c is the complex ratio of the RHCP/LHCP components
        ''' 
        return np.tan(self.gamma)*np.exp(1j*2*self.tau.u())
    
    @property
    def cirPolVec(self):
        return Voltage([np.cos(self.gamma)*np.exp(1j*self.tau.u()), np.sin(self.gamma)*np.exp(-1j*self.tau.u())])
    
    @classmethod
    def dualPolGeneration(cls, pol1, pol2, relAmpp1_p2, relPhasep1_p2):
        '''
        Pol1:  first drive Polarization object
        pol2: second drive Polarization object
        relAmpp1_p2: A Voltage object whose magnitude is the ration of p1/p2
        relPhasep2: An Angle object which is the phase of p2 relative to p1
        returns a Polarization object
        given two drive polarizations pol1 and pol2 with the relative amplitude p1/p2
        and the phase of p2 (p1 phase is always zero) calculates and returns the resulting
        field polarization
        '''
        # print(f"Pol1 {pol1}")
        # print(f"Pol2 {pol2}")
        # print(f"E1/E2 = {relAmpp1_p2}, and relative E2 phase = {relPhasep2.u('deg')} deg")
        gamma = np.arctan(relAmpp1_p2)
        #print(f"gamma = { Angle(gamma).u('deg')}")
        pvec1 = np.cos(gamma)*pol1.cirPolVec()
        
        pvec2 = np.sin(gamma)*pol2.cirPolVec()
        #print(f"pvec2 = {pvec2}")
        pvec1 =  np.exp(1j*relPhasep1_p2.u()/2)*pvec1
        pvec2 =  np.exp(-1j*relPhasep1_p2.u()/2)*pvec2
        #print(f"pvec2 with phase applied = {pvec2}")
        pveccomb = pvec1 + pvec2
        #print(f"pvec1 + pvec2 (phase app) = {pveccomb}")
        rho = pveccomb[0] / pveccomb[1]
        return (cls.fromRho_c(rho), pveccomb[0], pveccomb[1])
        
    @classmethod
    def dualPolPwrIntensity(cls,  pol1, pol2, pxyz): 
        '''
        Given two drive polarizations and a Poincare sphere vector 
        return the power intensity at that vector
        ''' 
  
        # taupv = pxyz.phi / 2
        # lamdapv = (np.pi/2 - pxyz.theta)/2
        # polpv = Polarization(taupv, lamdapv)
        #
        # polvec_c = polpv.cirPolVec() #this is == to pvec1 + pvec2 but amplitude is off
        #
        # pol1vec_c = pol1.cirPolVec()
        # pol2vec_c = pol2.cirPolVec()

        pol1DotPvec = np.dot(pol1.poincareXyz().rv, pxyz.rv)
        pol2DotPvec = np.dot(pol2.poincareXyz().rv, pxyz.rv)
        pol1PvecZeta = np.arccos(pol1DotPvec) / 2
        pol2PvecZeta = np.arccos(pol2DotPvec) / 2
        
        # the bisector between the two pols 1 and 2
        #pol1_2Bisectorxyz = (pol1.poincareXyz() + pol2.poincareXyz()).norm
        
        # angle between bisector and pxyz
        #biAng = np.dot(pol1_2Bisectorxyz.rv, pxyz.rv)
        
        # MUST BE A BETTER WAY....
        
        pol1PvecEff = np.cos(pol1PvecZeta)**2
        pol2PvecEff = np.cos(pol2PvecZeta)**2
        
        print(pol1PvecEff + pol2PvecEff)
        
        
        maxVal = np.max((pol1PvecEff, pol2PvecEff)) * (pol1PvecEff + pol2PvecEff)
        
        #print(f" {pol1PvecEff}, {pol2PvecEff}")
        
        res = 10*np.log10((pol1PvecEff + pol2PvecEff) / (2*maxVal))
        
        #TODO:  Need to figure out the correct math
        #factor = 1 #+ (pol1DotPvec + pol2DotPvec)
         
        #
        # if pol1PvecAng == pol2PvecAng:
        #     weightPol1 = factor
        #     weightPol2 = factor
        # elif pol1PvecAng > pol2PvecAng:
        #     weightPol1 =  factor
        #     weightPol2 =  factor*pol2PvecAng/pol1PvecAng
        # else:
        #     weightPol1 = factor*pol1PvecAng/pol2PvecAng
        #     weightPol2 = factor
        #
        #
        # pvec1 = weightPol1*pol1.cirPolVec()
        # pvec2 = weightPol2*pol2.cirPolVec()
        #
        # pvecAdj = (pvec1 + pvec2)/2
        #

        
        #known quantities:
        # elh: a complex number
        # erh: a complex number
        # ep1lh: a complex number
        # ep1rh: a complex number
        # ep2lh: a complex number
        # ep2rh: a complex number
        # elh == np.cos(np.arctan(ra))*np.exp(1j*rp/2.0)*epilh +  np.sin(np.arctan(ra))*np.exp(-1j*rp/2.0)*ep2lh      
        # erh == np.cos(np.arctan(ra))*np.exp(1j*rp/2.0)*epirh +  np.sin(np.arctan(ra))*np.exp(-1j*rp/2.0)*ep2rh      
        #

        # what are the unknowns:
        # ra: a real number
        # rp: a real number
        



        # def equations(x, elh, erh, ep1lh, ep1rh, ep2lh, ep2rh):
        #     ra, rp = x
        #     eq1 = elh - (np.cos(np.arctan(ra)) * np.exp(1j * rp / 2.0) * ep1lh + np.sin(np.arctan(ra)) * np.exp(-1j * rp / 2.0) * ep2lh)
        #     eq2 = erh - (np.cos(np.arctan(ra)) * np.exp(1j * rp / 2.0) * ep1rh + np.sin(np.arctan(ra)) * np.exp(-1j * rp / 2.0) * ep2rh)
        #     # eq1 = elh - (1 / np.sqrt(1 + ra**2)) * np.exp(1j * rp / 2.0) * ep1lh + (ra / np.sqrt(1 + ra**2)) * np.exp(-1j * rp / 2.0) * ep2lh
        #     # eq2 = erh - (1 / np.sqrt(1 + ra**2)) * np.exp(1j * rp / 2.0) * ep1rh + (ra / np.sqrt(1 + ra**2)) * np.exp(-1j * rp / 2.0) * ep2rh
        #     return np.concatenate((np.real(eq1), np.imag(eq1), np.real(eq2), np.imag(eq2))).reshape(4,)
        #
        #
        # # Initial guess for ra and rp
        # ra_guess = Voltage(1.0)
        # rp_guess = Angle(0.0)
        #
        # # Solve the equations
        # solution = least_squares(equations, 
        #                          (ra_guess, rp_guess), 
        #                          bounds=((0, 0),(np.inf, 2*np.pi)), 
        #                          args=(polvec_c[0], 
        #                                polvec_c[1], 
        #                                pol1vec_c[0], 
        #                                pol1vec_c[1], 
        #                                pol2vec_c[0], 
        #                                pol2vec_c[1])
        #                          )
        #
        # # Unpack the solution
        # ra, rp = solution.x
        #
        #
        # if ra > 1:
        #     weightPol1 = 1
        #     weightPol2 = 1 / ra
        # else:
        #     weightPol1 =  ra
        #     weightPol2 = 1
        #
        # res = 10*np.log10( (np.abs(weightPol1) + np.abs(weightPol2) )/2 )
        #
        # if pxyz.theta.u('deg') < 20:
        #     print(f"\ntheta = {pxyz.theta.u('deg')} deg, phi = {pxyz.phi.u('deg')} deg")
        #     print(f"ra = {ra}")
        #     print(f"rp = {rp*180/np.pi}")
        #     print(f"Relative Power = {res}")
            
        
        #return 10*np.log10(np.abs(pvecAdj[0]**2 + pvecAdj[1]**2))
        return res

        
    @classmethod
    def fromCirPolVec(cls, cirPolVec):
        gamma = np.arctan2(np.abs(cirPolVec[1]), np.abs(cirPolVec[0]))
        lamda = Angle((np.pi/2 -   2.0*gamma)/2.0 )
        tau = Angle((np.angle(cirPolVec[0])-np.angle(cirPolVec[1]))/2.0)
        return Polarization(tau, lamda)
                 
           
        
       
        
        
        
    @classmethod 
    def fromRho_c(cls, rho_c):
        '''
        rho_c = tan((pi/2 - 2*lamda)/2)
        arctan(rho_c) = (pi/2 - 2*lamda)/2)
        2*arctan(rho_c) = pi/2 - 2*lamda
        2*lamda = pi/2 - 2*arctan(rho_c)
        lamda = (pi/2 - 2*arctan(rho_c))/2
        
        ARG!!!!
        ''' 
        lamda = Angle( (np.pi/2.0 - 2.0*np.arctan(np.abs(rho_c)) ) / 2.0)
        tau = Angle(np.angle(rho_c)/2.0)
        return Polarization(tau, lamda)
                   
        
        
        
        
        
if __name__ == "__main__":
    import EMLib as em
    tau1 = em.Angle(90, 'deg')
    lamda1 = em.Angle(0, 'deg')
    pv = em.Polarization(tau1, lamda1)
    tau2 = em.Angle(0, 'deg')
    lamda2 = em.Angle(0, 'deg')
    ph = em.Polarization(tau2, lamda2)
    
    relAmp = Voltage(1)
    relPhase = Angle(-90, 'deg')
    
    pt = Polarization.dualPolGeneration(pv, ph, relAmp, relPhase)
    print(f"Pt: tau = {pt.tau.u('deg')}, lamda = {pt.lamda.u('deg')}")



        
        
        
        
        