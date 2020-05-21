import numpy as np

'''
Solutions to Coupling of modes equations as P-matrix elements
Reproduced from "Surface Acoustic Wave Filters" by David Morgan, 2nd edition
Chapter 8, page 245

This code was written by Justin Lane.
'''

def delta(freq,v,lam):
    #returns the detuning wavenumber given a list of frequencies, a SAW velocity, and the lithographically
    #defined SAW wavelength
    omega_0 = 2*np.pi*v/lam
    omega = freq*2*np.pi
    return (omega-omega_0)/v

def K1(a1,c12, delta):
    # Returns particular integral solution 1
    s = np.sqrt(delta**2 - np.abs(c12)**2 + 0j)
    return (np.conj(a1)*c12 - 1j*delta*a1)/(delta**2 - np.abs(c12)**2)

def K2(a1,c12, delta):
    # Returns particular integral solution 2 
    s = np.sqrt(delta**2 - np.abs(c12)**2 + 0j)
    return (a1*np.conj(c12) + 1j*delta*np.conj(a1))/((delta**2 - np.abs(c12)**2))

def p11(c12,L,delta):
    s = np.sqrt(delta**2 - np.abs(c12**2) + 0j)
    # s =  np.real(s)*np.sign(delta) + 1j*np.imag(s)
    return -np.conj(c12)*np.sin(s*L)/(s*np.cos(s*L) + 1j*delta*np.sin(s*L))

def p22(c12,L,delta, lam):
    s = np.sqrt(delta**2 - np.abs(c12**2) + 0j)
    kc = 2*np.pi/lam
    return c12*np.sin(s*L)*np.exp(-2j*kc*L)/(s*np.cos(s*L) + 1j*delta*np.sin(s*L))

def p12(c12, L, delta, lam):
    s = np.sqrt(delta**2 - np.abs(c12**2) + 0j)
    kc = 2*np.pi/lam
    return s*np.exp(-1j*kc*L)/(s*np.cos(s*L) + 1j*delta*np.sin(s*L))

def p31(c12,a1,L, delta):
    K_2 = K2(a1,c12,delta)
    s = np.sqrt(delta**2 - np.abs(c12)**2 + 0j)
    P_31 = (2*np.conj(a1)*np.sin(s*L)- 2*s*K_2*(np.cos(s*L)-1))/(s*np.cos(s*L) + 1j*delta*np.sin(s*L))
    return P_31

def p32(c12,a1,L,delta,lam):
    K_1 = K1(a1,c12,delta)
    kc = 2*np.pi/lam
    s = np.sqrt(delta**2 - np.abs(c12)**2 + 0j)
    P_32 = (-2*a1*np.sin(s*L)- 2*s*K_1*(np.cos(s*L)-1))/(s*np.cos(s*L) + 1j*delta*np.sin(s*L))*np.exp(-1j*kc*L)
    return P_32

def p33(c12,a1,L,delta,lam):
    K_1 = K1(a1,c12,delta)
    K_2 = K2(a1,c12,delta)
    kc = 2*np.pi/lam
    P_33 = - K_2*p32(c12,a1,L,delta,lam)*np.exp(1j*kc*L) -K_1*p31(c12,a1,L, delta)  + 2*L*(np.conj(a1)*K_1 - a1*K_2)
    return P_33



'''
Define a P-matrix data structure, and a function for concatenating of two P-matrices
'''


class pmatrix:
    def __init__(self, lam, c12, a1, L, delta):
        self.p11 = p11(c12, L, delta)
        self.p22 = p22(c12, L, delta, lam)
        self.p12 = p12(c12, L, delta, lam)
        self.p21 = self.p12
        self.p31 = p31(c12,a1,L, delta)
        self.p13 = self.p31/(-2)
        self.p32 = p32(c12,a1,L,delta,lam)
        self.p23 = self.p32/(-2)       
        self.p33 = p33(c12,a1,L,delta,lam)
    
    @classmethod
    def blank(cls):
        #Return a blank p-matrix
        n = np.array([])
        return cls(n, n, n, n, n)
        

        
def concat(pl,pr):
    new = pmatrix.blank()
    D = 1-pl.p22*pr.p11
    new.p11 = pl.p11 + pr.p11*pl.p12**2/D
    new.p12 = pr.p12*pl.p12/D
    new.p21 = new.p12
    new.p22 = pr.p22 + pl.p22*pr.p12**2/D
    new.p13 = pl.p13 + pl.p12*(pr.p11*pl.p23 + pr.p13)/D
    new.p31 = -2*new.p13
    new.p23 = pr.p23 + pr.p12*(pr.p13*pl.p22 + pl.p23)/D
    new.p32 = -2*new.p23
    new.p33 = pl.p33 + pr.p33 - 2*pl.p23*(pr.p11*pl.p23 + pr.p13)/D - 2*pr.p13*(pl.p22*pr.p13 + pl.p23)/D
    return new



