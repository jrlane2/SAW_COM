import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import SAW_COM.sawcom as sc
import os






def generate_G(f, dd, pl, plot = False):
    '''
    Function that generates the conductance of a SAW resonator
    takes in some a frequency list, a dictionary defining the device, 
    and a constant that quantifies propagation loss.
    Im not very happy with the propagation loss thing right now.
    '''
    
    d = sc.delta(f,dd['v'],dd['IDTlam']) - pl
    dm = sc.delta(f,dd['v'],dd['Mirrorlam']) - pl
    
    IDT = sc.pmatrix(dd['IDTlam'],dd['c12'],dd['a1'],dd['IDTlen'],d) #transducer P_matrix
    Mir = sc.pmatrix(dd['Mirrorlam'],dd['c12'],0,dd['Mirrorlen'],dm) #transducer P_matrix
    freeR = sc.pmatrix(dd['IDTlam'], 0 , 0 ,dd['FreelenR'],d) #transducer P_matrix
    freeL = sc.pmatrix(dd['IDTlam'], 0 , 0 ,dd['FreelenL'],d) #transducer P_matrix
    
    dict_1 = {'x':IDT, 'y':Mir, 'R':freeR, 'L':freeL}
    str_1 = 'yLxRy'
    
    z = dict_1[str_1[0]]
    for i in range(1,len(str_1)):
        z = sc.concat(z,dict_1[str_1[i]])
        
    # Optional: plot the real part of G and compare the the real part of the IDT admittance
    if plot == True:
        fig, ax = plt.subplots(nrows = 1, ncols = 1 , figsize = (8,6))
    
        ax.plot(f/1E9,np.real(z.p33)*1E3, color = 'blue')
        ax.plot(f/1E9,np.imag(z.p33)*1E3, color = 'cyan')
        ax.plot(f/1E9,np.real(IDT.p33)*1E3, color = 'orange')
        ax.set_ylim([-1E-5,np.max(np.real(IDT.p33))*2E3])
        ax.set_xlabel('Frequency (GHz)', fontsize = 14)
        ax.set_ylabel('Conductance (mS)', fontsize = 14)
        
    sb = stopband_args(f,dd)
      
    return z.p33, sb
    
    
def stopband_args(f,dd):
    '''
    The admittance of a SAW resonator is generally well behaved inside the mirror
    stop-band. This function just picks out the arguments in a frequency array
    f that define the begining and ends of the mirror stop-band
    '''
    r = dd['c12']*(dd['IDTlam']/2)
    mirror_center = dd['v']/dd['Mirrorlam']
    stopbandw = mirror_center*2*np.abs(r)/np.pi

    startband = mirror_center - stopbandw/2
    endband = mirror_center + stopbandw/2

    startarg = np.argmin(np.abs(f - startband))
    endarg = np.argmin(np.abs(f - endband))

    return [startarg, endarg]






def resfinder(f, G, sba):
    '''
    Get the resonant frequencies of the Fabry-Perot modes
    takes in a frequency list, a pre-computed conductance
    and the stop band beginning/end arguments
    '''
    f_res = np.empty(1)
    fstep = f[1]-f[0]
    ft = f[sba[0]:sba[1]]
    Gt = G[sba[0]:sba[1]]

    zeroargs = np.where(np.diff(np.signbit(np.imag(Gt))))[0]

    #print(zeroargs)
    #print(ft[zeroargs[0]])
    for i in zeroargs:
        delta = np.imag(Gt)[i+1] - np.imag(Gt)[i]
        #print(delta, np.imag(Gt)[i+1],  np.imag(Gt)[i])
        if delta < 0:
            f_res = np.append(f_res,[ft[i] + fstep*(abs(np.imag(Gt)[i])/delta)])
    f_res = f_res[1:]
    
    return f_res

def BVDr(f, f0arr, Rarr, Carr):
    '''
    Function for fitting the (real part of the) conductance to its Butterworth-van Dyke equiv.
    inside the stop-band. Here, f0rr, Rarr, anc Carr are *arrays* with length equal to the number
    of resonances the structure has. f0 = resonant frequency, Rarr = BvD equiv resistance,
    Carr = BvD equiv capacitance
    '''
    w = 2*np.pi*f
    w0arr = 2*np.pi*f0arr
    Larr = 1/(w0arr**2*Carr)
    return sum([Rarr[i]*(w*Carr[i])**2/((Rarr[i]*w*Carr[i])**2 + 
                                         (1-Larr[i]*Carr[i]*w**2)**2) for i in range(len(f0arr))])


def BVDr_wrapper(x, *args):
    '''
    Since we don't know a priori how many resonances a structure will have, to get the BvD equiv.
    circuit we need to to fit a function with a variable number of variables.
    This wrapper serves as an intermediate between op.curve_fit and BVDr. BVDr takes in
    a series of arrays, but op.curve_fit fits a tuple of individual variables.
    The correct course of action is to flatten the lists into a single tuple, and use op.curve_fit
    to fit to this function, which takes in a tuple of variable length (indicated by *args
    in the argument... this is pythons way of passing a variable number of variables.) 
    Then, we break that tuple up into 3 lists and pass it to the original BVDr function. 
    '''
    N = int(len(args)/3)
    #print(N)
    #print(list(args[0:3]))
    a, b, c = np.array(args[:N]), np.array(args[N:2*N]), np.array(args[2*N:3*N])
    return BVDr(x, a, b, c)


def BvD_finder(f,G,sba, plot = False):
    '''
    Function to run when you actually want to find the equivalent circuit elements
    Takes in a frequency array, a conductance array, and the stop-band start/end arguments.
    Spits out 3 numpy arrays of length equal to the number of resonances. The ith element of 
    each array is the R, C, and L for the ith Butterworth-van Dyke equivalent circuit.
    '''
    f0s = resfinder(f, G, sba)
    if len(f0s) == 0:
    	return [], [], []
    ft = f[sba[0]:sba[1]]
    Gt = np.real(G[sba[0]:sba[1]])
    Rguess = [10 for i in range(len(f0s))]
    Cguess = [1E-16 for i in range(len(f0s))]
    p_0 = tuple(f0s) + tuple(Rguess) + tuple(Cguess)
    popt , popc  =  op.curve_fit(BVDr_wrapper, ft, Gt, p0 = p_0 )

    N = int(len(popt)/3)
    
    # Optional: plot to make sure the fit was done correctly
    if plot == True:
        fit = BVDr(ft, popt[:N], popt[N:2*N], popt[2*N:3*N])
        
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (9,6))
        plt.locator_params(axis='y', nbins=4)
        plt.locator_params(axis='x', nbins=4)
        ax.plot(ft/1E9,Gt*1E3, linewidth = 4, color = 'blue', label = r"Resonator Re$[P_{33}]$")
        ax.plot(ft/1E9,fit*1E3, color = 'orange', linestyle = '--', label = 'Re[BvD equiv.]')
        ax.legend(fontsize = 14)
        ax.set_xlabel('F (GHz)', size = 14)
        ax.set_ylabel(r'Re(P33) = Re(Y) (mS)', size = 14)
        ax.tick_params(labelsize = 14)
    
    r_array = np.array(popt[N:2*N])
    c_array = np.array(popt[2*N:3*N])
    l_array = 1/((2*np.pi*np.array(popt[:N]))**2*c_array)
    return r_array, c_array, l_array










def delta_to_Y(Z1,Z2,Z3):
    '''
    Does a "Delta-Wye" transformation (Delta -> Y)
    '''
    dn = Z1+Z2+Z3
    return Z1*Z2/dn, Z1*Z3/dn, Z2*Z3/dn

def parallel(Z1, Z2):
    return (Z1**-1 + Z2**-1)**-1

def Zind(L,f):
    w = 2*np.pi*f
    return 1j*L*w

def Zcap(C,f):
    w = 2*np.pi*f
    return (1j*C*w)**-1

def ZBvD(f, Rm, Lm, Cm, Cidt):
    '''
    Coputes the Butterworth-van Dyke equivalant impedance for a *single* resonance
    '''
    return parallel(Zcap(Cidt,f), Rm + Zind(Lm,f) + Zcap(Cm,f))

def ZBvDn(f, Rm, Lm, Cm, Cidt):
    '''
    Coputes the Butterworth-van Dyke equivalant impedance for an
    arbitrary number of parallel resonances
    '''
    Zi = [Rm[i] + Zcap(Cm[i],f) + Zind(Lm[i],f) for i in range(len(Rm))]
    Zf = Zcap(Cidt,f)
    for i in range(np.shape(Zi)[0]):
        Zf = parallel(Zf,Zi[i-1])
    return Zf

def Y_vs_Landf_capcup(f, lj, R, L, C, circparams):
    '''
    Takes in a dictionary called circparams, and 3 1xn arrays for the resistance,
    inductance, and capacitance of n Butterworth-van Dyke resonances. Returns the 
    admittance of a linear circuit of a LC resonator (the linearized qubit) capacitively
    coupled to the n-BvD resonances in parallel with the geometric capacitance of the 
    SAW structure. Y is a function of both frequency and the tunable inductance.
    '''
    # Generate impedance between each node, except for the josephson junction
    Z2 = Zcap(circparams['c1_2'], f)
    Z3 = Zcap(circparams['cp_2'], f)
    Z4 = ZBvDn(f, R, L, C, circparams['csaw'])
    Z5 = Zcap(circparams['c1_1'], f)
    Z6 = Zcap(circparams['cp_1'], f)
    
    # Transform impedances to get single impedance parallel to JJ
    Za, Zb, Zc = delta_to_Y(Z3, Z4, Z2)
    Z5a = Z5 + Za
    Z6c = Z6 + Zc
    Zn = parallel(Z6c, Z5a)
    Znb = Zn + Zb
    
    Y = np.zeros([len(lj), len(f)]) + 1j
    
    for i in range(len(lj)):
        Z1 = parallel(Zind(lj[i],f),Zcap(circparams['cj'],f))
        Y[i,:] = parallel(Z1, Znb) 
    Y = Y**-1
    
    return Y
    
    
    
    




def Y_zero_finder(Y_array, f_array, l_array):
    '''
    Find the zero crossings of the admittance parallel to the Josephson junction
    Finds only the crossings of negative slope in frequency
    returns list of inductances and frequencys at zero crossings
    '''
    f_zeros = np.empty(1)
    l_zeros = np.empty(1)
    fstep = f_array[1]-f_array[0]
    for i in range(len(l_array)):
        temp = np.where(np.diff(np.signbit(Y_array[i,:])))[0]
        #print(temp)
        for j in temp:
            delta = Y_array[i,j+1] - Y_array[i,j]
            #print(delta)
            if delta > 0:
                #print(j)
                l_zeros = np.append(l_zeros,[l_array[i]])
                f_zeros = np.append(f_zeros,[f_array[j] - fstep*(abs(Y_array[i,j])/delta)])
    f_zeros = f_zeros[1:]
    l_zeros = l_zeros[1:]
    
    return f_zeros, l_zeros

def Tmon_mode_cap_guess(f_array,l_array, fsaw):
    '''
    Generate a working guess for the mode capacitance of the transmon-like mode
    '''
    maxD = np.argmax(np.abs(f_array-fsaw))
    return (l_array[maxD]*(2*np.pi*f_array[maxD])**2)**-1


def band_split(flist, llist, fsaw, cguess):
    '''
    Splits avoided crossing data (flist and llist) into upper and lower
    bands of the avoided crossing. Takes in a guess of the SAW frequency
    and a guess of the mode capacitance to figure out where to split
    Returns 4 arrays: frequency and inductance for the lower (upper) bands
    '''
    EM_modeguess = 1/(2*np.pi*np.sqrt(cguess*llist))
    splitter_line = (EM_modeguess+fsaw)/2
    flow = np.empty(0)
    llow = np.empty(0)
    fhigh = np.empty(0)
    lhigh = np.empty(0)
    for i in range(len(flist)):
        if flist[i] < splitter_line[i]:
            flow = np.append(flow,[flist[i]])
            llow = np.append(llow, [llist[i]])
        else:
            fhigh = np.append(fhigh,[flist[i]])
            lhigh = np.append(lhigh, [llist[i]])
    return flow, llow, fhigh, lhigh 


def minus_band(llist, fsaw, g, c):
    EMmode = (2*np.pi*np.sqrt(llist*c))**-1
    delta = fsaw - EMmode
    return -np.sqrt(g**2 + (delta/2)**2) + (fsaw+EMmode)/2  

def plus_band(llist, fsaw, g, c):
    EMmode = (2*np.pi*np.sqrt(llist*c))**-1
    delta = fsaw - EMmode
    return +np.sqrt(g**2 + (delta/2)**2) + (fsaw+EMmode)/2  


def single_crossing_g_finder(Yarray, flist, llist, r, l, c, plot = False, return_eigs = False, debug = False):
    '''
    Takes in an array of complex admittances. Finds the zeros of the imaginary parts,
    and fits those to an avoided crossing. This only works for a single resonance.
    '''
    
    f_zeros, l_zeros = Y_zero_finder(np.imag(Yarray), flist, llist)

    
    # Generate some guess fit parameters 
    fsaw = float(1/(2*np.pi*np.sqrt(c*l))) # note that here the function should fail if n SAW resonances != 1
    g_guess = 10E6
    cap_guess = Tmon_mode_cap_guess(f_zeros,l_zeros, fsaw)
    
    fminus, lminus, fplus, lplus = band_split(f_zeros, l_zeros, fsaw, cap_guess)


    # Don't fit the data too close to the avoided crossing: it's not necessary for a good fit
    # and since there's a lot of it, it can easily fuck up the fit
    #f_fit = np.array([f_zeros[i] for i in range(len(f_zeros)) if np.abs(f_zeros[i] - fsaw) > 2E6 ])
    #l_fit = np.array([l_zeros[i] for i in range(len(l_zeros)) if np.abs(f_zeros[i] - fsaw) > 2E6 ])
    
    if debug == True:
        fig, ax = plt.subplots()
        ax.plot(l_zeros,f_zeros, 'ko', markersize = 10)
        ax.plot(lminus, fminus, 'bo', markersize = 5)
        ax.plot(lplus, fplus, 'ro', markersize = 5)
        ax.plot(l_zeros, 1/(2*np.pi*np.sqrt(l_zeros*cap_guess)))
        #print(type(fsaw), type(g_guess), type(cap_guess), type(f_fit), type(l_fit))

    
    #popt, popc =  op.curve_fit(inverted_crossing, f_zeros, l_zeros, p0 = [fsaw, g_guess, cap_guess])
    popt, popc =  op.curve_fit(minus_band, lminus, fminus, p0 = [fsaw, g_guess, cap_guess])
    
    if plot == True:
        f_fitm =minus_band(lminus, popt[0], popt[1], popt[2])
        f_fitp =plus_band(lplus, popt[0], popt[1], popt[2])
        
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (9,6))
        plt.locator_params(axis='y', nbins=4)
        plt.locator_params(axis='x', nbins=4)
        ax.plot(l_zeros*1E9,f_zeros/1E9, 'bo', markersize = 4, label = "Eigenfrequencies")
        ax.plot(lminus*1E9,f_fitm/1E9, 'r-', markersize = 2, label = "Fit to avoided crossing")
        ax.plot(lplus*1E9,f_fitp/1E9, 'r-', markersize = 2)#, label = "Fit to avoided crossing")
        ax.legend(fontsize = 14)
        ax.set_xlabel(r'$L_J$ (nH)', size = 14)
        ax.set_ylabel('F (GHz)', size = 14)
        ax.tick_params(labelsize = 14)
        
    if return_eigs == True:
        return f_zeros, l_zeros, popt
    else:
        return popt

