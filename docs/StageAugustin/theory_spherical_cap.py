import scipy.special as scsp
import numpy as np

import sys

toolboxes_path = './toolboxes/'
sys.path.insert(1, toolboxes_path)

def P(n,x) :

    return scsp.eval_legendre(n,x)

def spherical_hn(n, z, derivative = False):
    nu = n + 1/2
    return scsp.hankel2(nu, z)*np.sqrt(np.pi/(2*z))
def spherical_hn_diff(n,z):    
    hn_diff = 1/(2*n+1)*(n*spherical_hn(n-1,z) - (n+1)*spherical_hn(n+1,z))
    return hn_diff  

def W(n, theta_0) :
    # return 1/2*((n+1)/(2*n+3)*(P(n,np.cos(theta_0))-P(n+2,np.cos(theta_0)))+(n/(2*n-1)*(P(n-2,np.cos(theta_0))-P(n,np.cos(theta_0)))))
    return 1/2*(P(n-1,np.cos(theta_0))-P(n+1,np.cos(theta_0)))

def cart2sph(arraypos) :
    r = np.sqrt(np.sum(arraypos**2, axis=1))
    theta = np.arccos(arraypos[:,2]/r)
    return r, theta

def sph_to_cart(theta, phi, r) :
    z = r*np.cos(theta)*np.cos(phi)
    y = r*np.cos(theta)*np.sin(phi)
    x = r*np.sin(theta)
    return np.dstack((x,y,z))[0]

def comp_pressure(arraypos, k, N, cap_radius = 0.08, theta_0 = np.pi/9, c0 = 340, rho0 = 1.293, array_type = 'cart', distance= 10) :
    if array_type == 'cart' : r, theta = cart2sph(arraypos)
    else : 
        theta = arraypos
        r = distance*np.ones(theta.size)
    sum = np.sum([W(n, theta_0)*P(n,np.cos(theta))*spherical_hn(n,k*r)/spherical_hn_diff(n,k*cap_radius) for n in range(N+1)], axis=0)
    
    p = (-1j*rho0*c0*sum)

    return p

def comp_dir(f,N,theta,R = 0.08,c0 = 340, rho0 = 1.293, theta_0 = np.pi/9) :
    k = 2*np.pi*f/c0
    sum = 0
    for n in range(N+1): 
        sum += W(n, theta_0)*P(n,np.cos(theta))*1j**(n+1)/spherical_hn_diff(n,k*R)
    
    dir = -1j*rho0*c0*sum

    return dir