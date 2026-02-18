from scipy.special import eval_jacobi, factorial
import numpy as np

import sys, os
package_dir = os.path.abspath("C:/Users/froll/Documents/Labo/Projets/Outils/swd")
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

from swd import spherical_processing as sp
from swd import geotools as geom
from swd import plots as splots
import swd as swd
import numpy as np
import plotly.graph_objs as go


def x_rotate_coordinates(coords, a):
    # Rotation matrix around the x-axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a), np.cos(a)]
    ])
    
    # Apply the rotations
    rotated_coords = np.dot(R_x, coords)
    return rotated_coords
def y_rotate_coordinates(coords, a):
    # Rotation matrix around the x-axis
    R_y = np.array([
        [np.cos(a),0, -np.sin(a)],
        [1, 0, 0],       
        [np.sin(a), 0, np.cos(a)]
    ])
    
    # Apply the rotations
    rotated_coords = np.dot(R_y, coords)    
    return rotated_coords

def z_rotate_coordinates(coords, a):
    # Rotation matrix around the z-axis
    R_z = np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a), np.cos(a), 0],
        [0, 0, 1]
    ])
    # Apply the rotations
    rotated_coords = np.dot(R_z, coords)    
    return rotated_coords

def Dinf_from_meas(x_meas, XYZ_Mems, freqvect, N, nbtheta = 51, nbphi = 103, rmin = 1.5,
                   lambda_reg = 1e-4, c0 = 343, SH_Center = np.array([0,0,0])):
    kvect = 2*np.pi*freqvect/c0   
    N_SH_vect = swd.compute_N_SH_vect(freqvect, N, rmin = 1.5)
    H_array = swd.compute_SphericalWavesbasis_origin_to_field(XYZ_Mems, kvect, N, SH_Center)
    cmn = swd.compute_SHcoefs(x_meas, H_array, N_SH_vect = N_SH_vect, lambda_reg=lambda_reg)
    angles = geom.create_equal_angle_grid(nbtheta, nbphi)
    Dinf_meas = swd.compute_Dinf_from_SH_coefs_at_origin(cmn, angles, kvect)

    return(cmn, Dinf_meas, angles)


#####################################################################
def create_filled_circle(radius, plane, fill_color):
    alpha = np.linspace(0, 2 * np.pi, 100)
    r = np.linspace(0, radius, 50)
    R, Alpha = np.meshgrid(r, alpha)
    X = R * np.cos(Alpha)
    Y = R * np.sin(Alpha)
    Z = np.zeros_like(X)
    if plane == 'x0z':
        return go.Surface(x=X, y=Z, z=Y, colorscale=[[0, fill_color], [1, fill_color]], showscale=False)#, opacity=0.25)
    elif plane == 'y0z':
        return go.Surface(x=Z, y=X, z=Y, colorscale=[[0, fill_color], [1, fill_color]], showscale=False)#, opacity=0.25)
    elif plane == 'x0y':
        return go.Surface(x=X, y=Y, z=Z, colorscale=[[0, fill_color], [1, fill_color]], showscale=False)#, opacity=0.25)
#####################################################################
def plot_3D_Diag(Diag, angles, fig, row, col, clims,cscale, Ro):
    
    NbTh, NbPh = int(np.sqrt(angles.shape[0])), int(np.sqrt(angles.shape[0]))
    if Ro : 
        alpha = Ro/3
    else:
        alpha = 1       
    XYZDg = geom.sph2cart(angles.reshape(-1,2), Ro + alpha*Diag)
    XDg = XYZDg[:, 0].reshape(NbPh, NbTh)
    YDg = XYZDg[:, 1].reshape(NbPh, NbTh)
    ZDg = XYZDg[:, 2].reshape(NbPh, NbTh)

    surface = go.Surface(
        x=XDg, y=YDg, z=ZDg,
        surfacecolor=Diag.reshape(NbPh, NbTh),
        colorscale=cscale, cmin= clims[0], cmax=clims[1],
        showscale=False  # Hide the color scale
    )    
    fig.add_trace(surface, row=row, col=col)
    
    # Add arrows
    arrow_length = Ro *1.5
    head_length = 0.2*arrow_length
    body_length = 0.8*arrow_length
    arrows = [
        go.Cone(x=[body_length], y=[0], z=[0], u=[head_length], v=[0], w=[0], showscale=False, colorscale=[[0, 'rgba(255, 100, 100, 0.75)'], [1, 'rgba(255, 100, 100, 0.75)']], sizemode='absolute', sizeref=head_length),
        go.Cone(x=[0], y=[body_length], z=[0], u=[0], v=[head_length], w=[0], showscale=False, colorscale=[[0, 'rgba(100, 255, 100, 0.75)'], [1, 'rgba(100, 255, 100, 0.75)']], sizemode='absolute', sizeref=head_length),
        go.Cone(x=[0], y=[0], z=[body_length], u=[0], v=[0], w=[head_length], showscale=False, colorscale=[[0, 'rgba(100, 100, 255, 0.75)'], [1, 'rgba(100, 100, 255, 0.75)']], sizemode='absolute', sizeref=head_length)
    ]
    lines = [
        go.Scatter3d(x=[0, body_length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='rgba(255, 100, 100, 0.75)', width=5)),
        go.Scatter3d(x=[0, 0], y=[0, body_length], z=[0, 0], mode='lines', line=dict(color='rgba(100, 255, 100, 0.75)', width=5)),
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, body_length], mode='lines', line=dict(color='rgba(100, 100, 255, 0.75)', width=5))
    ]    
    for arrow in arrows:
        fig.add_trace(arrow, row=row, col=col)
    
    for line in lines:
        fig.add_trace(line, row=row, col=col)  
    # Add filled circles
    circles = [
        create_filled_circle(arrow_length, 'x0z', 'rgba(255, 100, 100, 0.3)'),
        create_filled_circle(arrow_length, 'y0z', 'rgba(100, 255, 100, 0.3)'),
        create_filled_circle(arrow_length, 'x0y', 'rgba(100, 100, 255, 0.3)')
    ]
    for circle in circles:
        fig.add_trace(circle, row=row, col=col)
    # Define a common camera view
    camera = dict(eye=dict(x=1., y=1., z=1.))
    fig.update_scenes(
        dict(
            aspectmode='cube',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera
        ),
        row=row, col=col
    )
#####################################################################
def plot_3D_Diag_30dB(Diag, angles, fig, row, col, clims,cscale):
        
    NbTh, NbPh = angles.shape
    Diag30dB = 20*np.log10(np.abs(Diag)/np.max(np.abs(Diag)))
    Diag30dB[Diag30dB<-30]=-30
    Diag30dB += 30   
    Diag30dB /= 30
    XYZDg = geom.sph2cart(angles.reshape(-1,2), Diag30dB)
    XDg = XYZDg[:, 0].reshape(NbPh, NbTh)
    YDg = XYZDg[:, 1].reshape(NbPh, NbTh)
    ZDg = XYZDg[:, 2].reshape(NbPh, NbTh)

    surface = go.Surface(
        x=XDg, y=YDg, z=ZDg,
        surfacecolor=Diag.reshape(NbPh, NbTh),
        colorscale=cscale, cmin= clims[0], cmax=clims[1],
        showscale=False  # Hide the color scale
    )    
    fig.add_trace(surface, row=row, col=col)
    
    # Add arrows
    arrow_length = 1.5
    head_length = 0.2*arrow_length
    body_length = 0.8*arrow_length
    arrows = [
        go.Cone(x=[body_length], y=[0], z=[0], u=[head_length], v=[0], w=[0], showscale=False, colorscale=[[0, 'rgba(255, 100, 100, 0.75)'], [1, 'rgba(255, 100, 100, 0.75)']], sizemode='absolute', sizeref=head_length),
        go.Cone(x=[0], y=[body_length], z=[0], u=[0], v=[head_length], w=[0], showscale=False, colorscale=[[0, 'rgba(100, 255, 100, 0.75)'], [1, 'rgba(100, 255, 100, 0.75)']], sizemode='absolute', sizeref=head_length),
        go.Cone(x=[0], y=[0], z=[body_length], u=[0], v=[0], w=[head_length], showscale=False, colorscale=[[0, 'rgba(100, 100, 255, 0.75)'], [1, 'rgba(100, 100, 255, 0.75)']], sizemode='absolute', sizeref=head_length)
    ]
    lines = [
        go.Scatter3d(x=[0, body_length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='rgba(255, 100, 100, 0.75)', width=5)),
        go.Scatter3d(x=[0, 0], y=[0, body_length], z=[0, 0], mode='lines', line=dict(color='rgba(100, 255, 100, 0.75)', width=5)),
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, body_length], mode='lines', line=dict(color='rgba(100, 100, 255, 0.75)', width=5))
    ]    
    for arrow in arrows:
        fig.add_trace(arrow, row=row, col=col)
    
    for line in lines:
        fig.add_trace(line, row=row, col=col)  
    # Add filled circles
    circles = [
        create_filled_circle(arrow_length, 'x0z', 'rgba(255, 100, 100, 0.3)'),
        create_filled_circle(arrow_length, 'y0z', 'rgba(100, 255, 100, 0.3)'),
        create_filled_circle(arrow_length, 'x0y', 'rgba(100, 100, 255, 0.3)')
    ]
    for circle in circles:
        fig.add_trace(circle, row=row, col=col)
    # Define a common camera view
    camera = dict(eye=dict(x=1., y=1., z=1.))
    fig.update_scenes(
        dict(
            aspectmode='cube',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera
        ),
        row=row, col=col
    )

def wigner_D_matrix(j, alpha, beta, gamma):
    """
    Compute the Wigner D-matrix D^j(alpha, beta, gamma).
    
    This function computes the (2j+1)x(2j+1) Wigner D-matrix for a given angular momentum j 
    and Euler angles (alpha, beta, gamma) in radians.
    
    Args:
        j (int): Angular momentum quantum number (must be integer >= 0).
        alpha (float): Euler angle alpha (rotation around Z) in radians.
        beta (float): Euler angle beta (rotation around Y) in radians.
        gamma (float): Euler angle gamma (rotation around Z) in radians.
        
    Returns:
        numpy.ndarray: The (2j+1)x(2j+1) complex Wigner D-matrix.
    """
    size = int(2 * j + 1)
    
    # Pre-compute trigonometric terms
    sb2 = np.sin(beta / 2.0)
    cb2 = np.cos(beta / 2.0)
    x = np.cos(beta)
    
    # Precompute factorials up to 2j
    # Note: For large j, use gammaln to avoid overflow, but for typical SH orders (j<50), factorial is fine.
    # fact[n] needs to accommodate indices up to 2j
    fact = factorial(np.arange(0, int(2 * j) + 5))

    d_mat = np.zeros((size, size))
    m = np.arange(-j, j + 1)
    
    for i, mp in enumerate(m):
        for k, mv in enumerate(m):
            # Symmetry handling to map to valid Jacobi polynomial indices (n>=0, a>-1, b>-1)
            # m' -> mp, m -> mv.
            
            target_mp, target_mv = mp, mv
            factor = 1.0
            
            # 1. Use d_{m',m}^j = (-1)^(m'-m) d_{m,m'}^j to ensure m' >= m
            if target_mp < target_mv:
                factor *= (-1.0)**(target_mp - target_mv)
                target_mp, target_mv = target_mv, target_mp
                
            # 2. Use d_{m',m}^j = (-1)^(m'-m) d_{-m',-m}^j to ensure m' + m >= 0
            if target_mp + target_mv < 0:
                 factor *= (-1.0)**(target_mp - target_mv)
                 target_mp, target_mv = -target_mv, -target_mp 
            
            # Now target_mp >= target_mv and target_mp + target_mv >= 0.
            # a = m' - m >= 0
            # b = m' + m >= 0
            n = int(j - target_mp)
            a = int(target_mp - target_mv)
            b = int(target_mp + target_mv)
            
            # Normalization factor
            # For this standard Jacobi definition:
            # d_mm' = sqrt( (j+m')! (j-m')! / ( (j+m)! (j-m)! ) )  * ...
            # Wait, using factorials computed earlier:
            # fact[n] corresponds to n!
            val_num = fact[int(j + target_mp)] * fact[int(j - target_mp)]
            val_den = fact[int(j + target_mv)] * fact[int(j - target_mv)]
            norm = np.sqrt(val_num / val_den)
            
            # Compute element
            val = norm * (sb2**a) * (cb2**b) * eval_jacobi(n, a, b, x)
            d_mat[i, k] = factor * val

    # Apply phases D_{m',m} = e^{-i m' alpha} * d_{m',m}(beta) * e^{-i m gamma}
    mp_grid, mv_grid = np.meshgrid(m, m, indexing='ij')
    phase = np.exp(-1j * (mp_grid * alpha + mv_grid * gamma))
    
    return phase * d_mat
