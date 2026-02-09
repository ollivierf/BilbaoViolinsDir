import sys, os
package_dir = os.path.abspath("C:/Users/froll/Documents/Labo/Projets/Violon/ManipViolon_Anech_08062023")
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)
from scripts import Tools3D as tools3d
package_dir = os.path.abspath("C:/Users/froll/Documents/Labo/Projets/Violon/ManipViolon_Anech_08062023/array-processing/toolboxes")
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)
import array_processing as ap
import numpy as np
import matplotlib.pyplot as plt
import utils_SHanalysis as SHutils
import utils_acoustics as acoustics
import utils_geometry as geom
import warnings
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
    N_SH_vect = SHutils.compute_N_SH_vect(freqvect, N, rmin = 1.5)
    H_array = SHutils.compute_SphericalWavesbasis_origin_to_field(XYZ_Mems, kvect, N, SH_Center)
    cmn = SHutils.compute_SHcoefs(x_meas, H_array, N_SH_vect = N_SH_vect, lambda_reg=lambda_reg)
    angles = geom.create_equal_angle_grid(nbtheta, nbphi)
    Dinf_meas = SHutils.compute_Dinf_from_SH_coefs_at_origin(cmn, angles, kvect)

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