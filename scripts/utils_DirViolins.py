import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sys, os
package_dir = os.path.abspath("C:/Users/froll/Documents/Labo/Projets/Outils/swd")
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

from swd import spherical_processing as sp
from swd import geotools as geom
from swd import plots as splots
import swd as swd
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
    # Rotation matrix around the y-axis
    R_y = np.array([
        [np.cos(a), 0, np.sin(a)],
        [0, 1, 0],       
        [-np.sin(a), 0, np.cos(a)]
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
    # Convert angles to Cartesian coordinates for field computation
    XYZ_angles = geom.sph2cart(angles, R=1)
    Dinf_meas = swd.compute_field_from_SH_coefs_at_origin(cmn, XYZ_angles, kvect, SH_center=np.array([0, 0, 0], dtype=complex))

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

#####################################################################
def load_violin_mesh(mesh_path, Wv=206.5e-3, Lv=485.7e-3, Hv=94.7e-3,
                     rx=np.pi/2, ry=0.0, rz=-np.pi/2):
    """Load, rotate, scale and centre a violin PLY mesh.

    The bridge top (SommetChevalet) is placed at the origin.
    Returns (vertices, faces).
    """
    import trimesh
    mesh = trimesh.load(mesh_path)
    verts = np.array(mesh.vertices, dtype=float)
    verts -= verts.mean(axis=0)
    verts = x_rotate_coordinates(verts.T, rx).T
    verts = y_rotate_coordinates(verts.T, ry).T
    verts = z_rotate_coordinates(verts.T, rz).T
    Dims0 = verts.max(axis=0) - verts.min(axis=0)
    DimsV = np.array([Hv, Wv, Lv])
    verts *= DimsV / Dims0
    SommetChevalet = verts[np.abs(verts[:, 0]).argmax()]
    verts -= SommetChevalet
    return verts, mesh.faces


#####################################################################
def plot_array_matplotlib(XYZm, XYZs, XYZHammerImpact, vertices, faces, NbMics,
                          L=0.6, elev=30, azim=210, mic_s=10, mic_c=None, mic_cmap='jet', mic_alpha=1.0):
    """Matplotlib 3D scatter of the microphone / source array with the violin mesh."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    XYZm = np.array(XYZm)
    
    if mic_c is None:
        mic_c = np.arange(NbMics)
        
    ax.scatter(*XYZm.T, c=mic_c, marker='o', edgecolor='k',
               s=mic_s, cmap=mic_cmap, alpha=mic_alpha, label='Mics')
    if XYZs is not None and len(XYZs) > 0:
        ax.scatter(*np.array(XYZs).T, marker='o', facecolor='lightgreen', edgecolor='k',
                   s=20, alpha=1, label='Srcs')
    if XYZHammerImpact is not None and len(XYZHammerImpact) > 0:
        XYZhi = np.atleast_2d(XYZHammerImpact)
        ax.scatter(*XYZhi.T, marker='o', facecolor='m', edgecolor='k',
                   s=30, label='Impact')
    for i in range(0, NbMics, 8):
        ax.plot(XYZm[i:i+8, 0], XYZm[i:i+8, 1], XYZm[i:i+8, 2],
                color='k', linewidth=0.5, alpha=0.5)
    light_brown = (0.87, 0.72, 0.53)
    verts = np.asarray(vertices)
    fac = np.asarray(faces)
    face_verts = verts[fac]
    ax.add_collection3d(Poly3DCollection(face_verts, alpha=0.1,
                                         edgecolor='k', facecolor=light_brown))
    ax.set_xlim([-L, L]); ax.set_ylim([-L, L]); ax.set_zlim([-L, L])
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.view_init(elev=elev, azim=azim, roll=0)
    ax.dist = 0.1
    ax.legend(loc='upper right', fontsize=10, frameon=False)
    plt.tight_layout()
    plt.show()
    return fig


#####################################################################
def plot_array_threejs(XYZm, XYZs, XYZHammerImpact, vertices, faces, NbMics):
    """Interactive pythreejs 3D view of the microphone / source array with the violin mesh."""
    try:
        from pythreejs import (
            SphereGeometry, MeshLambertMaterial, Mesh,
            BufferGeometry, BufferAttribute,
            LineBasicMaterial, LineSegments, WireframeGeometry,
            Scene, DirectionalLight, AmbientLight,
            PerspectiveCamera, OrbitControls, Renderer,
        )
        from IPython.display import display
    except ImportError:
        print("pythreejs is not installed. Please install it using: %pip install pythreejs")
        return None

    def _buf(arr):
        return BufferAttribute(array=arr.astype(np.float32, copy=False))

    scene_children = []

    mics_geo = SphereGeometry(radius=0.0075, widthSegments=8, heightSegments=8)
    mics_mat = MeshLambertMaterial(color='blue')
    for pos in XYZm:
        scene_children.append(Mesh(geometry=mics_geo, material=mics_mat,
                                   position=pos.tolist()))

    srcs_geo = SphereGeometry(radius=0.01, widthSegments=8, heightSegments=8)
    srcs_mat = MeshLambertMaterial(color='lightgreen')
    if XYZs is not None and len(XYZs) > 0:
        for pos in XYZs:
            scene_children.append(Mesh(geometry=srcs_geo, material=srcs_mat,
                                       position=pos.tolist()))

    impact_geo = SphereGeometry(radius=0.015, widthSegments=16, heightSegments=16)
    impact_mat = MeshLambertMaterial(color='magenta')
    if XYZHammerImpact is not None and len(XYZHammerImpact) > 0:
        scene_children.append(Mesh(geometry=impact_geo, material=impact_mat,
                                    position=np.array(XYZHammerImpact).flatten().tolist()))

    line_positions = []
    for i in range(0, NbMics, 8):
        for j in range(7):
            if i + j + 1 < NbMics:
                line_positions.extend([XYZm[i+j], XYZm[i+j+1]])
    if line_positions:
        lp = np.array(line_positions)
        lines_geo = BufferGeometry(attributes={'position': _buf(lp)})
        lines_mat = LineBasicMaterial(color='black', linewidth=1,
                                      transparent=True, opacity=0.5)
        scene_children.append(LineSegments(geometry=lines_geo, material=lines_mat))

    faces_flat = np.array(faces).flatten().astype(np.uint32)
    mesh_geo = BufferGeometry(attributes={
        'position': _buf(np.asarray(vertices)),
        'index': BufferAttribute(array=faces_flat, itemSize=1),
    })
    try:
        import trimesh as _trimesh
        tm = _trimesh.Trimesh(vertices=vertices, faces=faces)
        tm.fix_normals()
        mesh_geo.attributes['normal'] = _buf(tm.vertex_normals)
    except ImportError:
        pass
    except Exception:
        pass

    mesh_mat = MeshLambertMaterial(color='#DEB887', transparent=True,
                                   opacity=0.75, side='DoubleSide')
    scene_children.append(Mesh(geometry=mesh_geo, material=mesh_mat))
    wireframe_geo = WireframeGeometry(geometry=mesh_geo)
    wireframe_mat = LineBasicMaterial(color='black', linewidth=1,
                                      transparent=True, opacity=0.2)
    scene_children.append(LineSegments(geometry=wireframe_geo, material=wireframe_mat))

    scene = Scene(children=scene_children)
    key_light = DirectionalLight(color='white', position=[5, 5, 10], intensity=0.8)
    fill_light = DirectionalLight(color='white', position=[-5, 0, 5], intensity=0.5)
    scene.add([key_light, fill_light, AmbientLight(color='#777777')])

    camera = PerspectiveCamera(position=[-1, 1, 1], up=[0, 0, 1], aspect=1.0, fov=50)
    camera.lookAt([0, 0, 0])
    controls = OrbitControls(controlling=camera)
    renderer_widget = Renderer(camera=camera, scene=scene, controls=[controls],
                               width=800, height=800)
    display(renderer_widget)
    return renderer_widget


#####################################################################
def build_causal_excitation(frq, desired_duration_s=40.0e-3, f0=12_500.0, fs=1_000_000,
                            plot=True):
    """Build a bandlimited causal impulse excitation and compute its DFT on ``frq``.

    Parameters
    ----------
    frq : array-like
        Frequency vector (Hz) on which X_imp is evaluated.
    desired_duration_s : float
        Duration of the time signal (s).
    f0 : float
        Centre frequency of the Hann-windowed half-sine pulse (Hz).
    fs : int
        Oversampled time-domain sampling rate used for the DFT (Hz).
    plot : bool
        When True, display a 1x2 figure with the time-domain pulse and its spectrum.

    Returns
    -------
    X_imp : ndarray, shape (len(frq),)
        Complex DFT of the excitation evaluated at ``frq``.
    duration_s : float
        Actual signal duration used (equal to ``desired_duration_s``).
    """
    frq = np.asarray(frq)
    duration_s = desired_duration_s
    t_imp = np.arange(0, duration_s, 1.0 / fs)
    impulse_signal = np.zeros_like(t_imp)

    period_s = 1.0 / f0
    pulse_samples = max(8, int(np.round(period_s * fs)))
    t_pulse = np.arange(pulse_samples) / fs
    hann_window = np.hanning(pulse_samples)
    pulse_signal = hann_window * np.sin(2 * np.pi * f0 * t_pulse)
    pulse_signal = np.maximum(pulse_signal, 0.0)

    end_idx = pulse_samples // 2
    impulse_signal[:end_idx] = np.flip(pulse_signal[:end_idx])

    dt = t_imp[1] - t_imp[0]
    X_imp = np.sum(
        impulse_signal[None, :] * np.exp(-1j * 2 * np.pi * frq[:, None] * t_imp[None, :]),
        axis=1,
    ) * dt

    if plot:
        mag_db = 20 * np.log10(np.abs(X_imp) / (np.max(np.abs(X_imp)) + 1e-18) + 1e-18)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(t_imp * 1e3, impulse_signal, lw=1.3)
        axs[0].set_xlim(0.0, 2.0)
        axs[0].set_xlabel('Time (ms)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('Impulse Signal (Time Domain)')
        axs[0].grid(True, alpha=0.3)
        axs[1].plot(frq, mag_db, lw=1.3)
        axs[1].set_xlim(0.0, frq[-1])
        axs[1].set_ylim(-120, 5)
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Magnitude (dB, normalized)')
        axs[1].set_title('Impulse Spectrum')
        axs[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return X_imp, duration_s
