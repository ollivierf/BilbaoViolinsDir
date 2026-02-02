import numpy as np
import scipy.spatial.transform as trsf
import scipy.signal as sig
import meshio
import numpy as np
import plotly.graph_objs as go
import scipy.spatial.transform as trsf

def Plot_Isos(fig,S, Grid, Limits, Dyn) :
    S/=np.max(np.abs(S))
    Dyn = 10**(-Dyn/10)
    xmin, xmax, ymin, ymax, zmin, zmax = Limits 
    Lx = xmax-xmin
    Ly = ymax-ymin
    Lz = zmax-zmin
    Io = np.argmax(np.abs(S))
    Xo, Yo, Zo = Grid[Io,0], Grid[Io,1], Grid[Io,2]

    fig.add_trace(go.Isosurface( x = Grid[:,0],  y = Grid[:,1],  z = Grid[:,2], value = np.real(S), opacity = 0.5, 
                               surface_count=1,isomin=Dyn, isomax = Dyn, colorscale = 'Blues'))

    fig.add_trace(go.Isosurface( x = Grid[:,0],  y = Grid[:,1],  z = Grid[:,2], value = np.real(S), opacity = 0.5, 
                               surface_count=1,isomin=-Dyn, isomax = -Dyn, colorscale = 'Reds'))
    fig.update_layout(scene_aspectmode='manual', 
                      scene_aspectratio=dict(x=Lx, y=Ly, z=Lz))
    camera = dict(eye=dict(x=-Lx/2, y=2*Ly, z=Lz))
    fig.update_layout(scene_camera=camera)  
    fig.update_layout(scene = dict(xaxis = dict(nticks=5, range=[-Lx/2, Lx/2]),
                               yaxis = dict(nticks=5, range=[-Ly/2, Ly/2]),
                               zaxis = dict(nticks=5, range=[-Lz/2, Lz/2]))) 

def Plot_Diag(fig,S,Center, th, ph, Limits):
    Diag = np.real(S)/np.max(np.abs(S))
    xmin, xmax, ymin, ymax, zmin, zmax = Limits 
    Lx = xmax-xmin
    Ly = ymax-ymin
    Lz = zmax-zmin
   
    Diag = Diag.reshape(len(th),len(ph))
    Xs  = (Diag)*np.sin(th)*np.cos(ph)
    Ys  = (Diag)*np.sin(th)*np.sin(ph)
    Zs  = (Diag)*np.cos(th)
   
    Xo,Yo,Zo = Center 
    
    fig = go.Figure(data=[go.Surface(z=Zs+Zo, x=Xs+Xo, y=Ys+Yo, surfacecolor=Diag, opacity=0.5, colorscale='Reds')])
    camera = dict(eye=dict(x=-Lx/2, y=2*Ly, z=Lz))
    fig.update_layout(scene_camera=camera)
    fig.update_layout(scene_aspectmode='manual', 
                      scene_aspectratio=dict(x=Lx, y=Ly, z=Lz),
                      width=1000, height = 1000,
                      scene = dict(xaxis = dict(nticks=10, range=[0,Lx],),
                                yaxis = dict(nticks=10, range=[0,Ly],),
                                zaxis = dict(nticks=10, range=[0,Lz],),))
    fig.show()

#################################################
## Calcul du maillage regulier d'une sphère pour le BF et Directivité
#################################################
def CalcSphGrid(C, r=1, n=100):
    # Points sur la sphère de rayon r de centre C
    nth,nph = n,n
    ph, th = np.mgrid[0:2*np.pi:nph*1j, 0:np.pi:nth//2*1j]

    Xs  = r*np.sin(th)*np.cos(ph)
    Ys  = r*np.sin(th)*np.sin(ph)
    Zs  = r*np.cos(th)

    Xd = Xs+C[0]; Yd = Ys+C[1]; Zd = Zs+C[2]; 
    Sph = np.array([Xd.flatten(), Yd.flatten(), Zd.flatten()]).T
    ns = len(Sph)
    return Sph, ns, th, ph
#################################################
## Calcul du maillage d'un Volume pour le BF
#################################################
def CalcSqrGrid(vmin, vmax, dv):
    xv = np.arange(vmin[0], vmax[0]+dv, dv)
    yv = np.arange(vmin[1], vmax[1]+dv, dv)
    zv = np.arange(vmin[2], vmax[2]+dv, dv)
    nx = len(xv)
    ny = len(yv)
    nz = len(zv)
    [Xv, Yv, Zv] = np.meshgrid(xv,yv,zv)
    Xv = Xv.flatten()
    Yv = Yv.flatten()
    Zv = Zv.flatten()
    V = np.array([Xv, Yv, Zv]).T
    return V, nx, ny, nz

def BandPass(Mems, Fe = 50000, fcl = 20, fch = 10000):
    #Filtrage des signaux
    nyq = 0.5 * Fe
    fcn = fch / nyq
    sos = sig.butter(10, [fcl, fch], 'bp', fs=Fe, output='sos')
    MemsF = np.zeros_like(Mems)
    NbMems = Mems.shape[1]
    for ii in range(NbMems):
        MF = sig.sosfilt(sos, Mems[:,ii])
        MemsF[:,ii] = np.flipud(sig.sosfilt(sos,np.flipud(MF)))

    return MemsF

def BuildAvatar(FileName, Xa,Ya,Za,Ha, RotX, RotY, RotZ):    

    mesh= meshio.read(FileName)

    Scale = Ha/np.max(mesh.points[:,2])

    vertices = mesh.points * Scale
    vertices[:,2] -= np.min(vertices[:,2])
    if RotX : 
        rotdeg = RotX
        rotrad = np.radians(rotdeg)
        rotaxe = np.array([1, 0, 0])
        rotvec = rotrad * rotaxe
        rot = trsf.Rotation.from_rotvec(rotvec)
        vertices = rot.apply(vertices)    
    if RotY : 
        rotdeg = RotY
        rotrad = np.radians(rotdeg)
        rotaxe = np.array([0, 1, 0])
        rotvec = rotrad * rotaxe
        rot = trsf.Rotation.from_rotvec(rotvec)
        vertices = rot.apply(vertices)    
    if RotX : 
        rotdeg = RotZ
        rotrad = np.radians(rotdeg)
        rotaxe = np.array([0, 0, 1])
        rotvec = rotrad * rotaxe
        rot = trsf.Rotation.from_rotvec(rotvec)
        vertices = rot.apply(vertices)    
 
    triangles = np.vstack(np.array([cells.data for cells in mesh.cells if cells.type == "triangle"]))
    I, J, K = triangles.T
    tri_points = vertices[triangles]    
    xVx = vertices[:,0]+Xa
    yVx = vertices[:,1]+Ya
    zVx = vertices[:,2]+Za
    
    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k%3][0]+Xa for k in range(4)]+[ None])
        Ye.extend([T[k%3][1]+Ya for k in range(4)]+[ None])
        Ze.extend([T[k%3][2]+Za for k in range(4)]+[ None])   
    #define the trace for triangle sides
    XYZVx = [xVx, yVx, zVx, I, J,K]
    XYZe = [Xe, Ye, Ze]

    #define the trace for triangle sides
    xVx, yVx, zVx, I, J, K =  XYZVx
    Xe, Ye, Ze = XYZe
    avatar_mesh = go.Mesh3d(x=xVx, y=yVx, z= zVx, color ='white', opacity = 1, i=I, j=J, k=K,  showscale=False)    
    avatar_lines = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', name='', line=dict(color= 'black', width=1))  
    return avatar_mesh, avatar_lines

def DrawAvatar(fig, Mesh, Lines,  row = None, col = None) :
    fig.add_trace(Mesh, row = row, col = col)
    fig.add_trace(Lines, row = row, col = col)

def DrawSol(fig, L) : 
    x=np.linspace(0, L, 2)-L/2
    y=np.linspace(0, L, 2)-L/2
    x,y=np.meshgrid(x,y)
    z_level=0#plot a z-plane at height 1
    z=z_level*np.ones(x.shape)      
    Sol = go.Surface(x=x, y=y, z=z, colorscale='gray', showscale=False)
    fig.add_trace(Sol)


def rotate(XYZ, Rot, Dir) : 
    rotdeg = Rot
    rotrad = np.radians(rotdeg)
    rotaxe = np.array([0, 0, 0])
    rotaxe[Dir] = 1
    rotvec = rotrad * rotaxe
    rot = trsf.Rotation.from_rotvec(rotvec)
    XYZr = rot.apply(XYZ) 
    return XYZr   

def DrawArray(fig, XYZm = np.random.randn(256,3), 
              MkrSz = None, Level = None, Cols = None, opacity = 1, colscale = 'turbo',
              row = None, col = None) :
    Array = go.Scatter3d(x=XYZm[:,0], y=XYZm[:,1], z=XYZm[:,2],
                            showlegend = False, name = 'Micros',
                            mode = 'markers', 
                            marker = dict(line_width = 2, 
                                        size = MkrSz, 
                                        opacity = opacity, 
                                        color= Level,
                                       # cmin =-50,
                                       # cmax = 0,
                                        colorscale=colscale,
                             ))
                        
    if row is not None and col is not None:                         
        fig.add_trace(Array, row = row, col = col)
    else :
        fig.add_trace(Array)