import os
import numpy as np
import scipy.io as io
import scipy.signal as sig
import warnings
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots 
import plotly.graph_objs as go
from scipy.spatial.distance import cdist
import cupy
import cupyx.scipy.linalg as culin
import Tools3D as tools

warnings.filterwarnings("ignore")

external_stylesheets = [dbc.themes.YETI]

Path_Data = 'Victor/'
Path_Avatar = './'
CalibDataFile = 'XYZm_Calib3_aligne'

Tc = 21.5 
C = np.sqrt( 1.4 * 287 *(Tc + 273) )
FS = 2**24
NbVoies = 257
NbMems = 256
Fe = 50e3
NumRef = 1  

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, prevent_initial_callbacks=True)

server = app.server

app.layout = html.Div([
dbc.Container([
    dbc.Row([
        dbc.Col(html.Div(html.H1('Acoustic Twin V 0.5'))),
        ]),    
    dbc.Row([
        dbc.Col([
            html.H3('Choisir un avatar'),
            dcc.Dropdown(
                id='dropdown_avatar',
                options=[{'label': i[:-4], 'value': i[:-4]} for i in os.listdir(Path_Avatar) if i.endswith('ply')],
                value='Isa'
                ),   
            html.H4('Pseudo'),
            dcc.Input(id = 'input-pseudo', value = 'Isa'),
            html.H4(id = 'lbl-taille', children='Taille : 1.55 m'),
            dcc.Slider(id = 'slider-taille', value = 1.55, min = 1.20, max = 2.0, step = 0.05),                        
            html.H4(id = 'lbl-x',children='X : 0 m'),
            dcc.Slider(id = 'slider-x', value = 0, min = -1, max = 1, step = 0.1),                        
            html.H4(id = 'lbl-y',children='Y : 0 m'),
            dcc.Slider(id = 'slider-y', value = 0, min = -1, max = 1, step = 0.1),
            html.H4(id = 'lbl-r',children='Theta : 180°'),
            dcc.Slider(id = 'slider-r', value = 180, min = 0, max = 360, step = 5),
            dbc.Button(id = 'btn-Avatar', children='Valider'),
            ], width=2),    
        dbc.Col([
            html.H4('Choisir un enregistrement'),
            dcc.Dropdown(
                id='dropdown_record',
                options=[{'label': i, 'value': i} for i in os.listdir(Path_Data) if i.endswith('dat')],
                value='Lucile.dat'
                ), 
            #dcc.Graph(id = 'display-spectrogram'),
            dcc.Graph(id = 'display-signal'),
            #dcc.Graph(id = 'display-spectrum'),
            ], width = 4),
        dbc.Col([
            html.H4('Analyse du champ à (Hz)'), 
            dcc.Input('input-fo' ),   
            html.H4(id='lbl-dyn', children ='Isosurface à (dB)'),       
            dcc.Slider(id = 'slider-dyn', value = 3, min = 0, max = 20, step = 1),
            dbc.Button(id = 'btn-champ', children='Calculer'),
            dbc.Tabs(
                [
                dbc.Tab(
                    dbc.Card(
                       dbc.CardBody( dcc.Graph(id = 'display-blobi'),
                 )), label="BlobI"),
                dbc.Tab(dbc.Card(
                       dbc.CardBody( [
                                      dcc.Graph(id = 'display-blobp'),
                                      dcc.Interval(id = 'interval', interval = 100, max_intervals= 1000),
                                      ]
                 )), label="BlobP"),
                dbc.Tab(dbc.Card(
                       dbc.CardBody( dcc.Graph(id = 'display-dir'),                
                )),label="Directivite")
                ]),              
            ], width = 6),        
        ]),   
    ], fluid = True),
])
@app.callback(Output('lbl-taille', 'children'),              
              Input('slider-taille', 'value'))
def display_taille(tailleStr):
    return 'Taille : ' + str(tailleStr) + ' m'
@app.callback(Output('lbl-x', 'children'),              
              Input('slider-x', 'value'))
def display_x(xStr):
    return 'X : ' + str(xStr) + ' m'
@app.callback(Output('lbl-y', 'children'),              
              Input('slider-y', 'value'))
def display_y(yStr):
    return 'Y : ' + str(yStr) + ' m'
@app.callback(Output('lbl-r', 'children'),              
              Input('slider-r', 'value'))
def display_r(rStr):
    return 'Theta : ' + str(rStr) + '°'     
@app.callback(Output('lbl-dyn', 'children'),              
              Input('slider-dyn', 'value'))           
def display_dyn(dynStr):              
    return 'Dynamique : ' + str(dynStr) + ' dB' 

@app.callback(Output('display-signal', 'figure'),  
              Input('dropdown_record', 'value'))

def display_record(value):
    file = Path_Data + value
    SpcData = np.load(file[:-4]+'.spcr.npz')
    P = SpcData['P']
    Frq = SpcData['Frq']
    t = SpcData['t']
    pref = SpcData['pref']
    tt = SpcData['tt']
    ff = SpcData['ff'] 
    Spgr = SpcData['Spgr']
    fig = make_subplots(rows=3, cols=1)
    fig.update_layout(height=1000)
    fig.add_trace(go.Heatmap(x=tt,y=ff,z=20*np.log10(np.abs(Spgr)), zmin = 50, zmax = 120, colorscale='inferno', showscale=False,showlegend=False),row=1, col=1)
    fig.update_xaxes(title_text="Temps", row=1, col=1)
    fig.update_yaxes(title_text="Frequence", range = [0, 10000], row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=pref, showlegend=False), row=2, col=1)
    fig.update_xaxes(title_text="Temps", row=2, col=1)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    fig.add_trace(go.Scatter(x=Frq, y=20*np.log10(np.mean(np.abs(P),axis=1)),showlegend=False), row=3, col=1)
    fig.update_xaxes(title_text="Frequence", range = [0, 10000], row=3, col=1)

    return fig

@app.callback(Output('display-blobi', 'figure'), 
              Output('display-blobp', 'figure'),   
              Output('display-dir', 'figure'),                               
              Input('btn-champ', 'n_clicks'),
              Input('slider-taille', 'value'), 
              Input('slider-x', 'value'),Input('slider-y', 'value'), Input('slider-r', 'value'),
              Input('slider-dyn', 'value'),
              State('dropdown_avatar', 'value'),
              State('dropdown_record', 'value'),    
              State('input-fo', 'value'))
def display_field(n, taille, x,y,r , dyndB,avatar, record, foStr):
    
    avatar_mesh, avatar_lines = tools.CalcAvatar(avatar, x,y,0,taille, r)
    avatar_data = tools.DrawAvatar(avatar_mesh, avatar_lines) 
    
    layout = go.Layout(font=dict(size=12, color='black'),height=900,margin=dict(l=5, r=5, t=5, b=5),scene_xaxis_visible=True,scene_yaxis_visible=True, scene_zaxis_visible=True, paper_bgcolor='white',showlegend = False)
    fig1 = go.Figure(layout = layout)
    fig1.update(data=avatar_data)
    fig1.add_trace(tools.DrawSol(2))  
    fig2 = go.Figure(layout = layout)
    fig2.update(data=avatar_data)
    fig2.add_trace(tools.DrawSol(2))  
    fig3 = go.Figure(layout = layout)
    fig3.update(data=avatar_data)
    fig3.add_trace(tools.DrawSol(2))  
    
    
    FdS256Data = np.load('FdS256_ConfigMedium.npz')
    XYZm = FdS256Data['XYZm'] 
    memsoff = FdS256Data['memsoff'] 
    VCub = FdS256Data['VCub']  
    XYZm = tools.rotate(XYZm, 180, 2)
    Datafile = Path_Data + record
    
    camera = dict(up=dict(x=0, y=0, z=1),  center=dict(x=0, y=0, z=0), eye=dict(x=1, y=1, z=1))

    SpcData = np.load(Datafile[:-4]+'.spcr.npz')
    
    P = SpcData['P']
    P = np.delete(P,memsoff,1)
    Frq = SpcData['Frq']

    avatar_mesh, avatar_lines = tools.CalcAvatar(avatar, x,y,0,taille, r)
    avatar_data = tools.DrawAvatar(avatar_mesh, avatar_lines) 
    camera = dict(up=dict(x=0, y=0, z=1),  center=dict(x=0, y=0, z=0), eye=dict(x=2*np.cos(r*np.pi/180), y=2*np.sin(r*np.pi/180), z=0))

    pas = 0.02
    MinV = [np.min(XYZm[:,0]), np.min(XYZm[:,1]),np.min(XYZm[:,2])]
    MaxV = [np.max(XYZm[:,0]), np.max(XYZm[:,1]),np.max(XYZm[:,2])]
    
    VCub, nx, ny, nz = tools.CalcSqrGrid(MinV, MaxV, pas)
    nv = len(VCub)
    RCub = cupy.array(cdist(VCub, XYZm)) 
   #Fonction de Green pour une fréquence
    fo = float(foStr)
    ifo = np.argmin(np.abs(Frq-fo))
    omg = 2*np.pi*Frq[ifo]
    G = cupy.exp(-1j*omg*RCub/C)

    #Beamforming Light
    Pfo = cupy.array(P[ifo,:])
    S = cupy.dot(G, Pfo)
    BlobI = cupy.abs(S)**2
    BlobI /= cupy.max(BlobI)
    BlobP = cupy.real(S)/cupy.max(cupy.abs(S))

    DyndB = dyndB
    Dyn = 10**(-DyndB/10)

    Io = cupy.asnumpy(cupy.argmax(cupy.abs(BlobI)))
    Xo, Yo, Zo = VCub[Io,0], VCub[Io,1], VCub[Io,2]

    BlobI = cupy.asnumpy(BlobI)
    BlobP = cupy.asnumpy(BlobP)
    
    fig1.add_trace(go.Volume( x = VCub[:,0],  y = VCub[:,1],  z = VCub[:,2], value = BlobI, opacity = 0.5, surface_count = 2, isomin = Dyn/2, isomax = Dyn, showscale=False))    
    fig1.add_trace(go.Scatter3d(x = XYZm[:,0], y = XYZm[:,1], z = XYZm[:,2], mode = 'markers',  marker = dict(line_width=1, size=2, opacity = .5, color = 'black',showscale=False))) 
    fig1.update_layout(scene_camera=camera)

    fig2 = go.Figure(layout = layout)
    fig2.update(data=avatar_data)
    fig2.add_trace(tools.DrawSol(2))  
    fig2.add_trace(go.Isosurface( x = VCub[:,0],  y = VCub[:,1],  z = VCub[:,2], value = BlobP, opacity = 0.5, surface_count = 1, isomin = Dyn, isomax = Dyn, colorscale='reds',showscale=False))
    fig2.add_trace(go.Isosurface( x = VCub[:,0],  y = VCub[:,1],  z = VCub[:,2], value = BlobP, opacity = 0.5, surface_count = 1, isomin = -Dyn, isomax = -Dyn, colorscale='blues',showscale=False))
    fig2.add_trace(go.Scatter3d(x = XYZm[:,0], y = XYZm[:,1], z = XYZm[:,2], mode = 'markers', marker = dict(line_width=1, size=2, opacity = .5, color = 'black',showscale=False))) 
    fig2.update_layout(scene_camera=camera)                   
    
    ## Diagramme de directivité
    n = 100
    Cntr = [Xo, Yo, Zo]
    VSph = FdS256Data['VSph']

    VSph,nn,th,ph = tools.CalcSphGrid(Cntr, r = 0.25, n = n)     
    VSph = tools.rotate(VSph, 180, 2)
   
    RSph = cupy.array(cdist(VSph,XYZm))

    G = cupy.exp(-1j*RSph*omg/C)
    Diag = cupy.dot(G,Pfo)
    Diag /= cupy.max(cupy.abs(Diag))    
    Diag = cupy.asnumpy(Diag.reshape(n,n//2)/3)

    Ro = 0.
    Xs  = (Ro+np.abs(Diag))*np.sin(th)*np.cos(ph)
    Ys  = (Ro+np.abs(Diag))*np.sin(th)*np.sin(ph)
    Zs  = (Ro+np.abs(Diag))*np.cos(th)
    
    fig3 = go.Figure(avatar_data,layout = layout)
    fig3.add_trace(tools.DrawSol(2))  
    fig3.add_trace(go.Surface(z=Zs+Zo, x=Xs+Xo, y=Ys+Yo,colorscale = 'inferno', surfacecolor=np.abs(Diag)-Ro, opacity=1, showscale=False))
    fig3.add_trace(go.Scatter3d(x = XYZm[:,0], y = XYZm[:,1], z = XYZm[:,2], mode = 'markers', marker = dict(line_width=1, size=2, opacity = 0.5, color = 'black',showscale=False)))   
    fig3.update_layout(scene_camera=camera)

    return fig1, fig2, fig3

app.run_server(debug=True)    