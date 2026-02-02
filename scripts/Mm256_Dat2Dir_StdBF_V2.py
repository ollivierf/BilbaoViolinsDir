import os
import numpy as np
import scipy.io as io
import scipy.signal as sig
import warnings
import dash
from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots 
import plotly.graph_objs as go
from scipy.spatial.distance import cdist
import LAMTools as tools

warnings.filterwarnings("ignore")

external_stylesheets = [dbc.themes.YETI]

Path_Data = './marteau/'
#Path_Data = './'

MmOfst = [0,0,1.40]

Tc = 21.5 
C = np.sqrt( 1.4 * 287 *(Tc + 273) )
FS = 2**24
NbVoies = 260
NbMems = 256
Fe = 50e3
NumRef = 1  

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, prevent_initial_callbacks=True)

server = app.server

app.layout = html.Div([
dbc.Container([    
    dbc.Row([
        dbc.Col([
            html.H4('Choisir un enregistrement'),
            dcc.Dropdown(
                id='dropdown_record',
                options=[{'label': i, 'value': i} for i in os.listdir(Path_Data) if i.endswith('.dat')],
                value='hammer1.dat'
                ), 
            #dcc.Graph(id = 'display-spectrogram'),
            dcc.Graph(id = 'display-signal'),
            #dcc.Graph(id = 'display-spectrum'),
            ], width = 4),
        dbc.Col([
            html.H4('Analyse du champ à (Hz)'), 
            dcc.Input(id='input-fo', value=500),   
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
@app.callback(Output('display-signal', 'figure'),  
              Input('dropdown_record', 'value'))

def display_record(value):

    file = Path_Data + value

    SpcDataFileName = file[:-4]+'.spcr.npz'

    if SpcDataFileName in os.listdir(Path_Data) : 
        SpcData = np.load(SpcDataFileName)
        P = SpcData['P']
        Frq = SpcData['Frq']
        t = SpcData['t']
        pref = SpcData['pref']
        tt = SpcData['tt']
        ff = SpcData['ff'] 
        Spgr = SpcData['Spgr']
    else : 
        NbBytes = os.path.getsize(file)       
        NbSamples = NbBytes/4    
        NbTixels = int(NbSamples/NbVoies)

        dt = 1/Fe
        
        t = np.arange(NbTixels) * dt
        Sog = np.fromfile(file, dtype="int32")    
        Sig = np.reshape(Sog, (NbTixels,NbVoies))
        Sig = Sig.T     
        Mems = Sig[1:NbMems+1,:]     
        Mems = np.float64(Sig[1:NbMems+1,:])# (Pa)

        Nfft = Fe/10
        NTrt = 4096
        NOvlp = 2048
        pref = Mems[NumRef,:]
        ff,tt, Spgr = sig.spectrogram(pref, nfft = Nfft, fs = Fe, nperseg = NTrt,return_onesided=True,axis = 0,noverlap= NOvlp)
        Frq, RR = sig.welch(  pref, nfft = Nfft, fs = Fe, nperseg = NTrt,return_onesided=True,axis = 0,noverlap= NOvlp)
        Frq, PR = sig.csd(Mems.T, pref, nfft = Nfft, fs = Fe, nperseg = NTrt,return_onesided=True,axis = 0,noverlap= NOvlp)

        P = PR/np.sqrt(RR[:,np.newaxis])
        PAvg = np.mean(np.abs(P), axis = 1)
        np.savez(file[:-3]+'spcr',P = P, Frq=Frq, t = np.arange(NbTixels)/Fe, pref = pref, tt=tt, ff=ff, Spgr = Spgr)



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

def display_record(value):
    
    file = Path_Data + value

    NbBytes = os.path.getsize(file)       
    NbSamples = NbBytes/4    
    NbTixels = int(NbSamples/NbVoies)

    dt = 1/Fe
    
    t = np.arange(NbTixels) * dt
    Sog = np.fromfile(file, dtype="int32")    
    Sig = np.reshape(Sog, (NbTixels,NbVoies))
    Sig = Sig.T     
    Mems = Sig[1:NbMems+1,:]     
    Mems = np.float64(Sig[1:NbMems+1,:])# (Pa)

    Nfft = Fe/10
    NTrt = 4096
    NOvlp = 2048
    pref = Mems[NumRef,:]
    ff,tt, Spgr = sig.spectrogram(pref, nfft = Nfft, fs = Fe, nperseg = NTrt,return_onesided=True,axis = 0,noverlap= NOvlp)
    Frq, RR = sig.welch(  pref, nfft = Nfft, fs = Fe, nperseg = NTrt,return_onesided=True,axis = 0,noverlap= NOvlp)
    Frq, PR = sig.csd(Mems.T, pref, nfft = Nfft, fs = Fe, nperseg = NTrt,return_onesided=True,axis = 0,noverlap= NOvlp)

    P = PR/np.sqrt(RR[:,np.newaxis])
    PAvg = np.mean(np.abs(P), axis = 1)
    np.savez(file[:-3]+'spcr',P = P, Frq=Frq, t = np.arange(NbTixels)/Fe, pref = pref, tt=tt, ff=ff, Spgr = Spgr)

    fig2 = go.Figure(go.Heatmap(x=tt,y=ff,z=20*np.log10(np.abs(Spgr)), zmin = 50, zmax = 120, colorscale='inferno'))
    fig2.update_yaxes(title_text="Frequence", range = [0, 10000])
    fig3 = go.Figure(go.Scatter(x=t, y=pref))
    fig3.update_layout(#height = 300, 
                margin=dict(l=5, r=5, t=5, b=5))
    fig4 = go.Figure(go.Scatter(x=Frq, y=20*np.log10(PAvg)))
    fig4.update_layout(#height = 300, 
                        margin=dict(l=5, r=5, t=5, b=5))
    fig4.update_xaxes(title_text="Frequence", range = [0, 10000])

    return fig2 ,fig3 , fig4 

@app.callback(Output('display-blobi', 'figure'), 
              Output('display-blobp', 'figure'),   
              Output('display-dir', 'figure'),                               
              Input('btn-champ', 'n_clicks'),
              Input('slider-dyn', 'value'),
              State('dropdown_record', 'value'),    
              State('input-fo', 'value'))
def display_field(n, dyndB, record, foStr):
    r = 0
    layout = go.Layout(font=dict(size=12, color='black'),height=900,margin=dict(l=5, r=5, t=5, b=5),scene_xaxis_visible=True,scene_yaxis_visible=True, scene_zaxis_visible=True, paper_bgcolor='white',showlegend = False)
    fig1 = go.Figure(layout = layout)
    
    
    LAM256Data = np.load('XYZm_LAM.npz')
    XYZm = LAM256Data['XYZm'] 
    memsoff = LAM256Data['memsoff'] 
    XYZm = tools.rotate(XYZm, 180, 2)
    Datafile = Path_Data + record
    
    camera = dict(up=dict(x=0, y=0, z=1),  center=dict(x=0, y=0, z=0), eye=dict(x=1, y=1, z=1))

    SpcData = np.load(Datafile[:-4]+'.spcr.npz')
    
    P = SpcData['P']
    P = np.delete(P,memsoff,1)
    Frq = SpcData['Frq']

    camera = dict(up=dict(x=0, y=0, z=1),  center=dict(x=0, y=0, z=0), eye=dict(x=2*np.cos(r*np.pi/180), y=2*np.sin(r*np.pi/180), z=0))

    pas = 0.05
    MinV = [np.min(XYZm[:,0]), np.min(XYZm[:,1]),np.min(XYZm[:,2])]
    MaxV = [np.max(XYZm[:,0]), np.max(XYZm[:,1]),np.max(XYZm[:,2])]
    
    VCub, nx, ny, nz = tools.CalcSqrGrid(MinV, MaxV, pas)
    nv = len(VCub)
    RCub = np.array(cdist(VCub, XYZm)) 
   #Fonction de Green pour une fréquence
    fo = float(foStr)
    ifo = np.argmin(np.abs(Frq-fo))
    omg = 2*np.pi*Frq[ifo]
    G = np.exp(-1j*omg*RCub/C)

    #Beamforming Light
    Pfo = np.array(P[ifo,:])
    S = np.dot(G, Pfo)
    BlobI = np.abs(S)**2
    BlobI /= np.max(BlobI)
    BlobP = np.real(S)/np.max(np.abs(S))

    DyndB = dyndB
    Dyn = 10**(-DyndB/10)

    Io = np.asnumpy(np.argmax(np.abs(BlobI)))
    Xo, Yo, Zo = VCub[Io,0], VCub[Io,1], VCub[Io,2]

    BlobI = np.asnumpy(BlobI)
    BlobP = np.asnumpy(BlobP)
    
    fig1.add_trace(go.Volume( x = VCub[:,0],  y = VCub[:,1],  z = VCub[:,2], value = BlobI, opacity = 0.5, surface_count = 2, isomin = Dyn/2, isomax = Dyn, showscale=False))    
    fig1.add_trace(go.Scatter3d(x = XYZm[:,0], y = XYZm[:,1], z = XYZm[:,2], mode = 'markers',  marker = dict(line_width=1, size=2, opacity = .5, color = 'black',showscale=False))) 
    fig1.update_layout(scene_camera=camera)

    fig2 = go.Figure(layout = layout)
    fig2.add_trace(go.Isosurface( x = VCub[:,0],  y = VCub[:,1],  z = VCub[:,2], value = BlobP, opacity = 0.5, surface_count = 1, isomin = Dyn, isomax = Dyn, colorscale='reds',showscale=False))
    fig2.add_trace(go.Isosurface( x = VCub[:,0],  y = VCub[:,1],  z = VCub[:,2], value = BlobP, opacity = 0.5, surface_count = 1, isomin = -Dyn, isomax = -Dyn, colorscale='blues',showscale=False))
    fig2.add_trace(go.Scatter3d(x = XYZm[:,0], y = XYZm[:,1], z = XYZm[:,2], mode = 'markers', marker = dict(line_width=1, size=2, opacity = .5, color = 'black',showscale=False))) 
    fig2.update_layout(scene_camera=camera)                   
    
    ## Diagramme de directivité
    n = 100
    Cntr = [0,0,0]
    VSph,nn,th,ph = tools.CalcSphGrid(Cntr, r = 0.25, n = n)     
    VSph = tools.rotate(VSph, 180, 2)
   
    RSph = np.array(cdist(VSph,XYZm))

    G = np.exp(-1j*RSph*omg/C)
    Diag = np.dot(G,Pfo)
    Diag /= np.max(np.abs(Diag))    
    Diag = np.asnumpy(Diag.reshape(n,n//2)/3)

    Ro = 0.
    Xs  = (Ro+np.abs(Diag))*np.sin(th)*np.cos(ph)
    Ys  = (Ro+np.abs(Diag))*np.sin(th)*np.sin(ph)
    Zs  = (Ro+np.abs(Diag))*np.cos(th)
    
    fig3 = go.Figure(layout = layout)
    fig3.add_trace(go.Surface(z=Zs+Zo, x=Xs+Xo, y=Ys+Yo,colorscale = 'RdBu', surfacecolor=np.real(Diag)-Ro, opacity=1, showscale=False))
    fig3.add_trace(go.Scatter3d(x = XYZm[:,0], y = XYZm[:,1], z = XYZm[:,2], mode = 'markers', marker = dict(line_width=1, size=2, opacity = 0.5, color = 'black',showscale=False)))   
    fig3.update_layout(scene_camera=camera)

    return fig1, fig2, fig3

app.run_server(debug=True)    
