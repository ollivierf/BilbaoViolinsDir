import sys
sys.path.insert(0,'./src')
import bempp.api as bem
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from dash import Dash,dcc,html, Input, Output, State, callback, exceptions


import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

import utils_SHanalysis as sh
import utils_geometry as geo
import Tools3D as t3d

from bempp.api.linalg import gmres
from bempp.api.operators.boundary.sparse import identity as BSpI
from bempp.api.operators.boundary.helmholtz import single_layer as BHzL
from bempp.api.operators.boundary.helmholtz import double_layer as BHzM
from bempp.api.operators.potential.helmholtz import single_layer as DHzL
from bempp.api.operators.potential.helmholtz import double_layer as DHzM

load_figure_template(["darkly"])
global ShDt
#Milieu de propagation (air)
c = 340.
rho = 1
fo = 1000
k = 2*np.pi*fo/c
SHOrderMax = 7
SHNbMax = (SHOrderMax+1)**2

from bempp.api.grid.grid import Grid

# mesh = meshio.read('Source12.vtk')
# vertices = mesh.points.T
# elements = mesh.cells_dict["triangle"].T.astype("uint32")
# np.savez('DodecaMesh.npz', vertices = vertices, elements = elements)
mesh = np.load('DodecaMesh.npz')
DodecaGrid = Grid(mesh['vertices'], mesh['elements'], None)
spc = bem.function_space(DodecaGrid, "DP",0)
ShDt = dict(k = 2*np.pi*fo/c, 
            grid = DodecaGrid,
             spc = spc  )
ShDt['Vs'] = bem.GridFunction(spc, spc, coefficients = np.zeros(ShDt['grid'].number_of_elements))
ShDt['Ps'] = bem.GridFunction(spc, spc, coefficients = np.zeros(ShDt['grid'].number_of_elements))

ColScl1 = 'picnic'
ColScl2 = 'hsv'
# HS
def InitCmnGraph(SHOrderMax):
    SHOColors = []
    for o in range(SHOrderMax):
        for d in range(-o, o+1):
            SHOColors.append(px.colors.qualitative.Alphabet[o])
    return  SHOColors

def BuildR1mMesh():
    NbPho = 200
    NbTho = 100
    pho = np.linspace(0,2*np.pi,NbPho)
    tho = np.linspace(0,np.pi,NbTho)
    Pho,Tho = np.meshgrid(pho, tho)    
    Pho = Pho.flatten()
    Tho = Tho.flatten()
    Xo = np.cos(Pho)*np.sin(Tho)
    Yo = np.sin(Pho)*np.sin(Tho)
    Zo = np.cos(Tho)  
    return np.array([Xo,Yo,Zo]), np.array([Tho, Pho])

def BuildSrcMesh() : 
    global ShDt# Détermination de la position des HP et définition des domaines        
    grid = bem.import_grid('Source12.vtk')
    rElts = (np.linalg.norm(grid.centroids, axis=1))
    rmin = 0.06695
    rsph = 0.0730
    CentreHP = grid.centroids[rElts<rmin,:]
    nCentreHP = CentreHP/np.linalg.norm(CentreHP,axis=1)[:,None]
    dom = np.zeros(grid.number_of_elements, dtype='int32')
    for hp in range(12) :            
        for i in range(grid.number_of_elements):
            Vtcs = grid.vertices[:, grid.elements[:,i]]
            nVtcs = Vtcs/np.linalg.norm(Vtcs,axis = 0) #Normale aux sommets de l'élément       
            alpha = np.abs(np.degrees(np.arccos(np.dot(nCentreHP[hp,:], nVtcs))))       
            if np.all(alpha < 16) and rElts[i] < rsph:                                 
                dom[i] = int(hp+1)
    grid = bem.Grid(grid.vertices, grid.elements, dom)
    return grid, spc
#################################################################################################
DodecaBem = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
DodecaBem.title = 'DodécaBem_v0.1'
grid,spc = BuildSrcMesh()
Vs = bem.GridFunction(spc, coefficients = np.zeros(grid.number_of_elements))
Ps = Vs
PtsFlat, ViewAngles = BuildR1mMesh()
PtsMesh = PtsFlat.reshape(3,100,200)
    
ShDt = dict(grid=grid, spc=spc, fo=fo, k=k, Vs=Vs, Ps = Ps)

##MESHES INIT###############################################################################################
Src12Msh = t3d.DrawMeshSrc12()
camera = dict(up=dict(x=0, y=0, z=1),  center=dict(x=0, y=0, z=0), eye=dict(x=2, y=2, z=2))
HPNames = [f"vHP{i+1}" for i in range(12)]
CmnColors = InitCmnGraph(SHOrderMax)

figVs = go.Figure()
figVs.add_trace(Src12Msh[0])
#figVs.add_trace(Src12Msh[1])
figVs.update_layout(template="darkly",margin=dict(l=0, r=0, t=0, b=0), scene_camera=camera,
                    plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
figVs.update_traces(intensity = np.zeros_like(Src12Msh[0])  ,intensitymode = 'cell',         
                        cmin = -1, cmax = 1,
                        colorscale = ColScl1,  opacity=0.5, 
                        selector=dict(type="mesh3d"))
figPs = go.Figure()
figPs.add_trace(Src12Msh[0])
#figPs.add_trace(Src12Msh[1])
figPs.update_layout(template="darkly",margin=dict(l=0, r=0, t=0, b=0),scene_camera=camera,
                    plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
figPs.update_traces(intensity = np.zeros_like(Src12Msh[0])  ,intensitymode = 'cell',         
                        cmin = -1, cmax = 1,
                        colorscale = ColScl1, opacity=0.5,
                        selector=dict(type="mesh3d"))
#################################################################################################
figDiag = go.Figure()
figDiag.add_trace(go.Surface(colorscale = ColScl2, cmin=-np.pi, cmax = np.pi,
                            colorbar_orientation = 'h', opacity = 0.5))
figVs.add_trace(Src12Msh[0])

figDiag.update_layout(template="darkly", margin=dict(l=0, r=0, t=0, b=0), autosize = True,                        
                        scene=dict(aspectmode='cube'), showlegend = False,                
                        plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
#################################################################################################
figSpcHS = make_subplots(  rows=2, cols = 1) 
figSpcHS.add_trace(go.Bar(), row=1, col=1)
figSpcHS.add_trace(go.Bar(), row=2, col=1)
figSpcHS.update_layout(template="darkly", margin=dict(l=0, r=0, t=10, b=0), autosize = True,                        
                        showlegend = False,
                        plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
figSpcHS.update_xaxes(title_text = 'SH number', range= [-0.5,  (SHOrderMax+1)**2],row=1, col=1)
figSpcHS.update_yaxes(title_text = 'Amplitude normalisée', range= [0,1],row=1, col=1)
figSpcHS.update_xaxes(title_text = 'SH number', range= [-0.5,  (SHOrderMax+1)**2],row=2, col=1)
figSpcHS.update_yaxes(title_text = 'Phase', range= [-np.pi, np.pi],row=2, col=1)
#### LAYOUT ##########################################################################################################################################
options = [{'label': html.Div([str(val)]), 'value': val} for val in [-1, 0, 1]]
radio_items = [dcc.RadioItems(id=f'HP{i}', 
                              options=options, 
                              value=1, 
                              inline = True, labelStyle={'display':'block'},
                              className='d-flex justify-content-center') for i in range(1,13)]
Layout = [   
        html.H2('DodécaBem - Calcul de la Directivité de la source LMA-Dodécaèdrique ',style= {'width' : '100%', 'height': '5vh','background-color': 'transparent'}),   
           dbc.Row([
                dbc.Col([
                        dbc.Card([dbc.Label("HPs On/Off "),
                                  dbc.CardBody(radio_items),], className="fs-6"),
                        dbc.Card([dbc.Label("Fréquence (Hz) "), 
                                  dbc.Input(id = 'foInput', value = 1000),]),
                        dbc.Card([dbc.Button("Calcul de Ps", id = 'BIEBtn')]),
                        dbc.Card([dbc.Button("Calcul du rayonnement à 1m", id = 'HIEBtn')]),
                        ],
                        style={'min-height':'100vh','background-color': 'transparent'}
                        ,width=1),
                dbc.Col([ 
                        dbc.Card([dbc.Label("Champ de vitesse sur la source"),
                                  dcc.Graph(id='VsPlot',figure = figVs, responsive=True, style = {'height':'100%','background-color': 'transparent'})], 
                                 style = {'height' : '45vh','background-color': 'transparent'}), 
                        dbc.Spinner(html.Div(id='SpinnerBIE'), color="danger", type="grow"),
                        dbc.Card([dbc.Label("Champ de pression sur la source"),
                                  dcc.Graph(id='PsPlot',figure = figPs, responsive=True, style = {'height':'100%','background-color': 'transparent'}),],
                                 style = {'height' : '45vh','background-color': 'transparent'}),                        
                        ],width=2),
                dbc.Col([ 
                        dbc.Card([  dbc.Label("Directivité mesurée à 1m"),                  
                                    dcc.Graph(id='DiagPlot', figure=figDiag, responsive=True, style = {'height':'90%','background-color': 'transparent'})], 
                                    style= {'height' : '90vh','background-color': 'transparent'}),
                        ],width=5), 
                dbc.Col([ 
                        dbc.Spinner(html.Div(id='SpinnerHIE'), color="danger", type="grow"),  
                        dbc.Card([ dbc.Button("Fermer", id = 'CloseBtn', color='danger'),
                                  dbc.Label("Spectre HS"),                  
                                    dcc.Graph(id='HSPlot', figure=figSpcHS, responsive=True,  style= {'height' : '80vh','background-color': 'transparent'}),
                                    ],style= {'background-color': 'transparent'}),
                        ],width = 4),

            ], style={'min-height': '100%'}),    
    ]
        
DodecaBem.layout = Layout


@callback(Output("CloseBtn", "n_clicks"),
          Input("CloseBtn", "n_clicks"),
          prevent_initial_call = True)
          
def Close(n):        
    raise exceptions.PreventUpdate
    sys.exit()    
    
    
@callback(Output('VsPlot', 'figure'),          
         [Input(f'HP{i}', 'value') for i in range(1,13)])
          
def PlotVs(*values):        
    global ShDt
    grid, spc = BuildSrcMesh()
    vHP = [v  for v in values]
    V0 = 1 
    Vn = np.zeros(grid.number_of_elements)    
    for domix in range(1,13):        
        Vn[grid.domain_indices==domix] = vHP[domix-1]*V0
    Vs = bem.GridFunction(spc, coefficients = Vn)
    ShDt['grid'] = grid
    ShDt['spc'] = spc
    ShDt['Vs'] = Vs
    ShDt['Vn'] = Vn
    figVs.update_traces(intensity = Vn ,intensitymode = 'cell',         
                        cmin = -1, cmax = 1,
                        colorscale = ColScl1, opacity = 0.5,
                        selector=dict(type="mesh3d"))
    return figVs

@callback(Output('PsPlot', 'figure'),  
          Output('SpinnerBIE', 'children'),       
          Input('BIEBtn', 'n_clicks'),
          State('foInput','value'),
          prevent_initial_call = True)
          
def PlotPs(n,fo):   
    
    global ShDt
    #ShDt['Ps'] = bem.GridFunction(spc, spc, coefficients = np.zeros(ShDt['grid'].number_of_elements))
    
    Vs = ShDt['Vs']   
    ShDt['k'] = float(fo)*2*np.pi/c
    k =  ShDt['k'] 
    #############Calcul de la pression parietale Ps#######################
    print("Calcul de la pression sur la surface (BIE)")
    #Definition des opérateurs de frontière utiles
    I = BSpI(spc, spc, spc)
    L = BHzL(spc, spc, spc, k)
    M = BHzM(spc, spc, spc, k)
    #Formulation de l'Equation intégrale de surface  : Formulation directe élémentaire
    # (M - 0.5*I) Phi = L Vs
    A =  (M - 0.5 * I) 
    B = L*Vs
    Phis,_ = gmres(A, B, tol = 1E-5)
    Ps=1j*rho*c*k*Phis
    ShDt['Ps'] = Ps
    ShDt['Phis'] = Phis
    
    Psmax = np.max(np.abs(np.real(Ps.coefficients))) 
    figPs.update_traces(intensity = np.real(Ps.coefficients) ,
                        intensitymode = 'cell',         
                         cmin = -Psmax, cmax = Psmax,
                         colorscale = ColScl1, opacity = 0.5,
                         selector=dict(type="mesh3d"))
    return figPs, ''

@callback(Output('DiagPlot', 'figure'),          
          Output('SpinnerHIE', 'children'), 
          Output('HSPlot', 'figure')      ,
          Input('HIEBtn', 'n_clicks'),
          prevent_initial_call = True)
          
def PlotDiag(n):   
    global ShDt
    Phis = ShDt['Phis']
    Vs = ShDt['Vs']
    Vn = ShDt['Vn']
    spc = ShDt['spc']
 
    print('Calcul du Champ à 1m')
    
    Lv = DHzL(spc, PtsFlat, k)
    Mv = DHzM(spc, PtsFlat, k)
    P1m =  1j*rho*c*k*(Mv.evaluate(Phis) - Lv.evaluate(Vs))
    
    Diag = P1m.reshape(100, 200)    
    R = np.abs(Diag)/np.max(np.abs(Diag))  
    ShDt['P1m'] = P1m
    figDiag.data[0].update(x = R*PtsMesh[0,:], y= R*PtsMesh[1,:], z = R*PtsMesh[2,:], 
                          surfacecolor = np.angle(Diag),
                          colorscale = ColScl2, cmin=-np.pi, cmax = np.pi,
                         colorbar_orientation = 'h'
                           )  
    DMax = np.max(np.abs(R))
    figDiag.update_layout(title='Phase (radians)',
                            scene=dict(aspectmode='cube', 
                                    xaxis=dict(nticks= 10, range=[-DMax,DMax]),
                                    yaxis=dict(nticks= 10, range=[-DMax,DMax]),
                                    zaxis=dict(nticks= 10, range=[-DMax,DMax])),
                        height = 1000
                        )
    figDiag.add_trace(Src12Msh[0])
    figDiag.update_traces(intensity = Vn ,intensitymode = 'cell',         
                        cmin = -1, cmax = 1,
                        colorscale = ColScl1, 
                        selector=dict(type="mesh3d"))
    #figDiagVs.add_trace(Src12Msh[1])
    ShDt['P1m'] = P1m

    print("Calcul du spectre HS à l'origine")
    NbPts = PtsFlat.shape[1]
    Ymn = sh.compute_SphericalHarmonics_matrix(ViewAngles.T, SHOrderMax)
    hnkl = sh.compute_SphericalHankel_matrix(np.ones(NbPts,), np.array([k]), SHOrderMax).squeeze()
    H = Ymn*hnkl
    P1m = ShDt['P1m']
    Cmn = sh.compute_SHcoefs(P1m.T, H[:,:,None]) 
    NCmn = Cmn/np.max(np.abs(Cmn))
    figSpcHS.data[0]['x'] = np.arange(SHNbMax)
    figSpcHS.data[0]['y'] = np.abs(NCmn).squeeze()
    figSpcHS.data[0]['marker_color'] = CmnColors
    
    ACmnp = [np.angle(a) if np.abs(a)>0.1 else 0 for a in Cmn.squeeze()]
    figSpcHS.data[1]['x'] = np.arange(SHNbMax)
    figSpcHS.data[1]['y'] = ACmnp
    figSpcHS.data[1]['marker_color'] = CmnColors
    
    return figDiag,'', figSpcHS

if __name__ == "__main__":
    
    DodecaBem.run_server(debug=True)

