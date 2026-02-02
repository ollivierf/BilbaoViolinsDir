#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import warnings
import plotly.graph_objs as go
from scipy.spatial.distance import cdist
import scipy.signal as sig
import os
import Tools3D as tools3d
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
#Vitesse du son au moment de la mesure, dependant de la temperature:
Tc = 21.5 
C = np.sqrt( 1.4 * 287 *(Tc + 273) )
Path = './'
CalibData = np.load(Path+'XYZm_Calib3_aligne.npz')
Xm = CalibData['XYZm']
Nbm = len(Xm)
L = 1
file = 'Victor/violon4/gamme_forte.dat'
Avat = "Hugo"
XA,YA,ZA, HA, RA = 0,-0.1,-1.5, 1.7,-90
FS = 2**23/2.5# 2.5V = dynamique du CAN
NbMems = 256
NbVoies = 259
Fe = 50e3

Avatar = tools3d.BuildAvatar(Avat+'.ply', XA, YA, ZA, HA, RA)


# In[3]:


Nfft = int(Fe/10)
NTrt = 512  
NOvlp = int(0.5*NTrt)
fen = sig.windows.hann(NTrt)

NbEch = os.stat(file).st_size // (NbVoies*4)
NOvlp = NTrt//2
NSTFT = int(NbEch//(NTrt-NOvlp))-1
RMS = np.zeros(NbMems)
fid = open(file,'rb')
jj = 0
for ii in range(NSTFT) : 
    Ofst = 0 if not ii else -(NOvlp*4*NbVoies)
    data = np.fromfile(fid, count = NTrt*NbVoies, offset = Ofst, dtype='int32')
    
    Sigs = np.reshape(data,(-1, NbVoies)).T
    Mems = np.float64(Sigs[1:-2,:]) /FS          
            
    #Niveaux RMS    
    RMS += np.sum(Mems**2,1) #V
fid.close()


# In[4]:


#########################################################
## Vérification des micros
##########################################################
from plotly.subplots import make_subplots
rms = RMS.reshape(-1,8)
rmsdB = 20*np.log10(rms/np.max(rms))
maxdB = np.max(rmsdB)
fig = go.Figure()
fig = make_subplots(rows=1, cols=2, specs=[[{}, {"type":"scene"}]])
tools3d.DrawArray(fig, XYZm = Xm, MkrSz = 6 , Level=rmsdB.flatten(),   row=1, col=2)
fig.add_trace(go.Heatmap(z = rmsdB.T, colorscale='hot'), row=1, col=1)
fig.update_layout(yaxis = dict(scaleanchor = 'x',constrain = "domain"))
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
fig.show()


# In[5]:


##################################################
# Calcul du Spectrogramme
#####################################################
Nfft = 4096
NTrt = 1024
NOvlp = int(0.25*NTrt)
fen = sig.windows.hann(NTrt)

NbEch = os.stat(file).st_size // (NbVoies*4)
NOvlp = NTrt//2
NSTFT = int(NbEch//(NTrt-NOvlp))-1

NSample = NSTFT
Spgr = np.zeros((NSTFT,int(Nfft/2+1)), dtype='complex128')
MM = np.zeros((int(Nfft/2+1)), dtype='complex128')
fid = open(file,'rb')
jj = 0
Sig = []
for ii in range(NSTFT) : 
    Ofst = 0 if not ii else -(NOvlp*4*NbVoies)
    data = np.fromfile(fid, count = NTrt*NbVoies, offset = Ofst, dtype='int32')    
    Sigs = np.reshape(data,(-1, NbVoies)).T #V
    Mems = np.float64(Sigs[128,:])              
    Spgr[ii,:] = np.fft.rfft(Mems*fen, Nfft)    
    Sig.append(Mems[NOvlp:].T)
    Lp = np.sqrt(np.std(np.abs(Spgr[ii,:]))/ ((ii+1)*NTrt) ) 
    print("\r{:4d}/{:4d} : Lp  =  {:.4f} Pa ".format(ii, NSTFT, Lp), end="")  
fid.close()


# In[6]:


t = np.arange(NSTFT*(NTrt-NOvlp))/Fe
dtt = (NTrt-NOvlp)/Fe
tt = np.arange(NSTFT)*dtt
Sog = np.array(Sig).reshape(-1,1)
freq = np.fft.rfftfreq(Nfft,1/Fe)
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=t,y=Sog.ravel()), row=1, col=1)
fig.add_trace(go.Heatmap(x=tt,y=freq,z=20*np.log10(np.abs(Spgr.T)),zmin = 80, zmax = 120, colorscale='inferno'), row=2, col=1)
fig.update_yaxes(title_text="Frequence", range = [0, 20000], row=2, col=1)
#fig.update_layout(template="plotly_dark")
fig.show()


# In[7]:


Son = np.round(Sog/np.max(np.abs(Sog))*2**23)[:,0]
from IPython.display import Audio
Audio(Son, rate=Fe)


# In[8]:


tmin, tmax = 1.5, 14.2
SogZ = Sog[np.logical_and(t >= tmin, t<tmax)]
SpgZ = Spgr[np.logical_and(tt >= tmin, tt<tmax),:]
tZ = t[np.logical_and(t >= tmin, t<tmax)]
ttZ =tt[np.logical_and(tt >= tmin, tt<tmax)] 
freq = np.fft.rfftfreq(Nfft,1/Fe)
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=tZ,y=SogZ.ravel()), row=1, col=1)
fig.add_trace(go.Heatmap(x=ttZ,y=freq,z=20*np.log10(np.abs(SpgZ.T)),zmin = 80, zmax = 120, colorscale='inferno'), row=2, col=1)
fig.update_yaxes(title_text="Frequence", range = [0, 10000], row=2, col=1)

fig.show()
Son = np.round(SogZ/np.max(np.abs(SogZ))*2**23)[:,0]
Audio(Son, rate=Fe)


# In[9]:


fid = open(file,'rb')
NbTixels = int((tmax-tmin)*Fe)
Ofst = int(tmin*Fe)*4*NbVoies
p = np.fromfile(fid, count = NbTixels*NbVoies, offset = Ofst, dtype='int32').reshape((-1, NbVoies))[:,1:-2]
fid.close()
tz = np.arange(NbTixels)/Fe


# In[10]:


df = 10
Nfft = int(Fe/df)
NTrt = Nfft
NOvlp = 0; int(Nfft//2)
w = sig.windows.hann(NTrt)
SFT = sig.ShortTimeFFT(w, hop = NTrt-NOvlp, fs=Fe, mfft=Nfft, scale_to='magnitude')
P = SFT.stft(p.T)
Frq = SFT.f
tt = SFT.t(NbTixels)
np.save("P_Sph10_100cm.npy",P)


# In[11]:


Nf =int(Nfft/2+1)
n = 100
V,nn,th,ph = tools3d.CalcSphGrid([0, 0, 0], r = 1, n = n)
R = cdist(V,Xm[:256,:])
omg = 2*np.pi*Frq
G = np.exp(-1j*np.einsum('f,sm->fsm', omg, R)/C)
np.save("G_Sph10_100cm.npy",G)
# In[12]:


S = np.einsum('fsm,mft->st', G, P)
np.save("S_Sph10_100cm.npy",S)


# In[13]:


#Directivité brute
S = np.load("S_Sph100_100cm.npy")
S = S/np.max(np.abs(S))

Diag = (np.abs(S)**2).reshape(n,int(n/2),-1)

#Ici le diagramme de directivité Diag est normalisé par rapport au max sur toute la durée du signal
#et varie entre 0 et 1


# In[14]:


import sys
sys.path.insert(0, './array-processing/toolboxes')
from utils_SHanalysis import *

#Directivité tronquée en HS
Ordermax = 10
Omega = np.array([th.flatten(),ph.flatten()]).T
Ymn = compute_SphericalHarmonics_matrix(Omega, Ordermax)
Nt = S.shape[1]
H = np.tile(Ymn[:,:,None],(1,1,Nt))


# In[15]:


# Cmn = compute_SHcoefs(S, H) 
# plt.pcolormesh(np.log10((np.abs(Cmn)**2)))


# In[16]:


import plotly.express as px

#np.save('Cmn_Sph100_100cm.npy',Cmn)
Cmn = np.load('Cmn_Sph100_100cm.npy')
C = (np.mean(np.abs(Cmn),axis=1)**2).squeeze()
C /= np.max(np.abs(C))

SHOrders = []
OColors = []
CmnColors = []
OEnergy = []
n=0
for o in range(10):
    OColors.append(px.colors.qualitative.Alphabet[o])
    Energy=0
    for d in range(-o, o+1):
        SHOrders.append(o)
        CmnColors.append(px.colors.qualitative.Alphabet[o])
        Energy += np.abs(C[n])**2
        n+=1
    OEnergy.append(Energy)

EdB = 10*np.log10(OEnergy).squeeze()
EdB +=30
EdB[EdB<0]=0
nE = np.arange(10)
fig = go.Figure(layout = go.Layout(template='plotly_dark'))  
fig.add_traces(go.Bar(x =nE, y = EdB ,  marker_color=OColors))
fig.update_layout(title="Energie dans les ordres SH")
      


# In[23]:


Diag_r = np.einsum('mnt,nt->mt', H, Cmn)
Diag_r = (np.abs(Diag_r)**2).reshape((100,50,-1))


# In[24]:


DyndB = 30 # Dynamique en dB
Ro = 1     # Rayon de base du diagramme de directivité 

#figW = go.FigureWidget()
figW = go.FigureWidget(make_subplots(rows=4, cols=1, specs=[[{}], [{}],[{'type': 'surface','rowspan':2}],[{}]]))
figW.update_layout(width =1000, height = 1000)

DiagdB = 10*np.log10(Diag_r)
D = DiagdB + DyndB
D /= DyndB
D[D<0]=0
#A ce stade, la Dynamique de DyndB du diagramme est telle que 0dB ---> 1m et -DyndB --->0m
#On introduit un paramètre pour réduire ou augmenter la longeur de représentation : AmpliDir = 1
# Alors la Dynamique de DyndB du diagramme est telle que 0dB ---> AmpliDir m et -DyndB --->0m
AmpliDir = 1

Xs  = (Ro+D*AmpliDir)*np.sin(th[:,:,None])*np.cos(ph[:,:,None])
Ys  = (Ro+D*AmpliDir)*np.sin(th[:,:,None])*np.sin(ph[:,:,None])
Zs  = (Ro+D*AmpliDir)*np.cos(th[:,:,None])

figW.add_trace(go.Surface(x = Xs[:,:,0], y = Ys[:,:,0], z =  Zs[:,:,0],
                surfacecolor = DiagdB[:,:,0], opacity=0.9, colorscale='magma',
                cmin = -DyndB,cmax = 0), row=3, col = 1)


figW.add_trace(go.Scatter(x=tZ, y=SogZ.ravel()), row=1, col=1)
figW.add_trace(go.Heatmap(x=ttZ,y=freq,z=20*np.log10(np.abs(SpgZ.T)),zmin = 80, zmax = 120, colorscale='inferno'), 
                        row=2, col=1)
figW.update_yaxes(title_text="Frequence", range = [0, 10000], row=2, col=1)


                   
tools3d.DrawAvatar(figW, Avatar[0], Avatar[1],row=3,col = 1)
tools3d.DrawArray(figW, Xm, MkrSz=1,row=3,col = 1)                                
L*=2
Limits = [-L, L, -L, L, -L, L]
figW.update_layout(scene_aspectmode='manual', 
                   scene_aspectratio=dict(x=L, y=L, z=L),
                   width=1000, height = 1000,
                   scene = dict(xaxis = dict(nticks=10, range=[-L,L],),
                                yaxis = dict(nticks=10, range=[-L,L],),
                                zaxis = dict(nticks=10, range=[-L,L],),))
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-2, y=-2, z=1)
)
figW.update_layout(scene_camera=camera, title='Violon4 - Directivité large bande - Gamme Forte') 

figW


# In[ ]:


for n in range(Diag.shape[2]):
    
    figW.data[0]['x']=Xs[:,:,n] 
    figW.data[0]['y']=Ys[:,:,n]
    figW.data[0]['z']=Zs[:,:,n]
    figW.data[0]['surfacecolor'] = DiagdB[:,:,n]
    figW['layout']['title'] = f"Violon 4 - Directivité large bande - Gamme Forte - {n/382.*12.:.2f}/12s" 

