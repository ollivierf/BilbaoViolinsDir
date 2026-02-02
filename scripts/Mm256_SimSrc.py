import os
import numpy as np
import h5py as h5
import scipy.signal as sig
import warnings
from scipy.spatial.distance import cdist
import scipy.special as spe
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
from scipy.spatial.transform import Rotation as R
C = 340

#%% Chargement des coordonnées des micros
XYZm = np.load('XYZm_Calib3_aligne.npz')['XYZm']
    
#%% Simulation du signal d'une source sur les micros
fsim = 100
lbda = C/fsim
k = 2*np.pi/lbda
print("Simulation des signaux aux micros")
SrcTxt = "Dipole"
## Monopole 
if SrcTxt == "Monopole" : 
    S = np.array([[0,0,0]])    
    PosEx = S
    Rsim = cdist(XYZm,S)
    P = (np.exp(-1j*2*np.pi*fsim*Rsim/C)/Rsim).T
# Dipole constitué de deux monopoles en opposition de phase
elif SrcTxt == "Dipole":
    d = lbda/30
    S = np.array([[-d/2, 0, 0],
                  [ d/2, 0, 0]])
    PosEx = np.array([0., 0., 0.])
    S += PosEx    
    R1sim = cdist(XYZm, S[0,:][None,:])
    R2sim = cdist(XYZm, S[1,:][None,:])
    P = np.exp(-1j*2*np.pi*fsim*R1sim/C)/R1sim - np.exp(-1j*2*np.pi*fsim*R2sim/C)/R2sim
# Quadripole constitué de deux Dipoles et en opposition de phase
elif SrcTxt == "QuadripoleT":
    d = lbda/30
    dT = lbda/20
    S = np.array([[-d/2, -dT/2, 0],
                  [ d/2, -dT/2, 0],
                  [-d/2,  dT/2, 0], 
                  [ d/2, dT/2,  0]])
    
    PosEx = np.array([0., 0., 0.])
    S += PosEx
    
    R11sim = cdist(S[0,:][None,:],XYZm)
    R12sim = cdist(S[1,:][None,:],XYZm)
    R21sim = cdist(S[2,:][None,:],XYZm)
    R22sim = cdist(S[3,:][None,:],XYZm)
    
    P1sim = np.exp(-1j*2*np.pi*fsim*R11sim/C)/R11sim - np.exp(-1j*2*np.pi*fsim*R12sim/C)/R12sim
    P2sim = -(np.exp(-1j*2*np.pi*fsim*R21sim/C)/R21sim - np.exp(-1j*2*np.pi*fsim*R22sim/C)/R22sim)
    P = P1sim + P2sim

elif SrcTxt == "QuadripoleL":
    d = lbda/30
    dT = lbda/20
    S = np.array([[-d/2-dT/2, 0 , 0],
                  [ d/2-dT/2, 0 , 0],
                  [-d/2+dT/2, 0 , 0],
                  [ d/2+dT/2, 0 , 0]
                  ])
    
    PosEx = np.array([0., 0., 0.])
    S += PosEx
    
    R11sim = cdist(S[0,:][None,:],XYZm)
    R12sim = cdist(S[1,:][None,:],XYZm)
    R21sim = cdist(S[2,:][None,:],XYZm)
    R22sim = cdist(S[3,:][None,:],XYZm)
    
    P1sim = np.exp(-1j*2*np.pi*fsim*R11sim/C)/R11sim - np.exp(-1j*2*np.pi*fsim*R12sim/C)/R12sim
    P2sim = -(np.exp(-1j*2*np.pi*fsim*R21sim/C)/R21sim - np.exp(-1j*2*np.pi*fsim*R22sim/C)/R22sim)
    P = P1sim + P2sim
    
#%%
import plotly.graph_objects as go
figy = go.Figure()
figy.add_trace(go.Scatter3d(x = XYZm[:,0],y =XYZm[:,1] ,z =XYZm [:,2] , mode='markers', opacity=1))   
figy.add_trace(go.Scatter3d(x = S[:,0],y = S[:,1] ,z =S[:,2] , mode='markers', opacity=1))   

figy.update_layout(autosize=False, width=1000, height=1000, margin=dict(l=50, r=50, b=100, t=100, pad=4), paper_bgcolor="Black",)
figy.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(XYZm[:,0], XYZm[:,1],XYZm[:,2], s = 20, c = np.real(P))
ax.scatter(S[:,0], S[:,1],S[:,2], s = 30 , c = 'r', alpha = 1)
 #%%   f = 1kHz
 #Source 2 : Dipole x en [0,0,0]
 #Source 1 : Monopole en [0,0,0]
np.save('Src1_1kHz.npy', P)