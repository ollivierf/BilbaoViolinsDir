#%%
import matplotlib.pyplot as plt
from matplotlib import ticker
import h5py as h5
import numpy as np
from numpy.fft import rfft, irfft
import scipy.signal as sig
import os
import glob
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
def on_press(event):
    print('you pressed', event.button, event.xdata, event.ydata)
#%%
Tc = 26 
C = np.sqrt( 1.4 * 287 *(Tc + 273) )
Fe = 50e3
dt = 1/Fe
Po = 20e-6

NbCmpt = 1
NbMems0 = 256
NbMics = 257
NbAnal = 1
SensMems = 2.83e5
SAna = 2**23/2.5 #sensibilite des voies analogiques

NbVoies = NbCmpt + NbAnal   + NbMems0
NumSnd = 257
#%%
Pathnpz = '../marteau'
Files = [f for f in os.listdir(Pathnpz) if f.endswith('.npz') and f.startswith('hammer')]
NbFiles = len(Files)
toa = np.zeros((5, 257, NbFiles))
PingsData = np.load(Pathnpz + '/' + 'PingsAll.npz')
PMics = PingsData['Mms']
PMicref = PingsData['Mcr']
PHam = PingsData['Hmr']
plt.close('all')
#%%
Lmax = 3
tmax = Lmax/C
lmax = 1.
tfmax = lmax/C
ifmax = tfmax*Fe
   
for ff in range(NbFiles):    
    
    Mics = np.concatenate((PMics[ff,:,:,:], PMicref[ff,:,:,:]), axis=2)    
    Ham = PHam[ff,:,:].squeeze()
    NbPings = Ham.shape[0]
    NbTixels = np.shape(Mics)[1]
    t = np.arange(NbTixels)/Fe

    NFFT = NbTixels
    df = Fe/NFFT
    T = np.zeros(NbMics)
    
    SHam = rfft(Ham, NFFT, axis = 1)                          
    SMics =rfft(Mics, NFFT, axis = 1)    
    
    NUp = 10*NFFT
    dtUp = 1/(NUp*df)    
    MicsUp = irfft(SMics, NUp,axis = 1)
    HamUp = irfft(SHam, NUp, axis = 1)
    tt = np.arange(NUp)*dtUp
    MicsUp = MicsUp[:,tt<tmax,:]
    MicsUp = np.abs(sig.hilbert(MicsUp, axis = 1))
    HamUp = HamUp[:,tt<tmax] 
    tt = tt[tt<tmax]
    maxMicsUp = np.max(MicsUp, axis = 1)[:,None,:]
    maxHamUp  = np.max(HamUp, axis = 1)[:,None]
    MicsUp = MicsUp/maxMicsUp
    HamUp = HamUp/maxHamUp
    plt.figure()
    plt.plot(HamUp.T)    
    plt.plot(MicsUp[:,:,-1].T)
 #   plt.show()
    
    TrigLvl = 0.1
    NbF = NbMics//8    
    
    fig = plt.figure(ff)
    for p in range(NbPings):
        iTrigHam  = np.argwhere(HamUp[p, :]>TrigLvl)[0]  # Get the first index in HamUp[p, :] > TrigLvl
        iTrigMics = np.array([np.argwhere(MicsUp[p, 150:, m] > TrigLvl)[0]+150 for m in range(NbMics)])
        toa[p, :, ff] = (iTrigMics - iTrigHam).squeeze() * dtUp
        plt.subplot(NbPings, 1, p + 1)
        plt.pcolormesh(range(NbMics + 1), tt * C, MicsUp[p, :-1, :], cmap='Greys')
        plt.plot(range(NbMics), toa[p, :, ff] * C, '.r')
    plt.show()
    cid = fig.canvas.mpl_connect('button_press_event', on_press)
toa = np.rollaxis(toa,2,0)
#toa = toa[(0,3,4,5,1,2),:,:]
toa = toa.reshape(-1,257)
# toa = np.concatenate((toa[:16], toa[20:]), axis = 0)
plt.figure()
plt.pcolormesh(toa, cmap='jet')
plt.colorbar()
plt.xlabel('Mics')
plt.ylabel('Pings')
plt.title('TOA')

plt.xticks(np.arange(0,NbMics, 8))
plt.grid()
plt.show()       
np.save('toaUp_hammer.npy', toa)

# %%
