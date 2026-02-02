import matplotlib.pyplot as plt
from matplotlib import ticker
import h5py as h5
import numpy as np
import scipy.signal as sig
import os
import glob
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
def on_press(event):
    print('you pressed', event.button, event.xdata, event.ydata)


Tc = 26 
C = np.sqrt( 1.4 * 287 *(Tc + 273) )
Fe = 50e3
dt = 1/Fe
Po = 20e-6

NbCmpt = 1
NbMems0 = 256
NbMics = 257
NbAnal = 1
SensMems = 3.54e-6#Pa/digital unit
SAna = 2**23/2.5 #sensibilite des voies analogiques

NbVoies = NbCmpt + NbAnal   + NbMems0
NumSnd = 257

PathCalib = '../DataCalib4'
Files = os.listdir(PathCalib)
NbFiles = len(Files)
#toa = np.zeros((257, NbFiles))

plt.close('all')

Lmax = 3
tmax = Lmax/C
lmax = 1.
tfmax = lmax/C
ifmax = tfmax*Fe
   
ff = -1   
toa = []
for File in Files[1:]:    
    ff += 1 
    print(File)
    data = h5.File(PathCalib+ '/' + File, 'r')  
    
    Secs = [int(i) for i in data['muh5'].keys()] 
    NbSecs = len(data['muh5'].keys())
    
    Sig = np.concatenate([data['muh5'][str(i)]['sig'][:] for i in range(NbSecs)], axis=1)
    
    Cmptr =  Sig[0,:]
    NbTixels = len(Cmptr)
    Chk = np.sum(np.diff(Cmptr))
    if Chk != NbTixels-1:
        print("Bad Counter - Skipping file "+ File)
        continue
    Mics = Sig[np.arange(1,257) ,:]
    Mics = np.vstack((Mics,Sig[259 ,:]))
    Ref = -Sig[257,:]  #Volts
    t = np.arange(NbTixels)/Fe
    
    del Sig
    
    NFFT = NbTixels
    df = Fe/NFFT
    T = np.zeros(NbMics)
    
    SRef = np.fft.rfft(Ref, NFFT)                          
    SMics = np.fft.rfft(Mics, NFFT)    
    GCS = np.conj(SRef)[None,:] * (SMics)/(np.abs(SMics) * np.abs(SRef)[None,:])
    NUp = 10*NFFT
    dtUp = 1/(NUp*df)    
    GCC = np.fft.irfft(GCS, NUp)
    tt = np.arange(NUp)*dtUp
    GCCE = np.abs(sig.hilbert(GCC[:,tt<tmax], axis = 1))
    tt = tt[tt<tmax]
    
    imax = np.argmax(GCCE[:,500:-100], axis = 1)+500
    NbF = NbMics//8    
    toas =np.array([tt[i] for i in imax])
    toa= np.append(toa,toas) 
    fig = plt.figure(ff)
    plt.pcolormesh(range(NbMics),tt*C, GCCE.T, cmap='Greys')
    plt.plot(range(NbMics),np.array(toas)*C, '.r')
    plt.show()
    # cid = fig.canvas.mpl_connect('button_press_event', on_press)
toa = np.array(toa).reshape((-1, 257))
plt.figure()
plt.pcolormesh(toa)
plt.yticks(np.arange(0,NbMics, 8))
plt.grid()
       
np.save('toaUp_violon.npy', toa)