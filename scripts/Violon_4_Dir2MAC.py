import sys, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from swd import spherical_processing as sp

# Add directories to sys.path
package_dir = os.path.abspath("C:/Users/froll/Documents/Labo/Projets/Violon/ManipViolon_Anech_08062023")
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)
package_dir = os.path.abspath("C:/Users/froll/Documents/Labo/Projets/Outils/swd")
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# Constants
Tc = 21.5
C = np.sqrt(1.4 * 287 * (Tc + 273))
NbViol = 6
OSH = 7

NbSH = (OSH + 1) ** 2
frq = np.load('ViolinsPm.npz')['frq']
NbTh = 180 // 5  # Nombre de theta
NbPh = 360 // 5  # Nombre de phi
frqMax = 10000
NumViolon = [1, 4, 5, 9, 11, 13]
Band = (frq > 0) & (frq < frqMax)
f = frq[Band]
Nbf = len(f)
kvect = 2 * np.pi * f.T / C
NbDirs = NbTh * NbPh
dPh = 2 * np.pi / NbPh
dTh = np.pi / NbTh

Cmn = np.load('Cmn.npz')['Cmn']
Diag = np.load('Diag.npz')['Diag']
angles_look = np.load('Diag.npz')['angles_look']
f1 = 100
f2 = 510
df = f[2] - f[1]
ixf1 = np.argmin(np.abs(f - f1))
ixf2 = np.argmin(np.abs(f - f2))

fvect = np.arange(f1, f2, df)
kvect = 2 * np.pi * fvect.T / C
Nbf = len(fvect)
Cmn[Cmn == 0] = 1e-10

SelectedViol = np.array([0, 1, 2, 3, 4, 5])
NbViol = len(SelectedViol)
ListViol = "".join([f"-{NumViolon[n]}" for n in range(NbViol)])

FileName = f'MaxXC{ListViol:s}_{f1}_{f2}_lin{Nbf}.npz'
if os.path.exists(FileName):
    MaxXC = np.load(FileName)['MaxXC']
    PHC = np.load(FileName)['PHC']
    THC = np.load(FileName)['THC']
    mnToFill = np.argwhere(MaxXC == 0)
    np.random.shuffle(mnToFill)
    K = np.arange(len(mnToFill))
    M = mnToFill[:, 0]
    N = mnToFill[:, 1]
    W = M // Nbf
    Ixg = M % Nbf + ixf1
    V = N // Nbf
    Ixf = N % Nbf + ixf1
    I = V * Nbf + Ixf - ixf1
    J = W * Nbf + Ixg - ixf1
else:
    MaxXC = np.zeros((Nbf * NbViol, Nbf * NbViol))
    PHC = np.zeros((Nbf * NbViol, Nbf * NbViol))
    THC = np.zeros((Nbf * NbViol, Nbf * NbViol))
    K = np.arange((NbViol * Nbf) ** 2)
    np.random.shuffle(K)
    I = np.arange(NbViol * Nbf)
    J = np.arange(NbViol * Nbf)
    IJ = np.meshgrid(I, J)
    I = IJ[0].flatten()
    J = IJ[1].flatten()

# Function to process k
def process_k(k, I, J, Nbf, ixf1, Cmn, angles_look, kvect):
    i = I[k]
    j = J[k]

    v = i // Nbf
    ixf = i % Nbf + ixf1
    w = j // Nbf
    ixg = j % Nbf + ixf1
    C1 = Cmn[w, :, ixg]
    C2 = Cmn[v, :, ixf]
    XC = sp.spherical_correlation(C1, C2, angles_look, kvect=np.array([0]))
    mac = np.max(np.abs(XC))
    m = w * Nbf + ixg - ixf1
    n = v * Nbf + ixf - ixf1
    thc = angles_look[np.argmax(np.abs(XC)), 0]
    phc = angles_look[np.argmax(np.abs(XC)), 1]
    return m, n, mac, thc, phc

from concurrent.futures import ThreadPoolExecutor
# Parallelize the loop over K
batch_size = 1000  # Adjust this based on your system's capacity
with ThreadPoolExecutor(max_workers=4) as executor:
    for i in tqdm(range(0, len(K), batch_size)):
        batch = K[i:i + batch_size]
        results = list(executor.map(
            partial(process_k, I=I, J=J, Nbf=Nbf, ixf1=ixf1, Cmn=Cmn, angles_look=angles_look, kvect=kvect),
            batch
        ))
        # Update matrices with batch results
        for m, n, mac, thc, phc in results:
            MaxXC[m, n] = mac
            THC[m, n] = thc
            PHC[m, n] = phc

# Save the results
np.savez(FileName, MaxXC=MaxXC, THC=THC, PHC=PHC)
