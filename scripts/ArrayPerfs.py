import numpy as np
from scipy.special import sph_harm, hankel2

def cart2sph(x, y, z):
    """Converts Cartesian coordinates to Spherical (r, theta, phi)."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # Elevation from z-axis [0, pi]
    phi = np.arctan2(y, x)    # Azimuth [0, 2pi]
    return r, theta, phi

def spherical_hankel2(n, z):
    """Spherical Hankel function of the 2nd kind for outward waves."""
    return np.sqrt(np.pi / (2 * z)) * hankel2(n + 0.5, z)

def process_array_quality(xyz, freq, c=343.0):
    """
    xyz: (Q, 3) array of microphone coordinates in meters.
    freq: Frequency in Hz.
    """
    Q = xyz.shape[0]
    k = 2 * np.pi * freq / c
    N = int(np.floor(np.sqrt(Q) - 1))  # Max order N based on Q >= (N+1)^2
    
    # 1. Coordinate Conversion
    r, theta, phi = np.zeros(Q), np.zeros(Q), np.zeros(Q)
    for q in range(Q):
        r[q], theta[q], phi[q] = cart2sph(xyz[q,0], xyz[q,1], xyz[q,2])
        
    # 2. Build Matrix B (Q x (N+1)^2)
    num_coeffs = (N + 1)**2
    B = np.zeros((Q, num_coeffs), dtype=complex)
    
    idx = 0
    for n in range(N + 1):
        for m in range(-n, n + 1):
            # Y_nm follows scipy convention: (m, n, azimuth, polar)
            Y_nm = sph_harm(m, n, phi, theta)
            for q in range(Q):
                # Radial function for internal source: bn = 4*pi*(-i)^n * hn2(kr)
                bn_q = 4 * np.pi * (1j)**(-n) * spherical_hankel2(n, k * r[q])
                B[q, idx] = bn_q * Y_nm[q]
            idx += 1

    # --- Metrics Evaluation ---
    
    # A. Condition Number (κ)
    # Measures sensitivity to positioning errors and noise amplification [1, 8].
    cond_num = np.linalg.cond(B)
    
    # B. White Noise Gain (WNG)
    # Measures robustness against sensor noise. 
    # For a Regular (PWD) beamformer, WNG relates to the sum of weights [2, 9].
    # Using the matrix form S*S^H for nearly-uniform sampling [10, 11].
    S = np.linalg.pinv(B) # Discrete Spherical Fourier Transform matrix
    # WNG = 1 / ||w||^2 for unit signal gain. 
    # Here we estimate the mean WNG across the decomposition order.
    wng_linear = 1.0 / np.trace(S @ S.conj().T)
    wng_db = 10 * np.log10(np.real(wng_linear) * Q) # Normalized to Q [9, 10]

    # C. Directivity Index (DI)
    # Spatial resolution gain. Theoretical max DI = 10*log10((N+1)^2) [3].
    di_db = 10 * np.log10((N + 1)**2)
    
    return f"Max Order (N): {N}\n" \
           f"Condition Number: {cond_num:.2e}\n" \
           f"WNG (dB): {wng_db:.2f}\n" \
           f"Directivity Index (dB): {di_db:.2f}\n" \
           f"Robustness: {'High' if cond_num < 20 else 'Low'}"
"""""

    Condition Number (κ):   This index quantifies how errors in microphone signals or positions are amplified during decomposition. 
                            A low condition number (close to 1) indicates high numerical stability. 
                            If κ>100, the high-order coefficients will likely be dominated by noise.
    White Noise Gain (WNG): This reflects the array's resistance to spatially white sensor noise. 
                            At low frequencies (small kr), the magnitude of bn​(kr) for high orders n is very small, 
                            forcing the weights 1/bn​ to become very large. 
                            This results in a poor WNG (often <0 dB), indicating that the array is amplifying sensor noise 
                            rather than signal.
    Directivity Index (DI): This represents the array's ability to suppress diffuse ambient noise. 
                            It is primarily a function of the decomposition order N, where DI=10log10​(N+1)2. 
                            While a high order improves resolution, the script shows the trade-off: 
                            higher orders at low frequencies drastically increase the condition number and decrease WNG.
    Frequency Sensitivity:  For non-spherical arrays, the varying radii rq​ help avoid the "Bessel nulls" 
                            that occur in single-radius open arrays. 
                            Distributing microphones across a spherical shell volume stabilizes the condition number 
                            across a wider frequency range compared to a single sphere.
"""
