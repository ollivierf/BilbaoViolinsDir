"""
Excess Phase Decomposition for 3D Acoustic Directivity Patterns
================================================================
Given complex impulse responses H(f, Omega) measured on a sphere
(e.g. 256-mic array), this module decomposes the excess phase into:

  1. Common (direction-independent) component  -> monopole resonances
  2. Acoustic center shift                     -> dipole phase, l=1
  3. Spatially structured residual             -> SH decomposition l>=2
  4. Noise floor estimation

Usage
-----
See bottom of file for a worked example with synthetic data.
"""

import numpy as np
from scipy.signal import hilbert
from scipy.linalg import lstsq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Compatibility for scipy >= 1.15 which removed sph_harm
try:
    from scipy.special import sph_harm as _sph_harm_orig
    def _sph(m, l, phi_az, theta):
        return _sph_harm_orig(m, l, phi_az, theta)
except ImportError:
    from scipy.special import sph_harm_y
    def _sph(m, l, phi_az, theta):
        return sph_harm_y(l, m, theta, phi_az)


# ---------------------------------------------------------------------------
# 1.  CORE PHASE UTILITIES
# ---------------------------------------------------------------------------

def compute_minimum_phase(H):
    """
    Compute the minimum-phase equivalent of each transfer function.

    Parameters
    ----------
    H : ndarray, shape (n_freq, n_mics)
        Complex transfer functions (single-sided FFT, DC to Nyquist).

    Returns
    -------
    phi_min : ndarray, shape (n_freq, n_mics)
        Minimum-phase response in radians.
    """
    log_mag = np.log(np.abs(H) + 1e-12)
    # Hilbert transform along frequency axis:
    #   phi_min(f) = -imag{ hilbert( log|H(f)| ) }
    phi_min = -np.imag(hilbert(log_mag, axis=0))
    return phi_min


def compute_excess_phase(H):
    """
    Compute actual, minimum-phase, and excess phase.

    Returns
    -------
    phi_actual, phi_min, phi_excess : ndarray, shape (n_freq, n_mics)
    """
    phi_actual = np.unwrap(np.angle(H), axis=0)
    phi_min    = compute_minimum_phase(H)

    # Align DC constant
    offset  = phi_actual[0, :] - phi_min[0, :]
    phi_min = phi_min + offset[np.newaxis, :]

    phi_excess = phi_actual - phi_min
    return phi_actual, phi_min, phi_excess


# ---------------------------------------------------------------------------
# 2.  SPHERICAL HARMONICS BASIS
# ---------------------------------------------------------------------------

def real_sh_basis(theta, phi_az, l_max):
    """
    Build real spherical harmonics basis matrix Y, shape (n_dirs, n_sh).

    Parameters
    ----------
    theta  : colatitude in [0, pi]   (n_dirs,)
    phi_az : azimuth   in [0, 2pi]   (n_dirs,)
    l_max  : maximum SH order

    Returns
    -------
    Y      : ndarray (n_dirs, (l_max+1)**2)
    labels : list of (l, m) tuples
    """
    n_dirs = len(theta)
    n_sh   = (l_max + 1) ** 2
    Y      = np.zeros((n_dirs, n_sh))
    labels = []

    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m < 0:
                Ylm = np.sqrt(2) * (-1)**m * np.imag(_sph(abs(m), l, phi_az, theta))
            elif m == 0:
                Ylm = np.real(_sph(0, l, phi_az, theta))
            else:
                Ylm = np.sqrt(2) * (-1)**m * np.real(_sph(m, l, phi_az, theta))
            Y[:, idx] = Ylm
            labels.append((l, m))
            idx += 1

    return Y, labels


# ---------------------------------------------------------------------------
# 3.  DECOMPOSITION STEPS
# ---------------------------------------------------------------------------

def remove_common_phase(phi_excess):
    """
    Step 1: Direction-averaged (common) excess phase.

    Returns
    -------
    phi_common   : ndarray (n_freq,)        – mean over directions
    phi_residual : ndarray (n_freq, n_mics) – phi_excess - phi_common
    """
    phi_common   = np.mean(phi_excess, axis=1)
    phi_residual = phi_excess - phi_common[:, np.newaxis]
    return phi_common, phi_residual


def fit_acoustic_center(phi_residual, freqs, theta, phi_az, c=343.0):
    """
    Step 2: Fit direction-dependent linear-in-frequency phase (acoustic center).

    Model: phi_residual(f, Omega) ≈ -2*pi*f * (d(f)·Omega_hat) / c

    Returns
    -------
    tau_excess    : ndarray (n_freq, n_dirs) – group-delay field (seconds)
    d_vec         : ndarray (n_freq, 3)      – acoustic center (meters)
    phi_dipole    : ndarray (n_freq, n_dirs) – fitted dipole phase
    phi_residual2 : ndarray (n_freq, n_dirs) – residual after dipole removal
    """
    n_freq, n_dirs = phi_residual.shape

    df         = freqs[1] - freqs[0]
    dphi       = np.gradient(phi_residual, df, axis=0)
    tau_excess = -dphi / (2 * np.pi)    # seconds

    # Direction cosines
    sin_t = np.sin(theta)
    Omega = np.column_stack([sin_t * np.cos(phi_az),
                             sin_t * np.sin(phi_az),
                             np.cos(theta)])           # (n_dirs, 3)

    d_vec      = np.zeros((n_freq, 3))
    phi_dipole = np.zeros_like(phi_residual)

    for fi, f in enumerate(freqs):
        if f < 1e-6:
            continue
        d, _, _, _ = lstsq(Omega, tau_excess[fi])
        d_vec[fi]  = d * c
        phi_dipole[fi] = -2 * np.pi * f * (Omega @ d) / c

    phi_residual2 = phi_residual - phi_dipole
    return tau_excess, d_vec, phi_dipole, phi_residual2


def sh_decomposition(phi_residual2, theta, phi_az, l_max=6):
    """
    Step 3: Decompose remaining excess phase into spherical harmonics.

    Returns
    -------
    coeffs         : ndarray (n_freq, n_sh)
    Y              : ndarray (n_dirs, n_sh)
    labels         : list of (l, m)
    power_by_order : ndarray (n_freq, l_max+1) – energy per SH order
    """
    Y, labels = real_sh_basis(theta, phi_az, l_max)
    n_freq = phi_residual2.shape[0]

    coeffs, _, _, _ = lstsq(Y, phi_residual2.T)
    coeffs = coeffs.T   # (n_freq, n_sh)

    power_by_order = np.zeros((n_freq, l_max + 1))
    for i, (l, m) in enumerate(labels):
        power_by_order[:, l] += coeffs[:, i] ** 2

    return coeffs, Y, labels, power_by_order


def excess_phase_norm(phi_excess):
    """RMS norm over frequency for each direction."""
    return np.sqrt(np.mean(phi_excess ** 2, axis=0))


# ---------------------------------------------------------------------------
# 4.  FULL PIPELINE
# ---------------------------------------------------------------------------

def decompose_directivity(H, freqs, theta, phi_az, c=343.0, l_max=6):
    """
    Full excess phase decomposition pipeline.

    Parameters
    ----------
    H      : complex ndarray (n_freq, n_dirs)
    freqs  : ndarray (n_freq,)  in Hz
    theta  : colatitude (n_dirs,) in radians
    phi_az : azimuth    (n_dirs,) in radians
    c      : speed of sound (m/s)
    l_max  : max SH order for residual decomposition

    Returns
    -------
    dict with all intermediate and final quantities
    """
    phi_actual, phi_min, phi_excess = compute_excess_phase(H)

    phi_common, phi_res1 = remove_common_phase(phi_excess)

    tau_excess, d_vec, phi_dipole, phi_res2 = fit_acoustic_center(
        phi_res1, freqs, theta, phi_az, c=c)

    coeffs, Y, labels, power_by_order = sh_decomposition(
        phi_res2, theta, phi_az, l_max=l_max)

    # Noise estimate: residual after removing all fitted SH components
    high_l_mask = np.array([l for l, m in labels]) >= l_max
    phi_noise   = phi_res2 - (Y[:, ~high_l_mask] @ coeffs[:, ~high_l_mask].T).T

    return dict(
        freqs=freqs, theta=theta, phi_az=phi_az,
        phi_actual=phi_actual, phi_min=phi_min, phi_excess=phi_excess,
        phi_common=phi_common,
        phi_res1=phi_res1,
        tau_excess=tau_excess,
        d_vec=d_vec,
        phi_dipole=phi_dipole,
        phi_res2=phi_res2,
        coeffs=coeffs, Y=Y, labels=labels,
        power_by_order=power_by_order,
        phi_noise=phi_noise,
    )


# ---------------------------------------------------------------------------
# 5.  PLOTTING
# ---------------------------------------------------------------------------

def plot_decomposition(res, violin_id="", f_range=None, outfile=None):
    """
    Summary figure with 5 panels:
      A) Excess phase norm histogram (total vs components)
      B) Common phase group delay (monopole resonances)
      C) Acoustic center displacement |d(f)|
      D) SH power by order vs frequency
      E) Residual noise map on sphere (RMS, Mollweide projection)
    """
    freqs = res['freqs']
    mask  = ((freqs >= f_range[0]) & (freqs <= f_range[1])
             if f_range else np.ones(len(freqs), bool))
    f = freqs[mask]

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"Excess Phase Decomposition — Violin {violin_id}", fontsize=14)
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # A: Histogram of norms per component
    ax = fig.add_subplot(gs[0, :])
    norm_total    = excess_phase_norm(res['phi_excess'][mask])
    norm_dipole   = excess_phase_norm(res['phi_dipole'][mask])
    norm_residual = excess_phase_norm(res['phi_res2'][mask])
    bins = np.linspace(0, norm_total.max() * 1.1, 40)
    ax.hist(norm_total,    bins=bins, alpha=0.5, label='Total excess phase',       color='steelblue')
    ax.hist(norm_dipole,   bins=bins, alpha=0.6, label='Dipole (acoustic center)', color='orange')
    ax.hist(norm_residual, bins=bins, alpha=0.6, label='Structured residual (l≥2)',color='green')
    ax.set_xlabel("Excess Phase Norm (rad)")
    ax.set_ylabel("Count")
    ax.set_title("A — Norm distributions per component")
    ax.legend(fontsize=9)

    # B: Common group delay
    ax = fig.add_subplot(gs[1, 0])
    df = f[1] - f[0]
    gd = -np.gradient(res['phi_common'][mask], df) / (2 * np.pi) * 1e3  # ms
    ax.plot(f, gd, 'k')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Group delay (ms)")
    ax.set_title("B — Common group delay (monopole resonances)")

    # C: Acoustic center displacement
    ax = fig.add_subplot(gs[1, 1])
    d_norm = np.linalg.norm(res['d_vec'][mask], axis=1) * 100  # cm
    ax.plot(f, d_norm, 'r')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|d(f)| (cm)")
    ax.set_title("C — Acoustic center displacement")

    # D: SH power by order
    ax = fig.add_subplot(gs[2, 0])
    po      = res['power_by_order'][mask]
    po_norm = po / (po.sum(axis=1, keepdims=True) + 1e-12)
    l_max_p = po.shape[1]
    cmap    = plt.cm.viridis(np.linspace(0, 1, l_max_p))
    for l in range(l_max_p):
        ax.plot(f, po_norm[:, l], color=cmap[l], label=f"l={l}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Relative SH energy")
    ax.set_title("D — SH power by order (residual)")
    ax.legend(fontsize=7, ncol=2)

    # E: Residual noise on sphere (Mollweide)
    ax  = fig.add_subplot(gs[2, 1], projection='mollweide')
    rms = excess_phase_norm(res['phi_noise'][mask])
    lon = res['phi_az'].copy()
    lon[lon > np.pi] -= 2 * np.pi
    lat = np.pi / 2 - res['theta']
    sc  = ax.scatter(lon, lat, c=rms, cmap='hot_r', s=10)
    plt.colorbar(sc, ax=ax, label='RMS (rad)', shrink=0.7)
    ax.set_title("E — Noise/residual RMS on sphere")

    if outfile:
        plt.savefig(outfile, bbox_inches='tight')
        print(f"Saved: {outfile}")
    else:
        plt.savefig("excess_phase_decomposition.pdf", bbox_inches='tight')
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 6.  EXAMPLE WITH SYNTHETIC DATA
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    # Geometry: Fibonacci spiral (same as your 256-mic array)
    N      = 256
    i      = np.arange(N)
    theta  = np.arccos(1 - 2 * (i + 0.5) / N)
    phi_az = (2 * np.pi * i / ((1 + np.sqrt(5)) / 2)) % (2 * np.pi)

    # Frequency axis
    fs    = 48000
    NFFT  = 4096
    freqs = np.fft.rfftfreq(NFFT, 1 / fs)

    # Synthetic directivity
    # (a) Magnitude with resonance bumps
    magnitude = np.ones((len(freqs), N))
    for f0, Q, A in [(500, 20, 0.4), (1000, 15, 0.6), (2500, 10, 0.3)]:
        bw   = f0 / Q
        peak = A / (1 + ((freqs[:, None] - f0) / bw) ** 2)
        magnitude += peak * (1 + 0.2 * np.cos(theta[None, :]))

    # (b) Acoustic center shift: d = [2, 1, 0.5] cm
    c     = 343.0
    sin_t = np.sin(theta)
    Omega = np.column_stack([sin_t * np.cos(phi_az),
                             sin_t * np.sin(phi_az),
                             np.cos(theta)])
    d_true    = np.array([0.02, 0.01, 0.005])
    phi_delay = -2 * np.pi * freqs[:, None] * (Omega @ d_true)[None, :] / c

    # (c) Quadrupole residual (l=2, m=0)
    Y20      = np.real(_sph(0, 2, phi_az, theta))
    phi_quad = 0.3 * Y20[None, :] * np.sin(2 * np.pi * freqs[:, None] / 2000)

    # (d) Noise
    phi_noise_synth = 0.05 * rng.standard_normal((len(freqs), N))

    H = magnitude * np.exp(1j * (phi_delay + phi_quad + phi_noise_synth))

    # Run decomposition
    f_range = (200, 8000)
    res = decompose_directivity(H, freqs, theta, phi_az, c=c, l_max=6)

    # Report
    mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    print("=== Excess Phase Decomposition ===")
    print(f"Mean excess phase norm (total):    {excess_phase_norm(res['phi_excess'][mask]).mean():.3f} rad")
    print(f"Mean norm (dipole component):      {excess_phase_norm(res['phi_dipole'][mask]).mean():.3f} rad")
    print(f"Mean norm (structured residual):   {excess_phase_norm(res['phi_res2'][mask]).mean():.3f} rad")
    d_cm = np.linalg.norm(res['d_vec'][mask], axis=1).mean() * 100
    print(f"Mean acoustic center displacement: {d_cm:.1f} cm")
    print(f"True acoustic center:              {np.linalg.norm(d_true)*100:.1f} cm")

    plot_decomposition(res, violin_id="synthetic", f_range=f_range,
                       outfile="/mnt/user-data/outputs/excess_phase_decomposition.pdf")

