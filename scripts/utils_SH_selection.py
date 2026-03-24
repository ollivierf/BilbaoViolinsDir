import numpy as np
import sys
import os
from tqdm import tqdm
from joblib import Parallel, delayed

# Ensure swd is in path
package_dir = os.path.abspath("C:/Users/froll/Documents/Labo/Projets/Outils/swd")
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

try:
    from swd import spherical_processing as sp
except ImportError:
    import swd.spherical_processing as sp

def _process_cv_for_order(order, freqs, H_max, k_folds, folds, n_mics, P_meas, lambda_reg, return_per_freq):
    """
    Helper function to process a single SH order for cross-validation.
    """
    n_sh = (order + 1) ** 2
    fold_errors = []
    
    # Use a constant SH order for all frequencies in this loop iteration (since we are explicitly testing `order`).
    # This overrides optimal truncation based on radius/frequency because the goal is to cross-validate orders.
    N_SH_vect = np.full(len(freqs), order, dtype=int)
    
    # Slice H for current order
    # Ensure we take exactly (order+1)^2 columns which corresponds to the current tested order
    H_current = H_max[:, :n_sh, :]
    
    # Iterate over folds
    for k in range(k_folds):
        test_idx = folds[k]
        # Use boolean mask for training indices
        train_mask = np.ones(n_mics, dtype=bool)
        train_mask[test_idx] = False
        
        # Split Data
        P_train = P_meas[train_mask, :]
        P_test = P_meas[test_idx, :]
        
        H_train = H_current[train_mask, :, :]
        H_test = H_current[test_idx, :, :]
        
        # Compute Coefficients
        try:
            C_train = sp.compute_SHcoefs(
                P_train, H_train, N_SH_vect, lambda_reg=lambda_reg
            )
        except Exception as e:
            print(f"Error computing coefs for order {order}, fold {k}: {e}")
            if return_per_freq:
                fold_errors.append(np.full(len(freqs), np.inf))
            else:
                fold_errors.append(np.inf)
            continue
        
        # Predict P_pred: (n_test, n_freq)
        # H_test: (n_test, n_sh, n_freq)
        # C_train: (n_sh, n_freq)
        P_pred = np.einsum('msf,sf->mf', H_test, C_train)
        
        if return_per_freq:
            # Calculate error per frequency
            diff_sq = np.sum(np.abs(P_pred - P_test)**2, axis=0)
            sig_sq = np.sum(np.abs(P_test)**2, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_error = diff_sq / sig_sq
                rel_error[sig_sq <= 1e-15] = 0.0
            fold_errors.append(rel_error)
        else:
            # Global error
            diff_sq = np.sum(np.abs(P_pred - P_test)**2)
            sig_sq = np.sum(np.abs(P_test)**2)
            
            if sig_sq > 0:
                rel_error = diff_sq / sig_sq
            else:
                rel_error = 0.0
            fold_errors.append(rel_error)
        
    avg_error = np.mean(fold_errors, axis=0) # Axis 0 is over folds
    return avg_error

def select_optimal_sh_order_cv(P_meas, XYZ_mics, freqs, max_order=15, k_folds=5, c=343.0, rmin=0.1, lambda_reg=1e-4, return_per_freq=False):
    """
    Selects the optimal SH truncation order using K-Fold Cross Validation.
    
    Parameters:
    -----------
    P_meas : ndarray (n_mics, n_freqs)
        Measured pressure at microphones.
    XYZ_mics : ndarray (n_mics, 3)
        Cartesian coordinates of microphones.
    freqs : ndarray (n_freqs,)
        Frequency vector.
    max_order : int
        Maximum SH order to test.
    k_folds : int
        Number of folds for cross-validation.
    c : float
        Speed of sound in m/s.
    rmin : float
        Minimum radius (used for regularization/scaling in SH basis).
    lambda_reg : float
        Regularization parameter for Tikhonov regularization.
    return_per_freq : bool
        If True, returns errors and optimal orders per frequency.
        
    Returns:
    --------
    optimal_order : int or ndarray
        The truncation order with the lowest average CV error.
        If return_per_freq is True, returns an array of optimal orders for each frequency.
    orders : ndarray
        Orders tested.
    cv_errors : ndarray
        Average normalized reconstruction error for each order.
        If return_per_freq is True, shape is (n_orders, n_freqs).
    """
    
    # Handle single frequency case (1D input)
    if P_meas.ndim == 1:
        P_meas = P_meas[:, np.newaxis]
        
    n_mics, n_freqs = P_meas.shape
    
    # Handle scalar frequency
    if np.isscalar(freqs) or (isinstance(freqs, np.ndarray) and freqs.ndim == 0):
        freqs = np.atleast_1d(freqs)

    if len(freqs) != n_freqs:
        raise ValueError(f"Mismatch: P_meas has {n_freqs} frequencies (columns), but freqs has {len(freqs)} elements.")

    kvect = 2 * np.pi * freqs / c
    
    # Precompute basis for max_order
    print(f"Precomputing SH basis for max order {max_order}...")
    try:
        H_max = sp.compute_SphericalWavesbasis_origin_to_field(
            XYZ_mics, kvect, max_order, SH_center=np.array([0,0,0])
        )
        if H_max.ndim == 2:
            H_max = H_max[:, :, np.newaxis]
    except Exception as e:
        print(f"Error computing SH basis: {e}")
        return None, None, None
    
    orders = np.arange(1, max_order + 1)
    
    indices = np.arange(n_mics)
    np.random.seed(42) # For reproducibility
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)
    
    print(f"Starting Cross-Validation (Folds: {k_folds}) with parallel processing...")

    # Parallelize the loop over orders
    cv_errors = Parallel(n_jobs=-1)(
        delayed(_process_cv_for_order)(
            order, freqs, H_max, k_folds, folds, n_mics, P_meas, lambda_reg, return_per_freq
        ) for order in tqdm(orders, desc="Processing Orders")
    )

    cv_errors = np.array(cv_errors) # (n_orders, n_freqs) if return_per_freq else (n_orders,)
    
    if return_per_freq:
        optimal_order_idx = np.argmin(cv_errors, axis=0)
        optimal_order = orders[optimal_order_idx]
        return optimal_order, orders, cv_errors
    else:
        optimal_order_idx = np.argmin(cv_errors)
        optimal_order = orders[optimal_order_idx]
        print(f"Optimal order found: {optimal_order} with Error: {cv_errors[optimal_order_idx]:.3f}")
        return optimal_order, orders, cv_errors
