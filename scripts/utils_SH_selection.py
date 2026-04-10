import numpy as np
import sys
import os
from joblib import Parallel, delayed

# Ensure swd is in path
package_dir = os.path.abspath("C:/Users/froll/Documents/Labo/Projets/Outils/swd")
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

try:
    from swd import spherical_processing as sp
except ImportError:
    import swd.spherical_processing as sp

def compute_optimal_lambda_gcv(P_meas, H_max, freqs, lambda_grid=None):
    """
    Computes the optimal regularization parameter lambda using Generalized Cross Validation (GCV)
    for each frequency.

    Parameters:
    -----------
    P_meas : (n_mics, n_freqs)
        Measured pressure.
    H_max : (n_mics, n_sh_max, n_freqs)
        Transfer matrix for the maximum SH order considered.
    freqs : (n_freqs,)
        Frequency vector.
    lambda_grid : array-like, optional
        Grid of lambda values to search over. If None, uses logspace(-6, 0, 50).

    Returns:
    --------
    optimal_lambdas : (n_freqs,)
        Optimal lambda for each frequency.
    """
    if lambda_grid is None:
        lambda_grid = np.logspace(-8, 0, 100)
    
    n_freqs = len(freqs)
    optimal_lambdas = np.zeros(n_freqs)
    
    # Precompute SVD for each frequency to speed up GCV calculation
    # H_max is (n_mics, n_sh, n_freqs)
    
    for f_idx in range(n_freqs):
        H_f = H_max[:, :, f_idx] # (M, N)
        p_f = P_meas[:, f_idx]   # (M,)
        
        # SVD of H_f
        # H = U S V^H
        # We only need singular values S and U^H * p for the GCV formula
        try:
            U, s, _ = np.linalg.svd(H_f, full_matrices=False)
        except np.linalg.LinAlgError:
            optimal_lambdas[f_idx] = 1e-4 # Fallback
            continue

        # precompute U^H * p
        Up = U.conj().T @ p_f
        
        # Vectorized GCV calculation over lambda_grid
        # GCV(lambda) = ||(I - A_lambda) y||^2 / (Tr(I - A_lambda))^2
        # where A_lambda = H (H^H H + lambda I)^-1 H^H
        # In terms of SVD:
        # ||(I - A_lambda) y||^2 = sum_i ( lambda / (s_i^2 + lambda) * |u_i^H y| )^2 + ||y_perp||^2
        # Tr(I - A_lambda) = M - N + sum_i ( lambda / (s_i^2 + lambda) )
        # (Assuming M > N, effective degrees of freedom logic)
        
        M = H_f.shape[0]
        N = H_f.shape[1]
        
        # Calculate residual for components in range(U)
        # filter_factors = s^2 / (s^2 + lambda)
        # residual_factors = 1 - filter_factors = lambda / (s^2 + lambda)
        
        s2 = s**2
        s2_plus_lambda = s2[:, np.newaxis] + lambda_grid[np.newaxis, :] # (N, n_lambdas)
        residual_factors = lambda_grid[np.newaxis, :] / s2_plus_lambda # (N, n_lambdas)
        
        # Numerator: ||(I - A_lambda) y||^2
        # The projection onto range(U) is affected by residual_factors.
        # The projection onto kernel(U^H) (if M > N) is untouched (always 1).
        # norm_y_perp_sq = ||y||^2 - ||Up||^2
        norm_y_sq = np.sum(np.abs(p_f)**2)
        norm_Up_sq = np.sum(np.abs(Up)**2)
        norm_y_perp_sq = norm_y_sq - norm_Up_sq
        
        # proj_residual = sum( |residual_factors * Up|^2 )
        # term inside sum is (N, n_lambdas) * (N, 1) -> (N, n_lambdas)
        proj_residual_sq = np.sum( (residual_factors * np.abs(Up[:, np.newaxis]))**2, axis=0 ) # (n_lambdas,)
        
        numerator = proj_residual_sq + norm_y_perp_sq
        
        # Denominator: (Tr(I - A_lambda))^2
        # Tr(I - A_lambda) = Trace(I_M) - Trace(A_lambda)
        # Trace(A_lambda) = sum( s^2 / (s^2 + lambda) )
        # Tr(I - A_lambda) = M - sum( s^2 / (s^2 + lambda) )
        #                  = (M - N) + sum( 1 - s^2/(s^2+lambda) )
        #                  = (M - N) + sum( lambda / (s^2 + lambda) )
        
        trace_term = (M - N) + np.sum(residual_factors, axis=0) # (n_lambdas,)
        
        denominator = trace_term**2
        
        gcv_scores = numerator / denominator
        
        best_lambda_idx = np.argmin(gcv_scores)
        optimal_lambdas[f_idx] = lambda_grid[best_lambda_idx]
        
    return optimal_lambdas


def _process_cv_for_order(order, freqs, H_max, k_folds, folds, n_mics, P_meas, lambda_reg, return_per_freq):
    """
    Helper function to process a single SH order for cross-validation.
    """
    n_sh = (order + 1) ** 2
    n_freqs = len(freqs)
    fold_errors = []
    
    # Slice H for current order
    H_current = H_max[:, :n_sh, :]
    
    # Check if lambda_reg is array-like (per frequency)
    per_freq_lambda = isinstance(lambda_reg, (np.ndarray, list)) and len(lambda_reg) == n_freqs
    
    # Iterate over folds
    for k in range(k_folds):
        test_idx = folds[k]
        train_mask = np.ones(n_mics, dtype=bool)
        train_mask[test_idx] = False
        
        # Split Data
        P_train = P_meas[train_mask, :]
        P_test = P_meas[test_idx, :]
        
        H_train = H_current[train_mask, :, :]
        H_test = H_current[test_idx, :, :]
        
        # Compute Coefficients
        try:
            if per_freq_lambda:
                # Custom loop for per-frequency lambda
                C_train = np.zeros((n_sh, n_freqs), dtype=complex)
                for f in range(n_freqs):
                    y = P_train[:, f]
                    H_f = H_train[:, :, f]
                    lam = lambda_reg[f]
                    
                    # Tikhonov: x = (H^H H + lam I)^-1 H^H y
                    Gram = H_f.conj().T @ H_f
                    reg_matrix = Gram + lam * np.eye(n_sh)
                    # Use solve which is faster/more stable than inv
                    # H^H y
                    HTy = H_f.conj().T @ y
                    C_train[:, f] = np.linalg.solve(reg_matrix, HTy)
            else:
                # Use standard function with scalar lambda
                N_SH_vect = np.full(len(freqs), order, dtype=int)
                C_train = sp.compute_SHcoefs(
                    P_train, H_train, N_SH_vect, lambda_reg=lambda_reg
                )
        except Exception as e:
            if return_per_freq:
                fold_errors.append(np.full(len(freqs), np.inf))
            else:
                fold_errors.append(np.inf)
            continue
        
        # Predict P_pred: (n_test, n_freq)
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

def select_optimal_sh_order_cv(P_meas, XYZ_mics, freqs, max_order=15, k_folds=5, c=343.0, rmin=0.1, lambda_reg=1e-4, return_per_freq=False, H_max=None):
    """
    Selects the optimal SH truncation order using K-Fold Cross Validation.
    If lambda_reg is 'gcv' or 'auto', it first optimizes lambda per frequency using GCV on the max order model.
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
    
    # Precompute basis for max_order if not provided
    if H_max is None:
        try:
            H_max = sp.compute_SphericalWavesbasis_origin_to_field(
                XYZ_mics, kvect, max_order, SH_center=np.array([0,0,0])
            )
            if H_max.ndim == 2:
                H_max = H_max[:, :, np.newaxis]
        except Exception as e:
            return None, None, None

    # Optimization of Regularization Parameter (GCV) if requested
    if isinstance(lambda_reg, str) and lambda_reg.lower() in ['gcv', 'auto']:
        try:
             # Search range: 1e-8 to 1.0
             lambda_reg = compute_optimal_lambda_gcv(P_meas, H_max, freqs, lambda_grid=np.logspace(-8, 0, 100))
        except Exception as e:
            lambda_reg = 1e-4
    
    orders = np.arange(1, max_order + 1)
    
    indices = np.arange(n_mics)
    np.random.seed(42) # For reproducibility
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)
    
    # Parallelize the loop over orders
    cv_errors = Parallel(n_jobs=-1)(
        delayed(_process_cv_for_order)(
            order, freqs, H_max, k_folds, folds, n_mics, P_meas, lambda_reg, return_per_freq
        ) for order in orders
    )

    cv_errors = np.array(cv_errors) # (n_orders, n_freqs) if return_per_freq else (n_orders,)
    
    if return_per_freq:
        optimal_order_idx = np.argmin(cv_errors, axis=0)
        optimal_order = orders[optimal_order_idx]
        return optimal_order, orders, cv_errors
    else:
        optimal_order_idx = np.argmin(cv_errors)
        optimal_order = orders[optimal_order_idx]
        return optimal_order, orders, cv_errors
