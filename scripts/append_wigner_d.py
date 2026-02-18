
import os

file_path = "c:/Users/froll/Documents/Labo/Projets/Violon/scripts/utils_DirViolins.py"

new_function = """

def wigner_D_matrix(j, alpha, beta, gamma):
    \"\"\"
    Compute the Wigner D-matrix D^j(alpha, beta, gamma).
    
    This function computes the (2j+1)x(2j+1) Wigner D-matrix for a given angular momentum j 
    and Euler angles (alpha, beta, gamma) in radians.
    
    Args:
        j (int): Angular momentum quantum number (must be integer >= 0).
        alpha (float): Euler angle alpha (rotation around Z) in radians.
        beta (float): Euler angle beta (rotation around Y) in radians.
        gamma (float): Euler angle gamma (rotation around Z) in radians.
        
    Returns:
        numpy.ndarray: The (2j+1)x(2j+1) complex Wigner D-matrix.
    \"\"\"
    size = int(2 * j + 1)
    
    # Pre-compute trigonometric terms
    sb2 = np.sin(beta / 2.0)
    cb2 = np.cos(beta / 2.0)
    x = np.cos(beta)
    
    # Precompute factorials up to 2j
    # Note: For large j, use gammaln to avoid overflow, but for typical SH orders (j<50), factorial is fine.
    # fact[n] needs to accommodate indices up to 2j
    fact = factorial(np.arange(0, int(2 * j) + 5))

    d_mat = np.zeros((size, size))
    m = np.arange(-j, j + 1)
    
    for i, mp in enumerate(m):
        for k, mv in enumerate(m):
            # Symmetry handling to map to valid Jacobi polynomial indices (n>=0, a>-1, b>-1)
            # m' -> mp, m -> mv.
            
            target_mp, target_mv = mp, mv
            factor = 1.0
            
            # 1. Use d_{m',m}^j = (-1)^(m'-m) d_{m,m'}^j to ensure m' >= m
            if target_mp < target_mv:
                factor *= (-1)**(target_mp - target_mv)
                target_mp, target_mv = target_mv, target_mp
                
            # 2. Use d_{m',m}^j = (-1)^(m'-m) d_{-m',-m}^j to ensure m' + m >= 0
            if target_mp + target_mv < 0:
                 factor *= (-1)**(target_mp - target_mv)
                 target_mp, target_mv = -target_mv, -target_mp 
            
            # Now target_mp >= target_mv and target_mp + target_mv >= 0.
            # a = m' - m >= 0
            # b = m' + m >= 0
            n = int(j - target_mp)
            a = int(target_mp - target_mv)
            b = int(target_mp + target_mv)
            
            # Normalization factor
            # For this standard Jacobi definition:
            # d_mm' = sqrt( (j+m')! (j-m')! / ( (j+m)! (j-m)! ) )  * ...
            # Wait, using factorials computed earlier:
            # fact[n] corresponds to n!
            val_num = fact[int(j + target_mp)] * fact[int(j - target_mp)]
            val_den = fact[int(j + target_mv)] * fact[int(j - target_mv)]
            norm = np.sqrt(val_num / val_den)
            
            # Compute element
            val = norm * (sb2**a) * (cb2**b) * eval_jacobi(n, a, b, x)
            d_mat[i, k] = factor * val

    # Apply phases D_{m',m} = e^{-i m' alpha} * d_{m',m}(beta) * e^{-i m gamma}
    mp_grid, mv_grid = np.meshgrid(m, m, indexing='ij')
    phase = np.exp(-1j * (mp_grid * alpha + mv_grid * gamma))
    
    return phase * d_mat
"""

with open(file_path, "a", encoding="utf-8") as f:
    f.write(new_function)
print("Function appended successfully.")
