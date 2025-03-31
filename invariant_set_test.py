import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from control import dare
# -------------------------
# 1) Define the Double Integrator system
# -------------------------
A = np.array([[1, 1],
              [0, 1]])
B = np.array([[0],
              [1]])

# Stage cost Q = I, R = 3
Q = np.eye(2)
R = np.array([[3]])

# Solve DARE => P, K (sign convention: K = - (R + B^T P B)^-1 (B^T P A))
P, _, K = dare(A, B, Q, R)
K = -K

# We'll use P = 0.5 * P_dare in the terminal cost, or just P_dare.
# The problem statement says "Vf(x) = 1/2 x^T P x", but let's just use P_dare for the ellipsoid.
# (This factor of 1/2 won't change the shape of the ellipsoid, just scales alpha by 2.)

# -------------------------
# 2) Define constraints
# -------------------------
# State constraints: |x1| <= 3/4, |x2| <= 3/4
x_max = 0.75

# Input constraint: |u| <= 1/2, but u = -K x => | -K x | <= 1/2
u_max = 0.5

# We'll write them in polyhedral form:
#   Hx x <= hx,   Hu_eff x <= hu
# For the state:  -x_max <= x1 <= x_max,  -x_max <= x2 <= x_max
# That is:
#   x1 <= 0.75
#   -x1 <= 0.75
#   x2 <= 0.75
#   -x2 <= 0.75

Hx = np.array([
    [1, 0],   # x1 <= 0.75
    [-1, 0],  # -x1 <= 0.75 => x1 >= -0.75
    [0, 1],   # x2 <= 0.75
    [0, -1],  # -x2 <= 0.75 => x2 >= -0.75
])
hx = np.array([x_max, x_max, x_max, x_max])

# For the input constraint: |u| <= 0.5 => -0.5 <= u <= 0.5
# But u = -Kx.  Let Kx = k1*x1 + k2*x2. We want -0.5 <= -(k1*x1 + k2*x2) <= 0.5
# i.e.  (k1*x1 + k2*x2) <= 0.5   and  -(k1*x1 + k2*x2) <= 0.5
# => [k1, k2] x <= 0.5,   and  [-k1, -k2] x <= 0.5

k_row = -K[0,:]  # Because K is 1x2, -K => k_row is a 1D array
Hu_eff = np.array([
    k_row,         #  k_row * x <= 0.5
    -k_row         # -k_row * x <= 0.5
])
hu = np.array([u_max, u_max])

# Combine them conceptually, but for the "largest ellipsoid" approach,
# we can just handle them as separate constraints. We'll compute alpha from each row.

# -------------------------
# 3) Compute alpha via row-by-row approach
# -------------------------
# The formula: For each row a^T x <= b, we need
#   sqrt(alpha * a^T P^-1 a) <= b
# => alpha <= b^2 / (a^T P^-1 a)
# We'll do this for all rows in Hx, Hu_eff.

A_combined = np.vstack([Hx, Hu_eff])
b_combined = np.hstack([hx, hu])

P_inv = np.linalg.inv(P)
alpha_candidates = []

for i in range(A_combined.shape[0]):
    a = A_combined[i,:]
    b_ = b_combined[i]
    denom = a @ P_inv @ a.T
    if denom <= 1e-12:
        # Something degenerate
        continue
    alpha_i = (b_**2) / denom
    alpha_candidates.append(alpha_i)

alpha_max = min(alpha_candidates) if len(alpha_candidates)>0 else 0.0

print(f"Computed alpha_max = {alpha_max:.4f}")
# => The terminal ellipsoid is x^T P x <= alpha_max.

# -------------------------
# 4) Plotting
# -------------------------
# We'll do:
#  - The feasible region by sampling (x1,x2) in [-x_max, x_max]^2
#    and also checking the input constraint.
#  - The ellipse x^T P x = alpha_max
#  - The bounding box lines for the state constraints
#  - The lines for the input constraint

def feasible_region_points(n_points=300):
    """Return array of feasible (x1,x2) by naive sampling in [-x_max, x_max]^2."""
    xs = []
    x1_vals = np.linspace(-x_max, x_max, n_points)
    x2_vals = np.linspace(-x_max, x_max, n_points)
    for x1 in x1_vals:
        for x2 in x2_vals:
            # state constraint is obviously satisfied, but let's check input constraint
            # u = -K x => must satisfy |u| <= 0.5
            u = -K @ np.array([x1, x2])
            if np.abs(u[0]) <= u_max:
                xs.append([x1, x2])
    return np.array(xs)

feasible_points = feasible_region_points(n_points=200)

# We'll define a function to parametric-plot the ellipse x^T P x = alpha_max
def ellipse_points(alpha, n_pts=200):
    """Parametrically generate points for x^T P x = alpha in R^2."""
    # Solve P = Q^T Q to get a sqrt factor, or do an eigen-decomp
    # We'll do an eigen-decomp for clarity:
    eigvals, eigvecs = np.linalg.eigh(P)
    # The ellipse x^T P x = alpha => z^T diag(eigvals) z = alpha, where x = eigvecs z
    # => z^T diag(eigvals) z = alpha => in local coords: (z1^2 * eigvals1 + z2^2 * eigvals2) = alpha
    # param eq for circle => z = sqrt(alpha)* diag(1/sqrt(eigvals)) [cos t; sin t]
    # => x = eigvecs z
    angles = np.linspace(0, 2*np.pi, n_pts)
    pts = []
    for theta in angles:
        # local coords
        r = np.array([np.cos(theta)/np.sqrt(eigvals[0]),
                      np.sin(theta)/np.sqrt(eigvals[1])])
        r_scaled = np.sqrt(alpha)*r
        x_local = eigvecs @ r_scaled
        pts.append(x_local)
    return np.array(pts)

ell_pts = ellipse_points(alpha_max, n_pts=300)

# Plot
plt.figure(figsize=(8,8))
plt.scatter(feasible_points[:,0], feasible_points[:,1], s=1, c='blue', alpha=0.2, label='Feasible region')

# Plot bounding box lines
plt.plot([ x_max,  x_max], [-x_max, x_max], 'k--', lw=1)
plt.plot([-x_max, -x_max], [-x_max, x_max], 'k--', lw=1)
plt.plot([-x_max,  x_max], [ x_max,  x_max], 'k--', lw=1)
plt.plot([-x_max,  x_max], [-x_max,-x_max], 'k--', lw=1)

# Plot ellipse boundary
plt.plot(ell_pts[:,0], ell_pts[:,1], 'r', lw=2, label=r'Ellipse $x^T P x = \alpha_{\max}$')

plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.axis('equal')
plt.xlim([-x_max-0.1, x_max+0.1])
plt.ylim([-x_max-0.1, x_max+0.1])
plt.title(f"Double Integrator: alpha_max={alpha_max:.4f},  feasible region & ellipse")
plt.legend()
plt.show()
