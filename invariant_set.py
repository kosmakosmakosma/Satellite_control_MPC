import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import solve_discrete_are
import pickle
from control import dare


def calculate_ellipsoid(u_max, x_max, P, K, dim_x, dim_u):
    Hx = np.vstack([np.eye(dim_x), -np.eye(dim_x)])  # (12 x 6)
    hx = x_max * np.ones(2*dim_x)

    # Input constraints: |u_j| <= u_max for all j
    Hu = np.vstack([np.eye(dim_u), -np.eye(dim_u)])  # (6 x 3)
    hu = u_max * np.ones(2*dim_u)

    # Effective input constraint: u = -K x  =>   Hu_eff x <= hu
    Hu_eff = Hu @ (-K)   # (6 x 6)
    # Combined constraint set for x:
    A_combined = np.vstack([Hx, Hu_eff])
    b_combined = np.hstack([hx, hu])

    # -------------------------
    # 4) Compute maximum c such that the ellipsoid E(c) = { x : x^T P x <= c } is contained in X.
    # -------------------------

    P_inv = np.linalg.inv(P)
    c_candidates = []
    for i in range(A_combined.shape[0]):
        a = A_combined[i, :]  # 1 x 6
        b_i = b_combined[i]
        denom = a @ P_inv @ a.T
        if denom <= 1e-12:
            continue
        c_val = (b_i**2) / denom
        c_candidates.append(c_val)
        #print(f"Constraint {i}: a = {a}, b = {b_i:.3f}, candidate c = {c_val:.3f}")

    if len(c_candidates) == 0:
        c_max = 0.0
    else:
        c_max = min(c_candidates)
    print(f"\nComputed maximum c (terminal ellipsoid scaling): {c_max:.4f}")

    return c_max

def get_point_within_ellipsoid(P, c_max, dim_x, x_max):   # Generate a random point within the ellipsoid
    print("params: ")
    print("P dimensions: ", P.shape)
    print("c_max: ", c_max)
    print("dim_x: ", dim_x)
    while True:
        z = np.random.randn(dim_x)  # Random point in R^n
        scale_factor = 0.9*np.sqrt(c_max / (z.T @ P @ z))
        x = scale_factor * z
        if np.all(np.abs(x) <= x_max):
            return x  # Return the point if it's within the state constraints
def check_point_within_ellipsoid(P, c_max, x):
    return x.T @ P @ x <= c_max

def plot_ellipsoid2():  # Plot the ellipsoid for the first 3 states
    # -------------------------
    # 5) Sample points on the ellipsoid E(c_max) in R^6 and project onto states 1,3,4 (indices 0,2,3).
    # -------------------------
    N_ell = 20000  # number of points to sample on the ellipsoid boundary
    points_6D = []
    for _ in range(N_ell):
        z = np.random.randn(n)
        # Scale z so that x = z_scaled lies on the boundary: x^T P x = c_max.
        # That is, let scaling factor = sqrt(c_max/(z^T P z))
        scale_factor = np.sqrt(c_max / (z.T @ P @ z))
        x_boundary = scale_factor * z
        points_6D.append(x_boundary)
    points_6D = np.array(points_6D)
    # Projection indices: 1st, 3rd, 4th state (i.e., indices 0, 2, 3)
    proj_indices = [0, 2, 3]
    ellipsoid_proj = points_6D[:, proj_indices]

    # -------------------------
    # 6) Sample feasible points from the constraint set X.
    # -------------------------
    N_sample = 10000
    feasible_points = []
    for _ in range(N_sample):
        x_rand = np.random.uniform(-x_max, x_max, n)
        if np.all(A_combined @ x_rand <= b_combined + 1e-6):
            feasible_points.append(x_rand)
    feasible_points = np.array(feasible_points)
    print(f"Number of feasible samples: {feasible_points.shape[0]}")
    feasible_proj = feasible_points[:, proj_indices]

    # -------------------------
    # 7) Plot the projection (for states 1, 3, 4) of the ellipsoid and the feasible set.
    # -------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feasible_proj[:, 0], feasible_proj[:, 1], feasible_proj[:, 2],
            color='blue', s=1, alpha=0.3, label='Feasible region (projection)')
    ax.scatter(ellipsoid_proj[:, 0], ellipsoid_proj[:, 1], ellipsoid_proj[:, 2],
            color='red', s=5, alpha=0.6, label='Ellipsoid boundary')

    # Also, plot the state constraints for these coordinates.
    # Since state constraints are |x_i| <= x_max for each state,
    # the projection is simply the box: for x1, x3, x4: [-x_max, x_max].
    r = [-x_max, x_max]
    # Create grid lines for the box edges (approximate)
    X_box, Y_box = np.meshgrid(r, r)
    # For constant z = x3 for one face
    ax.plot_wireframe(X_box, Y_box, np.full_like(X_box, -x_max), color='k', linewidth=0.5, alpha=0.5)
    ax.plot_wireframe(X_box, Y_box, np.full_like(X_box, x_max), color='k', linewidth=0.5, alpha=0.5)
    # Similarly for other two faces (you can add more if desired)

    ax.set_xlabel('State 1')
    ax.set_ylabel('State 3')
    ax.set_zlabel('State 4')
    ax.set_title(f'Projection of Terminal Ellipsoid (E(c_max)) and Feasible Region\n c_max = {c_max:.4f}')
    ax.legend()
    plt.show()

if __name__ == "__main__":

    dim_x = 6 # states
    dim_u = 3 # inputs
    dim_y = 3 # outputs

    with open('state_space_matrices.pkl', 'rb') as f:
        matrices = pickle.load(f)

    A = matrices['Ad']
    B = matrices['Bd']
    C = matrices['Cd']
    D = matrices['Dd']

    Q = np.eye(dim_x)
    R = np.eye(dim_u)

    P, _, K = dare(A, B, Q, R)
    K = -K


    #Define constraints.
   
    x_max = 1000000000
    u_max = 100

    c_max = calculate_ellipsoid(u_max, x_max, P, K, dim_x, dim_u)
    print("c_max: ", c_max)
    x_0 = get_point_within_ellipsoid(P, c_max, dim_x, x_max)
    print("x_0: ", x_0)
    print("Point within allipsoid? ", check_point_within_ellipsoid(P, c_max, x_0))
    print("Random point within the ellipsoid:", x_0.T @ P @ x_0 < c_max)