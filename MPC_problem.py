import cvxpy as cp
import numpy as np
import pickle
import matplotlib.pyplot as plt
from control import dare
from scipy.linalg import solve_discrete_are
from invariant_set import calculate_ellipsoid, get_point_within_ellipsoid, check_point_within_ellipsoid
from control import place
import sys
import os
from scipy.io import loadmat
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

dim_x = 6*2    # number of states
dim_u = 3    # number of inputs

# with open(script_dir+'\state_space_matrices.pkl', 'rb') as f:   # Load state matrices
#         matrices = pickle.load(f)

# Ad = matrices['Ad']
# Bd = matrices['Bd']
# Cd = matrices['Cd']
# Dd = matrices['Dd']

# desired_poles = np.array([0.1 + 0.1j, 0.1 - 0.1j, 0.2 + 0.2j, 0.2 - 0.2j, 0.3 + 0.3j, 0.3 - 0.3j])
# L = place(Ad.T, Cd.T, desired_poles).T

# Load the MATLAB file containing the observer gain matrix
mat_data = loadmat(script_dir + '/observer_gain_L.mat')
mat_data_2 = loadmat(script_dir + '/state_space_matrices.mat')

# Extract the observer gain matrix and convert it to a numpy array
L = np.array(mat_data['L'])
Ad = np.array(mat_data_2['Ad'])
Bd = np.array(mat_data_2['Bd'])
Cd = np.array(mat_data_2['Cd'])
Dd = np.zeros((dim_u, dim_u))


# print(L)


# observer_poles = np.linalg.eigvals(Ad - L @ Cd)
# print("Poles of A - LC:", observer_poles)


A_tilda = np.block([
    [Ad, np.zeros_like(Ad)],
    [np.zeros_like(Ad), Ad - L @ Cd]
])

B_tilda = np.block([[Bd], [Bd]])

Q = np.eye(dim_x)
R = np.eye(dim_u)

# Solve discrete ARE: P = solve_discrete_are(A, B, Q, R)
P, _, K = dare(A_tilda, B_tilda, Q, R)
K = -K

# Define constraints.
x_max = 10000000
u_max = 500

# Calculate the terminal set X_f
c_max =  calculate_ellipsoid(u_max, x_max, P, K, dim_x, dim_u)
#c_max = 10
print("c_max: ", c_max)

# Generate a random point within X_f
#x0 = get_point_within_ellipsoid(P, c_max, dim_x, x_max)
#print("x0: ", x0)
x0 = np.array([100000, 1000, 1000, 20, 20, 20, 100000, 1000, 1000, 20, 20, 20])  # Example initial state
# Constraints for states and inputs (example bounds)
x_lb = -x_max * np.ones(dim_x)
x_ub =  x_max * np.ones(dim_x)
u_lb = -u_max * np.ones(dim_u)
u_ub =  u_max * np.ones(dim_u)


# Prediction horizon and initial state
N = 40       # MPC horizon length

def solve_mpc(dim_x, dim_u, N, x0, Ad, Bd, Cd, Dd, Q, R, P, c_max):
    # ==============================
    # 2. Setup the MPC Optimization Problem
    # ==============================

    # Create lists of CVXPY variables for the predicted state and control trajectories
    x_var = [cp.Variable(dim_x) for _ in range(N + 1)]  # x[0], x[1], ..., x[N]
    u_var = [cp.Variable(dim_u) for _ in range(N)]        # u[0], u[1], ..., u[N-1]

    # Initialize cost and constraint list
    cost = 0
    constraints = []

    # Initial condition constraint
    constraints.append(x_var[0] == x0)

    # Dynamics and path constraints for each time step k = 0,...,N-1:
    for k in range(N):
        # System dynamics: x[k+1] = A x[k] + B u[k]
        constraints.append(x_var[k+1] == Ad @ x_var[k] + Bd @ u_var[k])
        
        # State constraints: x_lb <= x[k] <= x_ub 
        constraints.append(x_var[k] >= x_lb)
        constraints.append(x_var[k] <= x_ub)
        
        # Input constraints: u_lb <= u[k] <= u_ub
        constraints.append(u_var[k] >= u_lb)
        constraints.append(u_var[k] <= u_ub)
        
        # Stage cost: quadratic cost for state and input
        cost += cp.quad_form(x_var[k], Q) + cp.quad_form(u_var[k], R)

    # Terminal constraints:
    # (a) Terminal state constraint: the terminal state must lie within the ellipsoidal terminal set.
    constraints.append(cp.quad_form(x_var[N], P) <= c_max)
    # (b) Optional: Add a terminal cost term. Here we use the same matrix P as the terminal weight.
    cost += cp.quad_form(x_var[N], P)

    # ==============================
    # 3. Formulate and Solve the Optimization Problem
    # ==============================

    # Define the optimization problem: minimize the cost subject to all constraints
    prob = cp.Problem(cp.Minimize(cost), constraints)

    # Solve the problem using an appropriate solver (ECOS is a good choice for convex QPs)
    solution = prob.solve(solver=cp.ECOS, verbose=True)

    Us = np.array([u_var[k].value for k in range(N)])   # Optimal control trajectory
    Xs = np.array([x_var[k].value for k in range(N + 1)])   # Optimal state trajectory
    # Check terminal constraint value (should be <= c_max)
    print("Terminal state within Xf: ", check_point_within_ellipsoid(P, c_max, x_var[N].value))
    return Xs, Us

def plot_results(Xs, Us, dim_x, dim_u):
    # Plot each state value as a subplot
    time_steps = np.arange(N + 1)
    plt.figure(figsize=(12, 8))

    for i in range(dim_x):
        plt.subplot(dim_x, 1, i + 1)
        plt.step(time_steps, Xs[:, i], where='post', label=f"x[{i}]")
        plt.ylabel(f"x[{i}]")
        plt.grid()
        if i == 0:
            plt.title("State Trajectories")
        if i == dim_x - 1:
            plt.xlabel("Time step")

        # Mark the point where x^T P x < c_max with a vertical dashed line
        label_added = False
        for t in range(N + 1):
            if Xs[t].T @ P @ Xs[t] < c_max:
                if not label_added and i == 0:  # Add label only in the first subplot
                    plt.axvline(x=t, color='r', linestyle='--', label="State within X_f")
                    label_added = True
                else:
                    plt.axvline(x=t, color='r', linestyle='--')
                break  # Mark the first occurrence only

        plt.legend()  # Ensure the legend is added after marking the line

    plt.tight_layout()
    plt.show()

Xs, Us = solve_mpc(dim_x, dim_u, N, x0, A_tilda, B_tilda, Cd, Dd, Q, R, P, c_max)
plot_results(Xs, Us, dim_x, dim_u)