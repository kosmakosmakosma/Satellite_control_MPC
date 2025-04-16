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
dim_x_dyn = 6    # number of states in the dynamics
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
Dd = np.array(mat_data_2['Dd'])

# print(L)


# observer_poles = np.linalg.eigvals(Ad - L @ Cd)
# print("Poles of A - LC:", observer_poles)

A_tilda = np.block([
    [Ad, np.zeros_like(Ad)],
    [np.zeros_like(Ad), Ad - L @ Cd]
])

B_tilda = np.block([
    [Bd],
    [Bd]
])


#A_tilda = Ad
#B_tilda = Bd

Q = np.eye(dim_x_dyn)
R = np.eye(dim_u)

# Solve discrete ARE: P = solve_discrete_are(A, B, Q, R)
P, _, K = dare(Ad, Bd, Q, R)
K = -K

# Define constraints.
#x_max = 10000000
#u_max = 500





x0 = np.array([100, -100, 30, 2, 1, 1, 15, 15, 15, 2, 2, 2])  # Initial state

x_ub =  np.array([200, 200, 150, 10, 10, 10])   # Position and velocity upper constraints
x_lb = -x_ub
u_ub = np.array([0.5, 0.5, 0.5])    # Acceleration upper constraints. Slightly unrealistic but for testing purposes
u_lb = -u_ub

# Calculate the terminal set X_f
c_max =  calculate_ellipsoid(u_ub, x_ub, P, K, dim_x_dyn, dim_u)
#c_max = 10
print("c_max: ", c_max)

N = 20       # MPC horizon length
num_steps = 100  # Number of steps to run the MPC loop
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
        constraints.append(x_var[k][:6] + x_var[k][6:] >= x_lb[:6])
        constraints.append(x_var[k][:6] + x_var[k][6:] <= x_ub[:6])
        
        # Input constraints: u_lb <= u[k] <= u_ub
        constraints.append(u_var[k] >= u_lb)
        constraints.append(u_var[k] <= u_ub)
        
        # Stage cost: quadratic cost for state and input
        cost += cp.quad_form(x_var[k][:6] + x_var[k][6:], Q) + cp.quad_form(u_var[k], R)

    # Terminal constraints:
    # (a) Terminal state constraint: the terminal state must lie within the ellipsoidal terminal set.
    #constraints.append(cp.quad_form(x_var[N][:6] + x_var[N][6:], P) <= c_max)
    # (b) Optional: Add a terminal cost term. Here we use the same matrix P as the terminal weight.
    cost += cp.quad_form(x_var[N][:6] + x_var[N][6:], P)

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
    #print("Terminal state within Xf: ", check_point_within_ellipsoid(P, c_max, x_var[N][:6].value))
    return Xs, Us
def run_mpc_loop(dim_x, dim_u, N, x0, A_tilda, B_tilda, Cd, Dd, Q, R, P, c_max, num_steps):
    """
    Runs the MPC loop for a given number of steps. At each step, it solves the MPC problem,
    applies the first control input to the system, and updates the state.

    Args:
        dim_x: Dimension of the state vector.
        dim_u: Dimension of the control input vector.
        N: MPC horizon length.
        x0: Initial state vector.
        A_tilda: System dynamics matrix.
        B_tilda: Input matrix.
        Cd, Dd: Not used in this implementation but kept for compatibility.
        Q: State cost matrix.
        R: Input cost matrix.
        P: Terminal cost matrix.
        c_max: Terminal set constraint.
        num_steps: Number of steps to run the MPC loop.

    Returns:
        Xs_all: List of all state trajectories over the MPC loop.
        Us_all: List of all control inputs over the MPC loop.
    """
    Xs_all = []
    Us_all = []
    x_current = x0

    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")
        print("Current state:", x_current)
        # Solve the MPC problem
        Xs, Us = solve_mpc(dim_x, dim_u, N, x_current, A_tilda, B_tilda, Cd, Dd, Q, R, P, c_max)

        # Store the results
        Xs_all.append(x_current)
        Us_all.append(Us[0])

        # Apply the first control input to the system dynamics
        u_current = Us[0]
        x_next = A_tilda @ x_current + B_tilda @ u_current

        # Update the current state
        x_current = x_next

    return Xs_all, Us_all

def plot_results(Xs, Us, dim_x, dim_u, num_steps):
    # Ensure Xs is a numpy array with the correct shape
    Xs = np.array(Xs)
    if Xs.shape != (num_steps, dim_x):
        raise ValueError(f"Expected Xs to have shape ({num_steps}, {dim_x}), but got {Xs.shape}")

    # Plot each state value as a subplot
    time_steps = np.arange(num_steps)
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

    plt.tight_layout()
    plt.show()
Xs, Us = run_mpc_loop(dim_x, dim_u, N, x0, A_tilda, B_tilda, Cd, Dd, Q, R, P, c_max, num_steps)

# Print the shape of Xs
print("Shape of Xs:", np.array(Xs).shape)
plot_results(Xs, Us, dim_x, dim_u, num_steps)

# Export Xs to a file for later use
output_file_observer = script_dir + '/Xs_data_observer.npy'
np.save(output_file_observer, Xs)
print(f"Optimal state trajectory Xs has been saved to {output_file_observer}")

#output_file = script_dir + '/Xs_data_.npy'
#np.save(output_file, Xs)
#print(f"Optimal state trajectory Xs has been saved to {output_file}")

