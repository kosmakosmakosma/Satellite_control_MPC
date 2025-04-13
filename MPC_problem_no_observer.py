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

dim_x = 6    # number of states
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


#A_tilda = np.block([[Ad, np.zeros_like(Ad)], [np.zeros_like(Ad), Ad - L @ Cd]])

#B_tilda = np.block([[Bd],[Bd]])

A_tilda = Ad
B_tilda = Bd

Q = 1*np.eye(dim_x_dyn)
R = 1*np.eye(dim_u)
#R = 10*np.matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

# Solve discrete ARE: P = solve_discrete_are(A, B, Q, R)
P, _, K = dare(Ad, Bd, Q, R)
K = -K


# Define constraints.
#x_max = 10000000
#u_max = 500




x0 = np.array([100, -100, 30, 2, 1, 1])  # Initial state

x_ub =  np.array([200, 200, 150, 10, 10, 10])  # Position and velocity upper constraints
x_lb = -x_ub
u_ub = 1*np.array([0.5, 0.5, 0.5])     # Acceleration upper constraints. Slightly unrealistic but for testing purposes
u_lb = -u_ub

# Calculate the terminal set X_f
c_max =  calculate_ellipsoid(u_ub, x_ub, P, K, dim_x_dyn, dim_u)
#c_max = 10
print("c_max: ", c_max)

N = 50  # MPC horizon length

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
        constraints.append(x_var[k][:6] >= x_lb[:6])
        constraints.append(x_var[k][:6] <= x_ub[:6])
        
        # Input constraints: u_lb <= u[k] <= u_ub
        constraints.append(u_var[k] >= u_lb)
        constraints.append(u_var[k] <= u_ub)
        
        # Stage cost: quadratic cost for state and input
        cost += cp.quad_form(x_var[k][:6], Q) + cp.quad_form(u_var[k], R)

    # Terminal constraints:
    # (a) Terminal state constraint: the terminal state must lie within the ellipsoidal terminal set.
    constraints.append(cp.quad_form(x_var[N][:6], P) <= c_max)
    # (b) Optional: Add a terminal cost term. Here we use the same matrix P as the terminal weight.
    cost += cp.quad_form(x_var[N][:6], P)

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
            if Xs[t][:6].T @ P @ Xs[t][:6] < c_max:
                if not label_added and i == 0:  # Add label only in the first subplot
                    plt.axvline(x=t, color='r', linestyle='--', label="State within X_f")
                    label_added = True
                else:
                    plt.axvline(x=t, color='r', linestyle='--')
                break  # Mark the first occurrence only

        plt.legend()  # Ensure the legend is added after marking the line

    plt.tight_layout()
    plt.show()
# Xs, Us = solve_mpc(dim_x, dim_u, N, x0, A_tilda, B_tilda, Cd, Dd, Q, R, P, c_max)
# plot_results(Xs, Us, dim_x, dim_u)

simulation_length = 50
x_simulated = np.zeros((simulation_length, dim_x, 3))
u_simulated = np.zeros((simulation_length, dim_u, 3))
x0_2 = x0.copy()
x0_05 = x0.copy()
for i in range(simulation_length):
    _, u_bar_cd = solve_mpc(dim_x, dim_u, N, x0, A_tilda, B_tilda, Cd, Dd, Q, R, P, c_max)
    # Solve DARE to get the LQR solution
    _, _, K_lqr = dare(A_tilda, B_tilda, Q, R)
    K_lqr = -K_lqr

    # Compute the LQR control input
    u_bar_cd_2 = K_lqr @ x0_2
    # Bound u_bar_cd_2 by u_ub and u_lb
    u_bar_cd_2 = np.clip(u_bar_cd_2, u_lb, u_ub)

    x0 = np.dot(A_tilda, x0) + np.dot(B_tilda, u_bar_cd[0, :].T)
    x0_2 = np.dot(A_tilda, x0_2) + np.dot(B_tilda, u_bar_cd_2)
    # x0 = np.dot(A_tilda, x0) + np.dot(B_tilda, u_bar_cd[0,:].T)
    x_simulated[i] = np.column_stack((x0, x0_2, x0_05))
    #u_simulated[i] = u_bar_cd[0,:]

# Export Xs to a file for later use

# Calculate Euclidean distances along the second axis for x_simulated and u_simulated
x_distances_distances = np.linalg.norm(x_simulated[:, :3, 0], axis=1)
x_distances_distances_2 = np.linalg.norm(x_simulated[:, :3, 1], axis=1)
#x_distances_distances_05 = np.linalg.norm(x_simulated[:, :3, 2], axis=1)

#x_velocities_distances = np.linalg.norm(x_simulated[:, 3:6], axis=1)
#u_distances = np.linalg.norm(u_simulated, axis=1)

# Save the distances to files for later use
# x_distances_file = script_dir + '/x_distances.npy'
# u_distances_file = script_dir + '/u_distances.npy'
# np.save(x_distances_file, x_distances)
# np.save(u_distances_file, u_distances)

# print(f"Euclidean distances for x_simulated saved to {x_distances_file}")
# print(f"Euclidean distances for u_simulated saved to {u_distances_file}")

# Plot the Euclidean distances for x_simulated and u_simulated

# plt.figure(figsize=(4, 2))
# time_steps = np.arange(simulation_length)

# plt.plot(time_steps, x_distances_distances, label="MPC")
# plt.plot(time_steps, x_distances_distances_2, label="LQR")
# #plt.plot(time_steps, x_distances_distances_05, label="N = 25", marker='^')

# plt.title("Euclidean Distances of x_simulated")
# plt.xlabel("Time step")
# plt.ylabel("Euclidean Distance")
# plt.legend()
# plt.tight_layout()
# plt.show()

# Plot the first three states of x_simulated[:, :3, 0]
time_steps = np.arange(simulation_length)
plt.figure(figsize=(4, 4))

for i in range(3):  # Plot the first three states
    plt.plot(time_steps, x_simulated[:, i, 0], label=f"x[{i}] (MPC)")

#plt.title("State Trajectories (First Three States) for x_simulated[:, :3, 0]")
plt.xlabel("Time step")
plt.ylabel("State Value")
plt.legend()
plt.tight_layout()
plt.show()
# output_file = script_dir + '/Xs_data_.npy'
# np.save(output_file, Xs)
# print(f"Optimal state trajectory Xs has been saved to {output_file}")

