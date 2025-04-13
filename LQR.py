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

A_tilda = Ad
B_tilda = Bd

Q = 1*np.eye(dim_x_dyn)
R = 10*np.eye(dim_u)
#R = 10*np.matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

# Solve discrete ARE: P = solve_discrete_are(A, B, Q, R)
P, _, K = dare(Ad, Bd, Q, R)
K = -K

x0 = np.array([100, -100, 30, 2, 1, 1])  # Initial state

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
Xs, Us = solve_mpc(dim_x, dim_u, N, x0, A_tilda, B_tilda, Cd, Dd, Q, R, P, c_max)
plot_results(Xs, Us, dim_x, dim_u)