import numpy as np
from quadprog import solve_qp
from scipy.signal import lsim, StateSpace
from scipy.linalg import expm
import pickle
import matplotlib.pyplot as plt
from control import dare
from invariant_set import calculate_ellipsoid, get_point_within_ellipsoid, check_point_within_ellipsoid

# Double integrator dynamics
dim_x = 6
dim_u = 3
dim_y = 3

miu = 398600.4418
a = 6778
n = (miu/a**3)**0.5
# CONTINUOUS-TIME SYSTEM
# State-space representation
Ac = np.array([[0, 0, 0, 1, 0, 0], 
               [0, 0, 0, 0, 1, 0], 
               [0, 0, 0, 0, 0, 1], 
               [3*n**2, 0, 0, 0, 2*n, 0],
               [0, 0, 0, -2*n, 0, 0],
               [0, 0, -n**2, 0, 0, 0]])

Bc = np.array([[0, 0, 0], 
               [0, 0, 0], 
               [0, 0, 0],                
               [1, 0, 0], 
               [0, 1, 0],
               [0, 0, 1]])

# Position output matrix
Cc = np.array([[1, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 1, 0]]) 

Dc = np.zeros((3,3))

assert Ac.shape == (dim_x, dim_x)
assert Bc.shape == (dim_x, dim_u)
assert Cc.shape == (dim_y, dim_x)
# DISCRETE-TIME SYSTEM

dt = 1  # Disc seems to match cont system only for dt=1

ABCDc = np.zeros((dim_y + dim_u + dim_x, dim_y + dim_u + dim_x))
ABCDc[:dim_x, :dim_x] = Ac
ABCDc[:dim_x, (dim_x+dim_y):] = Bc
ABCDc[dim_x:(dim_x+dim_y), :dim_x] = Cc
ABCDc[dim_x:(dim_x+dim_y), (dim_x+dim_y):] = Dc
expm_ABCDc = expm(ABCDc*dt)

Ad = expm_ABCDc[:dim_x, :dim_x]
Bd = expm_ABCDc[:dim_x, (dim_x+dim_y):]
Cd = expm_ABCDc[dim_x:(dim_x+dim_y), :dim_x]
Dd = expm_ABCDc[dim_x:(dim_x+dim_y), (dim_x+dim_y):] 

# Save Ad, Bd, Cd, Dd as Python objects

with open('state_space_matrices.pkl', 'wb') as f:
    pickle.dump({'Ad': Ad, 'Bd': Bd, 'Cd': Cd, 'Dd': Dd}, f)

Q = np.eye(dim_x)
R = np.eye(dim_u)

# Solve discrete ARE: P = solve_discrete_are(A, B, Q, R)
P, _, K = dare(Ad, Bd, Q, R)
K = -K

# Define constraints.
x_max = 100000
u_max = 100

# Calculate the terminal set X_f
c_max =  calculate_ellipsoid(u_max, x_max, P, K, dim_x, dim_u)
print("c_max: ", c_max)

# Generate a random point within X_f
x0 = get_point_within_ellipsoid(P, c_max, dim_x, x_max)
print("x0: ", x0)

# Discretize the input signal for the discrete-time system
t_disc = np.arange(0, 100, dt)
u_disc = np.zeros((len(t_disc), 3))  # Input signal for discrete system
Ad_K = Ad+Bd @ K

def simulate_discrete_system(Ad, Bd, Cd, Dd, x0, u_disc, t_disc):
    # Simulate the discrete-time system
    y_out_disc = []
    x_out_disc = []
    for i in range(len(t_disc)):
        y_out_disc.append(Cd @ x0 + Dd @ u_disc[i])
        x_out_disc.append(x0)
        x0 = Ad @ x0 + Bd @ u_disc[i]
    y_out_disc = np.array(y_out_disc)
    x_out_disc = np.array(x_out_disc)
    # Plot the results
    plt.figure(figsize=(10, 6))
    for j in range(y_out_disc.shape[1]):
        plt.step(t_disc, y_out_disc[:, j], where='post', linestyle='--', label=f'Discrete Output {i+1}')
    plt.title('Discrete State-Space System Simulation')
    plt.xlabel('Time (s)')
    plt.ylabel('Output Values')
    plt.legend()
    plt.grid()
    plt.show()

def simulate_cont_vs_disc_system(Ad, Bd, Cd, Dd, x0, u, t):

    # Continuous-time system
    system_cont = StateSpace(Ac, Bc, Cc, Dc)

    # Simulate the continuous-time system
    t_out_cont, y_out_cont, x_out_cont = lsim(system_cont, U=u, T=t, X0=x0)

    # Discretize the input signal for the discrete-time system
    t_disc = np.arange(0, 10, dt)
    u_disc = np.ones((len(t_disc), 3))  # Input signal for discrete system

    # Simulate the discrete-time system
    x_disc = x0
    y_out_disc = []
    x_out_disc = []
    for i in range(len(t_disc)):
        y_out_disc.append(Cd @ x_disc + Dd @ u_disc[i])
        x_out_disc.append(x_disc)
        x_disc = Ad @ x_disc + Bd @ u_disc[i]
    y_out_disc = np.array(y_out_disc)
    x_out_disc = np.array(x_out_disc)

    # Plot the results
    plt.figure(figsize=(10, 6))
    for i in range(y_out_cont.shape[1]):
        plt.plot(t_out_cont, y_out_cont[:, i], label=f'Continuous Output {i+1}')
        plt.step(t_disc, y_out_disc[:, i], where='post', linestyle='--', label=f'Discrete Output {i+1}')
    plt.title('Continuous vs Discrete State-Space System Simulation')
    plt.xlabel('Time (s)')
    plt.ylabel('Output Values')
    plt.legend()
    plt.grid()
    plt.show()

simulate_discrete_system(Ad_K, Bd, Cd, Dd, x0, u_disc, t_disc)