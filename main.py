import numpy as np
from quadprog import solve_qp
from scipy.signal import lsim, StateSpace
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Double integrator dynamics
dim_x = 6
dim_u = 3
dim_y = 3

miu = 1
a = 1
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
# State-space representation
dt = 0.01

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

# Define initial state and input
x0 = np.array([0, 0, 0, 0, 0, 0])  # Initial state
t = np.linspace(0, 10, 1000)       # Time vector
print(t)
u = np.ones((len(t), 3))           # Input signal (zero input)

# Continuous-time system
system_cont = StateSpace(Ac, Bc, Cc, Dc)

# Simulate the continuous-time system
t_out_cont, y_out_cont, x_out_cont = lsim(system_cont, U=u, T=t, X0=x0)

# Discretize the input signal for the discrete-time system
t_disc = np.arange(0, 10, dt)
print(t_disc)
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
    plt.step(t_disc, y_out_cont[:, i], where='post', linestyle='--', label=f'Discrete Output {i+1}')
plt.title('Continuous vs Discrete State-Space System Simulation')
plt.xlabel('Time (s)')
plt.ylabel('Output Values')
plt.legend()
plt.grid()
plt.show()