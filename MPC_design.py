import cvxpy as cp
import numpy as np
from quadprog import solve_qp
from qpsolvers import solve_qp
from control import dare
import pickle
import gurobipy as gp
from gurobipy import GRB
from control import place
from pprint import pprint

def gen_prediction_matrices(Ad, Bd, N):
    dim_x = Ad.shape[0]
    dim_u = Bd.shape[1]
    
    T = np.zeros(((dim_x * (N + 1), dim_x)))
    S = np.zeros(((dim_x * (N + 1), dim_u * N)))
    # Condensing
    power_matricies = []    # power_matricies = [I, A, A^2, ..., A^N]
    power_matricies.append(np.eye(dim_x))
    for k in range(N):
        power_matricies.append(power_matricies[k] @ Ad)
    
    for k in range(N + 1):
        T[k * dim_x: (k + 1) * dim_x, :] = power_matricies[k]
        for j in range(N):
            if k > j:
                S[k * dim_x:(k + 1) * dim_x, j * dim_u:(j + 1) * dim_u] = power_matricies[k - j - 1] @ Bd
                
    return T, S

def gen_cost_matrices(Q, R, P, T, S, x0, N):
    dim_x = Q.shape[0]
    
    Q_bar = np.zeros(((dim_x * (N + 1), dim_x * (N + 1))))
    Q_bar[-dim_x:, -dim_x:] = P
    Q_bar[:dim_x * N, :dim_x * N] = np.kron(np.eye(N), Q) 
    R_bar = np.kron(np.eye(N), R)
    
    H = S.T @ Q_bar @ S + R_bar
    h = S.T @ Q_bar @ T @ x0
    
    H = 0.5 * (H + H.T) # Ensure symmetry!
    
    return H, h

def gen_constraint_matrices(u_lb, u_ub, N):
    dim_u = u_lb.shape[0]
    
    Gu = np.zeros((2 * dim_u, dim_u))
    Gu[0: dim_u, :] = np.eye(dim_u)
    Gu[dim_u: , :] = -np.eye(dim_u)
    
    gu = np.zeros(2 * dim_u)
    gu[0: dim_u] = u_ub
    gu[dim_u:] = -u_lb    
    
    Gu_bar = np.kron(np.eye(N), Gu)
    gu_bar = np.kron(np.ones(N), gu)
    
    return Gu_bar, gu_bar


def solve_mpc_condensed(Ad, Bd, Q, R, P, x0, N, u_lb, u_ub):
    dim_x = Ad.shape[0]
    dim_u = Bd.shape[1]
    T, S = gen_prediction_matrices(Ad, Bd, N)
    H, h = gen_cost_matrices(Q, R, P, T, S, x0, N)
    Gu_bar, gu_bar = gen_constraint_matrices(u_lb, u_ub, N)
    
    u_bar = solve_qp(H, h, G=Gu_bar, h=gu_bar, solver='quadprog') 
    x_bar = T @ x0 + S @ u_bar
    x_bar = x_bar.reshape((N + 1, dim_x))
    u_bar = u_bar.reshape((N, dim_u))
    
    return x_bar, u_bar


# Load matrices from the pickle file
with open('state_space_matrices.pkl', 'rb') as f:
    matrices = pickle.load(f)

Ad = matrices['Ad']
Bd = matrices['Bd']
Cd = matrices['Cd']
Dd = matrices['Dd']
Dd = np.zeros((3, 3))

x = np.array([1, 1, -10, 0, 0, 0]) 

u_lb = 1000*np.array([-1, -1, -1])
u_ub = 1000*np.array([1, 1, 1])
N = 20

Q = 100*np.eye(6)
R = np.eye(3)
#P = np.eye(6)*1000
P_inf, _, K_inf = dare(Ad, Bd, Q, R)
P = P_inf

noise_std = 0.001*np.eye(6)
distnaces = []
force_applied = []

# create new LTI system with estimated the original states and estimation error stacked vertically

# Define desired pole locations for the observer
desired_poles = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])	

# Compute the observer gain L using pole placement
L = place(Ad.T, Cd.T, desired_poles).T

observer_poles = np.linalg.eigvals(Ad - L @ Cd)
print("Poles of A - LC:", observer_poles)


A_tilda = np.block([
    [Ad, np.zeros_like(Ad)],
    [np.zeros_like(Ad), Ad - L @ Cd]
])

B_tilda = np.block([[Bd], [Bd]])

x_bar = 1*np.ones((Ad.shape[0] * 2))


for i in range(1000):
    _, u_bar_cd = solve_mpc_condensed(Ad, Bd, Q, R, P, x_bar[6:12] + x_bar[:6], N, u_lb, u_ub)

    # print(x.shape)
    # print(x_bar[6:12].shape)
    # print(x_bar.shape)
    # print(A_tilda.shape)
    # print(B_tilda.shape)
    # print(u_bar_cd[0,:].T.shape)
    x_bar = np.dot(A_tilda, x_bar) + np.dot(B_tilda, u_bar_cd[0,:].T)
    # x_bar = np.dot(A_tilda, x_bar) + np.dot(B_tilda, u_bar_cd[0,:].T)
    # x = np.dot(Ad, x.T) + np.dot(Bd, u_bar_cd[0,:].T) + np.random.multivariate_normal(np.zeros(6), noise_std)
    # y = np.dot(Cd, x) + np.dot(Dd, u_bar_cd[0,:])
    distnaces.append(np.linalg.norm(x_bar[0:3]))
    force_applied.append(np.linalg.norm(u_bar_cd[0,:]))
    

# Calculate the Euclidean distance of the first three elements of the states over time

import matplotlib.pyplot as plt

plt.figure()
plt.plot(distnaces, marker='o')
plt.title('Euclidean Distance of First Three States Over Time')
plt.xlabel('Time Step')
plt.ylabel('Euclidean Distance')
plt.grid()
plt.show()

# Plot the Euclidean distances
plt.figure()
plt.plot(force_applied, marker='o')
plt.title('Euclidean Distance of Control Inputs Over Time')
plt.xlabel('Time Step')
plt.ylabel('Euclidean Distance')
plt.grid()
plt.show()