# This script attempts to solve an RBC model using the
# Sequence-Space-Jacobian package 
# (https://github.com/shade-econ/sequence-jacobian/tree/ac77f11b90fc70b94633de0526ffa40fec046966)
# 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
from sequence_jacobian.utilities.drawdag import drawdag
from sequence_jacobian import simple, create_model


# Firm block
@simple 
def firm(A, N, P):
    Y = A * N
    W = A * P
    return Y, W

# Household block
@simple 
def household(M, P, W, chi, N, phi, theta, nu, gamma):
    
    def labor_leisure(C, W, P, chi, N, phi, theta, nu, gamma):
        X = ((1-theta) * C ** (1-nu) + theta * (M/P) ** (1-nu)) ** (1/(1-nu))
        return (W / P) - (chi * N**phi) / ((1 - theta) * C**(-nu) * X**(-gamma + nu))

    def solve_for_C(W, P, chi, N, phi, theta, nu, gamma, initial_guess=1.0):
        # Use fsolve to solve for C_t
        C_solution = fsolve(labor_leisure, initial_guess, args=(W, P, chi, N, phi, theta, nu, gamma))
        return C_solution[0]
    
    C = solve_for_C(W, P, chi, N, phi, theta, nu, gamma)
    Q = 1 / (1 - ((1-theta)/theta*M/P/C)**(-nu))
    X = ((1-theta) * C ** (1-nu) + theta * (M/P) ** (1-nu)) ** (1/(1-nu))
    
    return C, Q, X

# Market clearings
@simple
def mkt_clearing(Y, C, Q, P, X, beta):
    goods_mkt = Y - C
    euler = beta * Q * P / P(+1) * (X(+1) / X) ** (-gamma + nu) * (C(+1) / C) ** (-nu)
    return goods_mkt, euler


# Set up RBC model
rbc = create_model([household, firm, mkt_clearing], name="RBC")
print(rbc)
print(f"Blocks: {rbc.blocks}")


# Draw DAG 
unknowns = ['N', 'P']
targets = ['euler', 'goods_mkt']
inputs = ['M']

drawdag(rbc, inputs, unknowns, targets)


# Parameters
gamma = 1
phi = 1
chi = 1
beta = 0.99
rho = 0.99
sigma_m = 0.01
nu_list = [0.25, 0.5, 0.999, 2, 4]


# Calibration
nu_len = len(nu_list)
theta_list = np.zeros(nu_len)

def solve_for_theta(theta, nu):
    return (1 - beta)**(-1 / nu) * (theta / (1 - theta))**(1 / nu) * ((1 - theta) / chi * ((1 - theta + theta * (1 - beta)**(-(1 - nu) / nu) * (theta / (1 - theta))**((1 - nu) / nu)))**((nu - gamma) / (1 - nu)))**(1 / (phi + gamma)) - 1

for i in range(nu_len):
    nu = nu_list[i]
    sol = root(solve_for_theta, 0.01, args=(nu))
    theta_list[i] = sol.x[0]

calibration = {"N": 1, "A": 1, "M": 1, "P": 1, "r": 0.01, "gamma": 1, 
               "chi": 1, "phi": 1, "nu": nu_list, 
               "beta": 0.99, "theta": theta_list}
unknowns_ss = {"C": 0, "Q": 0.5}
targets_ss = {"goods_mkt": 0, "euler": 0}


# Solve for steady state
ss = rbc.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")


# Solve for impulse responses
G = rbc.solve_jacobian(ss, unknowns, targets, inputs, T=300)
T, impact, rho = 300, 0.01, 0.8
dM = np.empty((T, 1))
dM[:, 0] = impact * ss['M'] * rho ** np.arange(T)
dC = 100 * G['C']['M'] @ dM / ss['C']
dP = 100 * G['P']['M'] @ dM / ss['C']
dQ = 100 * G['Q']['M'] @ dM / ss['C']

