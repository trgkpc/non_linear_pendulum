#!/usr/bin/python3.6
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import math

log_skip_mode = False
t_max = 8.0
simulation_range = 0.2
alpha_for_realtime_mpc = 3.0
x0 = np.array([0.0, 30*math.pi/180, 0.0, 0.0])
mpc_iteration_time = 100
mpc_U0_update_type = "lqr"

import scipy
from scipy import linalg
import equation_of_state as source
# f(x,u,params): dx/dt
# A(x,  params): fをx近傍で近似1
# B(x,  params): fをx近傍で近似2
def runge_kutta(x, u, params, h):
    k1 = source.f(x           , u, params)
    k2 = source.f(x + (h/2)*k1, u, params)
    k3 = source.f(x + (h/2)*k2, u, params)
    k4 = source.f(x + h*k3    , u, params)
    return (h/6) * (k1 + 2.0*k2 + 2.0*k3 + k4)

params = {"M":2.0, "m":1.0, "l":1.2, "Dx":0.2, "Dtheta":0.1, "g":9.8}
dt = 1e-3
stable_state = [0., 0., 0., 0.]

A0 = source.A(stable_state, params)
B0 = source.B(stable_state, params)
Q = np.array([
    [1.0, 0.1, 0.1, 0.1],
    [0.1, 2.0, 0.1, 0.1],
    [0.1, 0.1, 0.6, 0.1],
    [0.1, 0.1, 0.1, 1.2]])
R = np.array([
    [0.1]])
P = scipy.linalg.solve_continuous_are(A0, B0, Q, R)
F = -LA.inv(R)@(B0.T)@P
print("P")
print(P)
print("Pの固有値")
print(LA.eig(P)[0])
print("ゲイン")
print(F)
print("システムの極")
poles,EigenVectors = LA.eig(A0+B0@F)
ReverseVectors = LA.inv(EigenVectors)
print(poles)
print("システムの固有ベクトル")
print(EigenVectors)
"""
appended_system = A0 + B0@F
pe = EigenVectors@np.diag(poles)@ReverseVectors
print("拡張システムと対角化との差分")
print(LA.norm(pe-appended_system))
"""
# "A":A0+B0@F = EigenVectors@np.diag(poles)@ReverseVectors
# "P": EigenVectors
b_tilde = EigenVectors.T @ (Q+F.T@R@F) @ EigenVectors
S = np.array([[-b_tilde[j][i]/(poles[i]+poles[j]) for i in range(4)] for j in range(4)])
FinalCostMatrix = ReverseVectors.T @ S @ ReverseVectors
print("LQRコスト計算行列")
print(FinalCostMatrix.tolist())
print("コスト行列の固有値")
print(LA.eig(FinalCostMatrix)[0])


log_file = 0
def create_log_file(filename):
    if log_skip_mode:
        return None
    global log_file
    print("create log file", filename)
    log_file = open(filename, "w")

def write_log(*args):
    if log_skip_mode:
        return None
    for p in args:
        log_file.write(str(p)+" ")
    log_file.write("\n")

def simulate_lqr():
    create_log_file("lqr_result.log")

    print("==== start lqr simulation ====")
    history = []
    t = 0.0
    x = np.array(x0)
    cost = 0.0
    while t <= t_max:
        u = F@x
        L = x@Q@x + u@R@u
    
        write_log(t,L,x[0],x[1],x[2],x[3],u[0])
        history.append([t, L, x[0], x[1]])
     
        x += runge_kutta(x, u, params, dt)
        t += dt
        cost += L * dt
    print("~~~ end lqr simulation ~~~")
    return [history, cost]

def realtime_mpc(current_state, init_answer, verbose=False):
    n = len(init_answer)
    m = len(init_answer[0])
    U = np.ravel(init_answer)
    max_exceed = True
    alpha = alpha_for_realtime_mpc
    for iteration_times in range(mpc_iteration_time): # 適当な回数反復
        x_history = []
        
        # 状態シミュレーション
        x = np.array(current_state)
        x_history.append(x)
        for i in range(n):
            x += source.f(x, U[m*i:m*(i+1)], params) * dt
            x_history.append(x)
        
        # 共状態シミュレーション
        #l = 0.0 * current_state
        # 終端コストを ∫xQx+uRuで置けば良さそう(LQRのコスト)
        # u = F@xだから、x@Q@x + u@R@u = x@Q@x + x@F.T@R@F@x
        l = 2.0 * FinalCostMatrix @ x
        l = np.array([element.real for element in l])
        grad_ = []
        for j in range(n):
            i = n-j
            x = x_history[i]
            u = U[m*(i-1):m*i]
            A = source.A(x, params)
            B = source.B(x, params)
            dHdx = 2.0 * (Q@x) + l@A
            l += dHdx * dt # l += -dHdx * (-dt)
            dHdu = 2.0 * (R@u) + l@B
            grad_.insert(0,dHdu) 
        grad = np.ravel(grad_)

        # 制御入力アップデート
        U -= alpha * grad
        #print(LA.norm(grad))
        if LA.norm(grad) < 1e-5:
            max_exceed = False
            break
    if max_exceed:
        print("mpc max iteration exceeded",LA.norm(dHdu))
    if verbose:
        print("iteration:",iteration_times)
    return U.reshape(n,m)

def receeding_mpc(current_state, init_answer, verbose=False):
    n = len(init_answer)
    m = len(init_answer[0])
    U = np.ravel(init_answer)
    max_exceed = True
    #不明

import time

def conv_center(x):
    poyo = np.array(x)
    poyo[0] = 0.0
    return poyo

def simulate_slqr():
    create_log_file("slqr_result.log")

    print("==== start slqr simulation ====")
    history = []
    t = 0.0
    x = np.array(x0)
    cost = 0.0
    while t <= t_max: 
        x_center = conv_center(x)
        A = source.A(x_center, params)
        B = source.B(x_center, params)
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        F = -LA.inv(R)@(B.T)@P
        u = F@x
        L = x@Q@x + u@R@u

        write_log(t,L,x[0],x[1],x[2],x[3],u[0])
        history.append([t, L, x[0], x[1]])

        x += runge_kutta(x, u, params, dt)
        t += dt
        cost += L * dt
    print("~~~ end slqr simulation ~~~")
    return [history,cost]


    

def simulate_mpc():
    mpc = realtime_mpc
    create_log_file("mpc_result.log")
    
    print("===== start mpc simulation =====")
    history = []
    t = 0.0
    x = np.array(x0)
    n = int(simulation_range/dt)
    cost = 0.0

    # MPCの初期解を作る
    U = []
    extended_system = A0 + B0@F
    x_ = np.array(x)
    for i in range(n):
        U.append(F@x_)
        x_ += dt * (extended_system@x_)
    U = np.array(U)

    real_time = time.time()
    while t <= t_max:
        start_time = time.time()
        # 解の更新
        ## MPCの初期解を作る
        last_U = U
        U = []
        extended_system = A0 + B0@F
        x_ = np.array(x)
        if mpc_U0_update_type == "lqr":
            for i in range(n):
                U.append(F@x_)
                x_ += dt * (extended_system@x_)
        elif mpc_U0_update_type == "default":
            U = last_U
        elif mpc_U0_update_type == "zeros":
            U = np.zeros((n,1))
        else:
            print("[[Warning]] わ")
            U = last_U

        U = np.array(U)
        U = mpc(x, U)
        u = U[0]

        L = x@Q@x + u@R@u
        end_time = time.time()

        write_log(t,L,x[0],x[1],x[2],x[3],u[0])
        history.append([t, L, x[0], x[1]])
        print(t, time.time()-real_time, end_time-start_time,L)

        x += runge_kutta(x, u, params, dt)
        t += dt
        cost += L * dt
    print("~~~ end mpc simulation ~~~")
    return [history, cost]

def compare(with_slqr=False,with_mpc=False):
    lqr = np.array(simulate_lqr()[0]).T
    if with_slqr:
        slqr = np.array(simulate_slqr()[0]).T 
    if with_mpc:
        mpc = np.array(simulate_mpc()[0]).T

    plt.plot(lqr[0], lqr[1], label="lqr")
    if with_slqr:
        plt.plot(slqr[0], slqr[1], label="slqr")
    if with_mpc:
        plt.plot(mpc[0], mpc[1], label="non-linear")
    plt.legend()
    plt.show()

    plt.plot(lqr[2], lqr[3], label="lqr")
    if with_slqr:
        plt.plot(slqr[2], slqr[3], label="slqr")
    if with_mpc:
        plt.plot(mpc[2], mpc[3], label="mpc")
    plt.legend()
    plt.show()

def mpc_debug():
    U = [[0.0] for i in range(int(simulation_range/dt))]
    realtime_mpc(x0, U,verbose=True)

def final_cost_matrix_debug():
    cost = simulate_lqr()[1]
    theoritical = x0@FinalCostMatrix@x0
    print("真値:",cost)
    print("理論値:",theoritical)

if __name__ == '__main__':
    compare(with_slqr=True,with_mpc=False)
    #final_cost_matrix_debug()
    #mpc_debug()

