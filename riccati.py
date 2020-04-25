#!/usr/bin/python3.6
import numpy as np
import numpy.linalg as LA

# 連続時間Riccati代数方程式を解くプログラム
#  式： A X + X A^T + Q - X B R^{-1} B^T X = 0
def solve_care(A, B, Q, R):
    # 1. ハミルトンマトリクスを置く
    H = np.block([[A.T, -B @ LA.inv(R) @ B.T],
                  [-Q , -A]])
    # 2.固有値分解する
    eigenvalue, w = LA.eig(H)
    # 3.補助行列を置く
    Y_, Z_ = [], []
    n = len(eigenvalue)//2
    for i in range(2*n):
        if eigenvalue[i].real < 0.0:
            Y_.append(w.T[i][:n])
            Z_.append(w.T[i][n:])
    # 3'.補助行列のランクが足りない場合補助してあげる
    """
    i = 0
    print(w)
    while len(Y_) < n:
        if abs(eigenvalue[i]) < 1e-4:
            y_norm = LA.norm(w.T[i][:n])
            z_norm = LA.norm(w.T[i][n:])
            if y_norm > z_norm:
                Y_.append(w.T[i][:n])
                Z_.append(w.T[i][n:])
            else:
                print(i,"ビミョー",y_norm,z_norm)
        else:
            print(i,"skipp")
        i += 1
    """
    Y = np.array(Y_).T
    Z = np.array(Z_).T
    # 4.Pが求まる
    return Z @ LA.inv(Y)

# 反復法により定常カルマンフィルタのゲインを求めるプログラム
def discrete_kalman_gain(StateDevelopmentMatrix, StateDevelopmentNoise, ObservationMatrix, ObservationNoise, verbose=False):
    CovarianceMatrix = np.array(StateDevelopmentNoise)
    max_iteration = True
    for i in range(200):
        CovarianceMatrixPrediction = StateDevelopmentMatrix @ CovarianceMatrix @ StateDevelopmentMatrix.T + StateDevelopmentNoise
        KalmanGain = CovarianceMatrixPrediction @ ObservationMatrix.T @ LA.inv(ObservationMatrix@CovarianceMatrixPrediction@ObservationMatrix.T + ObservationNoise)
        newCovarianceMatrix = CovarianceMatrixPrediction - KalmanGain @ ObservationMatrix @ CovarianceMatrixPrediction
        if LA.det(newCovarianceMatrix - CovarianceMatrix) < 1e-10:
            CovarianceMatrix = np.array(newCovarianceMatrix)
            max_iteration = False
            break
        else:
            CovarianceMatrix = np.array(newCovarianceMatrix)
            continue
    if max_iteration:
        print("dare max 反復 exceeded")
    if verbose:
        print("DAREの解P")
        print(CovarianceMatrixPrediction)
        print("固有値")
        print(LA.eig(CovarianceMatrixPrediction)[0])
    KalmanGain = CovarianceMatrixPrediction @ ObservationMatrix.T @ LA.inv(ObservationMatrix@CovarianceMatrixPrediction@ObservationMatrix.T + ObservationNoise)
    return KalmanGain

def test_code1():
    A = np.array([[3., 1.],[0., 1.]])
    B = np.array([[1.2], [1.]])
    Q = np.array([[1., 0.2], [0.2, 1.0]])
    R = np.array([[1.]])
    P = solve_care(A, B, Q, R)
    Value = A@P + P@A.T + Q - P@B@LA.inv(R)@B.T@P
    print("P")
    print(P)
    print("Riccati代数方程式の左辺")
    print(Value)
    print(LA.norm(Value))

def test_code2():
    A = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, -3.2666666666666666, -0.08888888888888888, 0.027777777777777776], [0.0, 8.166666666666666, 0.05555555555555555, -0.06944444444444445]])
    B = np.array([[0.0], [0.0], [0.4444444444444444], [-0.27777777777777773]])
    dt = 1e-3
    Gain = discrete_kalman_gain(
        np.eye(4) + A * dt,
        np.eye(4),
        B.T,
        np.eye(1),
        True)
    print(Gain)

def test_code3():
    A = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, -3.2666666666666666, -0.08888888888888888, 0.027777777777777776], [0.0, 8.166666666666666, 0.05555555555555555, -0.06944444444444445]])
    B = np.array([[0.0], [0.0], [0.4444444444444444], [-0.27777777777777773]])
    P = solve_care(A, B, np.eye(4), 0.01 * np.eye(1))
    print(P)

if __name__ == "__main__":
    test_code3()
