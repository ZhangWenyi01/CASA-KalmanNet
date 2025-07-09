import numpy as np
import pandas as pd
from sklearn.datasets import make_spd_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

plt.style.use('bmh')
#Check whether a matrix is positive definite
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

#Force a matrix to be positive definite if its not
def force_SPD(A, c = 1e-10):
    A = (A + A.T)/2
    if np.amin(np.linalg.eigvals(A)) < 0:
        A += (np.abs(np.amin(np.linalg.eigvals(A))) + c)*np.identity(A.shape[0])
    if np.amin(np.linalg.eigvals(A)) == 0:
        A += c*np.identity(A.shape[0])
    return(A)


def KalmanFilter(F, Q, H, R, z, x_0, P_0):
    T = len(z)
    # Check the dimension of the hidden state
    if isinstance(x_0, np.ndarray) == True:
        n = len(x_0)
    else:
        n = 1
    # Check the dimension of the measurements
    if isinstance(z[0], np.ndarray) == True:
        p = len(z[0])
    else:
        p = 1

    # Multidimensional case
    if n > 1 and p > 1:
        x_hat_minus = np.empty((1, n))
        x_hat = np.array([x_0])
        P_minus = np.empty((1, n, n))
        P = np.array([P_0])
        K = np.empty((1, n, p))

        for i in range(T):
            # State extrapolation equation
            x_hat_minus = np.append(x_hat_minus, [F @ x_hat[i]], axis=0)

            # Covariance extrapolation equation
            P_minus = np.append(P_minus, [F @ P[i] @ F.T + Q], axis=0)

            # Kalman gain
            K = np.append(K, [P_minus[i + 1] @ H.T @ np.linalg.inv(H @ P_minus[i + 1] @ H.T + R)], axis=0)

            # State update equation
            x_hat = np.append(x_hat, [x_hat_minus[i + 1] + K[i + 1] @ (z[i] - H @ x_hat_minus[i + 1])], axis=0)

            # Covariance update equation
            P = np.append(P, [
                (np.identity(n) - K[i + 1] @ H) @ P_minus[i + 1] @ (np.identity(n) - K[i + 1] @ H).T + K[i + 1] @ R @ K[
                    i + 1].T], axis=0)

    # One-dimensional measurements case
    elif n > 1 and p == 1:
        x_hat_minus = np.empty((1, n))
        x_hat = np.array([x_0])
        P_minus = np.empty((1, n, n))
        P = np.array([P_0])
        K = np.empty((1, n, p))

        for i in range(T):
            # State extrapolation equation
            x_hat_minus = np.append(x_hat_minus, [F @ x_hat[i]], axis=0)

            # Covariance extrapolation equation
            P_minus = np.append(P_minus, [F @ P[i] @ F.T + Q], axis=0)

            # Kalman gain
            K = np.append(K, [P_minus[i + 1] @ H.T @ np.linalg.inv(H @ P_minus[i + 1] @ H.T + R)], axis=0)

            # State update equation
            x_hat = np.append(x_hat, [x_hat_minus[i + 1] + K[i + 1] @ (z[i] - H @ x_hat_minus[i + 1])], axis=0)

            # Covariance update equation
            P = np.append(P, [
                (np.identity(n) - K[i + 1] @ H) @ P_minus[i + 1] @ (np.identity(n) - K[i + 1] @ H).T + R * K[i + 1] @ K[
                    i + 1].T], axis=0)

    # One-dimensional case
    else:
        x_hat_minus = np.empty(1)
        x_hat = np.array([x_0])
        P_minus = np.empty(1)
        P = np.array([P_0])
        K = np.empty(1)

        for i in range(T):
            # State extrapolation equation
            x_hat_minus = np.append(x_hat_minus, [F * x_hat[i]], axis=0)

            # Covariance extrapolation equation
            P_minus = np.append(P_minus, [F ** 2 * P[i] + Q], axis=0)

            # Kalman gain
            K = np.append(K, [P_minus[i + 1] * H / (H ** 2 * P_minus[i + 1] + R)], axis=0)

            # State update equation
            x_hat = np.append(x_hat, [x_hat_minus[i + 1] + K[i + 1] * (z[i] - H * x_hat_minus[i + 1])], axis=0)

            # Covariance update equation
            P = np.append(P, [(1 - K[i + 1] * H) ** 2 * P_minus[i + 1] + K[i + 1] ** 2 * R], axis=0)

    x_hat_minus = np.delete(x_hat_minus, 0, axis=0)
    P_minus = np.delete(P_minus, 0, axis=0)
    K = np.delete(K, 0, axis=0)

    return (x_hat_minus, P_minus, K, x_hat, P)


def KalmanSmoother(F, x_hat_minus, P_minus, x_hat, P):
    T = len(x_hat_minus)
    if isinstance(x_hat[0], np.ndarray) == True:
        n = len(x_hat[0])
    else:
        n = 1

    # Multidimensional case
    if n > 1:
        x_tilde = np.array([x_hat[T]])
        P_tilde = np.array([P[T]])
        S = np.empty((1, n, n))

        for i in reversed(range(T)):
            # Smoothing gain
            S = np.insert(S, 0, [P[i] @ F.T @ np.linalg.inv(P_minus[i])], axis=0)

            # State correction
            x_tilde = np.insert(x_tilde, 0, [x_hat[i] + S[0] @ (x_tilde[0] - x_hat_minus[i])], axis=0)

            # Covariance correction
            P_tilde = np.insert(P_tilde, 0, [P[i] + S[0] @ (P_tilde[0] - P_minus[i]) @ S[0].T], axis=0)

    # One-dimensional case
    else:
        x_tilde = np.array([x_hat[T]])
        P_tilde = np.array([P[T]])
        S = np.empty(1)

        for i in reversed(range(T)):
            # Smoothing gain
            S = np.insert(S, 0, [P[i] * F / P_minus[i]], axis=0)

            # State correction
            x_tilde = np.insert(x_tilde, 0, [x_hat[i] + S[0] * (x_tilde[0] - x_hat_minus[i])], axis=0)

            # Covariance correction
            P_tilde = np.insert(P_tilde, 0, [P[i] + S[0] ** 2 * (P_tilde[0] - P_minus[i])], axis=0)

    S = np.delete(S, len(S) - 1, axis=0)

    return (S, x_tilde, P_tilde)


def Lag1AutoCov(K, S, F, H, P):
    T = len(P) - 1
    if isinstance(F, np.ndarray) == True:
        n = F.shape[0]
    else:
        n = 1

    # Multidimensional case
    if n > 1:
        V = np.array([(np.identity(n) - K[T - 1] @ H) @ F @ P[T - 1]])

        for i in reversed(range(1, T)):
            V = np.insert(V, 0, [P[i] @ S[i - 1].T + S[i] @ (V[0] - F @ P[i]) @ S[i - 1].T], axis=0)

    # One-dimensional case
    else:
        V = np.array([(1 - K[T - 1] * H) * F * P[T - 1]])

        for i in reversed(range(1, T)):
            V = np.insert(V, 0, [P[i] * S[i - 1].T + S[i] * (V[0] - F * P[i]) * S[i - 1].T], axis=0)

    return (V)


def ell(H, R, z, x, P):
    T = len(z)
    if isinstance(x[0], np.ndarray) == True:
        n = len(x[0])
    else:
        n = 1
    if isinstance(z[0], np.ndarray) == True:
        p = len(z[0])
    else:
        p = 1

    likelihood = -T * p / 2 * np.log(2 * np.pi)

    # Multidimensional case
    if n > 1 and p > 1:
        for i in range(T):
            likelihood -= 0.5 * (np.log(np.linalg.det(H @ P[i] @ H.T + R)) + (z[i] - H @ x[i]).T @ np.linalg.inv(
                H @ P[i] @ H.T + R) @ (z[i] - H @ x[i]))

    # One-dimensional measurements case
    elif n > 1 and p == 1:
        for i in range(T):
            likelihood -= 0.5 * (
                        np.log(np.linalg.det(H @ P[i] @ H.T + R)) + (z[i] - H @ x[i]) ** 2 / (H @ P[i] @ H.T + R))
        likelihood = likelihood[0][0]

    # One-dimensional case
    else:
        for i in range(T):
            likelihood -= 0.5 * (np.log(H ** 2 * P[i] + R) + (z[i] - H * x[i]) ** 2 / (H ** 2 * P[i] + R))

    return (likelihood)


def EMKF(F_0, Q_0, H_0, R_0, z, xi_0, L_0, max_it=1000, tol_likelihood=0.01, tol_params=0.005,
         em_vars=["F", "Q", "H", "R", "xi", "L"]):
    T = len(z)
    if isinstance(xi_0, np.ndarray) == True:
        n = len(xi_0)
    else:
        n = 1
    if isinstance(z[0], np.ndarray) == True:
        p = len(z[0])
    else:
        p = 1

    # Initialization
    F = np.array([F_0])
    Q = np.array([Q_0])
    H = np.array([H_0])
    R = np.array([R_0])
    xi = np.array([xi_0])
    L = np.array([L_0])

    likelihood = np.empty(1)

    # Multidimensional case
    if n > 1 and p > 1:
        A_5 = np.zeros((p, p))
        for j in range(T):
            A_5 += np.outer(z[j], z[j])

        for i in range(max_it):
            if i > 0 and i % 50 == 0:
                print(f"Iteration {i}")
            # E-step
            x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[i], Q[i], H[i], R[i], z, xi[i], L[i])
            S, x_tilde, P_tilde = KalmanSmoother(F[i], x_hat_minus, P_minus, x_hat, P)
            V = Lag1AutoCov(K, S, F[i], H[i], P)

            likelihood = np.append(likelihood, [ell(H[i], R[i], z, x_hat_minus, P_minus)], axis=0)

            # Convergence check for likelihood
            convergence_count = 0
            if i >= 1 and likelihood[i + 1] - likelihood[i] < tol_likelihood:
                convergence_count += 1

            # M-step
            A_1 = np.zeros((n, n))
            A_2 = np.zeros((n, n))
            A_3 = np.zeros((n, n))
            A_4 = np.zeros((p, n))

            for j in range(T):
                A_1 += np.outer(x_tilde[j + 1], x_tilde[j]) + V[j]
                A_2 += np.outer(x_tilde[j], x_tilde[j]) + P_tilde[j]
                A_3 += np.outer(x_tilde[j + 1], x_tilde[j + 1]) + P_tilde[j + 1]
                A_4 += np.outer(z[j], x_tilde[j + 1])

            if "F" in em_vars:
                # Update equation for F
                F = np.append(F, [A_1 @ np.linalg.inv(A_2)], 0)

                # Convergence check for F
                if i >= 1 and np.all(np.abs(F[i + 1] - F[i]) < tol_params):
                    convergence_count += 1
            else:
                F = np.append(F, [F_0], 0)

            if "Q" in em_vars:
                # Update equation for Q
                if "F" in em_vars:
                    Q_i = (A_3 - F[i + 1] @ A_1.T) / T
                else:
                    Q_i = (A_3 - A_1 @ np.linalg.inv(A_2) @ A_1.T) / T

                    # Check whether the updated estimate for Q is positive definite
                if is_pos_def(Q_i) == False:
                    Q_i = force_SPD(Q_i)

                Q = np.append(Q, [Q_i], 0)

                # Convergence check for Q
                if i >= 1 and np.all(np.abs(Q[i + 1] - Q[i]) < tol_params):
                    convergence_count += 1
            else:
                Q = np.append(Q, [Q_0], 0)

            if "H" in em_vars:
                # Update equation for H
                H = np.append(H, [A_4 @ np.linalg.inv(A_3)], 0)

                # Convergence check for H
                if i >= 1 and np.all(np.abs(H[i + 1] - H[i]) < tol_params):
                    convergence_count += 1
            else:
                H = np.append(H, [H_0], 0)

            if "R" in em_vars:
                # Update equation for R
                if "H" in em_vars:
                    R_i = (A_5 - H[i + 1] @ A_4.T) / T
                else:
                    R_i = (A_5 - A_4 @ np.linalg.inv(A_3) @ A_4.T) / T
                # Check whether the updated estimate of R is positive definite
                if is_pos_def(R_i) == False:
                    R_i = force_SPD(R_i)

                R = np.append(R, [R_i], axis=0)

                # Convergence check for R
                if i >= 1 and np.all(np.abs(R[i + 1] - R[i]) < tol_params):
                    convergence_count += 1
            else:
                R = np.append(R, [R_0], 0)

            if "xi" in em_vars:
                # Update equation for xi
                xi = np.append(xi, [x_tilde[0]], axis=0)

                # Convergence check for xi
                if i >= 1 and np.all(np.abs(xi[i + 1] - xi[i]) < tol_params):
                    convergence_count += 1
            else:
                xi = np.append(xi, [xi_0], 0)

            if "L" in em_vars:
                # Update equation for Lambda
                L = np.append(L, [P_tilde[0]], axis=0)

                # Convergence check for Lambda
                if i >= 1 and np.all(np.abs(L[i + 1] - L[i]) < tol_params):
                    convergence_count += 1
            else:
                L = np.append(L, [L_0], axis=0)

            if convergence_count == len(em_vars) + 1:
                break

        iterations = i + 1

    # One-dimensional measurements case
    elif n > 1 and p == 1:
        A_5 = 0
        for j in range(T):
            A_5 += z[j] ** 2

        for i in range(max_it):
            if i > 0 and i % 50 == 0:
                print(f"Iteration {i}")
            # E-step
            x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[i], Q[i], H[i], R[i], z, xi[i], L[i])
            S, x_tilde, P_tilde = KalmanSmoother(F[i], x_hat_minus, P_minus, x_hat, P)
            V = Lag1AutoCov(K, S, F[i], H[i], P)

            likelihood = np.append(likelihood, [ell(H[i], R[i], z, x_hat_minus, P_minus)], axis=0)

            convergence_count = 0
            if i >= 1 and likelihood[i + 1] - likelihood[i] < tol_likelihood:
                convergence_count += 1

            # M-step
            A_1 = np.zeros((n, n))
            A_2 = np.zeros((n, n))
            A_3 = np.zeros((n, n))
            A_4 = np.zeros((p, n))

            for j in range(T):
                A_1 += np.outer(x_tilde[j + 1], x_tilde[j]) + V[j]
                A_2 += np.outer(x_tilde[j], x_tilde[j]) + P_tilde[j]
                A_3 += np.outer(x_tilde[j + 1], x_tilde[j + 1]) + P_tilde[j + 1]
                A_4 += z[j] * x_tilde[j + 1]

            if "F" in em_vars:
                F = np.append(F, [A_1 @ np.linalg.inv(A_2)], 0)

                if i >= 1 and np.all(np.abs(F[i + 1] - F[i]) < tol_params):
                    convergence_count += 1
            else:
                F = np.append(F, [F_0], 0)

            if "Q" in em_vars:
                if "F" in em_vars:
                    Q_i = (A_3 - F[i + 1] @ A_1.T) / T
                else:
                    Q_i = (A_3 - A_1 @ np.linalg.inv(A_2) @ A_1.T) / T
                if is_pos_def(Q_i) == False:
                    Q_i = force_SPD(Q_i)

                Q = np.append(Q, [Q_i], 0)

                if i >= 1 and np.all(np.abs(Q[i + 1] - Q[i]) < tol_params):
                    convergence_count += 1
            else:
                Q = np.append(Q, [Q_0], 0)

            if "H" in em_vars:
                H = np.append(H, [A_4 @ np.linalg.inv(A_3)], 0)

                if i >= 1 and np.all(np.abs(H[i + 1] - H[i]) < tol_params):
                    convergence_count += 1
            else:
                H = np.append(H, [H_0], 0)

            if "R" in em_vars:
                if "H" in em_vars:
                    R_i = float((A_5 - H[i + 1] @ A_4.T) / T)
                else:
                    R_i = float((A_5 - A_4 @ np.linalg.inv(A_3) @ A_4.T) / T)

                R = np.append(R, [R_i], axis=0)
                if i >= 1 and np.abs(R[i + 1] - R[i]) < tol_params:
                    convergence_count += 1
            else:
                R = np.append(R, [R_0], 0)

            if "xi" in em_vars:
                xi = np.append(xi, [x_tilde[0]], axis=0)

                if i >= 1 and np.all(np.abs(xi[i + 1] - xi[i]) < tol_params):
                    convergence_count += 1
            else:
                xi = np.append(xi, [xi_0], 0)

            if "L" in em_vars:
                L = np.append(L, [P_tilde[0]], axis=0)

                if i >= 1 and np.all(np.abs(L[i + 1] - L[i]) < tol_params):
                    convergence_count += 1
            else:
                L = np.append(L, [L_0], axis=0)

            if convergence_count == len(em_vars) + 1:
                break

        iterations = i + 1

    # One-dimensional case
    else:
        A_5 = 0
        for j in range(T):
            A_5 += z[j] ** 2

        for i in range(max_it):
            if i > 0 and i % 50 == 0:
                print(f"Iteration {i}")
            # E-step
            x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[i], Q[i], H[i], R[i], z, xi[i], L[i])
            S, x_tilde, P_tilde = KalmanSmoother(F[i], x_hat_minus, P_minus, x_hat, P)
            V = Lag1AutoCov(K, S, F[i], H[i], P)

            likelihood = np.append(likelihood, [ell(H[i], R[i], z, x_hat_minus, P_minus)], axis=0)

            convergence_count = 0
            if i >= 1 and likelihood[i + 1] - likelihood[i] < tol_likelihood:
                convergence_count += 1

            # M-step
            A_1 = 0
            A_2 = 0
            A_3 = 0
            A_4 = 0

            for j in range(T):
                A_1 += x_tilde[j + 1] * x_tilde[j] + V[j]
                A_2 += x_tilde[j] ** 2 + P_tilde[j]
                A_3 += x_tilde[j + 1] ** 2 + P_tilde[j + 1]
                A_4 += z[j] * x_tilde[j + 1]

            if "F" in em_vars:
                F = np.append(F, [A_1 / A_2], 0)

                if i >= 1 and np.abs(F[i + 1] - F[i]) < tol_params:
                    convergence_count += 1
            else:
                F = np.append(F, [F_0], 0)

            if "Q" in em_vars:
                if "F" in em_vars:
                    Q_i = (A_3 - F[i + 1] * A_1) / T
                else:
                    Q_i = (A_3 - A_1 ** 2 / A_2) / T

                Q = np.append(Q, [Q_i], 0)

                if i >= 1 and np.abs(Q[i + 1] - Q[i]) < tol_params:
                    convergence_count += 1
            else:
                Q = np.append(Q, [Q_0], 0)

            if "H" in em_vars:
                H = np.append(H, [A_4 / A_3], 0)

                if i >= 1 and np.abs(H[i + 1] - H[i]) < tol_params:
                    convergence_count += 1
            else:
                H = np.append(H, [H_0], 0)

            if "R" in em_vars:
                if "H" in em_vars:
                    R_i = (A_5 - H[i + 1] * A_4) / T
                else:
                    R_i = (A_5 - A_4 ** 2 / A_3) / T

                R = np.append(R, [R_i], axis=0)
                if i >= 1 and np.all(np.abs(R[i + 1] - R[i]) < tol_params):
                    convergence_count += 1
            else:
                R = np.append(R, [R_0], 0)

            if "xi" in em_vars:
                xi = np.append(xi, [x_tilde[0]], axis=0)

                if i >= 1 and np.abs(xi[i + 1] - xi[i]) < tol_params:
                    convergence_count += 1
            else:
                xi = np.append(xi, [xi_0], 0)

            if "L" in em_vars:
                L = np.append(L, [P_tilde[0]], axis=0)

                if i >= 1 and np.abs(L[i + 1] - L[i]) < tol_params:
                    convergence_count += 1
            else:
                L = np.append(L, [L_0], axis=0)

            if convergence_count == len(em_vars) + 1:
                break

        iterations = i + 1

    likelihood = np.delete(likelihood, 0, axis=0)

    return (F, Q, H, R, xi, L, likelihood, iterations)


def grad(F, Q, H, R, xi, L, z, x, P, V, em_vars=["F", "Q", "H", "R", "xi", "L"]):
    T = len(z)
    n = len(x[0])
    p = len(z[0])

    A_1 = np.zeros((n, n))
    A_2 = np.zeros((n, n))
    A_3 = np.zeros((n, n))
    A_4 = np.zeros((p, n))
    A_5 = np.zeros((p, p))

    for j in range(T):
        A_1 += np.outer(x[j + 1], x[j]) + V[j]
        A_2 += np.outer(x[j], x[j]) + P[j]
        A_3 += np.outer(x[j + 1], x[j + 1]) + P[j + 1]
        A_4 += np.outer(z[j], x[j + 1])
        A_5 += np.outer(z[j], z[j])

    gradient = np.empty(1)

    if "F" in em_vars:
        gradient = np.append(gradient, np.ndarray.flatten(A_1 - F @ A_2, order='F'), axis=0)

    if "Q" in em_vars:
        gradient = np.append(gradient,
                             np.ndarray.flatten((T * Q.T - A_3 + 2 * A_1 @ F.T - F @ A_2 @ F.T) / 2, order='F'), axis=0)

    if "H" in em_vars:
        gradient = np.append(gradient, np.ndarray.flatten(A_4 - H @ A_3, order='F'), axis=0)

    if "R" in em_vars:
        gradient = np.append(gradient,
                             np.ndarray.flatten((T * R.T - A_5 + 2 * A_4 @ H.T - H @ A_3 @ H.T) / 2, order='F'), axis=0)

    if "xi" in em_vars:
        gradient = np.append(gradient, np.ndarray.flatten((x[0] - xi).T, order='F'), axis=0)

    if "L" in em_vars:
        gradient = np.append(gradient, np.ndarray.flatten(np.outer(x[0] - xi, x[0] - xi) + P[0], order='F'), axis=0)

    gradient = np.delete(gradient, 0, axis=0)
    return (gradient)


def HessianApprox(F, Q, H, R, xi, L, z, x, P, V, em_vars=["F", "Q", "H", "R", "xi", "L"], shift=0.5, scale=1000,
                  MC_size=10000):
    T = len(z)
    n = len(x[0])
    p = len(z[0])

    gen = np.random.default_rng(seed=None)

    d = 0
    for var in em_vars:
        d += np.prod(locals()[var].shape)

    Hessian = np.empty((1, d, d))

    for i in range(MC_size):
        if i % 100 == 0:
            print(f"Simulation {i}")
        delta = np.empty(1)

        if "F" in em_vars:
            delta = np.append(delta, (gen.binomial(1, 0.5, n ** 2) - shift) / scale, axis=0)
            F_per_plus = F + np.reshape(delta[len(delta) - n ** 2:], (n, n), order='F')
            F_per_minus = F - np.reshape(delta[len(delta) - n ** 2:], (n, n), order='F')
        else:
            F_per_plus = F
            F_per_minus = F

        if "Q" in em_vars:
            delta = np.append(delta, (gen.binomial(1, 0.5, n ** 2) - shift) / scale, axis=0)
            Q_per_plus = Q + np.reshape(delta[len(delta) - n ** 2:], (n, n), order='F')
            Q_per_minus = Q - np.reshape(delta[len(delta) - n ** 2:], (n, n), order='F')
        else:
            Q_per_plus = Q
            Q_per_minus = Q

        if "H" in em_vars:
            delta = np.append(delta, (gen.binomial(1, 0.5, p * n) - shift) / scale, axis=0)
            H_per_plus = H + np.reshape(delta[len(delta) - p * n:], (p, n), order='F')
            H_per_minus = H - np.reshape(delta[len(delta) - p * n:], (p, n), order='F')
        else:
            H_per_plus = H
            H_per_minus = H

        if "R" in em_vars:
            delta = np.append(delta, (gen.binomial(1, 0.5, p ** 2) - shift) / scale, axis=0)
            R_per_plus = R + np.reshape(delta[len(delta) - p ** 2:], (p, p), order='F')
            R_per_minus = R - np.reshape(delta[len(delta) - p ** 2:], (p, p), order='F')
        else:
            R_per_plus = R
            R_per_minus = R

        if "xi" in em_vars:
            delta = np.append(delta, (gen.binomial(1, 0.5, n) - shift) / scale, axis=0)
            xi_per_plus = xi + np.reshape(delta[len(delta) - n:], n, order='F')
            xi_per_minus = xi - np.reshape(delta[len(delta) - n:], n, order='F')
        else:
            xi_per_plus = xi
            xi_per_minus = xi

        if "L" in em_vars:
            delta = np.append(delta, (gen.binomial(1, 0.5, n ** 2) - shift) / scale, axis=0)
            L_per_plus = L + np.reshape(delta[len(delta) - n ** 2:], (n, n), order='F')
            L_per_minus = L - np.reshape(delta[len(delta) - n ** 2:], (n, n), order='F')
        else:
            L_per_plus = L
            L_per_minus = L

        delta = np.delete(delta, 0, axis=0)

        grad_per_plus = grad(F_per_plus, Q_per_plus, H_per_plus, R_per_plus, xi_per_plus, L_per_plus, z, x, P, V,
                             em_vars)
        grad_per_minus = grad(F_per_minus, Q_per_minus, H_per_minus, R_per_minus, xi_per_minus, L_per_minus, z, x, P, V,
                              em_vars)
        delta_inv = 1 / delta

        grad_diff = grad_per_plus - grad_per_minus
        Hessian = np.append(Hessian,
                            [0.5 * (np.outer(grad_diff / 2, delta_inv) + np.outer(grad_diff / 2, delta_inv).T)], axis=0)

    Hessian = np.delete(Hessian, 0, axis=0)

    return (np.mean(Hessian, axis=0))


## SIMULATION STUDY

## STUDY 1

T = 100
n = 2
p = n

state = 1
gen = np.random.default_rng(seed=state)

# True parameters
F_sim1 = np.identity(n)
Q_sim1 = np.identity(n) / 10
H_sim1 = np.identity(p)
R_sim1 = np.identity(p) / 10
xi_sim1 = np.zeros(n)
L_sim1 = np.identity(n) / 10

# Defining the true hidden state and the measurements
x_sim1 = np.array([gen.multivariate_normal(xi_sim1, L_sim1)])
z_sim1 = np.empty((1, p))

for t in range(T):
    # Generating the true hidden state and measurement trajectory
    x_sim1 = np.append(x_sim1, [F_sim1 @ x_sim1[t] + gen.multivariate_normal(np.zeros(n), Q_sim1)], axis=0)
    z_sim1 = np.append(z_sim1, [H_sim1 @ x_sim1[t + 1] + gen.multivariate_normal(np.zeros(p), R_sim1)], axis=0)

z_sim1 = np.delete(z_sim1, 0, axis=0)

# Plotting the generated true hidden state
for i in range(n):
    plt.plot(x_sim1[:, i], lw=0.75, label=f"$x_{{t{i + 1}}}$")
plt.title("The Simulated Hidden State")
plt.xlabel(f"$t$")
plt.ylabel(f"$x_t$", rotation=0)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.figure()

# Plotting the generated measurements
for i in range(p):
    plt.plot(np.arange(1, T + 1), z_sim1[:, i], lw=0.75, label=f"$z_{{t{i + 1}}}$")
plt.title("The Simulated Measurements")
plt.xlabel(f"$t$")
plt.ylabel(f"$z_t$", rotation=0)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Applying the Kalman- filter and smoother with the true parameters
x_hat_minus_sim1, P_minus_sim1, K_sim1, x_hat_sim1, P_sim1 = KalmanFilter(F_sim1, Q_sim1, H_sim1, R_sim1, z_sim1,
                                                                          xi_sim1, L_sim1)
S_sim1, x_tilde_sim1, P_tilde_sim1 = KalmanSmoother(F_sim1, x_hat_minus_sim1, P_minus_sim1, x_hat_sim1, P_sim1)

# Applying EMKF with the true parameters as initial estiamates
MLE_sim1 = EMKF(F_sim1, Q_sim1, H_sim1, R_sim1, z_sim1, xi_sim1, L_sim1)

it = MLE_sim1[7]
print(it)
F_MLE_sim1 = MLE_sim1[0][it]
Q_MLE_sim1 = MLE_sim1[1][it]
H_MLE_sim1 = MLE_sim1[2][it]
R_MLE_sim1 = MLE_sim1[3][it]
R_MLE_sim1 = MLE_sim1[3][it]
xi_MLE_sim1 = MLE_sim1[4][it]
L_MLE_sim1 = MLE_sim1[5][it]

ell2 = "\ell"
theta = "\u03B8"
bold_z = "\mathbf{{z}}"
## STUDY 2

T = 1000
n = 3
p = 2

state = 1
gen = np.random.default_rng(seed=state)

F_sim2 = np.diag(np.repeat(0.9, n)) + gen.uniform(low=-0.05, high=0.05, size=(n, n))
Q_sim2 = make_spd_matrix(n, random_state=state) / 1000
H_sim2 = gen.uniform(low=-1, high=1, size=(p, n))
R_sim2 = make_spd_matrix(p, random_state=state) / 1000
xi_sim2 = gen.uniform(low=-0.2, high=0.2, size=n)
L_sim2 = make_spd_matrix(n, random_state=state) / 100

x_sim2 = np.array([gen.multivariate_normal(xi_sim2, L_sim2)])
z_sim2 = np.empty((1, p))

for t in range(T):
    x_sim2 = np.append(x_sim2, [F_sim2 @ x_sim2[t] + gen.multivariate_normal(np.zeros(n), Q_sim2)], axis=0)
    z_sim2 = np.append(z_sim2, [H_sim2 @ x_sim2[t + 1] + gen.multivariate_normal(np.zeros(p), R_sim2)], axis=0)

z_sim2 = np.delete(z_sim2, 0, axis=0)

print(f"F = {F_sim2}")
print(f"Q = {Q_sim2}")
print(f"H = {H_sim2}")
print(f"R = {R_sim2}")
print(f"xi = {xi_sim2}")
print(f"Lambda = {L_sim2}")

for i in range(n):
    plot_x = plt.plot(x_sim2[:, i], lw=0.75, label=f"$x_{{t{i + 1}}}$")
plt.title("The Simulated Hidden State")
plt.xlabel(f"$t$")
plt.ylabel(f"$x_t$", rotation=0)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.figure()

for i in range(p):
    plt.plot(np.arange(1, T + 1), z_sim2[:, i], lw=0.75, label=f"$z_{{t{i + 1}}}$")
plt.title("The Simulated Measurements")
plt.xlabel(f"$t$")
plt.ylabel(f"$z_t$", rotation=0)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

x_hat_minus_sim2, P_minus_sim2, K_sim2, x_hat_sim2, P_sim2 = KalmanFilter(F_sim2, Q_sim2, H_sim2, R_sim2, z_sim2,
                                                                          xi_sim2, L_sim2)
S_sim2, x_tilde_sim2, P_tilde_sim2 = KalmanSmoother(F_sim2, x_hat_minus_sim2, P_minus_sim2, x_hat_sim2, P_sim2)

for i in range(n):
    plt.plot(x_sim2[:, i], lw=0.75, label=f"$x_{{t{i + 1}}}$")
    plt.plot(x_hat_sim2[:, i], lw=0.75, label=r"$\hat{x}_"f"{{t{i + 1}}}$")
    ci = 1.96 * np.sqrt(P_sim2[:, i, i])
    plt.fill_between(np.arange(T + 1), (x_hat_sim2[:, i] - ci), (x_hat_sim2[:, i] + ci), color='#A60628', alpha=.25)
    plt.title(f"Estimation of Hidden State {i + 1}")
    plt.xlabel(f"$t$")
    plt.ylabel(f"$x_t$", rotation=0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.figure()
    print(mean_squared_error(x_sim2[:, i], x_hat_sim2[:, i]))

for i in range(n):
    plt.plot(x_sim2[:, i], lw=0.75, label=f"$x_{{t{i + 1}}}$")
    plt.plot(x_tilde_sim2[:, i], lw=0.75, label=r"$\tilde{x}_"f"{{t{i + 1}}}$")
    ci = 1.96 * np.sqrt(P_tilde_sim2[:, i, i])
    plt.fill_between(np.arange(T + 1), (x_tilde_sim2[:, i] - ci), (x_tilde_sim2[:, i] + ci), color='#A60628', alpha=.25)
    plt.title(f"Smoothed Estimation of Hidden State {i + 1}")
    plt.xlabel(f"$t$")
    plt.ylabel(f"$x_t$", rotation=0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.figure()
    print(mean_squared_error(x_sim2[:, i], x_tilde_sim2[:, i]))

print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# Perturb the true parameters
F_0_sim2 = F_sim2 + gen.uniform(low=-0.1, high=0.1, size=(n, n))
Q_0_sim2 = gen.uniform(0.9, 1.1) * Q_sim2
H_0_sim2 = H_sim2 + gen.uniform(low=-0.1, high=0.1, size=(p, n))
R_0_sim2 = gen.uniform(0.9, 1.1) * R_sim2
xi_0_sim2 = xi_sim2 + gen.uniform(low=-0.005, high=0.005, size=n)
L_0_sim2 = gen.uniform(0.9, 1.1) * L_sim2

MLE_sim2 = EMKF(F_0_sim2, Q_0_sim2, H_0_sim2, R_0_sim2, z_sim2, xi_0_sim2, L_0_sim2)

it = MLE_sim2[7]
print(it)
F_MLE_sim2 = MLE_sim2[0][it]
Q_MLE_sim2 = MLE_sim2[1][it]
H_MLE_sim2 = MLE_sim2[2][it]
R_MLE_sim2 = MLE_sim2[3][it]
R_MLE_sim2 = MLE_sim2[3][it]
xi_MLE_sim2 = MLE_sim2[4][it]
L_MLE_sim2 = MLE_sim2[5][it]

print(f"diff F = {F_MLE_sim2 - F_sim2}")
print(f"diff Q = {Q_MLE_sim2 - Q_sim2}")
print(f"diff H = {H_MLE_sim2 - H_sim2}")
print(f"diff R = {R_MLE_sim2 - R_sim2}")
print(f"diff xi = {xi_MLE_sim2 - xi_sim2}")
print(f"diff L = {L_MLE_sim2 - L_sim2}")
print(np.linalg.norm(F_MLE_sim2 - F_sim2, ord='fro'))
print(np.linalg.norm(Q_MLE_sim2 - Q_sim2, ord='fro'))
print(np.linalg.norm(H_MLE_sim2 - H_sim2, ord='fro'))
print(np.linalg.norm(R_MLE_sim2 - R_sim2, ord='fro'))
print(np.linalg.norm(xi_MLE_sim2 - xi_sim2))
print(np.linalg.norm(L_MLE_sim2 - L_sim2, ord='fro'))

plt.figure()
plt.plot(MLE_sim2[6])
plt.title("Log-likelihood of the Measurements")
plt.xlabel("Iteration")
ell2 = "\ell"
theta = "\u03B8"
bold_z = "\mathbf{{z}}"
plt.ylabel(f"${ell2}$", rotation=0)

names = ["F", "Q", "H", "R", "xi", "Lambda"]

# Plotting entry-wise convergence of the parameter estimates
for k in range(len(MLE_sim2) - 2):
    plt.figure()
    if k == 0 or k == 2:
        for i in range(MLE_sim2[k][0].shape[0]):
            for j in range(MLE_sim2[k][0].shape[1]):
                plt.plot(MLE_sim2[k][:, i, j], label=f"${names[k]}_{{{i + 1}{j + 1}}}$")
        plt.title(f"${names[k]}$")
        plt.xlabel("Iteration")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    elif k == 1 or k == 3:
        for i in range(MLE_sim2[k][0].shape[0]):
            for j in range(i + 1):
                plt.plot(MLE_sim2[k][:, i, j], label=f"${names[k]}_{{{i + 1}{j + 1}}}$")
        plt.title(f"${names[k]}$")
        plt.xlabel("Iteration")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    elif k == 4:
        xi_code = "\u03BE"
        for i in range(MLE_sim2[k][0].shape[0]):
            plt.plot(MLE_sim2[k][:, i], label=f"{xi_code}$_{i + 1}$")
        plt.title(f"{xi_code}")
        plt.xlabel("Iteration")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        Lambda_code = "\u039B"
        for i in range(MLE_sim2[k][0].shape[0]):
            for j in range(i + 1):
                plt.plot(MLE_sim2[k][:, i, j], label=f"{Lambda_code}$_{{{i + 1}{j + 1}}}$")
        plt.title(f"{Lambda_code}")
        plt.xlabel("Iteration")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
## EXAMPLE: Example 6.8 from Time Series and its Applications, Shumway and Stoffer (2017)

T = 100

f_E2 = 0.8
q_E2 = 1
h_E2 = 1
r_E2 = 1
xi_E2 = 0
l_E2 = r_E2 / (1 - f_E2 ** 2)

x_E2 = np.array(
    [-1.48536717, 0.55611786, 0.81098876, 0.58198108, 0.74819733, 1.16625304, -0.34621346, 0.15839804, -0.43878255,
     -1.27433740, 0.14548409, 1.15845595, 2.03059312, 1.60589954, 0.13776196, -1.29796995, -1.32070469, -1.47433377,
     -0.18280350, -0.25252858, -0.27142502, 0.73256109, 0.16951606, 1.10961326, 0.94998203, 1.29840768, -1.02609711,
     -0.38444603, -0.46778352, -1.01714954, 0.17157892, -1.09131019, -0.78782349, -1.83435262, -1.84432986, -0.11181531,
     -0.34233500, 0.73262198, 1.02324673, 2.47713581, 2.00939386, 0.62879551, 1.78676555, 0.29967083, 1.28639439,
     0.45118331, 0.06715235, -0.18805455, 0.06950418, -1.59535186, -0.79808080, -1.44374704, -1.83432717, 0.13987709,
     -2.48358925, -1.69672316, 0.02628140, -1.48902914, -1.86852187, -1.79278914, -2.95335067, -3.27451586, -3.45549335,
     -2.98154414, -3.45626762, -1.81996610, -0.32797608, -1.54102374, -0.77518771, -1.08911748, -0.75732814,
     -2.73085520, -3.95112008, -1.93630530, -1.37118823, -0.83262658, -0.47326733, 0.69143681, 0.77856347, 0.17830003,
     -0.02257237, 0.28286559, -0.22620858, -0.95109335, -0.15872756, -1.44181828, -2.03571804, -2.53298797, -0.94778013,
     -0.80688583, 0.77498781, 0.53807713, 1.32420866, 1.57634707, 0.67544987, 1.63105335, 1.40251480, 0.29897036,
     1.24797924, 1.99353481, 3.00468407])
z_E2 = np.array(
    [0.66036390, 1.57150402, 1.47781865, 0.65038867, 0.99769674, 1.13375776, 1.57269715, -1.01821000, -1.21081015,
     0.17759712, 0.19598912, 1.53609156, 0.42982525, 1.17053789, 0.13294586, -0.44848732, -2.13610480, -0.30855829,
     -0.86710284, -0.57248573, 0.95243452, 0.36919170, 1.12825663, 0.78560276, -0.03529483, -0.60347080, -1.58118042,
     -0.27610934, -1.70997969, 0.63782066, -0.27420433, -0.10451360, -2.31334478, -1.41120264, -1.94991989, -0.41327865,
     1.18065103, 0.80120589, 2.04530768, 2.86883629, 0.44435485, 1.62997579, -0.62720584, 1.77412792, 0.83881532,
     -0.92459220, -1.00945110, 0.13408006, -3.07751775, 0.72319888, -1.04155172, -1.36542580, 1.05971365, -1.71920838,
     -1.56730187, -0.66014813, -1.09825872, -2.59271482, -0.22317913, -3.17365379, -2.98702309, -5.33537370,
     -4.91623055, -2.83902187, -3.00231787, -0.05061561, -2.24796983, -0.35489788, -0.35210939, -1.88766166,
     -3.58840598, -4.38685520, -0.87709202, -3.57479724, 0.66317866, 0.98440079, -0.31336108, 1.95956079, 1.00954968,
     0.02830260, 0.96104463, 0.22581783, -2.46029034, 0.59854055, -2.59410509, -1.89493662, -4.07474021, -2.89897110,
     -1.40001630, 1.67388366, 2.26214104, 3.55610879, 2.43946840, -0.34929684, 1.92444965, 2.32745827, -0.50747941,
     -1.34046713, 2.23099767, 3.41580131])

f_E2_0 = 0.8371595
q_E2_0 = 0.6335816
r_E2_0 = 1.244884
xi_E2_0 = 0
l_E2_0 = 2.8

MLE_E2 = EMKF(f_E2_0, q_E2_0, h_E2, r_E2_0, z_E2, xi_E2_0, l_E2_0, em_vars=["F", "Q", "R", "xi", "L"])
max_it = MLE_E2[7]
print(max_it)

plt.figure()
plt.plot(MLE_E2[6])

xi_code = "\u03BE"
lambda_code = "\u03BB"
plt.figure()
plt.plot(MLE_E2[0], label=f"$f$")
plt.plot(MLE_E2[1], label=f"$q^2$")
plt.plot(MLE_E2[3], label=f"$r^2$")
plt.plot(MLE_E2[4], label=f"${xi_code}$")
plt.plot(MLE_E2[5], label=f"${lambda_code}^2$")
plt.title("Parameter Estimates for the Example")  # CHANGE LATER
plt.xticks(np.arange(max_it + 1, step=5))
plt.xlabel("Iteration")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

f_E2_MLE = MLE_E2[0][max_it]
q_E2_MLE = MLE_E2[1][max_it]
r_E2_MLE = MLE_E2[3][max_it]
xi_E2_MLE = MLE_E2[4][max_it]
l_E2_MLE = MLE_E2[5][max_it]

print(f"f: {f_E2_MLE}")
print(f"q: {q_E2_MLE}")
print(f"r: {r_E2_MLE}")
print(f"xi: {xi_E2_MLE}")
print(f"l: {l_E2_MLE}")

print(f"diff f: {f_E2_MLE - f_E2}")
print(f"diff q: {q_E2_MLE - q_E2}")
print(f"diff r: {r_E2_MLE - r_E2}")
print(f"diff f: {xi_E2_MLE - xi_E2}")
print(f"diff f: {l_E2_MLE - l_E2}")
## ANALYSIS

# Importing the data
apple = yf.download('AAPL', '2016-01-01', '2019-08-01')
netflix = yf.download('NFLX', '2016-01-01', '2019-08-01')
tesla = yf.download('TSLA', '2016-01-01', '2019-08-01')
google = yf.download('GOOGL', '2016-01-01', '2019-08-01')

# Processing the data
data = pd.concat((apple["Adj Close"], tesla["Adj Close"], netflix["Adj Close"], google["Adj Close"]),
                 keys=["Apple", "Tesla", "Netflix", "Google"], axis=1)
colors = {"Apple": "dimgray", "Tesla": "lightsteelblue", "Netflix": "firebrick", "Google": "mediumblue"}
names = ["Apple", "Tesla", "Netflix", "Google"]
T_train = 850
train = data.iloc[:T_train]
test = data.iloc[T_train:]
p = data.shape[1]

# Plotting the data
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4))
data.plot(kind="line", lw=0.75, color=['dimgray', 'lightsteelblue', 'firebrick', 'mediumblue'], legend=False,
          ax=axes[0])
axes[0].set_title("The Stock Price Data")
data[["Apple", "Tesla", "Google"]].plot(kind="line", lw=0.75, color=['dimgray', 'lightsteelblue', 'mediumblue'],
                                        legend=False, ax=axes[1])
axes[1].set_title("The Stock Price Data Excluding Netflix")
fig.supylabel("Stock Price (€)")
plt.tight_layout()
fig.legend(["Apple", "Tesla", "Netflix", "Google"], loc='center left', bbox_to_anchor=(1, 0.5))

# Standardizing the training data
z = train.to_numpy()
train_means = np.mean(z, axis=0)
train_std = np.std(z, axis=0)
z = (z - train_means) / train_std
T = len(z)

# Plotting the standardized data
plt.figure()
for i in range(p):
    plt.plot(z[:, i], lw=0.75, color=colors[names[i]], label=names[i])
fig.autofmt_xdate()
plt.title("The Standardized Stock Price Data")
plt.xlabel(f"$t$")
plt.ylabel("€", rotation=0)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Standardizing the test data
z_test = test.to_numpy()
test_means = np.mean(z_test, axis=0)
test_std = np.std(z_test, axis=0)
z_test = (z_test - test_means) / test_std
# Exploratory Analysis

diff = np.empty((1, T - 1))

# Plots for first differences
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.8, 6), sharex=True)
for i in range(p):
    # Augmented Dickey-Fuller test for stationarity
    print(f"({i + 1}, ADF): {adfuller(z[:, i], regression='ct')[0:3]}")
    diff = np.append(diff, [np.diff(z[:, i], axis=0)], axis=0)
    axes.flatten()[i].plot(diff[i + 1], lw=0.75, color=colors[names[i]])
    if i < 2:
        axes.flatten()[i].set_title(names[i])
    else:
        axes.flatten()[i].set_title(names[i], y=-0.2)
fig.supylabel("€", rotation=0)
plt.tight_layout()
diff = np.delete(diff, 0, axis=0)


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff


# Computing and plotting the weekly differences
diff = np.empty((1, T - 5))
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.8, 6), sharex=True)
for i in range(p):
    diff = np.append(diff, [difference(z[:, i], 5)], axis=0)
    axes.flatten()[i].plot(diff[i + 1], lw=0.75, color=colors[names[i]])
    if i < 2:
        axes.flatten()[i].set_title(names[i])
    else:
        axes.flatten()[i].set_title(names[i], y=-0.2)
fig.supylabel("€", rotation=0)
plt.tight_layout()

# Computing and plotting the monthly differences
diff = np.empty((1, T - 20))
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.8, 6), sharex=True)
for i in range(p):
    diff = np.append(diff, [difference(z[:, i], 20)], axis=0)
    axes.flatten()[i].plot(diff[i + 1], lw=0.75, color=colors[names[i]])
    if i < 2:
        axes.flatten()[i].set_title(names[i])
    else:
        axes.flatten()[i].set_title(names[i], y=-0.2)
fig.supylabel("€", rotation=0)
plt.tight_layout()

# Computing and plotting the annual differences
diff = np.empty((1, T - 260))
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.8, 6), sharex=True)
for i in range(p):
    diff = np.append(diff, [difference(z[:, i], 260)], axis=0)
    axes.flatten()[i].plot(diff[i + 1], lw=0.75, color=colors[names[i]])
    if i < 2:
        axes.flatten()[i].set_title(names[i])
    else:
        axes.flatten()[i].set_title(names[i], y=-0.2)
fig.supylabel("€", rotation=0)
plt.tight_layout()
## Modelling and Estimation

max_it = 2
runs = 2
MC = 3

MLE_filter = []
MLE_smoother = []
V = []
MLE = []
likelihoods = []

F = []
Q_MLE = []
H_MLE = []
R_MLE = []
xi_MLE = []
L_MLE = []

for n in [2, 4, 6, 8]:
    print(n)
    n_half = int(n / 2)
    # Defining F
    F_n = np.identity(n)
    for i in range(n):
        F_n[:n_half, n_half:] = np.identity(n_half)
    F.append(F_n)

    # Defining the remainder parameter estimates
    Q_0 = np.identity(n) / 10

    H_0 = np.zeros((p, n))
    H_0[:, :n_half] = np.ones((p, n_half)) / n_half
    if n == 8:
        H_0[:p, :p] = np.identity(p)
        H = H_0

    R_0 = np.zeros((p, p))
    for i in range(p):
        R_0[i, i] = np.var(z[:, i])

    xi_0 = np.zeros(n)
    if n != 8:
        xi_0[0:n_half] = np.mean(z[:100])
        xi_0[n_half:] = (np.max(z) - np.min(z)) / T
    else:
        for i in range(p):
            xi_0[i] = np.mean(z[:100, i])
            xi_0[i + p] = (np.max(z[:, i]) - np.min(z[:, i])) / T

    L_0 = np.identity(n) * 100

    MLE.append([])
    for i in range(runs):
        if i % 10 == 0:
            print(i)
        # Perturbing the initial parameter estimates for each run of EMKF
        Q_0_EM = Q_0 * gen.uniform(low=0.4, high=1.6)
        R_0_EM = R_0 * gen.uniform(low=0.4, high=1.6)
        xi_0_EM = np.zeros(n)
        for j in range(n):
            xi_0_EM[j] = xi_0[j] + gen.uniform(-np.amin(xi_0[j]), np.amin(xi_0[j]))
        L_0_EM = L_0 * gen.uniform(low=0.4, high=1.6)
        if n != 8:
            H_0_EM = H_0 * gen.uniform(low=0.3, high=3, size=(p, n))
            MLE[n_half - 1].append(EMKF(F[n_half - 1], Q_0_EM, H_0_EM, R_0_EM, z, xi_0_EM, L_0_EM, max_it,
                                        em_vars=["Q", "H", "R", "xi", "L"]))
        else:
            MLE[n_half - 1].append(
                EMKF(F[n_half - 1], Q_0_EM, H, R_0_EM, z, xi_0_EM, L_0_EM, max_it, em_vars=["Q", "R", "xi", "L"]))

    # Extracting the log-likelihood value
    likelihoods.append([])
    for i in range(runs):
        if MLE[n_half - 1][i][7] < max_it:
            likelihoods[n_half - 1].append(MLE[n_half - 1][i][6][MLE[n_half - 1][i][7] - 1])

    maxer = np.argmax(likelihoods[n_half - 1])
    it = MLE[n_half - 1][maxer][7]

    Q_MLE.append(MLE[n_half - 1][maxer][1][it])
    H_MLE.append(MLE[n_half - 1][maxer][2][it])
    R_MLE.append(MLE[n_half - 1][maxer][3][it])
    xi_MLE.append(MLE[n_half - 1][maxer][4][it])
    L_MLE.append(MLE[n_half - 1][maxer][5][it])

    # Applying the Kalman- filter and smoother with the maximum likelihood estimates
    if n != 8:
        x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[n_half - 1], Q_MLE[n_half - 1], H_MLE[n_half - 1],
                                                         R_MLE[n_half - 1], z, xi_MLE[n_half - 1], L_MLE[n_half - 1])
        S, x_tilde, P_tilde = KalmanSmoother(F[n_half - 1], x_hat_minus, P_minus, x_hat, P)
        V.append(Lag1AutoCov(K, S, F[n_half - 1], H_MLE[n_half - 1], P))
    else:
        x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[n_half - 1], Q_MLE[n_half - 1], H, R_MLE[n_half - 1], z,
                                                         xi_MLE[n_half - 1], L_MLE[n_half - 1])
        S, x_tilde, P_tilde = KalmanSmoother(F[n_half - 1], x_hat_minus, P_minus, x_hat, P)
        V.append(Lag1AutoCov(K, S, F[n_half - 1], H, P))

    MLE_filter.append([x_hat_minus, P_minus, K, x_hat, P])
    MLE_smoother.append([S, x_tilde, P_tilde])
# Hessian computation for all likelihoods until a local maximum is idenitfied
likelihoods = []
index = []
j = []

Hess = []

MC = 1000

for n in [2, 4, 6, 8]:
    print(n)
    n_half = int(n / 2)

    # Sorting the log-likelihoods descending
    likelihoods.append([])
    j.append([])
    for i in range(runs):
        if MLE[n_half - 1][i][7] < max_it:
            likelihoods[n_half - 1].append(MLE[n_half - 1][i][6][MLE[n_half - 1][i][7] - 1])
            j[n_half - 1].append(i)
    for i in range(len(likelihoods[n_half - 1])):
        likelihoods[n_half - 1][i] = [likelihoods[n_half - 1][i], j[n_half - 1][i]]
    likelihoods[n_half - 1].sort(reverse=True)
    index.append([])
    for x in likelihoods[n_half - 1]:
        index[n_half - 1].append(x[1])

    Hess.append([])
    i = 0
    local_max = False
    while local_max == False:
        # Computing the Hessian for each likelihood value until a local maximum is identified
        if i > 0 and i % 20 == 0:
            print(i, n)
        j = likelihoods[n_half - 1][i][1]
        it = MLE[n_half - 1][j][7]
        Q_MLE = MLE[n_half - 1][j][1][it]
        H_MLE = MLE[n_half - 1][j][2][it]
        R_MLE = MLE[n_half - 1][j][3][it]
        xi_MLE = MLE[n_half - 1][j][4][it]
        L_MLE = MLE[n_half - 1][j][5][it]

        if n != 8:
            x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[n_half - 1], Q_MLE, H_MLE, R_MLE, z, xi_MLE, L_MLE)
            S, x_tilde, P_tilde = KalmanSmoother(F[n_half - 1], x_hat_minus, P_minus, x_hat, P)
            V2 = Lag1AutoCov(K, S, F[n_half - 1], H_MLE, P)
        else:
            x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[n_half - 1], Q_MLE, H, R_MLE, z, xi_MLE, L_MLE)
            S, x_tilde, P_tilde = KalmanSmoother(F[n_half - 1], x_hat_minus, P_minus, x_hat, P)
            V2 = Lag1AutoCov(K, S, F[n_half - 1], H, P)

        if n != 8:
            Hess[n_half - 1].append(
                HessianApprox(F[n_half - 1], Q_MLE, H_MLE, R_MLE, xi_MLE, L_MLE, z, x_tilde, P_tilde, V2,
                              em_vars=["Q", "H", "R", "xi", "L"], MC_size=MC))
            if is_pos_def(-Hess[n_half - 1][i]) == True:
                print(f"{n_half}) Local maximum reached with MLE entry {j}")
                local_max = True
        else:
            Hess[n_half - 1].append(
                HessianApprox(F[n_half - 1], Q_MLE, H, R_MLE, xi_MLE, L_MLE, z, x_tilde, P_tilde, V2,
                              em_vars=["Q", "R", "xi", "L"], MC_size=MC))
            if is_pos_def(-Hess[n_half - 1][i]) == True:
                print(f"{n_half}) Local maximum reached with MLE entry {j}")
                local_max = True

        if i == len(likelihoods[n_half - 1]) - 1:
            print(f"Hessians for n = {n} not negative definite: no local maximum reached")
            break
        else:
            i = i + 1
# Generating the results
for n in [2, 4, 6, 8]:
    n_half = int(n / 2)
    maxer = likelihoods[n_half - 1][0][1]
    it = MLE[n_half - 1][maxer][7]

    Q_MLE = MLE[n_half - 1][maxer][1][it]
    H_MLE = MLE[n_half - 1][maxer][2][it]
    H_mu = H_MLE[:, :n_half]
    R_MLE = MLE[n_half - 1][maxer][3][it]
    xi_MLE = MLE[n_half - 1][maxer][4][it]
    L_MLE = MLE[n_half - 1][maxer][5][it]

    print(f"Q: {Q_MLE}")
    print(f"H: {H_MLE}")
    print(f"R: {R_MLE}")
    print(f"xi: {xi_MLE}")
    print(f"Lambda: {L_MLE}")

    # Testing for observability by the Kalman rank test
    ob_matrix = H_MLE
    for i in range(n - 1):
        ob_matrix = np.append(ob_matrix, H_MLE @ np.linalg.matrix_power(F[n_half - 1], i + 1), axis=0)
    if np.linalg.matrix_rank(ob_matrix) == n:
        print("System observable")

    x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[n_half - 1], Q_MLE, H_MLE, R_MLE, z, xi_MLE, L_MLE)
    S, x_tilde, P_tilde = KalmanSmoother(F[n_half - 1], x_hat_minus, P_minus, x_hat, P)
    if n == 2 or n == 8:
        x_tilde_uns = np.array([x_tilde[j, :n_half] * train_std + train_means for j in range(1, T + 1)])
    else:
        x_tilde_uns = np.array([H_mu @ x_tilde[j, :n_half] * train_std + train_means for j in range(1, T + 1)])
    y = np.array([H_MLE @ x_tilde[j] * train_std + train_means for j in range(1, T + 1)])
    ci = np.array([1.96 * np.sqrt(P_tilde[j, 0, 0] * train_std ** 2) for j in range(1, T + 1)])

    # print(f"EV Hessian: {np.sort(np.linalg.eigvals(Hess2[n_half - 1][0]))}")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.8, 6), sharex=True)
    handles = []
    for i in range(p):
        axes.flatten()[i].plot(train.to_numpy()[:, i], lw=0.75, color=colors[names[i]], label=names[i])
        axes.flatten()[i].plot(x_tilde_uns[:, i], lw=0.75, color="green", label="Trend")
        axes.flatten()[i].fill_between(np.arange(T), (x_tilde_uns[:, i] - ci[:, i]), (x_tilde_uns[:, i] + ci[:, i]),
                                       color='#90EE90', alpha=.25)
        axes.flatten()[i].plot(y[:, i], lw=0.75, color="orange", label="Approximation")
        handles.append(axes.flatten()[i].get_legend_handles_labels()[0][0])
        print(f"{names[i]}: {mean_absolute_percentage_error(train.to_numpy()[:, i], y[:, i]) * 100}")
    handles.append(axes[0, 0].get_legend_handles_labels()[0][1])
    handles.append(axes[0, 0].get_legend_handles_labels()[0][2])  # Comment if there is no trend
    fig.supxlabel(f"$t$")
    fig.supylabel("€", rotation=0)
    fig.suptitle("Trend- and Price Estimation")
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()

    y_prior = np.array([H_MLE @ x_hat_minus[j] for j in range(T)])
    inno = z - y_prior
    plt.figure()
    for i in range(p):
        plt.plot(inno[:, i], lw=0.75, color=colors[names[i]], label=names[i])
    plt.title("Innovation Sequence")
    plt.xlabel(f"$t$")
    plt.ylabel("Innovation")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    print(f"P: {P[T]}")
    print(f"K: {K[T - 1]}")

    x_hat_minus_test, P_minus_test, K_test, x_hat_test, P_test = KalmanFilter(F[n_half - 1], Q_MLE, H_MLE, R_MLE,
                                                                              z_test, x_tilde[T], P_tilde[T])

    preds = np.array([H_MLE @ x_hat_minus_test[j] * test_std + test_means for j in range(len(test))])

    if n == 2 or n == 8:
        pred_trend = np.array(
            [x_hat_minus_test[j, :n_half] * test_std + test_means for j in range(len(x_hat_minus_test))])
    else:
        pred_trend = np.array(
            [H_mu @ x_hat_minus_test[j, :n_half] * test_std + test_means for j in range(len(x_hat_minus_test))])

    handles = []
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.8, 6), sharex=True)
    for i in range(p):
        axes.flatten()[i].plot(pred_trend[:, i], lw=0.75, label="Forecasted Trend", color="green")
        axes.flatten()[i].plot(preds[:, i], lw=0.75, label="Forecasted Price", color="orange")
        axes.flatten()[i].plot(test.to_numpy()[:, i], lw=0.75, label=names[i], color=colors[names[i]])
        handles.append(axes.flatten()[i].get_legend_handles_labels()[0][2])  # Comment if no trend
        print(f"{names[i]}: {mean_absolute_percentage_error(test.to_numpy()[:, i], preds[:, i]) * 100}")
    handles.append(axes[0, 0].get_legend_handles_labels()[0][0])
    handles.append(axes[0, 0].get_legend_handles_labels()[0][1])
    fig.supxlabel(f"$t$")
    fig.supylabel("€", rotation=0)
    fig.suptitle("Trend- and Price Forecasting")
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()