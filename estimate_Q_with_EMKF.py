import numpy as np
import matplotlib.pyplot as plt

# Include all necessary functions to avoid import issues

def is_pos_def(x):
    """Check if matrix is positive definite"""
    return np.all(np.linalg.eigvals(x) > 0)

def force_SPD(A, c=1e-10):
    """Force matrix to be positive definite"""
    A = (A + A.T)/2
    if np.amin(np.linalg.eigvals(A)) < 0:
        A += (np.abs(np.amin(np.linalg.eigvals(A))) + c)*np.identity(A.shape[0])
    if np.amin(np.linalg.eigvals(A)) == 0:
        A += c*np.identity(A.shape[0])
    return A

def KalmanFilter(F, Q, H, R, z, x_0, P_0):
    """Kalman Filter"""
    T = len(z)
    # Check dimension of hidden state
    if isinstance(x_0, np.ndarray):
        n = len(x_0)
    else:
        n = 1
    # Check dimension of observation
    if isinstance(z[0], np.ndarray):
        p = len(z[0])
    else:
        p = 1

    # Multi-dimensional case
    if n > 1 and p > 1:
        x_hat_minus = np.empty((1, n))
        x_hat = np.array([x_0])
        P_minus = np.empty((1, n, n))
        P = np.array([P_0])
        K = np.empty((1, n, p))

        for i in range(T):
            # State prediction equation
            x_hat_minus = np.append(x_hat_minus, [F @ x_hat[i]], axis=0)
            # Covariance prediction equation
            P_minus = np.append(P_minus, [F @ P[i] @ F.T + Q], axis=0)
            # Kalman gain
            K = np.append(K, [P_minus[i + 1] @ H.T @ np.linalg.inv(H @ P_minus[i + 1] @ H.T + R)], axis=0)
            # State update equation
            x_hat = np.append(x_hat, [x_hat_minus[i + 1] + K[i + 1] @ (z[i] - H @ x_hat_minus[i + 1])], axis=0)
            # Covariance update equation
            P = np.append(P, [
                (np.identity(n) - K[i + 1] @ H) @ P_minus[i + 1] @ (np.identity(n) - K[i + 1] @ H).T + K[i + 1] @ R @ K[i + 1].T], axis=0)

    # One-dimensional observation case
    elif n > 1 and p == 1:
        x_hat_minus = np.empty((1, n))
        x_hat = np.array([x_0])
        P_minus = np.empty((1, n, n))
        P = np.array([P_0])
        K = np.empty((1, n, p))

        for i in range(T):
            x_hat_minus = np.append(x_hat_minus, [F @ x_hat[i]], axis=0)
            P_minus = np.append(P_minus, [F @ P[i] @ F.T + Q], axis=0)
            K = np.append(K, [P_minus[i + 1] @ H.T @ np.linalg.inv(H @ P_minus[i + 1] @ H.T + R)], axis=0)
            x_hat = np.append(x_hat, [x_hat_minus[i + 1] + K[i + 1] @ (z[i] - H @ x_hat_minus[i + 1])], axis=0)
            P = np.append(P, [
                (np.identity(n) - K[i + 1] @ H) @ P_minus[i + 1] @ (np.identity(n) - K[i + 1] @ H).T + R * K[i + 1] @ K[i + 1].T], axis=0)

    # One-dimensional case
    else:
        x_hat_minus = np.empty(1)
        x_hat = np.array([x_0])
        P_minus = np.empty(1)
        P = np.array([P_0])
        K = np.empty(1)

        for i in range(T):
            x_hat_minus = np.append(x_hat_minus, [F * x_hat[i]], axis=0)
            P_minus = np.append(P_minus, [F ** 2 * P[i] + Q], axis=0)
            K = np.append(K, [P_minus[i + 1] * H / (H ** 2 * P_minus[i + 1] + R)], axis=0)
            x_hat = np.append(x_hat, [x_hat_minus[i + 1] + K[i + 1] * (z[i] - H * x_hat_minus[i + 1])], axis=0)
            P = np.append(P, [(1 - K[i + 1] * H) ** 2 * P_minus[i + 1] + K[i + 1] ** 2 * R], axis=0)

    x_hat_minus = np.delete(x_hat_minus, 0, axis=0)
    P_minus = np.delete(P_minus, 0, axis=0)
    K = np.delete(K, 0, axis=0)

    return x_hat_minus, P_minus, K, x_hat, P

def KalmanSmoother(F, x_hat_minus, P_minus, x_hat, P):
    """Kalman Smoother"""
    T = len(x_hat_minus)
    if isinstance(x_hat[0], np.ndarray):
        n = len(x_hat[0])
    else:
        n = 1

    # Multi-dimensional case
    if n > 1:
        x_tilde = np.array([x_hat[T]])
        P_tilde = np.array([P[T]])
        S = np.empty((1, n, n))

        for i in reversed(range(T)):
            S = np.insert(S, 0, [P[i] @ F.T @ np.linalg.inv(P_minus[i])], axis=0)
            x_tilde = np.insert(x_tilde, 0, [x_hat[i] + S[0] @ (x_tilde[0] - x_hat_minus[i])], axis=0)
            P_tilde = np.insert(P_tilde, 0, [P[i] + S[0] @ (P_tilde[0] - P_minus[i]) @ S[0].T], axis=0)

    # One-dimensional case
    else:
        x_tilde = np.array([x_hat[T]])
        P_tilde = np.array([P[T]])
        S = np.empty(1)

        for i in reversed(range(T)):
            S = np.insert(S, 0, [P[i] * F / P_minus[i]], axis=0)
            x_tilde = np.insert(x_tilde, 0, [x_hat[i] + S[0] * (x_tilde[0] - x_hat_minus[i])], axis=0)
            P_tilde = np.insert(P_tilde, 0, [P[i] + S[0] ** 2 * (P_tilde[0] - P_minus[i])], axis=0)

    S = np.delete(S, len(S) - 1, axis=0)
    return S, x_tilde, P_tilde

def Lag1AutoCov(K, S, F, H, P):
    """Lag-1 auto-covariance"""
    T = len(P) - 1
    if isinstance(F, np.ndarray):
        n = F.shape[0]
    else:
        n = 1

    # Multi-dimensional case
    if n > 1:
        V = np.array([(np.identity(n) - K[T - 1] @ H) @ F @ P[T - 1]])
        for i in reversed(range(1, T)):
            V = np.insert(V, 0, [P[i] @ S[i - 1].T + S[i] @ (V[0] - F @ P[i]) @ S[i - 1].T], axis=0)
    # One-dimensional case
    else:
        V = np.array([(1 - K[T - 1] * H) * F * P[T - 1]])
        for i in reversed(range(1, T)):
            V = np.insert(V, 0, [P[i] * S[i - 1].T + S[i] * (V[0] - F * P[i]) * S[i - 1].T], axis=0)

    return V

def ell(H, R, z, x, P):
    """Log-likelihood function"""
    T = len(z)
    if isinstance(x[0], np.ndarray):
        n = len(x[0])
    else:
        n = 1
    if isinstance(z[0], np.ndarray):
        p = len(z[0])
    else:
        p = 1

    likelihood = -T * p / 2 * np.log(2 * np.pi)

    # Multi-dimensional case
    if n > 1 and p > 1:
        for i in range(T):
            # Compute innovation covariance with numerical stability
            S = H @ P[i] @ H.T + R
            # Ensure positive definiteness for numerical stability
            if np.any(np.linalg.eigvals(S) <= 0):
                S = S + np.eye(S.shape[0]) * 1e-10
            
            det_S = np.linalg.det(S)
            if det_S <= 0:
                det_S = 1e-10  # Prevent log of non-positive values
                
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)  # Use pseudo-inverse for singular matrices
            
            innovation = z[i] - H @ x[i]
            likelihood -= 0.5 * (np.log(det_S) + innovation.T @ S_inv @ innovation)
    # One-dimensional observation case
    elif n > 1 and p == 1:
        for i in range(T):
            # Compute innovation covariance with numerical stability
            S = H @ P[i] @ H.T + R
            # Ensure positive definiteness for numerical stability
            if np.any(np.linalg.eigvals(S) <= 0):
                S = S + np.eye(S.shape[0]) * 1e-10
            
            det_S = np.linalg.det(S)
            if det_S <= 0:
                det_S = 1e-10  # Prevent log of non-positive values
            
            likelihood -= 0.5 * (np.log(det_S) + (z[i] - H @ x[i]) ** 2 / S)
        likelihood = likelihood[0][0]
    # One-dimensional case
    else:
        for i in range(T):
            # Compute innovation variance with numerical stability
            variance = H ** 2 * P[i] + R
            if variance <= 0:
                variance = 1e-10  # Prevent log of non-positive values and division by zero
            
            innovation = z[i] - H * x[i]
            likelihood -= 0.5 * (np.log(variance) + innovation ** 2 / variance)

    return likelihood

def EMKF(F_0, Q_0, H_0, R_0, z, xi_0, L_0, max_it=1000, tol_likelihood=0.01, tol_params=0.005, em_vars=["F", "Q", "H", "R", "xi", "L"]):
    """EM algorithm Kalman Filter"""
    T = len(z)
    if isinstance(xi_0, np.ndarray):
        n = len(xi_0)
    else:
        n = 1
    if isinstance(z[0], np.ndarray):
        p = len(z[0])
    else:
        p = 1

    # Initialize
    F = np.array([F_0])
    Q = np.array([Q_0])
    H = np.array([H_0])
    R = np.array([R_0])
    xi = np.array([xi_0])
    L = np.array([L_0])

    likelihood = np.empty(1)

    # Multi-dimensional case
    if n > 1 and p > 1:
        A_5 = np.zeros((p, p))
        for j in range(T):
            A_5 += np.outer(z[j], z[j])

        for i in range(max_it):

            
            # E-step
            x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[i], Q[i], H[i], R[i], z, xi[i], L[i])
            S, x_tilde, P_tilde = KalmanSmoother(F[i], x_hat_minus, P_minus, x_hat, P)
            V = Lag1AutoCov(K, S, F[i], H[i], P)

            likelihood = np.append(likelihood, [ell(H[i], R[i], z, x_hat_minus, P_minus)], axis=0)

            # Convergence check
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

                if not is_pos_def(Q_i):
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
                    R_i = (A_5 - H[i + 1] @ A_4.T) / T
                else:
                    R_i = (A_5 - A_4 @ np.linalg.inv(A_3) @ A_4.T) / T
                
                R_i = force_SPD(R_i)

                R = np.append(R, [R_i], axis=0)
                if i >= 1 and np.all(np.abs(R[i + 1] - R[i]) < tol_params):
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

    # One-dimensional measurements case
    elif n > 1 and p == 1:
        A_5 = 0
        for j in range(T):
            A_5 += z[j] ** 2

        for i in range(max_it):
            # E-step
            x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[i], Q[i], H[i], R[i], z, xi[i], L[i])
            S, x_tilde, P_tilde = KalmanSmoother(F[i], x_hat_minus, P_minus, x_hat, P)
            V = Lag1AutoCov(K, S, F[i], H[i], P)

            likelihood = np.append(likelihood, [ell(H[i], R[i], z, x_hat_minus, P_minus)], axis=0)

            # Convergence check
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

                if not is_pos_def(Q_i):
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
            # E-step
            x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[i], Q[i], H[i], R[i], z, xi[i], L[i])
            S, x_tilde, P_tilde = KalmanSmoother(F[i], x_hat_minus, P_minus, x_hat, P)
            V = Lag1AutoCov(K, S, F[i], H[i], P)

            likelihood = np.append(likelihood, [ell(H[i], R[i], z, x_hat_minus, P_minus)], axis=0)

            # Convergence check
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
                if i >= 1 and np.abs(R[i + 1] - R[i]) < tol_params:
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
    return F, Q, H, R, xi, L, likelihood, iterations

def estimate_Q_from_observations(observations, F_true=None, H_true=None, R_true=None, 
                                xi_true=None, L_true=None, Q_initial=None, 
                                estimate_params=["Q"], max_iterations=100):
    """
    Use EMKF function to estimate state noise covariance matrix Q from noisy observation data
    """
    
    # Get dimensions of observation data
    T = len(observations)
    if observations.ndim == 1:
        p = 1
        observations = observations.reshape(-1, 1)
    else:
        p = observations.shape[1]
    
    # Assume state dimension equals observation dimension
    n = p
    
    # Set default parameters
    if F_true is None:
        F_true = np.eye(n) * 0.9
    
    if H_true is None:
        H_true = np.eye(p, n)
    
    if R_true is None:
        R_true = np.eye(p) * 0.1
    
    if xi_true is None:
        xi_true = np.zeros(n)
    
    if L_true is None:
        L_true = np.eye(n) * 1.0
    
    if Q_initial is None:
        Q_initial = np.eye(n) * 0.1
    
    
    # Call EMKF function
    results = EMKF(F_true, Q_initial, H_true, R_true, observations, 
                   xi_true, L_true, max_it=max_iterations, 
                   em_vars=estimate_params)
    
    # Parse results
    F_estimates, Q_estimates, H_estimates, R_estimates, xi_estimates, L_estimates, likelihood, iterations = results
    

    
    # Get final estimates
    final_F = F_estimates[iterations]
    final_Q = Q_estimates[iterations]
    final_H = H_estimates[iterations]
    final_R = R_estimates[iterations]
    final_xi = xi_estimates[iterations]
    final_L = L_estimates[iterations]
    
    
    return {
        'Q_estimated': final_Q,
        'F_estimated': final_F,
        'H_estimated': final_H,
        'R_estimated': final_R,
        'xi_estimated': final_xi,
        'L_estimated': final_L,
        'likelihood': likelihood,
        'iterations': iterations,
        'all_estimates': results
    }

def generate_sample_data(T=100, n=2, Q_true=None, F_true=None, H_true=None, 
                        R_true=None, xi_true=None, L_true=None, seed=42):
    """Generate sample data for testing"""
    
    np.random.seed(seed)
    
    # Set true parameters
    if Q_true is None:
        Q_true = np.array([[0.1, 0.05, 0.02], 
                          [0.05, 0.1, 0.03], 
                          [0.02, 0.03, 0.08]])
    
    if F_true is None:
        F_true = np.array([[0.9, 0.1, 0.05], 
                          [0.0, 0.8, 0.1], 
                          [0.0, 0.0, 0.85]])
    
    if H_true is None:
        H_true = np.eye(n)
    
    if R_true is None:
        p = H_true.shape[0]  # Get observation dimension
        R_true = np.eye(p) * 0.05
    
    if xi_true is None:
        xi_true = np.zeros(n)
    
    if L_true is None:
        L_true = np.eye(n) * 0.1
    
    # Get observation dimension from H_true
    p = H_true.shape[0]  # Number of observations
    
    # Generate state and observation sequences
    states = np.zeros((T+1, n))
    observations = np.zeros((T, p))
    
    # Initial state
    states[0] = np.random.multivariate_normal(xi_true, L_true)
    
    # Generate state and observation sequences
    for t in range(T):
        # State transition
        process_noise = np.random.multivariate_normal(np.zeros(n), Q_true)
        states[t+1] = F_true @ states[t] + process_noise
        
        # Observation
        observation_noise = np.random.multivariate_normal(np.zeros(p), R_true)
        observations[t] = H_true @ states[t+1] + observation_noise
    
    true_params = {
        'Q_true': Q_true,
        'F_true': F_true,
        'H_true': H_true,
        'R_true': R_true,
        'xi_true': xi_true,
        'L_true': L_true
    }
    
    return observations, states[1:], true_params

def plot_results(observations, true_states, estimated_results, true_params):
    """Plot result charts"""
    
    # Use estimated parameters for Kalman filtering
    Q_est = estimated_results['Q_estimated']
    F_est = estimated_results['F_estimated']
    H_est = estimated_results['H_estimated']
    R_est = estimated_results['R_estimated']
    xi_est = estimated_results['xi_estimated']
    L_est = estimated_results['L_estimated']
    
    # Apply Kalman filter
    x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(
        F_est, Q_est, H_est, R_est, observations, xi_est, L_est
    )
    
    # Apply Kalman smoother
    S, x_tilde, P_tilde = KalmanSmoother(F_est, x_hat_minus, P_minus, x_hat, P)
    
    # Create simplified figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 3D State Trajectory Comparison
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot 3D trajectories
    ax1.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2], 
             'b-', label='True State', linewidth=2, alpha=0.8)
    ax1.plot(x_tilde[1:, 0], x_tilde[1:, 1], x_tilde[1:, 2], 
             'r--', label='Estimated State', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('State 1')
    ax1.set_ylabel('State 2')
    ax1.set_zlabel('State 3')
    ax1.set_title('3D State Trajectory Comparison')
    ax1.legend()
    
    # 2. Q Matrix Comparison
    ax2 = fig.add_subplot(132)
    
    Q_true = true_params['Q_true']
    Q_est = estimated_results['Q_estimated']
    
    # Create a visual comparison of Q matrices
    n_states = Q_true.shape[0]
    x_pos = np.arange(n_states * n_states)
    
    # Flatten matrices for comparison
    Q_true_flat = Q_true.flatten()
    Q_est_flat = Q_est.flatten()
    
    # Create labels for matrix elements
    labels = [f'Q[{i},{j}]' for i in range(n_states) for j in range(n_states)]
    
    # Bar plot comparison
    width = 0.35
    ax2.bar(x_pos - width/2, Q_true_flat, width, label='True Q', alpha=0.8, color='blue')
    ax2.bar(x_pos + width/2, Q_est_flat, width, label='Estimated Q', alpha=0.8, color='red')
    
    ax2.set_xlabel('Matrix Elements')
    ax2.set_ylabel('Values')
    ax2.set_title('Q Matrix Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Single Estimation - Show Error Metrics
    ax3 = fig.add_subplot(133)
    
    error_norm = np.linalg.norm(Q_est - Q_true, 'fro')
    ax3.text(0.1, 0.8, f'Q Matrix Error Metrics:', fontsize=12, weight='bold')
    ax3.text(0.1, 0.7, f'Frobenius Error: {error_norm:.6f}', fontsize=11)
    ax3.text(0.1, 0.6, f'Max Absolute Error: {np.max(np.abs(Q_est - Q_true)):.6f}', fontsize=11)
    ax3.text(0.1, 0.5, f'Mean Squared Error: {np.mean((Q_est - Q_true)**2):.6f}', fontsize=11)
    ax3.text(0.1, 0.4, f'Relative Error: {error_norm/np.linalg.norm(Q_true, "fro")*100:.2f}%', fontsize=11)
    
    ax3.set_title('Estimation Quality')
    ax3.axis('off')
    
    plt.tight_layout(pad=2.0)  # Add more padding for better layout
    plt.show()
    
    # Calculate estimation error
    Q_error = np.linalg.norm(Q_est - Q_true, 'fro')
    print(f"\nQ matrix estimation Frobenius norm error: {Q_error:.6f}")
    
    # Also print the matrices for reference
    print(f"\nTrue Q matrix:")
    print(Q_true)
    print(f"\nEstimated Q matrix:")
    print(Q_est)

def adaptive_Q_estimation(observations, window_size=50, overlap=0, 
                          F_true=None, H_true=None, R_true=None, 
                          xi_true=None, L_true=None, Q_initial=None,
                          max_iterations=50):
    """
    Use sliding window method for adaptive Q matrix estimation
    
    Parameters:
    observations: long observation sequence, shape (T, p)
    window_size: size of each window
    overlap: overlap length between windows
    Other parameters same as estimate_Q_from_observations
    
    Returns:
    Q_history: historical changes of Q matrix
    filtered_states: filtered states using adaptive Q estimation
    window_results: detailed results for each window
    """
    
    T = len(observations)
    if observations.ndim == 1:
        p = 1
        observations = observations.reshape(-1, 1)
    else:
        p = observations.shape[1]
    
    n = p  # Assume state dimension equals observation dimension
    
    # Set default parameters
    if F_true is None:
        F_true = np.eye(n) * 0.9
    if H_true is None:
        H_true = np.eye(p, n)
    if R_true is None:
        R_true = np.eye(p) * 0.1
    if xi_true is None:
        xi_true = np.zeros(n)
    if L_true is None:
        L_true = np.eye(n) * 1.0
    if Q_initial is None:
        Q_initial = np.eye(n) * 0.1
    
    print(f"=== Sliding Window Adaptive Q Estimation ===")
    print(f"Observation sequence length: {T}")
    print(f"Window size: {window_size}")
    print(f"Overlap length: {overlap}")
    
    # Calculate window start positions
    step_size = window_size - overlap
    window_starts = list(range(0, T - window_size + 1, step_size))
    if window_starts[-1] + window_size < T:
        window_starts.append(T - window_size)  # Ensure coverage to the end
    
    print(f"Total number of windows: {len(window_starts)}")
    
    # Store results
    Q_history = []
    window_results = []
    filtered_states = np.zeros((T+1, n))
    filtered_covariances = np.zeros((T+1, n, n))
    
    # Currently used Q matrix
    current_Q = Q_initial.copy()
    current_xi = xi_true.copy()
    current_L = L_true.copy()
    
    for i, start_idx in enumerate(window_starts):
        end_idx = min(start_idx + window_size, T)
        window_obs = observations[start_idx:end_idx]
        
        print(f"\nWindow {i+1}/{len(window_starts)}: time steps {start_idx}-{end_idx-1}")
        
        # Use current Q estimate for EMKF
        try:
            results = estimate_Q_from_observations(
                observations=window_obs,
                F_true=F_true,
                H_true=H_true,
                R_true=R_true,
                xi_true=current_xi,
                L_true=current_L,
                Q_initial=current_Q,
                estimate_params=["Q"],
                max_iterations=max_iterations
            )
            
            # Update Q matrix
            new_Q = results['Q_estimated']
            
            # Smooth update (optional)
            alpha = 0.7  # Learning rate
            current_Q = alpha * new_Q + (1 - alpha) * current_Q
            
            Q_history.append(current_Q.copy())
            window_results.append(results)
            
            print(f"Window {i+1} Q matrix:")
            print(current_Q)
            
        except Exception as e:
            print(f"Window {i+1} estimation failed: {e}")
            Q_history.append(current_Q.copy())
            window_results.append(None)
    
    # Use adaptive Q matrix to filter the entire sequence
    print(f"\nUsing adaptive Q matrix for final filtering of entire sequence...")
    
    # For simplicity, use the last Q matrix to filter the entire sequence
    final_Q = Q_history[-1] if Q_history else Q_initial
    
    try:
        x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(
            F_true, final_Q, H_true, R_true, observations, xi_true, L_true
        )
        S, x_tilde, P_tilde = KalmanSmoother(F_true, x_hat_minus, P_minus, x_hat, P)
        
        filtered_states = x_tilde
        filtered_covariances = P_tilde
        
    except Exception as e:
        print(f"Final filtering failed: {e}")
    
    return {
        'Q_history': Q_history,
        'filtered_states': filtered_states,
        'filtered_covariances': filtered_covariances,
        'window_results': window_results,
        'window_starts': window_starts,
        'final_Q': final_Q
    }


def plot_adaptive_results(observations, true_states, adaptive_results, true_params):
    """Plot adaptive estimation results"""
    
    Q_history = adaptive_results['Q_history']
    filtered_states = adaptive_results['filtered_states']
    window_starts = adaptive_results['window_starts']
    final_Q = adaptive_results['final_Q']
    
    # Create simplified figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 3D State Trajectory Comparison
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot 3D trajectories
    ax1.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2], 
             'b-', label='True State', linewidth=2, alpha=0.8)
    ax1.plot(filtered_states[1:, 0], filtered_states[1:, 1], filtered_states[1:, 2], 
             'r--', label='Adaptive Estimated State', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('State 1')
    ax1.set_ylabel('State 2')
    ax1.set_zlabel('State 3')
    ax1.set_title('3D State Trajectory Comparison')
    ax1.legend()
    
    # 2. Q Matrix Comparison (Final vs True)
    ax2 = fig.add_subplot(132)
    
    Q_true = true_params['Q_true']
    
    # Create a visual comparison of Q matrices
    n_states = Q_true.shape[0]
    x_pos = np.arange(n_states * n_states)
    
    # Flatten matrices for comparison
    Q_true_flat = Q_true.flatten()
    Q_final_flat = final_Q.flatten()
    
    # Create labels for matrix elements
    labels = [f'Q[{i},{j}]' for i in range(n_states) for j in range(n_states)]
    
    # Bar plot comparison
    width = 0.35
    ax2.bar(x_pos - width/2, Q_true_flat, width, label='True Q', alpha=0.8, color='blue')
    ax2.bar(x_pos + width/2, Q_final_flat, width, label='Final Estimated Q', alpha=0.8, color='red')
    
    ax2.set_xlabel('Matrix Elements')
    ax2.set_ylabel('Values')
    ax2.set_title('Q Matrix Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Q Estimation Error Over Windows
    ax3 = fig.add_subplot(133)
    
    if Q_history:
        Q_errors = [np.linalg.norm(Q - Q_true, 'fro') for Q in Q_history]
        ax3.plot(Q_errors, 'o-', linewidth=2, color='red', markersize=6)
        ax3.set_title('Q Estimation Error Over Windows')
        ax3.set_xlabel('Window Number')
        ax3.set_ylabel('Frobenius Norm Error')
        ax3.grid(True, alpha=0.3)
        
        # Add final error as text
        if Q_errors:
            final_error = Q_errors[-1]
            ax3.text(0.02, 0.98, f'Final Error: {final_error:.6f}', 
                    transform=ax3.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(pad=2.0)  # Add more padding for better layout
    plt.show()
    
    # Calculate final estimation error
    if Q_history:
        final_error = np.linalg.norm(final_Q - Q_true, 'fro')
        print(f"\nFinal Q estimation Frobenius norm error: {final_error:.6f}")
        
        # Also print the matrices for reference
        print(f"\nTrue Q matrix:")
        print(Q_true)
        print(f"\nFinal estimated Q matrix:")
        print(final_Q)


def sliding_window_EMKF(observations, true_states, F_true, H_true, R_true, 
                        window_size=50, overlap=20, Q_initial=None, 
                        max_iterations=50, verbose=False, true_init_state=None,
                        allStates=True, init_covariance=None):
    """
    Sliding Window EMKF for adaptive Q matrix estimation
    
    Parameters:
    observations: observation data, shape (T, p) where T is time steps, p is observation dimension
    true_states: true state data, shape (T, n) where n is state dimension  
    F_true: state transition matrix (n, n)
    H_true: observation matrix (p, n)
    R_true: observation noise covariance matrix (p, p) or scalar
    window_size: size of each sliding window
    overlap: overlap length between windows
    Q_initial: initial Q matrix estimate (n, n), if None uses identity
    max_iterations: max EM iterations per window
    verbose: whether to print detailed progress
    true_init_state: true initial state, if None uses first true state
    allStates: if True, compute MSE on all states; if False, only position (same as KF test)
    init_covariance: initial covariance matrix (n, n), if None uses identity
    
    Returns:
    dict containing:
    - 'mse_loss': Mean Squared Error between estimated and true states
    - 'final_Q': Final estimated Q matrix
    - 'filtered_states': All filtered states using adaptive Q
    - 'Q_history': History of Q estimates per window
    """
    
    # Convert inputs to numpy if they're tensors
    if hasattr(observations, 'cpu'):
        observations = observations.cpu().numpy()
    if hasattr(true_states, 'cpu'):
        true_states = true_states.cpu().numpy()
    if hasattr(F_true, 'cpu'):
        F_true = F_true.cpu().numpy()
    if hasattr(H_true, 'cpu'):
        H_true = H_true.cpu().numpy()
    if hasattr(R_true, 'cpu'):
        R_true = R_true.cpu().numpy()
    
    # Get dimensions
    T = len(observations)
    if observations.ndim == 1:
        p = 1
        observations = observations.reshape(-1, 1)
    else:
        p = observations.shape[1]
    
    n = true_states.shape[1]  # State dimension from true states
    
    # Fix R_true to be matrix form
    if np.isscalar(R_true):
        R_true = np.array([[R_true]])
    elif R_true.ndim == 0:
        R_true = R_true.reshape(1, 1)
    
    # Set default parameters
    if Q_initial is None:
        Q_initial = np.eye(n) * 0.1
    
    # Use true initial state instead of zeros
    if true_init_state is not None:
        if hasattr(true_init_state, 'cpu'):
            xi_true = true_init_state.cpu().numpy().flatten()
        else:
            xi_true = true_init_state.flatten()
    else:
        xi_true = true_states[0]  # Use first true state as initial
    
    # 🔥 使用与KF测试相同的初始协方差矩阵
    if init_covariance is not None:
        if hasattr(init_covariance, 'cpu'):
            L_true = init_covariance.cpu().numpy()
        else:
            L_true = init_covariance.copy()
        
        # 处理零协方差矩阵的情况
        if np.allclose(L_true, 0):
            if verbose:
                print("⚠️  检测到零初始协方差矩阵，添加小的正则化项")
            L_true = np.eye(n) * 1e-6  # 添加小的正则化项避免数值问题
    else:
        L_true = np.eye(n) * 1.0  # 默认初始协方差
    
    if verbose:
        print(f"=== Sliding Window EMKF ===")
        print(f"Observation sequence length: {T}")
        print(f"State dimension: {n}, Observation dimension: {p}")
        print(f"Window size: {window_size}, Overlap: {overlap}")
        print(f"Initial state: {xi_true}")
        print(f"Initial covariance matrix:\n{L_true}")
        print(f"R matrix shape: {R_true.shape}")
        print(f"AllStates: {allStates}")
    
    # Use larger, less frequent windows for stability
    effective_window_size = max(window_size, min(50, T//2))
    effective_overlap = min(overlap, effective_window_size//3)
    
    # Calculate window start positions
    step_size = effective_window_size - effective_overlap
    window_starts = list(range(0, T - effective_window_size + 1, step_size))
    if len(window_starts) == 0 or window_starts[-1] + effective_window_size < T:
        window_starts.append(max(0, T - effective_window_size))  # Ensure coverage to the end
    
    if verbose:
        print(f"Adjusted window size: {effective_window_size}, overlap: {effective_overlap}")
        print(f"Total number of windows: {len(window_starts)}")
    
    # Store results
    Q_history = []
    current_Q = Q_initial.copy()
    
    for i, start_idx in enumerate(window_starts):
        end_idx = min(start_idx + effective_window_size, T)
        window_obs = observations[start_idx:end_idx]
        
        if verbose:
            print(f"Window {i+1}/{len(window_starts)}: time steps {start_idx}-{end_idx-1}")
        
        # Use current Q estimate for EMKF
        try:
            results = estimate_Q_from_observations(
                observations=window_obs,
                F_true=F_true,
                H_true=H_true,
                R_true=R_true,
                xi_true=xi_true,
                L_true=L_true,
                Q_initial=current_Q,
                estimate_params=["Q"],
                max_iterations=max_iterations
            )
            
            # Update Q matrix with conservative smoothing
            new_Q = results['Q_estimated']
            alpha = 0.3  # More conservative learning rate
            current_Q = alpha * new_Q + (1 - alpha) * current_Q
            
            Q_history.append(current_Q.copy())
            
            if verbose:
                print(f"Window {i+1} Q matrix diagonal: {np.diag(current_Q)}")
            
        except Exception as e:
            if verbose:
                print(f"Window {i+1} estimation failed: {e}")
            Q_history.append(current_Q.copy())
    
    # Use final Q matrix to filter the entire sequence
    final_Q = Q_history[-1] if Q_history else Q_initial
    
    if verbose:
        print(f"Final filtering with adaptive Q matrix...")
        print(f"Final Q matrix:\n{final_Q}")
    
    try:
        # Apply Kalman filter with final Q estimate
        x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(
            F_true, final_Q, H_true, R_true, observations, xi_true, L_true
        )
        # Apply Kalman smoother for best estimates
        S, x_tilde, P_tilde = KalmanSmoother(F_true, x_hat_minus, P_minus, x_hat, P)
        
        # Use smoothed estimates (x_tilde[1:] to align with true_states)
        filtered_states = x_tilde[1:]  # Remove initial state
        
        # **修复MSE计算，确保与KF测试完全一致**
        if allStates:
            # 计算所有状态的MSE (与KF测试的allStates=True一致)
            mse_linear = np.mean((filtered_states - true_states) ** 2)
            mse_loss = 10 * np.log10(mse_linear)
        else:
            # 只计算位置的MSE (与KF测试的allStates=False一致)
            # 使用与KF测试相同的mask: loc = [True, False, False]
            position_mse = np.mean((filtered_states[:, 0] - true_states[:, 0]) ** 2)
            mse_loss = 10 * np.log10(position_mse)
        
        # 为了兼容性，同时计算分别的MSE
        position_mse_linear = np.mean((filtered_states[:, 0] - true_states[:, 0]) ** 2)
        position_mse_db = 10 * np.log10(position_mse_linear)
        
        full_state_mse_linear = np.mean((filtered_states - true_states) ** 2)
        full_state_mse_db = 10 * np.log10(full_state_mse_linear)
        
        if verbose:
            print(f"Position-only MSE: {position_mse_db:.4f} dB")
            print(f"Full-state MSE: {full_state_mse_db:.4f} dB")
            print(f"Computed MSE ({'全状态' if allStates else '位置'}): {mse_loss:.4f} dB")
        
        return {
            'mse_loss': mse_loss,  # 主MSE，根据allStates参数计算
            'position_mse_db': position_mse_db,  # 位置MSE
            'full_state_mse_db': full_state_mse_db,  # 全状态MSE
            'final_Q': final_Q,
            'filtered_states': filtered_states,
            'Q_history': Q_history,
            'true_states': true_states,
            'observations': observations
        }
        
    except Exception as e:
        if verbose:
            print(f"Final filtering failed: {e}")
        
        # Return with high loss if filtering fails
        return {
            'mse_loss': float('inf'),
            'position_mse_db': float('inf'),
            'full_state_mse_db': float('inf'),
            'final_Q': final_Q,
            'filtered_states': None,
            'Q_history': Q_history,
            'true_states': true_states,
            'observations': observations
        }


def main():
    """Main function - demonstrates how to use EMKF to estimate Q matrix"""
    print("=== Example of using EMKF to estimate state noise covariance matrix Q ===\n")
    
    # Choose running mode
    mode = input("Choose running mode (1: Single estimation, 2: Sliding window adaptive estimation): ").strip()
    
    # 1. Generate sample data
    print("1. Generating sample data...")
    T = 1000 if mode == "2" else 200  # Sliding window mode uses longer sequences
    n = 3    # State dimension
    
    observations, true_states, true_params = generate_sample_data(T=T, n=n)
    
    print(f"True Q matrix:")
    print(true_params['Q_true'])
    
    if mode == "2":
        # Sliding window adaptive estimation
        print(f"\n2. Using sliding window adaptive EMKF to estimate Q matrix...")
        
        # Modifiable parameters
        window_size = int(input("Enter window size (default 100): ") or "100")
        overlap = int(input("Enter overlap length (default 20): ") or "20")
        
        Q_initial = np.eye(n) * 0.2  # Initial Q estimate
        
        adaptive_results = adaptive_Q_estimation(
            observations=observations,
            window_size=window_size,
            overlap=overlap,
            F_true=true_params['F_true'],
            H_true=true_params['H_true'], 
            R_true=true_params['R_true'],
            xi_true=true_params['xi_true'],
            L_true=true_params['L_true'],
            Q_initial=Q_initial,
            max_iterations=50
        )
        
        # 3. Display results
        print(f"\n3. Adaptive estimation results:")
        print(f"True Q matrix:")
        print(true_params['Q_true'])
        print(f"\nFinal estimated Q matrix:")
        print(adaptive_results['final_Q'])
        
        # 4. Plot results
        print(f"\n4. Plotting adaptive estimation result charts...")
        plot_adaptive_results(observations, true_states, adaptive_results, true_params)
        
    else:
        # Single estimation mode (original functionality)
        print(f"\n2. Using EMKF to estimate Q matrix...")
        
        Q_initial = np.eye(n) * 0.2  # Initial Q estimate
        
        estimated_results = estimate_Q_from_observations(
            observations=observations,
            F_true=true_params['F_true'],
            H_true=true_params['H_true'], 
            R_true=true_params['R_true'],
            xi_true=true_params['xi_true'],
            L_true=true_params['L_true'],
            Q_initial=Q_initial,
            estimate_params=["Q"],  # Only estimate Q matrix
            max_iterations=100
        )
        
        # 3. Display results
        print(f"\n3. Estimation results:")
        print(f"True Q matrix:")
        print(true_params['Q_true'])
        print(f"\nEstimated Q matrix:")
        print(estimated_results['Q_estimated'])
        
        # 4. Plot results
        print(f"\n4. Plotting result charts...")
        plot_results(observations, true_states, estimated_results, true_params)

if __name__ == "__main__":
    main() 