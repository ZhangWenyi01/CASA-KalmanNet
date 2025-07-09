import numpy as np
import matplotlib.pyplot as plt

# 包含所有必要的函数来避免导入问题

def is_pos_def(x):
    """检查矩阵是否正定"""
    return np.all(np.linalg.eigvals(x) > 0)

def force_SPD(A, c=1e-10):
    """强制矩阵为正定矩阵"""
    A = (A + A.T)/2
    if np.amin(np.linalg.eigvals(A)) < 0:
        A += (np.abs(np.amin(np.linalg.eigvals(A))) + c)*np.identity(A.shape[0])
    if np.amin(np.linalg.eigvals(A)) == 0:
        A += c*np.identity(A.shape[0])
    return A

def KalmanFilter(F, Q, H, R, z, x_0, P_0):
    """卡尔曼滤波器"""
    T = len(z)
    # 检查隐状态的维度
    if isinstance(x_0, np.ndarray):
        n = len(x_0)
    else:
        n = 1
    # 检查观测的维度
    if isinstance(z[0], np.ndarray):
        p = len(z[0])
    else:
        p = 1

    # 多维情况
    if n > 1 and p > 1:
        x_hat_minus = np.empty((1, n))
        x_hat = np.array([x_0])
        P_minus = np.empty((1, n, n))
        P = np.array([P_0])
        K = np.empty((1, n, p))

        for i in range(T):
            # 状态外推方程
            x_hat_minus = np.append(x_hat_minus, [F @ x_hat[i]], axis=0)
            # 协方差外推方程
            P_minus = np.append(P_minus, [F @ P[i] @ F.T + Q], axis=0)
            # 卡尔曼增益
            K = np.append(K, [P_minus[i + 1] @ H.T @ np.linalg.inv(H @ P_minus[i + 1] @ H.T + R)], axis=0)
            # 状态更新方程
            x_hat = np.append(x_hat, [x_hat_minus[i + 1] + K[i + 1] @ (z[i] - H @ x_hat_minus[i + 1])], axis=0)
            # 协方差更新方程
            P = np.append(P, [
                (np.identity(n) - K[i + 1] @ H) @ P_minus[i + 1] @ (np.identity(n) - K[i + 1] @ H).T + K[i + 1] @ R @ K[i + 1].T], axis=0)

    # 一维观测情况
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

    # 一维情况
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
    """卡尔曼平滑器"""
    T = len(x_hat_minus)
    if isinstance(x_hat[0], np.ndarray):
        n = len(x_hat[0])
    else:
        n = 1

    # 多维情况
    if n > 1:
        x_tilde = np.array([x_hat[T]])
        P_tilde = np.array([P[T]])
        S = np.empty((1, n, n))

        for i in reversed(range(T)):
            S = np.insert(S, 0, [P[i] @ F.T @ np.linalg.inv(P_minus[i])], axis=0)
            x_tilde = np.insert(x_tilde, 0, [x_hat[i] + S[0] @ (x_tilde[0] - x_hat_minus[i])], axis=0)
            P_tilde = np.insert(P_tilde, 0, [P[i] + S[0] @ (P_tilde[0] - P_minus[i]) @ S[0].T], axis=0)

    # 一维情况
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
    """滞后1自协方差"""
    T = len(P) - 1
    if isinstance(F, np.ndarray):
        n = F.shape[0]
    else:
        n = 1

    # 多维情况
    if n > 1:
        V = np.array([(np.identity(n) - K[T - 1] @ H) @ F @ P[T - 1]])
        for i in reversed(range(1, T)):
            V = np.insert(V, 0, [P[i] @ S[i - 1].T + S[i] @ (V[0] - F @ P[i]) @ S[i - 1].T], axis=0)
    # 一维情况
    else:
        V = np.array([(1 - K[T - 1] * H) * F * P[T - 1]])
        for i in reversed(range(1, T)):
            V = np.insert(V, 0, [P[i] * S[i - 1].T + S[i] * (V[0] - F * P[i]) * S[i - 1].T], axis=0)

    return V

def ell(H, R, z, x, P):
    """对数似然函数"""
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

    # 多维情况
    if n > 1 and p > 1:
        for i in range(T):
            likelihood -= 0.5 * (np.log(np.linalg.det(H @ P[i] @ H.T + R)) + (z[i] - H @ x[i]).T @ np.linalg.inv(H @ P[i] @ H.T + R) @ (z[i] - H @ x[i]))
    # 一维观测情况
    elif n > 1 and p == 1:
        for i in range(T):
            likelihood -= 0.5 * (np.log(np.linalg.det(H @ P[i] @ H.T + R)) + (z[i] - H @ x[i]) ** 2 / (H @ P[i] @ H.T + R))
        likelihood = likelihood[0][0]
    # 一维情况
    else:
        for i in range(T):
            likelihood -= 0.5 * (np.log(H ** 2 * P[i] + R) + (z[i] - H * x[i]) ** 2 / (H ** 2 * P[i] + R))

    return likelihood

def EMKF(F_0, Q_0, H_0, R_0, z, xi_0, L_0, max_it=1000, tol_likelihood=0.01, tol_params=0.005, em_vars=["F", "Q", "H", "R", "xi", "L"]):
    """EM算法的卡尔曼滤波器"""
    T = len(z)
    if isinstance(xi_0, np.ndarray):
        n = len(xi_0)
    else:
        n = 1
    if isinstance(z[0], np.ndarray):
        p = len(z[0])
    else:
        p = 1

    # 初始化
    F = np.array([F_0])
    Q = np.array([Q_0])
    H = np.array([H_0])
    R = np.array([R_0])
    xi = np.array([xi_0])
    L = np.array([L_0])

    likelihood = np.empty(1)

    # 多维情况
    if n > 1 and p > 1:
        A_5 = np.zeros((p, p))
        for j in range(T):
            A_5 += np.outer(z[j], z[j])

        for i in range(max_it):
            if i > 0 and i % 50 == 0:
                print(f"Iteration {i}")
            
            # E步
            x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[i], Q[i], H[i], R[i], z, xi[i], L[i])
            S, x_tilde, P_tilde = KalmanSmoother(F[i], x_hat_minus, P_minus, x_hat, P)
            V = Lag1AutoCov(K, S, F[i], H[i], P)

            likelihood = np.append(likelihood, [ell(H[i], R[i], z, x_hat_minus, P_minus)], axis=0)

            # 收敛性检查
            convergence_count = 0
            if i >= 1 and likelihood[i + 1] - likelihood[i] < tol_likelihood:
                convergence_count += 1

            # M步
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
                    print(f"Q NON-SPD at iteration {i}")
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
                
                if not is_pos_def(R_i):
                    print(f"R NON-SPD at iteration {i}")
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

    else:
        # 简化版本，暂时只处理多维情况
        raise NotImplementedError("暂时只支持多维观测情况")

    likelihood = np.delete(likelihood, 0, axis=0)
    return F, Q, H, R, xi, L, likelihood, iterations

def estimate_Q_from_observations(observations, F_true=None, H_true=None, R_true=None, 
                                xi_true=None, L_true=None, Q_initial=None, 
                                estimate_params=["Q"], max_iterations=100):
    """
    使用EMKF函数从有噪声的观测数据中估计状态噪声协方差矩阵Q
    """
    
    # 获取观测数据的维度
    T = len(observations)
    if observations.ndim == 1:
        p = 1
        observations = observations.reshape(-1, 1)
    else:
        p = observations.shape[1]
    
    # 假设状态维度等于观测维度
    n = p
    
    # 设置默认参数
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
    
    print(f"开始使用EMKF估计参数...")
    print(f"观测数据维度: {observations.shape}")
    print(f"状态维度: {n}")
    print(f"估计参数: {estimate_params}")
    
    # 调用EMKF函数
    results = EMKF(F_true, Q_initial, H_true, R_true, observations, 
                   xi_true, L_true, max_it=max_iterations, 
                   em_vars=estimate_params)
    
    # 解析结果
    F_estimates, Q_estimates, H_estimates, R_estimates, xi_estimates, L_estimates, likelihood, iterations = results
    
    print(f"EMKF迭代次数: {iterations}")
    print(f"最终似然值: {likelihood[-1]:.4f}")
    
    # 获取最终估计值
    final_F = F_estimates[iterations]
    final_Q = Q_estimates[iterations]
    final_H = H_estimates[iterations]
    final_R = R_estimates[iterations]
    final_xi = xi_estimates[iterations]
    final_L = L_estimates[iterations]
    
    print(f"\n估计的状态噪声协方差矩阵Q:")
    print(final_Q)
    
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
    """生成示例数据用于测试"""
    
    np.random.seed(seed)
    
    # 设置真实参数
    if Q_true is None:
        Q_true = np.array([[0.1, 0.05], [0.05, 0.1]])
    
    if F_true is None:
        F_true = np.array([[0.9, 0.1], [0.0, 0.8]])
    
    if H_true is None:
        H_true = np.eye(n)
    
    if R_true is None:
        R_true = np.eye(n) * 0.05
    
    if xi_true is None:
        xi_true = np.zeros(n)
    
    if L_true is None:
        L_true = np.eye(n) * 0.1
    
    # 生成状态和观测序列
    states = np.zeros((T+1, n))
    observations = np.zeros((T, n))
    
    # 初始状态
    states[0] = np.random.multivariate_normal(xi_true, L_true)
    
    # 生成状态和观测序列
    for t in range(T):
        # 状态转移
        process_noise = np.random.multivariate_normal(np.zeros(n), Q_true)
        states[t+1] = F_true @ states[t] + process_noise
        
        # 观测
        observation_noise = np.random.multivariate_normal(np.zeros(n), R_true)
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
    """绘制结果图表"""
    
    # 使用估计的参数进行卡尔曼滤波
    Q_est = estimated_results['Q_estimated']
    F_est = estimated_results['F_estimated']
    H_est = estimated_results['H_estimated']
    R_est = estimated_results['R_estimated']
    xi_est = estimated_results['xi_estimated']
    L_est = estimated_results['L_estimated']
    
    # 应用卡尔曼滤波器
    x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(
        F_est, Q_est, H_est, R_est, observations, xi_est, L_est
    )
    
    # 应用卡尔曼平滑器
    S, x_tilde, P_tilde = KalmanSmoother(F_est, x_hat_minus, P_minus, x_hat, P)
    
    # 绘制图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 绘制状态估计结果
    n = true_states.shape[1]
    for i in range(min(n, 2)):
        axes[0, i].plot(true_states[:, i], 'b-', label='真实状态', linewidth=2)
        axes[0, i].plot(x_hat[1:, i], 'r--', label='滤波估计', linewidth=1.5)
        axes[0, i].plot(x_tilde[1:, i], 'g:', label='平滑估计', linewidth=1.5)
        axes[0, i].set_title(f'状态 {i+1} 的估计')
        axes[0, i].set_xlabel('时间')
        axes[0, i].set_ylabel('状态值')
        axes[0, i].legend()
        axes[0, i].grid(True)
    
    # 绘制似然函数收敛过程
    axes[1, 0].plot(estimated_results['likelihood'])
    axes[1, 0].set_title('对数似然函数收敛过程')
    axes[1, 0].set_xlabel('迭代次数')
    axes[1, 0].set_ylabel('对数似然值')
    axes[1, 0].grid(True)
    
    # 绘制Q矩阵的比较
    Q_true = true_params['Q_true']
    Q_est = estimated_results['Q_estimated']
    
    axes[1, 1].text(0.1, 0.7, f'真实Q矩阵:\n{np.array2string(Q_true, precision=3)}', 
                   transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].text(0.1, 0.3, f'估计Q矩阵:\n{np.array2string(Q_est, precision=3)}', 
                   transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].set_title('Q矩阵比较')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 计算估计误差
    Q_error = np.linalg.norm(Q_est - Q_true, 'fro')
    print(f"\nQ矩阵估计的Frobenius范数误差: {Q_error:.6f}")

def adaptive_Q_estimation(observations, window_size=50, overlap=0, 
                          F_true=None, H_true=None, R_true=None, 
                          xi_true=None, L_true=None, Q_initial=None,
                          max_iterations=50):
    """
    使用滑动窗口方法进行自适应Q矩阵估计
    
    参数:
    observations: 长观测序列，形状为 (T, p)
    window_size: 每个窗口的大小
    overlap: 窗口之间的重叠长度
    其他参数同 estimate_Q_from_observations
    
    返回:
    Q_history: Q矩阵的历史变化
    filtered_states: 使用自适应Q估计的滤波状态
    window_results: 每个窗口的详细结果
    """
    
    T = len(observations)
    if observations.ndim == 1:
        p = 1
        observations = observations.reshape(-1, 1)
    else:
        p = observations.shape[1]
    
    n = p  # 假设状态维度等于观测维度
    
    # 设置默认参数
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
    
    print(f"=== 滑动窗口自适应Q估计 ===")
    print(f"观测序列长度: {T}")
    print(f"窗口大小: {window_size}")
    print(f"重叠长度: {overlap}")
    
    # 计算窗口的起始位置
    step_size = window_size - overlap
    window_starts = list(range(0, T - window_size + 1, step_size))
    if window_starts[-1] + window_size < T:
        window_starts.append(T - window_size)  # 确保覆盖到最后
    
    print(f"总窗口数: {len(window_starts)}")
    
    # 存储结果
    Q_history = []
    window_results = []
    filtered_states = np.zeros((T+1, n))
    filtered_covariances = np.zeros((T+1, n, n))
    
    # 当前使用的Q矩阵
    current_Q = Q_initial.copy()
    current_xi = xi_true.copy()
    current_L = L_true.copy()
    
    for i, start_idx in enumerate(window_starts):
        end_idx = min(start_idx + window_size, T)
        window_obs = observations[start_idx:end_idx]
        
        print(f"\n窗口 {i+1}/{len(window_starts)}: 时间步 {start_idx}-{end_idx-1}")
        
        # 使用当前Q估计进行EMKF
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
            
            # 更新Q矩阵
            new_Q = results['Q_estimated']
            
            # 平滑更新（可选）
            alpha = 0.7  # 学习率
            current_Q = alpha * new_Q + (1 - alpha) * current_Q
            
            Q_history.append(current_Q.copy())
            window_results.append(results)
            
            print(f"窗口 {i+1} Q矩阵:")
            print(current_Q)
            
        except Exception as e:
            print(f"窗口 {i+1} 估计失败: {e}")
            Q_history.append(current_Q.copy())
            window_results.append(None)
    
    # 使用自适应Q矩阵对整个序列进行滤波
    print(f"\n使用自适应Q矩阵对整个序列进行最终滤波...")
    
    # 为了简化，使用最后的Q矩阵对整个序列滤波
    final_Q = Q_history[-1] if Q_history else Q_initial
    
    try:
        x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(
            F_true, final_Q, H_true, R_true, observations, xi_true, L_true
        )
        S, x_tilde, P_tilde = KalmanSmoother(F_true, x_hat_minus, P_minus, x_hat, P)
        
        filtered_states = x_tilde
        filtered_covariances = P_tilde
        
    except Exception as e:
        print(f"最终滤波失败: {e}")
    
    return {
        'Q_history': Q_history,
        'filtered_states': filtered_states,
        'filtered_covariances': filtered_covariances,
        'window_results': window_results,
        'window_starts': window_starts,
        'final_Q': final_Q
    }


def plot_adaptive_results(observations, true_states, adaptive_results, true_params):
    """绘制自适应估计结果"""
    
    Q_history = adaptive_results['Q_history']
    filtered_states = adaptive_results['filtered_states']
    window_starts = adaptive_results['window_starts']
    final_Q = adaptive_results['final_Q']
    
    # 创建更大的图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 状态估计结果
    n = true_states.shape[1]
    for i in range(min(n, 2)):
        axes[0, i].plot(true_states[:, i], 'b-', label='True State', linewidth=2)
        axes[0, i].plot(filtered_states[1:, i], 'r--', label='Adaptive Filter', linewidth=1.5)
        axes[0, i].set_title(f'State {i+1} Estimation')
        axes[0, i].set_xlabel('Time')
        axes[0, i].set_ylabel('State Value')
        axes[0, i].legend()
        axes[0, i].grid(True)
    
    # 2. Q矩阵对角元素随时间的变化
    if Q_history:
        Q_diag_history = np.array([Q[np.diag_indices_from(Q)] for Q in Q_history])
        for i in range(min(n, 2)):
            axes[0, 2].plot(Q_diag_history[:, i], 'o-', label=f'Q[{i},{i}]', linewidth=1.5)
        
        # 标记窗口边界
        for j, start in enumerate(window_starts[1:], 1):
            axes[0, 2].axvline(x=j, color='gray', linestyle='--', alpha=0.5)
        
        axes[0, 2].set_title('Q Matrix Diagonal Elements Over Windows')
        axes[0, 2].set_xlabel('Window Number')
        axes[0, 2].set_ylabel('Q Value')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
    
    # 3. Q矩阵非对角元素随时间的变化（如果存在）
    if Q_history and n > 1:
        Q_offdiag_history = np.array([Q[0, 1] for Q in Q_history])
        axes[1, 0].plot(Q_offdiag_history, 'o-', label='Q[0,1]', linewidth=1.5)
        
        # 标记窗口边界
        for j, start in enumerate(window_starts[1:], 1):
            axes[1, 0].axvline(x=j, color='gray', linestyle='--', alpha=0.5)
        
        axes[1, 0].set_title('Q Matrix Off-diagonal Elements')
        axes[1, 0].set_xlabel('Window Number')
        axes[1, 0].set_ylabel('Q[0,1] Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 4. 最终Q矩阵与真实Q矩阵的比较
    Q_true = true_params['Q_true']
    axes[1, 1].text(0.1, 0.7, f'True Q:\n{np.array2string(Q_true, precision=3)}', 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.3, f'Final Estimated Q:\n{np.array2string(final_Q, precision=3)}', 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('Q Matrix Comparison')
    axes[1, 1].axis('off')
    
    # 5. 估计误差随时间的变化
    if Q_history:
        Q_errors = [np.linalg.norm(Q - Q_true, 'fro') for Q in Q_history]
        axes[1, 2].plot(Q_errors, 'o-', linewidth=1.5, color='red')
        axes[1, 2].set_title('Q Estimation Error Over Windows')
        axes[1, 2].set_xlabel('Window Number')
        axes[1, 2].set_ylabel('Frobenius Norm Error')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 计算最终估计误差
    if Q_history:
        final_error = np.linalg.norm(final_Q - Q_true, 'fro')
        print(f"\nFinal Q estimation Frobenius norm error: {final_error:.6f}")


def main():
    """主函数 - 演示如何使用EMKF估计Q矩阵"""
    print("=== 使用EMKF估计状态噪声协方差矩阵Q的示例 ===\n")
    
    # 选择运行模式
    mode = input("选择运行模式 (1: 单次估计, 2: 滑动窗口自适应估计): ").strip()
    
    # 1. 生成示例数据
    print("1. 生成示例数据...")
    T = 1000 if mode == "2" else 200  # 滑动窗口模式使用更长的序列
    n = 2    # 状态维度
    
    observations, true_states, true_params = generate_sample_data(T=T, n=n)
    
    print(f"真实的Q矩阵:")
    print(true_params['Q_true'])
    
    if mode == "2":
        # 滑动窗口自适应估计
        print(f"\n2. 使用滑动窗口自适应EMKF估计Q矩阵...")
        
        # 可以修改的参数
        window_size = int(input("输入窗口大小 (默认100): ") or "100")
        overlap = int(input("输入重叠长度 (默认20): ") or "20")
        
        Q_initial = np.eye(n) * 0.2  # 初始Q估计值
        
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
        
        # 3. 显示结果
        print(f"\n3. 自适应估计结果:")
        print(f"真实Q矩阵:")
        print(true_params['Q_true'])
        print(f"\n最终估计Q矩阵:")
        print(adaptive_results['final_Q'])
        
        # 4. 绘制结果
        print(f"\n4. 绘制自适应估计结果图表...")
        plot_adaptive_results(observations, true_states, adaptive_results, true_params)
        
    else:
        # 单次估计模式（原有功能）
        print(f"\n2. 使用EMKF估计Q矩阵...")
        
        Q_initial = np.eye(n) * 0.2  # 初始Q估计值
        
        estimated_results = estimate_Q_from_observations(
            observations=observations,
            F_true=true_params['F_true'],
            H_true=true_params['H_true'], 
            R_true=true_params['R_true'],
            xi_true=true_params['xi_true'],
            L_true=true_params['L_true'],
            Q_initial=Q_initial,
            estimate_params=["Q"],  # 只估计Q矩阵
            max_iterations=100
        )
        
        # 3. 显示结果
        print(f"\n3. 估计结果:")
        print(f"真实Q矩阵:")
        print(true_params['Q_true'])
        print(f"\n估计Q矩阵:")
        print(estimated_results['Q_estimated'])
        
        # 4. 绘制结果
        print(f"\n4. 绘制结果图表...")
        plot_results(observations, true_states, estimated_results, true_params)

if __name__ == "__main__":
    main() 