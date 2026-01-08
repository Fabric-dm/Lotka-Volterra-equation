"""
Lotka-Volterra方程PINNs求解：网络架构与采样策略探索
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# 检查当前后端并导入相应的模块
backend_name = dde.backend.backend_name
print(f"当前后端: {backend_name}")

if backend_name in ["tensorflow.compat.v1", "tensorflow", "jax"]:
    from deepxde.backend import tf
    concat_fn = tf.concat
    sin_fn = tf.sin
    tanh_fn = tf.tanh
elif backend_name == "pytorch":
    import torch
    concat_fn = torch.cat
    sin_fn = torch.sin
    tanh_fn = torch.tanh
elif backend_name == "paddle":
    import paddle
    concat_fn = paddle.concat
    sin_fn = paddle.sin
    tanh_fn = paddle.tanh
else:
    raise ValueError(f"不支持的后端: {backend_name}")

# 全局参数
ub = 200  # 种群上界
rb = 20   # 时间右边界

def generate_true_solution():
    """生成真实解用于对比"""
    t = np.linspace(0, 1, 100)

    def ode_func(t, r):
        x, y = r
        dx_t = 1/ub * rb * (2.0*ub*x - 0.04*ub*x*ub*y)
        dy_t = 1/ub * rb * (0.02*ub*x*ub*y - 1.06*ub*y)
        return dx_t, dy_t

    sol = integrate.solve_ivp(ode_func, (0, 10), (100/ub, 15/ub), t_eval=t)
    return sol.y[0].reshape(100,1), sol.y[1].reshape(100,1)

def lotka_volterra_ode(x, y):
    """Lotka-Volterra ODE系统"""
    r = y[:, 0:1]
    p = y[:, 1:2]
    dr_t = dde.grad.jacobian(y, x, i=0)
    dp_t = dde.grad.jacobian(y, x, i=1)

    return [
        dr_t - 1/ub * rb * (2.0*ub*r - 0.04*ub*r*ub*p),
        dp_t - 1/ub * rb * (0.02*r*ub*p*ub - 1.06*p*ub),
    ]

def build_pinn_model(layer_size=[1, 64, 64,64,64, 64, 2],
                    activation="tanh",
                    num_domain=3000,
                    num_boundary=2,
                    use_feature_transform=True,
                    use_output_transform=True,
                    epochs_adam=5000,
                    epochs_lbfgs=None):
    """构建并训练PINN模型"""

    # 定义几何域和时间域
    geom = dde.geometry.TimeDomain(0, 1.0)

    # 创建PDE数据
    data = dde.data.PDE(
        geom,
        lotka_volterra_ode,
        [],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_test=1000
    )

    # 根据是否使用特征变换调整输入层维度
    if use_feature_transform:
        # 特征变换将1维输入变为7维
        adjusted_layer_size = layer_size.copy()
        adjusted_layer_size[0] = 7  # 将输入层从1调整为7
    else:
        adjusted_layer_size = layer_size

    # 创建神经网络
    net = dde.nn.FNN(adjusted_layer_size, activation, "Glorot normal")

    # 应用特征变换（增强周期性特征）
    if use_feature_transform:
        def input_transform(t):
            return concat_fn(
                [t,
                 sin_fn(t),
                 sin_fn(2*t),
                 sin_fn(3*t),
                 sin_fn(4*t),
                 sin_fn(5*t),
                 sin_fn(6*t)],
                dim=1
            )
        net.apply_feature_transform(input_transform)

    # 应用输出变换（硬约束初始条件）
    if use_output_transform:
        def output_transform(t, y):
            y1 = y[:, 0:1]
            y2 = y[:, 1:2]
            return concat_fn(
                [y1 * tanh_fn(t) + 100/ub,
                 y2 * tanh_fn(t) + 15/ub],
                dim=1
            )
        net.apply_output_transform(output_transform)

    # 创建并编译模型
    model = dde.Model(data, net)

    # 第一阶段：Adam优化器
    model.compile("adam", lr=0.001)
    print("开始Adam优化阶段...")
    losshistory, _ = model.train(iterations=epochs_adam, display_every=2000)

    # 第二阶段：L-BFGS优化器（精细调优）
    if epochs_lbfgs is not None:
        try:
            model.compile("L-BFGS")
            print("开始L-BFGS优化阶段...")
            losshistory, train_state = model.train()
        except Exception as e:
            print(f"L-BFGS优化失败: {e}")
            print("继续使用Adam进行精细调优...")
            model.compile("adam", lr=0.0001)
            losshistory, train_state = model.train(iterations=1000)
    else:
        print("跳过L-BFGS阶段，使用Adam继续训练...")
        model.compile("adam", lr=0.0001)
        losshistory, train_state = model.train(iterations=1000)

    return model, losshistory, train_state

def evaluate_model(model, show_plot=True):
    """评估模型并可视化结果"""

    # 生成测试点
    t_test = np.linspace(0, 1, 100).reshape(100, 1)

    # 获取预测值
    sol_pred = model.predict(t_test)
    x_pred = sol_pred[:, 0:1]
    y_pred = sol_pred[:, 1:2]

    # 获取真实值
    x_true, y_true = generate_true_solution()

    # 计算误差
    mse_x = np.mean((x_pred - x_true)**2)
    mse_y = np.mean((y_pred - y_true)**2)
    relative_error_x = np.mean(np.abs(x_pred - x_true) / (np.abs(x_true) + 1e-8))
    relative_error_y = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8))

    print(f"预测结果评估:")
    print(f"  x的MSE: {mse_x:.6f}")
    print(f"  y的MSE: {mse_y:.6f}")
    print(f"  x的平均相对误差: {relative_error_x:.4%}")
    print(f"  y的平均相对误差: {relative_error_y:.4%}")

    # 可视化
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 种群随时间变化
        axes[0, 0].plot(t_test, x_true, 'b-', linewidth=2, label='x (猎物) 真实值')
        axes[0, 0].plot(t_test, y_true, 'g-', linewidth=2, label='y (捕食者) 真实值')
        axes[0, 0].plot(t_test, x_pred, 'r--', linewidth=2, label='x 预测值', alpha=0.8)
        axes[0, 0].plot(t_test, y_pred, 'orange', linestyle='--', linewidth=2, label='y 预测值', alpha=0.8)
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('种群数量')
        axes[0, 0].set_title('Lotka-Volterra方程解')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 误差分布
        axes[0, 1].plot(t_test, np.abs(x_pred - x_true), 'r-', linewidth=2, label='x绝对误差')
        axes[0, 1].plot(t_test, np.abs(y_pred - y_true), 'b-', linewidth=2, label='y绝对误差')
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].set_ylabel('绝对误差')
        axes[0, 1].set_title('预测误差随时间变化')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 相位图
        axes[1, 0].plot(x_true, y_true, 'b-', linewidth=2, label='真实相位轨迹')
        axes[1, 0].plot(x_pred, y_pred, 'r--', linewidth=2, label='预测相位轨迹', alpha=0.8)
        axes[1, 0].scatter(x_true[0], y_true[0], color='green', s=100, label='起始点', zorder=5)
        axes[1, 0].set_xlabel('猎物数量 (x)')
        axes[1, 0].set_ylabel('捕食者数量 (y)')
        axes[1, 0].set_title('相位空间轨迹')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 相对误差
        time_points = np.linspace(0, 1, 10)
        rel_error_x_points = []
        rel_error_y_points = []

        for t in time_points:
            idx = int(t * 99)
            # 将数组转换为标量
            rel_error_x = float(np.abs(x_pred[idx] - x_true[idx]) / (np.abs(x_true[idx]) + 1e-8))
            rel_error_y = float(np.abs(y_pred[idx] - y_true[idx]) / (np.abs(y_true[idx]) + 1e-8))
            rel_error_x_points.append(rel_error_x)
            rel_error_y_points.append(rel_error_y)

        axes[1, 1].bar(time_points - 0.02, rel_error_x_points, width=0.04, label='x相对误差', alpha=0.7)
        axes[1, 1].bar(time_points + 0.02, rel_error_y_points, width=0.04, label='y相对误差', alpha=0.7)
        axes[1, 1].set_xlabel('时间')
        axes[1, 1].set_ylabel('相对误差')
        axes[1, 1].set_title('不同时间点的相对误差')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {
        'mse_x': mse_x,
        'mse_y': mse_y,
        'relative_error_x': relative_error_x,
        'relative_error_y': relative_error_y,
        'x_pred': x_pred,
        'y_pred': y_pred,
        'x_true': x_true,
        'y_true': y_true
    }

def compare_architectures():
    """比较不同网络架构的效果"""

    architectures = [
        {
            'name': '浅层窄网络',
            'layers': [1, 32, 32, 2],
            'activation': 'tanh',
            'description': '3层，每层32个神经元'
        },
        {
            'name': '基准网络',
            'layers': [1, 64, 64, 64,64, 64, 2],
            'activation': 'tanh',
            'description': '7层，每层64个神经元'
        },
        {
            'name': '深层宽网络',
            'layers': [1, 128, 128, 128, 128,128, 2],
            'activation': 'tanh',
            'description': '8层，每层128个神经元'
        },
        {
            'name': 'ReLU网络',
            'layers': [1, 64, 64,64, 64,64, 2],
            'activation': 'relu',
            'description': '7层64神经元，ReLU激活'
        },
    ]

    results = {}

    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"训练架构: {arch['name']}")
        print(f"描述: {arch['description']}")
        print(f"激活函数: {arch['activation']}")
        print(f"{'='*60}")

        try:
            # 对于深层网络，减少训练迭代次数以节省时间
            if arch['name'] == '深层宽网络':
                epochs_adam = 1000
                epochs_lbfgs = None
            else:
                epochs_adam = 1500
                epochs_lbfgs = None

            model, losshistory, train_state = build_pinn_model(
                layer_size=arch['layers'],
                activation=arch['activation'],
                num_domain=3000,
                num_boundary=2,
                epochs_adam=epochs_adam,
                epochs_lbfgs=epochs_lbfgs
            )

            # 评估模型
            eval_results = evaluate_model(model, show_plot=False)

            # 计算参数量（考虑特征变换）
            if arch['name'] in ['基准网络', 'ReLU网络']:
                # 有特征变换，输入层为7
                total_params = 7*64 + 64 + sum([arch['layers'][i]*arch['layers'][i+1] + arch['layers'][i+1]
                                     for i in range(1, len(arch['layers'])-1)])
            elif arch['name'] == '深层宽网络':
                total_params = 7*128 + 128 + sum([arch['layers'][i]*arch['layers'][i+1] + arch['layers'][i+1]
                                     for i in range(1, len(arch['layers'])-1)])
            else:
                total_params = sum([arch['layers'][i]*arch['layers'][i+1] + arch['layers'][i+1]
                                     for i in range(len(arch['layers'])-1)])

            results[arch['name']] = {
                'layers': arch['layers'],
                'activation': arch['activation'],
                'relative_error_x': eval_results['relative_error_x'],
                'relative_error_y': eval_results['relative_error_y'],
                'total_params': total_params
            }

            print(f"完成! x相对误差: {eval_results['relative_error_x']:.4%}, "
                  f"y相对误差: {eval_results['relative_error_y']:.4%}")

        except Exception as e:
            print(f"训练失败: {e}")
            results[arch['name']] = {
                'layers': arch['layers'],
                'activation': arch['activation'],
                'relative_error_x': None,
                'relative_error_y': None,
                'error': str(e)
            }

    return results

def compare_sampling_strategies():
    """比较不同采样策略的效果"""

    sampling_configs = [
        {
            'name': '稀疏采样',
            'num_domain': 500,
            'num_boundary': 2,
            'description': '500个域点，2个边界点'
        },
        {
            'name': '基准采样',
            'num_domain': 3000,
            'num_boundary': 2,
            'description': '3000个域点，2个边界点'
        },
        {
            'name': '密集采样',
            'num_domain': 10000,
            'num_boundary': 2,
            'description': '10000个域点，2个边界点'
        },
        {
            'name': '边界加强',
            'num_domain': 2000,
            'num_boundary': 20,
            'description': '2000个域点，20个边界点'
        },
    ]

    results = {}

    for config in sampling_configs:
        print(f"\n{'='*60}")
        print(f"采样策略: {config['name']}")
        print(f"描述: {config['description']}")
        print(f"{'='*60}")

        try:
            # 对于密集采样，减少训练迭代次数以节省时间
            if config['name'] == '密集采样':
                epochs_adam = 10000
                epochs_lbfgs = None
            else:
                epochs_adam = 15000
                epochs_lbfgs = None

            model, losshistory, train_state = build_pinn_model(
                layer_size=[1, 64, 64, 64, 64, 64, 64, 2],
                activation='tanh',
                num_domain=config['num_domain'],
                num_boundary=config['num_boundary'],
                epochs_adam=epochs_adam,
                epochs_lbfgs=epochs_lbfgs
            )

            # 评估模型
            eval_results = evaluate_model(model, show_plot=False)

            results[config['name']] = {
                'num_domain': config['num_domain'],
                'num_boundary': config['num_boundary'],
                'relative_error_x': eval_results['relative_error_x'],
                'relative_error_y': eval_results['relative_error_y'],
                'total_points': config['num_domain'] + config['num_boundary']
            }

            print(f"完成! x相对误差: {eval_results['relative_error_x']:.4%}, "
                  f"y相对误差: {eval_results['relative_error_y']:.4%}")

        except Exception as e:
            print(f"训练失败: {e}")
            results[config['name']] = {
                'num_domain': config['num_domain'],
                'num_boundary': config['num_boundary'],
                'relative_error_x': None,
                'relative_error_y': None,
                'error': str(e)
            }

    return results

def visualize_comparison_results(arch_results, sampling_results):
    """可视化比较结果"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 网络架构误差对比
    if arch_results:
        arch_names = list(arch_results.keys())
        valid_archs = [name for name in arch_names
                      if arch_results[name].get('relative_error_x') is not None]

        if valid_archs:
            errors_x = [arch_results[name]['relative_error_x'] for name in valid_archs]
            errors_y = [arch_results[name]['relative_error_y'] for name in valid_archs]
            param_counts = [arch_results[name].get('total_params', 0) for name in valid_archs]

            x_pos = np.arange(len(valid_archs))
            width = 0.35

            axes[0, 0].bar(x_pos - width/2, errors_x, width, label='x相对误差', alpha=0.8)
            axes[0, 0].bar(x_pos + width/2, errors_y, width, label='y相对误差', alpha=0.8)
            axes[0, 0].set_xlabel('网络架构')
            axes[0, 0].set_ylabel('相对误差')
            axes[0, 0].set_title('不同网络架构的预测误差')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(valid_archs, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 参数数量与误差关系
            axes[0, 1].scatter(param_counts, errors_x, s=100, alpha=0.7, label='x误差')
            axes[0, 1].scatter(param_counts, errors_y, s=100, alpha=0.7, label='y误差')
            for i, name in enumerate(valid_archs):
                axes[0, 1].annotate(name, (param_counts[i], errors_x[i]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[0, 1].set_xlabel('网络参数量')
            axes[0, 1].set_ylabel('相对误差')
            axes[0, 1].set_title('网络复杂度 vs 预测误差')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

    # 2. 采样策略误差对比
    if sampling_results:
        sampling_names = list(sampling_results.keys())
        valid_sampling = [name for name in sampling_names
                         if sampling_results[name].get('relative_error_x') is not None]

        if valid_sampling:
            errors_x = [sampling_results[name]['relative_error_x'] for name in valid_sampling]
            errors_y = [sampling_results[name]['relative_error_y'] for name in valid_sampling]
            total_points = [sampling_results[name]['total_points'] for name in valid_sampling]

            x_pos = np.arange(len(valid_sampling))
            width = 0.35

            axes[1, 0].bar(x_pos - width/2, errors_x, width, label='x相对误差', alpha=0.8)
            axes[1, 0].bar(x_pos + width/2, errors_y, width, label='y相对误差', alpha=0.8)
            axes[1, 0].set_xlabel('采样策略')
            axes[1, 0].set_ylabel('相对误差')
            axes[1, 0].set_title('不同采样策略的预测误差')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(valid_sampling, rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 采样点数量与误差关系
            axes[1, 1].plot(total_points, errors_x, 'o-', linewidth=2, markersize=8, label='x误差')
            axes[1, 1].plot(total_points, errors_y, 's-', linewidth=2, markersize=8, label='y误差')
            axes[1, 1].set_xlabel('总采样点数量')
            axes[1, 1].set_ylabel('相对误差')
            axes[1, 1].set_title('采样点数量 vs 预测误差')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    """主函数：执行完整的探索实验"""

    print("="*70)
    print("Lotka-Volterra方程PINNs求解探索实验")
    print(f"当前后端: {backend_name}")
    print("探索不同网络架构和采样策略对预测效果的影响")
    print("="*70)

    # 选择实验模式
    print("\n请选择实验模式:")
    print("1. 基准模型测试")
    print("2. 网络架构比较")
    print("3. 采样策略比较")
    print("4. 完整探索实验")
    print("5. 快速基准测试（减少迭代次数）")

    choice = input("请输入选择 (1-5): ").strip()

    if choice == '1':
        print("\n运行基准模型测试...")
        model, losshistory, train_state = build_pinn_model()
        evaluate_model(model)

    elif choice == '2':
        print("\n比较不同网络架构...")
        arch_results = compare_architectures()

        # 显示结果表格
        print("\n" + "="*80)
        print("网络架构比较结果:")
        print("="*80)
        print(f"{'架构名称':<20} {'激活函数':<15} {'x相对误差':<15} {'y相对误差':<15} {'参数量':<10}")
        print("-"*80)

        for name, result in arch_results.items():
            if result.get('relative_error_x') is not None:
                print(f"{name:<20} {result.get('activation', 'N/A'):<15} "
                      f"{result.get('relative_error_x', 0):<15.4%} "
                      f"{result.get('relative_error_y', 0):<15.4%} "
                      f"{result.get('total_params', 0):<10}")
            else:
                print(f"{name:<20} {result.get('activation', 'N/A'):<15} "
                      f"{'训练失败':<15} {'训练失败':<15} {'N/A':<10}")

    elif choice == '3':
        print("\n比较不同采样策略...")
        sampling_results = compare_sampling_strategies()

        # 显示结果表格
        print("\n" + "="*90)
        print("采样策略比较结果:")
        print("="*90)
        print(f"{'策略名称':<15} {'域点数量':<10} {'边界点数量':<10} {'总点数':<10} {'x相对误差':<15} {'y相对误差':<15}")
        print("-"*90)

        for name, result in sampling_results.items():
            if result.get('relative_error_x') is not None:
                print(f"{name:<15} {result.get('num_domain', 0):<10} "
                      f"{result.get('num_boundary', 0):<10} "
                      f"{result.get('total_points', 0):<10} "
                      f"{result.get('relative_error_x', 0):<15.4%} "
                      f"{result.get('relative_error_y', 0):<15.4%}")
            else:
                print(f"{name:<15} {result.get('num_domain', 0):<10} "
                      f"{result.get('num_boundary', 0):<10} "
                      f"{result.get('total_points', 0):<10} "
                      f"{'训练失败':<15} {'训练失败':<15}")

    elif choice == '4':
        print("\n运行完整探索实验...")

        # 运行架构比较
        print("\n第一阶段：网络架构比较")
        arch_results = compare_architectures()

        # 运行采样策略比较
        print("\n第二阶段：采样策略比较")
        sampling_results = compare_sampling_strategies()

        # 可视化比较结果
        print("\n第三阶段：结果可视化")
        visualize_comparison_results(arch_results, sampling_results)

        # 总结分析
        print("\n" + "="*70)
        print("实验总结分析")
        print("="*70)

        # 找出最佳架构
        best_arch = None
        best_error = float('inf')

        for name, result in arch_results.items():
            if result.get('relative_error_x') is not None:
                avg_error = (result['relative_error_x'] + result['relative_error_y']) / 2
                if avg_error < best_error:
                    best_error = avg_error
                    best_arch = name

        # 找出最佳采样策略
        best_sampling = None
        best_sampling_error = float('inf')

        for name, result in sampling_results.items():
            if result.get('relative_error_x') is not None:
                avg_error = (result['relative_error_x'] + result['relative_error_y']) / 2
                if avg_error < best_sampling_error:
                    best_sampling_error = avg_error
                    best_sampling = name

        if best_arch:
            print(f"最佳网络架构: {best_arch}")
            print(f"  平均相对误差: {best_error:.4%}")

        if best_sampling:
            print(f"最佳采样策略: {best_sampling}")
            print(f"  平均相对误差: {best_sampling_error:.4%}")

        print("\n建议:")
        if best_arch and best_sampling:
            print(f"1. 使用{best_arch}网络架构")
            print(f"2. 使用{best_sampling}采样策略")
            print(f"3. 结合使用上述最佳配置可获得最佳效果")

    elif choice == '5':
        print("\n运行快速基准测试（减少迭代次数）...")
        model, losshistory, train_state = build_pinn_model(
            epochs_adam=5000,
            epochs_lbfgs=None
        )
        evaluate_model(model)

    else:
        print("无效选择，运行快速基准测试...")
        model, losshistory, train_state = build_pinn_model(
            epochs_adam=5000,
            epochs_lbfgs=None
        )
        evaluate_model(model)

    print("\n实验完成！")

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    np.random.seed(42)

    # 运行主程序
    main()