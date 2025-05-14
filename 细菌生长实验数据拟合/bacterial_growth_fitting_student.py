import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def load_bacterial_data(file_path):
    """
    加载细菌生长实验数据

    参数:
    file_path (str): 数据文件路径

    返回:
    tuple: 包含时间(t)和酶活性(y)的元组
    """
    data = np.loadtxt(file_path)
    t = data[:, 0]
    y = data[:, 1]
    return t, y


def V_model(t, tau):
    """
    V(t)模型：描述诱导分子TMG的渗透过程

    参数:
    t (array): 时间数组
    tau (float): 时间常数

    返回:
    array: V(t)模型计算结果
    """
    return 1 - np.exp(-t / tau)


def W_model(t, A, tau):
    """
    W(t)模型：描述β-半乳糖苷酶的合成过程

    参数:
    t (array): 时间数组
    A (float): 比例系数
    tau (float): 时间常数

    返回:
    array: W(t)模型计算结果
    """
    return A * (np.exp(-t / tau) - 1 + t / tau)


def fit_V_model(t, y):
    """
    拟合V(t)模型参数

    参数:
    t (array): 时间数组
    y (array): 实验数据

    返回:
    tuple: 包含拟合参数和参数误差的元组
    """
    # 初始猜测值
    p0 = [1.0]
    # 执行拟合
    popt, pcov = curve_fit(V_model, t, y, p0=p0)
    # 计算参数误差
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def fit_W_model(t, y):
    """
    拟合W(t)模型参数

    参数:
    t (array): 时间数组
    y (array): 实验数据

    返回:
    tuple: 包含拟合参数和参数误差的元组
    """
    # 初始猜测值
    p0 = [1.0, 1.0]
    # 执行拟合
    popt, pcov = curve_fit(W_model, t, y, p0=p0)
    # 计算参数误差
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def plot_fit_results(t_data, y_data, model_func, popt, perr, title, parameter_names, file_path=None):
    """
    绘制拟合结果图

    参数:
    t_data (array): 实验时间数据
    y_data (array): 实验测量数据
    model_func (function): 模型函数
    popt (array): 拟合参数
    perr (array): 参数误差
    title (str): 图表标题
    parameter_names (list): 参数名称列表
    file_path (str, optional): 保存图像的文件路径，默认不保存
    """
    # 创建时间点用于绘制平滑曲线
    t_plot = np.linspace(min(t_data), max(t_data), 1000)

    # 计算模型预测值
    y_model = model_func(t_plot, *popt)

    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制实验数据点
    plt.scatter(t_data, y_data, color='red', label='Experimental Data', alpha=0.6, s=30)

    # 绘制拟合曲线
    plt.plot(t_plot, y_model, color='blue', label='Fitted Model', linewidth=2)

    # 添加参数标注
    param_text = "\n".join([f"{name} = {val:.3f} ± {err:.3f}" for name, val, err in zip(parameter_names, popt, perr)])
    plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 设置图表属性
    plt.title(title, fontsize=14)
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Enzyme Activity', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # 保存图像（如果指定了路径）
    if file_path:
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()


def main():
    data_path = r'C:\Users\31025\OneDrive\桌面\t\g149novickA.txt'
    # 加载V(t)模型数据
    t_V, y_V = load_bacterial_data(data_path)

    # 拟合V(t)模型
    popt_V, perr_V = fit_V_model(t_V, y_V)
    print("V(t)模型拟合结果:")
    print(f"τ = {popt_V[0]:.3f} ± {perr_V[0]:.3f}")

    img_path_V = r'C:\Users\31025\OneDrive\桌面\t\V_model_fit.png'
    # 绘制V(t)模型拟合结果
    plot_fit_results(t_V, y_V, V_model, popt_V, perr_V,
                     'V(t) Model Fit: TMG Permeation Process',
                     ['τ'], img_path_V)

    data_path_W = r'C:\Users\31025\OneDrive\桌面\t\g149novickB.txt'
    # 加载W(t)模型数据
    t_W, y_W = load_bacterial_data(data_path_W)

    # 拟合W(t)模型
    popt_W, perr_W = fit_W_model(t_W, y_W)
    print("\nW(t)模型拟合结果:")
    print(f"A = {popt_W[0]:.3f} ± {perr_W[0]:.3f}")
    print(f"τ = {popt_W[1]:.3f} ± {perr_W[1]:.3f}")

    img_path_W = r'C:\Users\31025\OneDrive\桌面\t\W_model_fit.png'
    # 绘制W(t)模型拟合结果
    plot_fit_results(t_W, y_W, W_model, popt_W, perr_W,
                     'W(t) Model Fit: β-Galactosidase Synthesis Process',
                     ['A', 'τ'], img_path_W)


if __name__ == "__main__":
    main()
