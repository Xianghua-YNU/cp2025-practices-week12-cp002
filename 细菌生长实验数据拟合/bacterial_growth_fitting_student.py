
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 设置matplotlib后端，避免与PyCharm冲突
plt.switch_backend('Agg')

def load_bacterial_data(file_path):
    """
    加载细菌生长实验数据
    
    参数:
    file_path (str): 数据文件路径
    
    返回:
    tuple: 包含时间(t)和酶活性(y)的元组
    """
    try:
        # 尝试使用逗号作为分隔符加载数据
        data = np.loadtxt(file_path, delimiter=',')
    except ValueError:
        # 尝试更灵活的加载方式
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # 跳过空行和注释行
                
                # 处理包含逗号或空格的行
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        # 尝试将前两个值转换为浮点数
                        data.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        continue  # 跳过无法转换的行
        
        if not data:
            raise ValueError(f"无法从文件 {file_path} 中加载有效数据")
        
        data = np.array(data)
    
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
    fig = plt.figure(figsize=(10, 6))
    
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
    else:
        # 如果未指定路径，保存到当前目录
        plt.savefig('fit_result.png', bbox_inches='tight')
    
    plt.close(fig)  # 关闭图形以释放资源

def main():
    """
    主函数：执行完整的数据分析和可视化流程
    """
    # 文件路径
    data_path = r'C:\Users\31025\OneDrive\桌面\t\g149novickA.txt'
    img_path_V = r'C:\Users\31025\OneDrive\桌面\t\V_model_fit.png'
    
    # 加载V(t)模型数据
    try:
        t_V, y_V = load_bacterial_data(data_path)
    except Exception as e:
        print(f"加载V(t)模型数据时出错: {e}")
        return
    
    # 拟合V(t)模型
    try:
        popt_V, perr_V = fit_V_model(t_V, y_V)
        print("V(t)模型拟合结果:")
        print(f"τ = {popt_V[0]:.3f} ± {perr_V[0]:.3f}")
        
        # 绘制V(t)模型拟合结果
        plot_fit_results(t_V, y_V, V_model, popt_V, perr_V, 
                         'V(t) Model Fit: TMG Permeation Process', 
                         ['τ'], img_path_V)
        print(f"V(t)模型拟合图像已保存至: {img_path_V}")
    except Exception as e:
        print(f"绘制V(t)模型拟合图像时出错: {e}")
    
    # 加载W(t)模型数据
    data_path_W = r'C:\Users\31025\OneDrive\桌面\t\g149novickB.txt'
    img_path_W = r'C:\Users\31025\OneDrive\桌面\t\W_model_fit.png'
    
    try:
        t_W, y_W = load_bacterial_data(data_path_W)
    except Exception as e:
        print(f"加载W(t)模型数据时出错: {e}")
        return
    
    # 拟合W(t)模型
    try:
        popt_W, perr_W = fit_W_model(t_W, y_W)
        print("\nW(t)模型拟合结果:")
        print(f"A = {popt_W[0]:.3f} ± {perr_W[0]:.3f}")
        print(f"τ = {popt_W[1]:.3f} ± {perr_W[1]:.3f}")
        
        # 绘制W(t)模型拟合结果
        plot_fit_results(t_W, y_W, W_model, popt_W, perr_W, 
                         'W(t) Model Fit: β-Galactosidase Synthesis Process', 
                         ['A', 'τ'], img_path_W)
        print(f"W(t)模型拟合图像已保存至: {img_path_W}")
    except Exception as e:
        print(f"绘制W(t)模型拟合图像时出错: {e}")

if __name__ == "__main__":
    main()    
