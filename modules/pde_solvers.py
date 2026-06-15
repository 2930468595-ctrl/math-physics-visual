"""
偏微分方程求解与可视化模块

本模块实现两类重要的偏微分方程可视化：
1. 热传导方程（扩散方程）：∂u/∂t = α ∂²u/∂x²
2. 波动方程：∂²u/∂t² = c² ∂²u/∂x²

物理意义：
- 热传导方程描述热量在物体中的扩散过程
- 波动方程描述波的传播、反射、叠加等现象
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import io
import base64

# ==================== 辅助函数 ====================

def initialize_grid_1d(x_min, x_max, n_points):
    """
    初始化一维空间网格
    
    参数：
        x_min: 空间区间左端点
        x_max: 空间区间右端点
        n_points: 空间离散点数
    
    返回：
        x: 空间坐标数组
        dx: 空间步长
    """
    x = np.linspace(x_min, x_max, n_points)
    dx = x[1] - x[0]
    return x, dx


def initialize_grid_2d(x_min, x_max, y_min, y_max, n_points):
    """
    初始化二维空间网格（用于热传导方程的2D可视化）
    
    参数：
        x_min, x_max: x方向空间区间
        y_min, y_max: y方向空间区间
        n_points: 每个方向离散点数
    
    返回：
        x, y: 空间坐标数组
        X, Y: 网格矩阵
        dx, dy: 空间步长
    """
    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return x, y, X, Y, dx, dy


# ==================== 热传导方程求解器 ====================

def solve_heat_equation_1d(
    u0_func,
    x_min=0.0,
    x_max=1.0,
    n_points=100,
    alpha=0.01,
    t_max=1.0,
    n_time_steps=200,
    left_bc='dirichlet',
    right_bc='dirichlet',
    bc_left_value=0.0,
    bc_right_value=0.0
):
    """
    求解一维热传导方程（显式有限差分法）
    
   方程：∂u/∂t = α ∂²u/∂x²
    
    参数：
        u0_func: 初始条件函数 u(x, 0)
        x_min, x_max: 空间区间
        n_points: 空间离散点数
        alpha: 热扩散系数
        t_max: 最大模拟时间
        n_time_steps: 时间离散步数
        left_bc, right_bc: 边界条件类型（'dirichlet'或'neumann'）
        bc_left_value, bc_right_value: 边界条件值
    
    返回：
        x: 空间坐标
        t: 时间坐标
        u: 解的数组，形状为 (n_time_steps, n_points)
    """
    # 初始化空间网格
    x, dx = initialize_grid_1d(x_min, x_max, n_points)
    
    # 时间步长（满足稳定性条件：dt <= dx²/(2α)）
    dt = t_max / n_time_steps
    r = alpha * dt / (dx ** 2)  # 网格比
    
    # 稳定性条件检查
    if r > 0.5:
        # 自动调整时间步数确保稳定
        n_time_steps = int(2 * alpha * t_max / (dx ** 2)) + 1
        dt = t_max / n_time_steps
        r = alpha * dt / (dx ** 2)
    
    # 初始化时间数组
    t = np.linspace(0, t_max, n_time_steps)
    
    # 初始化解数组
    u = np.zeros((n_time_steps, n_points))
    
    # 设置初始条件
    u[0, :] = u0_func(x)
    
    # 设置边界条件（第一行和最后一列）
    if left_bc == 'dirichlet':
        u[:, 0] = bc_left_value
    if right_bc == 'dirichlet':
        u[:, -1] = bc_right_value
    
    # 显式差分迭代（ FTCS 格式）
    # u[i+1, j] = r * u[i, j-1] + (1 - 2r) * u[i, j] + r * u[i, j+1]
    for n in range(n_time_steps - 1):
        for j in range(1, n_points - 1):
            u[n + 1, j] = r * u[n, j - 1] + (1 - 2 * r) * u[n, j] + r * u[n, j + 1]
        
        # 边界条件处理
        if left_bc == 'neumann':
            # ∂u/∂x = 0 -> 虚节点方法
            u[n + 1, 0] = u[n + 1, 1]
        else:
            u[n + 1, 0] = bc_left_value
            
        if right_bc == 'neumann':
            u[n + 1, -1] = u[n + 1, -2]
        else:
            u[n + 1, -1] = bc_right_value
    
    return x, t, u


def solve_heat_equation_2d(
    u0_func,
    x_min=0.0,
    x_max=1.0,
    y_min=0.0,
    y_max=1.0,
    n_points=50,
    alpha=0.01,
    t_max=0.5,
    n_time_steps=100
):
    """
    求解二维热传导方程（显式有限差分法）
    
    方程：∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)
    
    参数：
        u0_func: 初始条件函数 u(x, y, 0)
        其他参数同上
    
    返回：
        x, y: 空间坐标
        X, Y: 网格矩阵
        u: 解数组，形状为 (n_time_steps, n_points, n_points)
    """
    # 初始化二维网格
    x, y, X, Y, dx, dy = initialize_grid_2d(x_min, x_max, y_min, y_max, n_points)
    
    # 时间步长
    dt = t_max / n_time_steps
    r_x = alpha * dt / (dx ** 2)
    r_y = alpha * dt / (dy ** 2)
    
    # 稳定性条件检查
    if r_x + r_y > 0.25:
        n_time_steps = int((r_x + r_y) * 4 * t_max / (alpha * 0.25)) + 1
        dt = t_max / n_time_steps
        r_x = alpha * dt / (dx ** 2)
        r_y = alpha * dt / (dy ** 2)
    
    t = np.linspace(0, t_max, n_time_steps)
    
    # 初始化解数组
    u = np.zeros((n_time_steps, n_points, n_points))
    u[0, :, :] = u0_func(X, Y)
    
    # 二维权重因子
    center = 1 - 2 * r_x - 2 * r_y
    
    # 显式差分迭代
    for n in range(n_time_steps - 1):
        u[n + 1, 1:-1, 1:-1] = (
            r_x * u[n, 1:-1, :-2] + r_x * u[n, 1:-1, 2:] +
            r_y * u[n, :-2, 1:-1] + r_y * u[n, 2:, 1:-1] +
            center * u[n, 1:-1, 1:-1]
        )
        # 边界条件（零迪利克雷）
        u[n + 1, :, 0] = 0
        u[n + 1, :, -1] = 0
        u[n + 1, 0, :] = 0
        u[n + 1, -1, :] = 0
    
    return x, y, X, Y, t, u


# ==================== 波动方程求解器 ====================

def solve_wave_equation_1d(
    u0_func,
    v0_func,
    x_min=0.0,
    x_max=1.0,
    n_points=200,
    c=1.0,
    t_max=2.0,
    n_time_steps=400,
    left_bc='fixed',
    right_bc='fixed'
):
    """
    求解一维波动方程（有限差分法）
    
    方程：∂²u/∂t² = c² ∂²u/∂x²
    
    参数：
        u0_func: 初始位移函数 u(x, 0)
        v0_func: 初始速度函数 ∂u/∂t (x, 0)
        x_min, x_max: 空间区间
        n_points: 空间离散点数
        c: 波速
        t_max: 最大模拟时间
        n_time_steps: 时间离散步数
        left_bc, right_bc: 边界条件（'fixed'或'free'）
    
    返回：
        x: 空间坐标
        t: 时间坐标
        u: 解的数组，形状为 (n_time_steps, n_points)
    """
    # 初始化空间网格
    x, dx = initialize_grid_1d(x_min, x_max, n_points)
    
    # 时间步长（满足 CFL 条件：c * dt / dx <= 1）
    dt = t_max / n_time_steps
    r = c * dt / dx  # CFL数
    
    # CFL稳定性检查
    if r > 1:
        n_time_steps = int(2 * c * t_max / dx) + 1
        dt = t_max / n_time_steps
        r = c * dt / dx
    
    t = np.linspace(0, t_max, n_time_steps)
    
    # 初始化解数组
    u = np.zeros((n_time_steps, n_points))
    
    # 设置初始条件：第一行和第二行
    u[0, :] = u0_func(x)
    
    # 使用中心差分计算初始速度来初始化第二行
    # ∂²u/∂t² = c² ∂²u/∂x² -> u[1] = u[0] + dt * v0 + (dt²/2) * c² * u_xx
    if v0_func is not None:
        # 计算二阶空间导数近似
        u_xx = np.zeros(n_points)
        u_xx[1:-1] = (u[0, :-2] - 2 * u[0, 1:-1] + u[0, 2:]) / (dx ** 2)
        # 边界处用一阶近似
        u_xx[0] = u_xx[1]
        u_xx[-1] = u_xx[-2]
        
        u[1, :] = u[0, :] + dt * v0_func(x) + 0.5 * (c ** 2) * (dt ** 2) * u_xx
    else:
        # 零初始速度：u[1] = u[0]
        u[1, :] = u[0, :]
    
    # 显式差分迭代（Wave Equation FTCS）
    # u[n+1, j] = 2*u[n, j] - u[n-1, j] + r²*(u[n, j+1] - 2*u[n, j] + u[n, j-1])
    for n in range(1, n_time_steps - 1):
        for j in range(1, n_points - 1):
            u[n + 1, j] = (
                2 * u[n, j] - u[n - 1, j] +
                (r ** 2) * (u[n, j + 1] - 2 * u[n, j] + u[n, j - 1])
            )
        
        # 边界条件处理
        if left_bc == 'fixed':
            # 固定端：u = 0
            u[n + 1, 0] = 0
        else:
            # 自由端：∂u/∂x = 0
            u[n + 1, 0] = u[n + 1, 1]
            
        if right_bc == 'fixed':
            u[n + 1, -1] = 0
        else:
            u[n + 1, -1] = u[n + 1, -2]
    
    return x, t, u, r


def solve_drumhead_equation(
    u0_func,
    x_min=-1.0,
    x_max=1.0,
    y_min=-1.0,
    y_max=1.0,
    n_points=50,
    c=1.0,
    t_max=2.0,
    n_time_steps=200
):
    """
    求解二维波动方程（圆形膜/鼓面）
    
    方程：∂²u/∂t² = c² (∂²u/∂x² + ∂²u/∂y²)
    
    参数：
        u0_func: 初始位移函数 u(x, y, 0)
        其他参数同上
    
    返回：
        x, y: 空间坐标
        X, Y: 网格矩阵
        t: 时间坐标
        u: 解数组
    """
    x, y, X, Y, dx, dy = initialize_grid_2d(x_min, x_max, y_min, y_max, n_points)
    
    dt = t_max / n_time_steps
    r_x = c * dt / dx
    r_y = c * dt / dy
    
    # CFL条件
    if r_x > 1 or r_y > 1:
        factor = max(r_x, r_y)
        n_time_steps = int(factor * 2 * t_max / c) + 1
        dt = t_max / n_time_steps
        r_x = c * dt / dx
        r_y = c * dt / dy
    
    t = np.linspace(0, t_max, n_time_steps)
    u = np.zeros((n_time_steps, n_points, n_points))
    
    # 圆形膜：计算径向距离
    R = np.sqrt(X ** 2 + Y ** 2)
    
    # 初始条件
    u[0, :, :] = u0_func(X, Y)
    u[1, :, :] = u[0, :, :]  # 零初始速度
    
    # 中心权重
    center = 2 - 2 * (r_x ** 2) - 2 * (r_y ** 2)
    
    # 差分迭代
    for n in range(1, n_time_steps - 1):
        u[n + 1, 1:-1, 1:-1] = (
            2 * u[n, 1:-1, 1:-1] - u[n - 1, 1:-1, 1:-1] +
            (r_x ** 2) * (u[n, 1:-1, :-2] - 2 * u[n, 1:-1, 1:-1] + u[n, 1:-1, 2:]) +
            (r_y ** 2) * (u[n, :-2, 1:-1] - 2 * u[n, 1:-1, 1:-1] + u[n, 2:, 1:-1])
        )
        
        # 圆形膜边界条件：固定边界 u=0
        u[n + 1, R > 0.95] = 0
    
    return x, y, X, Y, t, u


# ==================== 可视化生成函数 ====================

def generate_heat_equation_frames(x, t, u, n_frames=50):
    """
    为热传导方程生成动画帧
    
    参数：
        x, t, u: solve_heat_equation_1d 的输出
        n_frames: 生成的帧数
    
    返回：
        frames: 图像帧列表
    """
    # 选择时间索引
    time_indices = np.linspace(0, len(t) - 1, n_frames, dtype=int)
    
    frames = []
    for idx in time_indices:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, u[idx], 'b-', linewidth=2, label=f't = {t[idx]:.3f}')
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(np.min(u) - 0.1, np.max(u) + 0.1)
        ax.set_xlabel('位置 x', fontsize=12)
        ax.set_ylabel('温度 u', fontsize=12)
        ax.set_title('一维热传导方程解的演化', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 标记边界条件
        ax.axhline(y=u[idx, 0], color='r', linestyle='--', alpha=0.5, label='边界温度')
        ax.axhline(y=u[idx, -1], color='r', linestyle='--', alpha=0.5)
        
        fig.tight_layout()
        
        # 转换为base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        frames.append(img_str)
        plt.close(fig)
    
    return frames


def generate_wave_equation_frames(x, t, u, r, n_frames=50):
    """
    为波动方程生成动画帧
    
    参数：
        x, t, u: solve_wave_equation_1d 的输出
        r: CFL数
        n_frames: 生成的帧数
    
    返回：
        frames: 图像帧列表
    """
    time_indices = np.linspace(0, len(t) - 1, n_frames, dtype=int)
    
    frames = []
    y_min = np.min(u) - 0.2
    y_max = np.max(u) + 0.2
    
    for idx in time_indices:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 绘制当前时刻的波
        ax.plot(x, u[idx], 'b-', linewidth=2, label=f't = {t[idx]:.3f}')
        
        # 绘制平衡位置
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # 标记节点
        ax.set_xlim(x[0], x[-1])
        if y_min != y_max:
            ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel('位置 x', fontsize=12)
        ax.set_ylabel('位移 u', fontsize=12)
        ax.set_title(f'一维波动方程 (cΔt/Δx = {r:.3f})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        frames.append(img_str)
        plt.close(fig)
    
    return frames


def generate_heat_2d_frames(x, y, X, Y, t, u, n_frames=30):
    """
    为二维热传导方程生成动画帧
    
    返回：
        frames: base64编码的图像帧列表
    """
    time_indices = np.linspace(0, len(t) - 1, n_frames, dtype=int)
    
    frames = []
    vmin = np.min(u)
    vmax = np.max(u)
    
    for idx in time_indices:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.contourf(X, Y, u[idx], levels=20, cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'二维热传导方程 (t = {t[idx]:.3f})', fontsize=14)
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, label='温度')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        frames.append(img_str)
        plt.close(fig)
    
    return frames


def generate_drumhead_frames(x, y, X, Y, t, u, n_frames=50):
    """
    为鼓面振动生成动画帧（二维波动方程）
    
    返回：
        frames: base64编码的图像帧列表
    """
    time_indices = np.linspace(0, len(t) - 1, n_frames, dtype=int)
    
    frames = []
    vmax = np.max(np.abs(u)) + 0.01
    
    for idx in time_indices:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.contourf(X, Y, u[idx], levels=20, cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'鼓膜振动 (t = {t[idx]:.3f})', fontsize=14)
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, label='位移')
        
        # 绘制边界圆
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        frames.append(img_str)
        plt.close(fig)
    
    return frames


# ==================== 静态图生成函数 ====================

def plot_heat_equation_solution(x, t, u, time_idx=-1):
    """
    绘制热传导方程的解曲线
    
    参数：
        x, t, u: 解的数据
        time_idx: 要绘制的时间索引，-1表示最终时刻
    
    返回：
        fig: matplotlib图像对象
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # 绘制几个不同时刻的曲线
    n_curves = min(5, len(t))
    time_indices = np.linspace(0, len(t) - 1, n_curves, dtype=int)
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_curves))
    
    for i, idx in enumerate(time_indices):
        ax.plot(x, u[idx], color=colors[i], linewidth=2, 
                label=f't = {t[idx]:.3f}')
    
    ax.set_xlabel('位置 x', fontsize=12)
    ax.set_ylabel('温度 u', fontsize=12)
    ax.set_title('一维热传导方程：温度分布随时间演化', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_wave_equation_solution(x, t, u, time_indices=None):
    """
    绘制波动方程的解曲线（多个时刻）
    
    参数：
        x, t, u: 解的数据
        time_indices: 要绘制的时间索引列表，None表示自动选择
    
    返回：
        fig: matplotlib图像对象
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if time_indices is None:
        n_curves = min(6, len(t))
        time_indices = np.linspace(0, len(t) - 1, n_curves, dtype=int)
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(time_indices)))
    
    for i, idx in enumerate(time_indices):
        ax.plot(x, u[idx], color=colors[i], linewidth=1.5, 
                label=f't = {t[idx]:.3f}', alpha=0.8)
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('位置 x', fontsize=12)
    ax.set_ylabel('位移 u', fontsize=12)
    ax.set_title('一维波动方程：波传播过程', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_heat_2d_snapshot(X, Y, u, time_idx=-1):
    """
    绘制二维热传导方程的快照
    
    参数：
        X, Y: 网格坐标
        u: 解数组
        time_idx: 时间索引
    
    返回：
        fig: matplotlib图像对象
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    levels = np.linspace(np.min(u), np.max(u), 20)
    im = ax.contourf(X, Y, u[time_idx], levels=levels, cmap='hot')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'二维热传导方程快照 (t = {time_idx * 0.01:.3f})', fontsize=14)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax, label='温度')
    
    fig.tight_layout()
    return fig


def plot_drumhead_snapshot(X, Y, u, time_idx=-1):
    """
    绘制鼓膜振动的快照
    
    参数：
        X, Y: 网格坐标
        u: 解数组
        time_idx: 时间索引
    
    返回：
        fig: matplotlib图像对象
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    levels = np.linspace(-np.max(np.abs(u)), np.max(np.abs(u)), 20)
    im = ax.contourf(X, Y, u[time_idx], levels=levels, cmap='RdBu')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'鼓膜振动模式 (t = {time_idx * 0.01:.3f})', fontsize=14)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax, label='位移')
    
    # 绘制边界圆
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    
    fig.tight_layout()
    return fig


# ==================== 预设初始条件 ====================

# 一维热传导方程的初始条件
HEAT_INITIAL_CONDITIONS_1D = {
    'gaussian': {
        'name': '高斯分布',
        'func': lambda x: np.exp(-50 * (x - 0.5) ** 2),
        'description': '初始温度呈高斯分布（脉冲热源）'
    },
    'sine': {
        'name': '正弦分布',
        'func': lambda x: np.sin(np.pi * x),
        'description': '初始温度呈正弦波形分布'
    },
    'step': {
        'name': '阶梯分布',
        'func': lambda x: np.where(x < 0.5, 1.0, 0.0),
        'description': '左半部分为1，右半部分为0'
    },
    'double_peak': {
        'name': '双峰分布',
        'func': lambda x: np.exp(-50 * (x - 0.25) ** 2) + np.exp(-50 * (x - 0.75) ** 2),
        'description': '两个相邻的高斯脉冲'
    }
}

# 二维热传导方程的初始条件
HEAT_INITIAL_CONDITIONS_2D = {
    'point': {
        'name': '点热源',
        'func': lambda X, Y: np.exp(-100 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)),
        'description': '中心点热源'
    },
    'line': {
        'name': '线热源',
        'func': lambda X, Y: np.exp(-100 * (X - 0.5) ** 2),
        'description': '沿y方向的线热源'
    },
    'ring': {
        'name': '环形热源',
        'func': lambda X, Y: np.exp(-100 * ((np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.3) ** 2)),
        'description': '圆形环状热源'
    }
}

# 一维波动方程的初始条件
WAVE_INITIAL_CONDITIONS_1D = {
    'sine': {
        'name': '正弦波',
        'u0': lambda x: np.sin(np.pi * x),
        'v0': lambda x: np.zeros_like(x),
        'description': '单频正弦波初始位移'
    },
    'gaussian': {
        'name': '高斯波包',
        'u0': lambda x: np.exp(-50 * (x - 0.5) ** 2),
        'v0': lambda x: np.zeros_like(x),
        'description': '局域化的高斯波包'
    },
    'square': {
        'name': '方波脉冲',
        'u0': lambda x: np.where((x > 0.3) & (x < 0.7), 1.0, 0.0),
        'v0': lambda x: np.zeros_like(x),
        'description': '矩形区域的脉冲'
    },
    'wave_packet': {
        'name': '行波波包',
        'u0': lambda x: np.exp(-50 * (x - 0.5) ** 2) * np.sin(20 * np.pi * x),
        'v0': lambda x: np.zeros_like(x),
        'description': '带有振荡的高斯包络（波包）'
    }
}

# 二维波动方程（鼓膜）的初始条件
DRUMHEAD_INITIAL_CONDITIONS = {
    'circular': {
        'name': '圆形凸起',
        'func': lambda X, Y: np.where(np.sqrt(X ** 2 + Y ** 2) < 0.3, 
                                      np.cos(np.pi * np.sqrt(X ** 2 + Y ** 2) / 0.3), 0),
        'description': '中心圆形区域的余弦凸起'
    },
    'sine': {
        'name': '正弦模式',
        'func': lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y),
        'description': '二维正弦模式'
    },
    'cross': {
        'name': '十字形',
        'func': lambda X, Y: np.where(np.abs(X) < 0.1, 1.0, np.where(np.abs(Y) < 0.1, 1.0, 0.0)),
        'description': '十字形状的初始位移'
    }
}


# ==================== 获取可用的 PDE 类型 ====================

def get_available_pde_types():
    """
    获取所有可用的偏微分方程类型
    
    返回：
        dict: 包含方程类型和子类型的字典
    """
    return {
        'heat_1d': {
            'name': '一维热传导方程',
            'equation': '∂u/∂t = α ∂²u/∂x²',
            'description': '描述热量在一维杆中的扩散',
            'initial_conditions': HEAT_INITIAL_CONDITIONS_1D
        },
        'heat_2d': {
            'name': '二维热传导方程',
            'equation': '∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)',
            'description': '描述热量在二维平面中的扩散',
            'initial_conditions': HEAT_INITIAL_CONDITIONS_2D
        },
        'wave_1d': {
            'name': '一维波动方程',
            'equation': '∂²u/∂t² = c² ∂²u/∂x²',
            'description': '描述弦上的波传播',
            'initial_conditions': WAVE_INITIAL_CONDITIONS_1D
        },
        'drumhead': {
            'name': '二维波动方程（鼓膜）',
            'equation': '∂²u/∂t² = c² (∂²u/∂x² + ∂²u/∂y²)',
            'description': '描述圆形膜的振动',
            'initial_conditions': DRUMHEAD_INITIAL_CONDITIONS
        }
    }


# 实时 JavaScript Canvas 动画（位于 modules/realtime_animations.py）
from .realtime_animations import (
    generate_realtime_heat_1d_html,
    generate_realtime_heat_2d_html,
    generate_realtime_wave_1d_html,
    generate_realtime_drumhead_html,
)
