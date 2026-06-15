"""
特殊函数可视化模块

本模块实现数学物理中重要的特殊函数可视化：
1. 贝塞尔函数 J_n(x) - 柱坐标波动问题的解
2. 勒让德多项式 P_n(x) - 球坐标问题的解
3. 厄米特函数 H_n(x) - 量子谐振子
4. 拉盖尔多项式 L_n(x) - 氢原子问题
5. 切比雪夫多项式 T_n(x), U_n(x) - 逼近理论
6. 伽马函数 Γ(x) 和贝塔函数 B(x,y)

物理意义：
- 贝塞尔函数：圆形/柱形 membranes 的振动模式
- 勒让德多项式：球坐标拉普拉斯方程的解
- 厄米特函数：量子谐振子的波函数
- 拉盖尔多项式：氢原子波函数的角动量部分
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from matplotlib.animation import FuncAnimation
import io
import base64


# ==================== 贝塞尔函数 ====================

def plot_bessel_functions(n_max=4, x_max=20, n_points=500):
    """
    绘制第一类贝塞尔函数 J_n(x) 的图像
    
    参数：
        n_max: 最高阶数
        x_max: x轴最大取值
        n_points: 采样点数
    
    返回：
        fig: matplotlib图像对象
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.linspace(0.01, x_max, n_points)
    colors = plt.cm.viridis(np.linspace(0, 1, n_max + 1))
    
    for n in range(n_max + 1):
        # 使用 scipy.special.jn 计算第一类贝塞尔函数
        jn_values = special.jn(n, x)
        ax.plot(x, jn_values, color=colors[n], linewidth=2, label=f'$J_{n}(x)$')
        
        # 标记部分零点
        zeros = special.jn_zeros(n, 3)  # 每个阶数取前3个零点
        zeros = zeros[zeros <= x_max]
        ax.scatter(zeros, np.zeros_like(zeros), color=colors[n], s=30, zorder=5)
    
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$J_n(x)$', fontsize=14)
    ax.set_title('第一类贝塞尔函数 $J_n(x)$', fontsize=16)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.5, 1.1)
    
    fig.tight_layout()
    return fig


def plot_bessel_functions_2d(n_max=3, m_max=3, r_max=10):
    """
    绘制二维圆形膜的振动模式（贝塞尔函数的典型应用）
    
    参数：
        n_max: 径向阶数最大值
        m_max: 角向阶数最大值
        r_max: 径向最大距离
    
    返回：
        fig: matplotlib图像对象网格
    """
    fig, axes = plt.subplots(n_max, m_max, figsize=(3 * m_max, 3 * n_max))
    
    theta = np.linspace(0, 2 * np.pi, 100)
    r = np.linspace(0, r_max, 100)
    R, Theta = np.meshgrid(r, theta)
    
    # 将极坐标转换为笛卡尔坐标
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_max * m_max))
    
    idx = 0
    for n in range(n_max):
        for m in range(m_max):
            ax = axes[n, m] if n_max > 1 and m_max > 1 else axes[max(n, m)]
            
            # 计算 J_n(k*r) * cos(n*theta) 模式
            k = special.jn_zeros(n, 1)[0] / r_max  # 零点决定波数
            Z = special.jn(n, k * R) * np.cos(n * Theta)
            
            # 绘制填充等高线
            levels = np.linspace(-1, 1, 20)
            cf = ax.contourf(X, Y, Z, levels=levels, cmap='RdBu', extend='both')
            ax.set_title(f'模式 $({n},{m})$', fontsize=10)
            ax.set_aspect('equal')
            ax.axis('off')
            idx += 1
    
    fig.tight_layout()
    return fig


def plot_besselj_zero_convergence():
    """
    绘制贝塞尔函数零点随阶数增加的变化趋势
    帮助理解高阶贝塞尔函数的渐近行为
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：前几个阶数的零点分布
    ax1 = axes[0]
    n_values = np.arange(0, 6)
    zero_counts = 5  # 每个阶数取前5个零点
    
    for n in n_values:
        zeros = special.jn_zeros(n, zero_counts)
        ax1.scatter(np.full(zero_counts, n), zeros, s=30, alpha=0.7)
    
    ax1.set_xlabel('阶数 $n$', fontsize=12)
    ax1.set_ylabel('零点位置', fontsize=12)
    ax1.set_title('贝塞尔函数 $J_n(x)$ 的零点分布', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 右图：相邻零点间距
    ax2 = axes[1]
    n = 0
    zeros = special.jn_zeros(n, 20)
    spacing = np.diff(zeros)
    
    ax2.bar(range(1, len(spacing) + 1), spacing, alpha=0.7)
    ax2.axhline(y=np.pi, color='r', linestyle='--', label='渐近间距 $\\pi$')
    ax2.set_xlabel('第 $k$ 个间距', fontsize=12)
    ax2.set_ylabel('零点间距 $\\Delta x_k$', fontsize=12)
    ax2.set_title(f'$J_0(x)$ 零点间距（渐近 $\\pi \\approx 3.14$）', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


# ==================== 勒让德多项式 ====================

def plot_legendre_polynomials(n_max=7, x_range=(-1, 1)):
    """
    绘制勒让德多项式 P_n(x) 的图像
    
    参数：
        n_max: 最高阶数
        x_range: x轴取值范围
    
    返回：
        fig: matplotlib图像对象
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.linspace(x_range[0], x_range[1], 500)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_max + 1))
    
    for n in range(n_max + 1):
        # 使用 scipy.special.legendre 计算勒让德多项式
        pn_values = special.legendre(n)(x)
        ax.plot(x, pn_values, color=colors[n], linewidth=2, label=f'$P_{n}(x)$')
    
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$P_n(x)$', fontsize=14)
    ax.set_title('勒让德多项式 $P_n(x)$', fontsize=16)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_range)
    ax.set_ylim(-1.3, 1.3)
    
    # 在 x = ±1 处标记（勒让德多项式在这些点有确定值）
    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=-1, color='gray', linestyle=':', alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_associated_legendre(n_max=3, m_max=3, x_range=(-1, 1)):
    """
    绘制连带勒让德函数 P_n^m(x)
    
    参数：
        n_max: 最高阶数
        m_max: 最高次数
        x_range: x轴取值范围
    
    返回：
        fig: matplotlib图像对象网格
    """
    fig, axes = plt.subplots(n_max + 1, m_max + 1, figsize=(12, 10))
    
    x = np.linspace(x_range[0], x_range[1], 300)
    
    for n in range(n_max + 1):
        for m in range(m_max + 1):
            ax = axes[n, m]
            
            if m <= n:
                # 计算连带勒让德函数
                plm_values = special.sph_harm(0, n, 0, np.arccos(np.clip(x, -1, 1)))
                plm_values = np.real(plm_values)
                
                ax.plot(x, plm_values, linewidth=1.5)
                ax.set_title(f'$P_{{{n}}}^{{{m}}}(x)$' if m > 0 else f'$P_{n}(x)$', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax.set_ylim(-1.5, 1.5)
            else:
                ax.axis('off')
            
            if n == n_max:
                ax.set_xlabel('$x$', fontsize=10)
    
    fig.tight_layout()
    fig.suptitle('连带勒让德函数 $P_n^m(x)$', fontsize=14, y=1.02)
    return fig


# ==================== 厄米特函数（量子谐振子） ====================

def plot_hermite_functions(n_max=6, x_range=(-8, 8)):
    """
    绘制厄米特-高斯函数（量子谐振子波函数）
    
    波函数形式：ψ_n(x) = (1/√(2^n n!)) * H_n(x) * exp(-x²/2)
    
    参数：
        n_max: 最高量子数
        x_range: x轴取值范围
    
    返回：
        fig: matplotlib图像对象
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.linspace(x_range[0], x_range[1], 500)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, n_max + 1))
    
    # 绘制谐振子势阱
    potential = 0.5 * x ** 2
    ax.plot(x, potential, 'k-', linewidth=2, alpha=0.5, label='$V(x) = \\frac{1}{2}x^2$')
    
    # 绘制能级
    for n in range(n_max + 1):
        energy = n + 0.5
        ax.axhline(y=energy, color=colors[n], linestyle='--', alpha=0.5, linewidth=1)
        
        # 计算归一化的波函数
        hermite = special.hermite(n)
        psi = special.eval_hermite(n, x) * np.exp(-x ** 2 / 2) / np.sqrt(2 ** n * special.factorial(n) * np.sqrt(np.pi))
        
        ax.plot(x, energy + psi, color=colors[n], linewidth=2, label=f'$n={n}$, $E_{n}={energy}$')
    
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('能量 / 波函数', fontsize=14)
    ax.set_title('量子谐振子：厄米特-高斯函数 $\\psi_n(x)$', fontsize=16)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_range)
    ax.set_ylim(-0.5, n_max + 2)
    
    fig.tight_layout()
    return fig


def plot_hermite_polynomials(n_max=5, x_range=(-5, 5)):
    """
    绘制纯厄米特多项式 H_n(x)（不包含高斯因子）
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.linspace(x_range[0], x_range[1], 500)
    colors = plt.cm.cool(np.linspace(0, 1, n_max + 1))
    
    for n in range(n_max + 1):
        Hn = special.eval_hermite(n, x)
        ax.plot(x, Hn, color=colors[n], linewidth=2, label=f'$H_{n}(x)$')
    
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$H_n(x)$', fontsize=14)
    ax.set_title('厄米特多项式 $H_n(x)$', fontsize=16)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_range)
    
    fig.tight_layout()
    return fig


# ==================== 拉盖尔多项式 ====================

def plot_laguerre_polynomials(n_max=6, x_range=(0, 10)):
    """
    绘制（广义）拉盖尔多项式 L_n^α(x)
    
    参数：
        n_max: 最高阶数
        x_range: x轴取值范围（拉盖尔多项式定义在 x >= 0）
    
    返回：
        fig: matplotlib图像对象
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.linspace(max(0.01, x_range[0]), x_range[1], 500)
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, n_max + 1))
    
    for n in range(n_max + 1):
        # α = 0 (标准拉盖尔多项式)
        Ln = special.eval_genlaguerre(n, 0, x)
        ax.plot(x, Ln, color=colors[n], linewidth=2, label=f'$L_{n}(x)$')
    
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$L_n(x)$', fontsize=14)
    ax.set_title('拉盖尔多项式 $L_n(x)$（广义，$\\alpha=0$）', fontsize=16)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_range)
    
    fig.tight_layout()
    return fig


def plot_associated_laguerre(n_max=4, k_max=2, x_range=(0, 12)):
    """
    绘制连带拉盖尔多项式 L_n^k(x)
    """
    fig, axes = plt.subplots(1, k_max + 1, figsize=(4 * (k_max + 1), 5))
    
    x = np.linspace(max(0.01, x_range[0]), x_range[1], 500)
    
    if k_max == 0:
        axes = [axes]
    
    colors_set = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens]
    
    for k in range(k_max + 1):
        ax = axes[k]
        colors = colors_set[k % len(colors_set)](np.linspace(0.3, 0.9, n_max + 1))
        
        for n in range(n_max + 1):
            if n >= k:  # 拉盖尔多项式要求 n >= k
                Lnk = special.eval_genlaguerre(n - k, k, x)
                ax.plot(x, Lnk, color=colors[n - k], linewidth=2, 
                       label=f'$L_{{{n}-{k}}}^{{{k}}}$')
        
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel(f'$L_n^{{{k}}}(x)$', fontsize=12)
        ax.set_title(f'连带拉盖尔（$\\alpha={k}$）', fontsize=12)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_range)
    
    fig.tight_layout()
    return fig


# ==================== 切比雪夫多项式 ====================

def plot_chebyshev_Tn(n_max=7, x_range=(-1, 1)):
    """
    绘制第一类切比雪夫多项式 T_n(x) = cos(n arccos x)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.linspace(x_range[0], x_range[1], 500)
    colors = plt.cm.Spectral(np.linspace(0.1, 0.9, n_max + 1))
    
    for n in range(n_max + 1):
        Tn = special.chebyt(n)(x)
        ax.plot(x, Tn, color=colors[n], linewidth=2, label=f'$T_{n}(x)$')
    
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$T_n(x)$', fontsize=14)
    ax.set_title('第一类切比雪夫多项式 $T_n(x) = \\cos(n \\arccos x)$', fontsize=16)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=-1, color='gray', linestyle=':', alpha=0.3)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_range)
    ax.set_ylim(-1.5, 1.5)
    
    fig.tight_layout()
    return fig


def plot_chebyshev_Un(n_max=6, x_range=(-1, 1)):
    """
    绘制第二类切比雪夫多项式 U_n(x)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.linspace(x_range[0], x_range[1], 500)
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_max + 1))
    
    for n in range(n_max + 1):
        Un = special.chebyu(n)(x)
        ax.plot(x, Un, color=colors[n], linewidth=2, label=f'$U_{n}(x)$')
    
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$U_n(x)$', fontsize=14)
    ax.set_title('第二类切比雪夫多项式 $U_n(x)$', fontsize=16)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_range)
    
    fig.tight_layout()
    return fig


# ==================== 伽马函数和贝塔函数 ====================

def plot_gamma_function(x_range=(-5, 5), n_points=1000):
    """
    绘制伽马函数 Γ(x)
    
    性质：Γ(n) = (n-1)! for positive integers n
         Γ(z+1) = z * Γ(z) （泛函方程）
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 正半轴
    x1 = np.linspace(0.01, x_range[1], n_points)
    gamma1 = special.gamma(x1)
    
    axes[0].plot(x1, gamma1, 'b-', linewidth=2)
    axes[0].set_xlabel('$x$', fontsize=14)
    axes[0].set_ylabel('$\\Gamma(x)$', fontsize=14)
    axes[0].set_title('伽马函数 $\\Gamma(x)$ - 正半轴', fontsize=16)
    axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[0].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-10, 20)
    
    # 标记整数点的值
    for n in range(1, 7):
        axes[0].scatter([n], [special.gamma(n)], s=100, zorder=5)
        axes[0].annotate(f'$\\Gamma({n})={special.gamma(n):.0f}$', 
                        (n, special.gamma(n)), textcoords="offset points",
                        xytext=(10, 10), fontsize=10)
    
    # 负半轴（使用反射公式处理极点）
    x2 = np.linspace(x_range[0], -0.01, n_points)
    gamma2 = special.gamma(x2)
    
    # 使用反射公式避免数值问题
    # Γ(x)Γ(1-x) = π/sin(πx)
    
    axes[1].plot(x2, gamma2, 'r-', linewidth=2)
    axes[1].set_xlabel('$x$', fontsize=14)
    axes[1].set_ylabel('$\\Gamma(x)$', fontsize=14)
    axes[1].set_title('伽马函数 $\\Gamma(x)$ - 负半轴（注意极点）', fontsize=16)
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[1].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-10, 10)
    
    # 标记一些特殊点
    special_points = [-0.5, -1.5, -2.5]
    for sp in special_points:
        if sp > x_range[0]:
            axes[1].scatter([sp], [special.gamma(sp)], s=100, color='red', zorder=5)
    
    fig.tight_layout()
    return fig


def plot_beta_function():
    """
    绘制贝塔函数 B(x,y) = Γ(x)Γ(y)/Γ(x+y) 的等高线图
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = np.linspace(0.1, 5, 200)
    y = np.linspace(0.1, 5, 200)
    X, Y = np.meshgrid(x, y)
    
    # 计算贝塔函数值
    Z = special.beta(X, Y)
    
    # 使用对数刻度避免大值
    log_Z = np.log10(np.abs(Z))
    
    levels = np.linspace(-1, 2, 30)
    cf = ax.contourf(X, Y, log_Z, levels=levels, cmap='viridis')
    ax.contour(X, Y, log_Z, levels=levels, colors='k', alpha=0.3, linewidths=0.5)
    
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_title('贝塔函数 $\\log_{10}|B(x,y)|$ 的等高线图', fontsize=16)
    fig.colorbar(cf, ax=ax, label='$\\log_{10}|B(x,y)|$')
    
    # 标记一些特殊点
    special_points = [(1, 1), (0.5, 0.5), (2, 2), (0.5, 2)]
    for px, py in special_points:
        if px <= 5 and py <= 5:
            ax.scatter([px], [py], s=100, color='red', zorder=5)
            ax.annotate(f'B({px},{py})={special.beta(px,py):.2f}', 
                       (px, py), textcoords="offset points",
                       xytext=(10, 10), fontsize=9, color='white')
    
    fig.tight_layout()
    return fig


# ==================== 球谐函数 ====================

def plot_spherical_harmonics(l_max=3):
    """
    绘制球谐函数 Y_l^m(θ, φ) 的极坐标图
    球谐函数是角动量算符的本征函数
    """
    fig = plt.figure(figsize=(3 * (l_max + 1), 3 * l_max))
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            ax = fig.add_subplot(l_max + 1, 2 * l_max + 1, 
                                 (l) * (2 * l_max + 1) + (m + l) + 1, 
                                 projection='3d')
            
            # 创建球坐标网格
            theta = np.linspace(0, np.pi, 50)
            phi = np.linspace(0, 2 * np.pi, 50)
            Theta, Phi = np.meshgrid(theta, phi)
            
            # 计算球谐函数
            Y = special.sph_harm(m, l, Phi, Theta)
            
            # 转换为笛卡尔坐标用于3D绘图
            r = np.abs(Y)
            X = r * np.sin(Theta) * np.cos(Phi)
            Y_ = r * np.sin(Theta) * np.sin(Phi)
            Z = r * np.cos(Theta)
            
            # 使用颜色表示相位
            colors = np.angle(Y)
            
            surf = ax.plot_surface(X, Y_, Z, facecolors=plt.cm.twilight(colors), 
                                  linewidth=0, antialiased=True, alpha=0.8)
            
            ax.set_title(f'$Y_{{{l}}}^{{{m}}}$', fontsize=10)
            ax.set_axis_off()
    
    fig.suptitle('球谐函数 $Y_l^m(\\theta, \\phi)$（大小表示模，相位表示颜色）', 
                fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ==================== 获取可用函数列表 ====================

def get_available_special_functions():
    """
    获取所有可用的特殊函数类型
    """
    return {
        'bessel': {
            'name': '贝塞尔函数',
            'subtypes': ['j_n', 'zero_convergence'],
            'functions': {
                'plot_bessel_functions': '第一类贝塞尔函数 $J_n(x)$',
                'plot_bessel_functions_2d': '圆形膜振动模式',
                'plot_besselj_zero_convergence': '零点收敛分析'
            }
        },
        'legendre': {
            'name': '勒让德多项式',
            'subtypes': ['p_n', 'associated'],
            'functions': {
                'plot_legendre_polynomials': '勒让德多项式 $P_n(x)$',
                'plot_associated_legendre': '连带勒让德函数 $P_n^m(x)$'
            }
        },
        'hermite': {
            'name': '厄米特函数',
            'subtypes': ['psi_n', 'h_n'],
            'functions': {
                'plot_hermite_functions': '量子谐振子波函数 $\\psi_n(x)$',
                'plot_hermite_polynomials': '厄米特多项式 $H_n(x)$'
            }
        },
        'laguerre': {
            'name': '拉盖尔多项式',
            'subtypes': ['l_n', 'associated'],
            'functions': {
                'plot_laguerre_polynomials': '拉盖尔多项式 $L_n(x)$',
                'plot_associated_laguerre': '连带拉盖尔 $L_n^k(x)$'
            }
        },
        'chebyshev': {
            'name': '切比雪夫多项式',
            'subtypes': ['t_n', 'u_n'],
            'functions': {
                'plot_chebyshev_Tn': '第一类切比雪夫 $T_n(x)$',
                'plot_chebyshev_Un': '第二类切比雪夫 $U_n(x)$'
            }
        },
        'gamma_beta': {
            'name': '伽马与贝塔函数',
            'subtypes': ['gamma', 'beta'],
            'functions': {
                'plot_gamma_function': '伽马函数 $\\Gamma(x)$',
                'plot_beta_function': '贝塔函数 $B(x,y)$'
            }
        },
        'spherical_harmonics': {
            'name': '球谐函数',
            'subtypes': ['y_lm'],
            'functions': {
                'plot_spherical_harmonics': '球谐函数 $Y_l^m(\\theta, \\phi)$'
            }
        }
    }


def get_function_description(func_key):
    """
    获取特定函数的详细物理意义描述
    """
    descriptions = {
        'bessel': """
        **贝塞尔函数** 产生于柱坐标（极坐标）中的波动方程求解。
        
        **物理背景**：当求解圆形 membrane（如鼓面）的振动问题时，
        在极坐标下使用分离变量法，会得到关于径向部分方程：
        r²R'' + rR' + (k²r² - n²)R = 0
        其解即为 n 阶第一类贝塞尔函数 J_n(kr)。
        
        **零点物理意义**：J_n(x) = 0 的解给出圆形膜固定边界条件下的本征频率。
        """,
        
        'legendre': """
        **勒让德多项式** 产生于球坐标中的拉普拉斯方程求解。
        
        **物理背景**：求解氢原子的定态薛定谔方程时，角动量部分
        在球坐标下分离变量，θ 方程的解与勒让德多项式相关。
        
        **正交性**：∫_{-1}^{1} P_n(x) P_m(x) dx = δ_nm / (2n+1)
        这使得它们适合做球面上的傅里叶展开。
        """,
        
        'hermite': """
        **厄米特函数** 是量子谐振子的本征函数。
        
        **物理背景**：一维谐振子势 V(x) = mω²x²/2 的薛定谔方程：
        -ℏ²/2m · d²ψ/dx² + 1/2·mω²x²ψ = Eψ
        解得的能量本征值为 E_n = ℏω(n + 1/2)，
        对应的归一化波函数正是厄米特-高斯函数。
        
        **零点物理意义**：ψ_n(x) = 0 的点数随 n 增加而增加。
        """,
        
        'laguerre': """
        **拉盖尔多项式** 出现于氢原子波函数的径向部分。
        
        **物理背景**：氢原子薛定谔方程在球坐标分离变量后，
        径向方程的解包含连带拉盖尔多项式 L_{n-l-1}^{2l+1}(r)。
        
        **量子数关系**：主量子数 n ≥ l + 1
        """,
        
        'chebyshev': """
        **切比雪夫多项式** 是最佳一致逼近理论的核心工具。
        
        **特性**：T_n(x) 在 [-1,1] 上满足 |T_n(x)| ≤ 1，
        并且是最高次项系数为 2^{n-1} 的首一多项式中，
        对零偏差最小的。
        """,
        
        'gamma': """
        **伽马函数** 是阶乘的解析延拓。
        
        **性质**：Γ(n) = (n-1)! 对正整数成立
                 Γ(z+1) = z·Γ(z) 泛函方程
                 Γ(x)Γ(1-x) = π/sin(πx) 反射公式
        
        **应用**：多维积分、概率论中的 Gamma 分布
        """,
        
        'beta': """
        **贝塔函数** 与伽马函数密切相关。
        
        **定义**：B(a,b) = ∫₀¹ t^{a-1}(1-t)^{b-1} dt
        **关系**：B(a,b) = Γ(a)Γ(b)/Γ(a+b)
        
        **应用**：Beta 分布在贝叶斯统计中作为共轭先验
        """,
        
        'spherical_harmonics': """
        **球谐函数** 是角动量算符 L² 和 L_z 的共同本征函数。
        
        **量子数**：l = 0,1,2,... （角动量量子数）
                   m = -l,...,l （磁量子数）
        
        **性质**：∫ Y_l^m* Y_{l'}^{m'} dΩ = δ_ll' δ_mm'
        
        **应用**：原子轨道、球面调和分析
        """
    }
    
    return descriptions.get(func_key, "暂无详细描述")


# ==================== 工具函数 ====================

def generate_animation_frame_special(func, n, **kwargs):
    """
    为特殊函数生成单帧动画图像
    
    参数：
        func: 绘图函数名（字符串）
        n: 当前帧对应的阶数/量子数
        **kwargs: 传递给绘图函数的其他参数
    
    返回：
        fig: matplotlib图像对象
    """
    plot_funcs = {
        'bessel': (plot_bessel_functions, {'n_max': n}),
        'legendre': (plot_legendre_polynomials, {'n_max': n}),
        'hermite': (plot_hermite_functions, {'n_max': n}),
        'hermite_poly': (plot_hermite_polynomials, {'n_max': n}),
        'laguerre': (plot_laguerre_polynomials, {'n_max': n}),
        'chebyshev_T': (plot_chebyshev_Tn, {'n_max': n}),
        'chebyshev_U': (plot_chebyshev_Un, {'n_max': n}),
    }
    
    if func in plot_funcs:
        plot_fn, default_kwargs = plot_funcs[func]
        default_kwargs.update(kwargs)
        return plot_fn(**default_kwargs)
    else:
        return None


# ==================== 实时动画：贝塞尔函数鼓面振动 ====================

def generate_realtime_bessel_mode_html(n=1, m=1, R=1.0, grid_size=50, time_scale=1.0):
    """
    生成贝塞尔函数鼓面振动模式的实时Canvas动画 HTML
    
    物理原理：
        圆形薄膜的自由振动方程分离变量后可表示为：
            u(r,θ,t) = J_n(k_{nm}r) * cos(nθ) * cos(ω_{nm}t)
        其中 J_n 是 n 阶第一类贝塞尔函数，k_{nm} 是 J_n 的第 m 个零点，
        ω_{nm} = c * k_{nm} 是角频率，c 为波速。
    
    参数：
        n: 角向阶数（0,1,2,...）决定径向节线数量 
        m: 径向阶数（1,2,3,...）决定节点圆数量
        R: 圆膜半径
        grid_size: 计算网格大小
        time_scale: 时间演化速度
    
    返回：完整 HTML 字符串
    """
    # 贝塞尔函数 J_n 的零点（前6个零点近似值，用于 n=0..5）
    # 这些是 Bessel 函数 J_n 的前 m 个零点
    bessel_zeros = {
        0: [2.4048, 5.5201, 8.6537, 11.7915, 14.9309, 18.0711],
        1: [3.8317, 7.0156, 10.1735, 13.3237, 16.4706, 19.6159],
        2: [5.1356, 8.4172, 11.6198, 14.7959, 17.9598, 21.1170],
        3: [6.3802, 9.7610, 13.0152, 16.2235, 19.4094, 22.5827],
        4: [7.5883, 11.0647, 14.3725, 17.6160, 20.8269, 24.0190],
        5: [8.7715, 12.3386, 15.7002, 19.0036, 22.2220, 25.4304],
    }
    
    n = max(0, int(n))
    m = max(1, int(m))
    
    # 获取 J_n 的第 m 个零点
    if n in bessel_zeros and m - 1 < len(bessel_zeros[n]):
        k_nm = bessel_zeros[n][m - 1]
    else:
        # 渐近近似
        k_nm = (m + n / 2.0 - 0.25) * np.pi
    
    omega = k_nm  # ω = c*k/R，取 c=1, R=1
    
    canvas_size = 520
    
    html_parts = []
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html><head><meta charset="UTF-8">')
    html_parts.append('<style>')
    html_parts.append('* { box-sizing: border-box; }')
    html_parts.append('body { margin:0; padding:10px; background:#ffffff; font-family:"Segoe UI",sans-serif; }')
    html_parts.append('.container { width:100%; max-width:800px; margin:0 auto; padding:10px; }')
    html_parts.append('.title { font-size:22px; font-weight:bold; text-align:center; margin:10px 0; color:#333; }')
    html_parts.append('.subtitle { font-size:13px; text-align:center; color:#666; margin-bottom:8px; font-family:"Cambria Math",serif; }')
    html_parts.append('.time-display { font-size:15px; text-align:center; color:#333; margin:8px 0; font-family:monospace; }')
    html_parts.append('.controls { text-align:center; margin:10px 0; }')
    html_parts.append('.controls button { padding:10px 30px; font-size:14px; margin:0 10px; border:none; border-radius:20px; cursor:pointer; font-weight:bold; }')
    html_parts.append('.play-btn { background:#4CAF50; color:white; }')
    html_parts.append('.reset-btn { background:#2196F3; color:white; }')
    html_parts.append('.status { text-align:center; padding:6px 16px; border-radius:20px; font-size:13px; font-weight:bold; display:inline-block; margin:5px 0; }')
    html_parts.append('.status.running { background:#e8f5e9; color:#2e7d32; }')
    html_parts.append('.status.stopped { background:#ffebee; color:#c62828; }')
    html_parts.append('.canvas-wrapper { text-align:center; margin:15px 0; }')
    html_parts.append('canvas { border:2px solid #e0e0e0; border-radius:12px; box-shadow:0 4px 20px rgba(0,0,0,0.1); max-width:100%; }')
    html_parts.append('.mode-info { text-align:center; font-size:13px; color:#555; margin:10px 0; font-family:monospace; }')
    html_parts.append('</style></head>')
    html_parts.append('<body>')
    html_parts.append('<div class="container">')
    html_parts.append(f'  <div class="title">贝塞尔函数：圆形鼓面振动模式 (n={n}, m={m})</div>')
    html_parts.append(f'  <div class="subtitle">u(r,\u03b8,t) = J_{n}(k_{{{n}{m}}} r) · cos({n}\u03b8) · cos(\u03c9_{{{n}{m}}} t)</div>')
    html_parts.append(f'  <div class="mode-info">零点 k_{{{n}{m}}} = {k_nm:.4f}, 频率 \u03c9 = {omega:.4f}</div>')
    html_parts.append(f'  <div class="time-display" id="timeDisplay">时间: t = 0.00 s</div>')
    html_parts.append('  <div class="controls">')
    html_parts.append(f'    <button class="play-btn" id="playBtn" onclick="toggleAnimation()">▶ 开始仿真</button>')
    html_parts.append(f'    <button class="reset-btn" onclick="resetAnimation()">⟳ 重置</button>')
    html_parts.append('  </div>')
    html_parts.append('  <div style="text-align:center;"><span class="status stopped" id="status">状态: 已停止</span></div>')
    html_parts.append(f'  <div class="canvas-wrapper"><canvas id="besselCanvas" width="{canvas_size}" height="{canvas_size}"></canvas></div>')
    html_parts.append('</div>')
    
    html_parts.append('<script>')
    # 参数
    html_parts.append(f'  const N = {n};')
    html_parts.append(f'  const M = {m};')
    html_parts.append(f'  const K_NM = {k_nm:.6f};')
    html_parts.append(f'  const OMEGA = {omega:.4f};')
    html_parts.append(f'  const R = {R:.2f};')
    html_parts.append(f'  const GRID = {grid_size};')
    html_parts.append(f'  const TIME_SCALE = {time_scale:.3f};')
    
    # JavaScript: 通过递推公式计算贝塞尔函数 J_n(x)
    html_parts.append("""
  // 使用级数展开近似计算 J_n(x)
  // J_n(x) = (x/2)^n * Σ_{k=0}^{∞} (-x^2/4)^k / (k! * (n+k)!)
  function besselJ(n, x) {
    if (Math.abs(x) < 1e-10) { return (n === 0) ? 1.0 : 0.0; }
    const nInt = Math.floor(n);
    const halfX = x / 2.0;
    let sum = 0;
    let term;
    let logTerm = nInt * Math.log(halfX);
    let sign = 1;
    let logFactorialN = 0;
    for (let k = 2; k <= nInt; k++) logFactorialN += Math.log(k);
    let logFactorialK = 0;
    let logFactorialNK = logFactorialN;
    
    for (let k = 0; k < 50; k++) {
      // 计算 log(term)
      let lt = nInt * Math.log(halfX) - logFactorialNK;
      if (k > 0) {
        lt += k * Math.log(x*x/4.0);
        lt -= logFactorialK;
      }
      term = sign * Math.exp(lt);
      sum += term;
      if (Math.abs(term) < 1e-12 * Math.abs(sum)) break;
      // 更新阶数
      logFactorialK += Math.log(k + 1);
      logFactorialNK += Math.log(nInt + k + 1);
      sign = -sign;
    }
    return sum;
  }
  
  // 动画状态
  let isRunning = false;
  let simTime = 0;
  let lastTimestamp = null;
  let animationId = null;
  
  // 预计算网格
  const canvas = document.getElementById('besselCanvas');
  const ctx = canvas.getContext('2d');
  const CW = canvas.width, CH = canvas.height;
  const margin = 40;
  const plotSize = Math.min(CW, CH) - 2 * margin;
  const centerX = CW / 2;
  const centerY = CH / 2;
  const cellSize = plotSize / GRID;
  
  // 颜色映射 (RdBu风格: 正蓝, 负红)
  function colormap(v) {
    if (v > 1) v = 1; if (v < -1) v = -1;
    let r, g, b;
    if (v >= 0) {
      // 蓝白色系
      const t = v;
      r = Math.round(255 * (1 - t) + 30 * t);
      g = Math.round(255 * (1 - t) + 100 * t);
      b = Math.round(255 * (1 - t) + 250 * t);
    } else {
      const t = -v;
      r = Math.round(255 * (1 - t) + 250 * t);
      g = Math.round(255 * (1 - t) + 100 * t);
      b = Math.round(255 * (1 - t) + 30 * t);
    }
    return "rgb(" + r + "," + g + "," + b + ")";
  }
  
  // 主绘制函数
  function draw(t) {
    // 清屏
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, CW, CH);
    
    // 计算当前时刻的最大值（用于归一化）
    let maxVal = 0;
    const vals = new Float64Array(GRID * GRID);
    const inside = new Uint8Array(GRID * GRID);
    
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        const x = -1 + 2 * (j / (GRID - 1));
        const y = -1 + 2 * (i / (GRID - 1));
        const r = Math.sqrt(x*x + y*y);
        const idx = i * GRID + j;
        if (r > R) {
          inside[idx] = 0;
          continue;
        }
        inside[idx] = 1;
        const theta = Math.atan2(y, x);
        const besselArg = K_NM * r / R;
        let jn = besselJ(N, besselArg);
        let angular = (N === 0) ? 1.0 : Math.cos(N * theta);
        const temporal = Math.cos(OMEGA * t);
        const val = jn * angular * temporal;
        vals[idx] = val;
        if (Math.abs(val) > maxVal) maxVal = Math.abs(val);
      }
    }
    
    // 归一化 (如果 maxVal 为0则用1)
    const scale = (maxVal > 1e-10) ? 1.0 / maxVal : 1.0;
    
    // 绘制单元格
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        const idx = i * GRID + j;
        if (!inside[idx]) continue;
        const px = centerX + (j - GRID/2) * cellSize;
        const py = centerY + (i - GRID/2) * cellSize;
        ctx.fillStyle = colormap(vals[idx] * scale);
        ctx.fillRect(px, py, cellSize + 0.5, cellSize + 0.5);
      }
    }
    
    // 绘制圆形边界
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(centerX, centerY, plotSize/2, 0, 2 * Math.PI);
    ctx.stroke();
    
    // 色条
    const barX = CW - 25;
    const barYTop = margin;
    const barH = plotSize;
    const barW = 12;
    const nSeg = 40;
    for (let k = 0; k < nSeg; k++) {
      const v = 1 - 2 * (k / (nSeg - 1));
      ctx.fillStyle = colormap(v);
      ctx.fillRect(barX, barYTop + k * barH/nSeg, barW, barH/nSeg + 1);
    }
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.strokeRect(barX, barYTop, barW, barH);
    ctx.fillStyle = "#333";
    ctx.font = "12px sans-serif";
    ctx.fillText("+max", barX - 45, barYTop + 10);
    ctx.fillText("0", barX - 20, barYTop + barH/2);
    ctx.fillText("-max", barX - 45, barYTop + barH - 5);
    
    // 中心标记
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX - 4, centerY); ctx.lineTo(centerX + 4, centerY);
    ctx.moveTo(centerX, centerY - 4); ctx.lineTo(centerX, centerY + 4);
    ctx.stroke();
    
    // 时间显示
    document.getElementById("timeDisplay").textContent = "时间: t = " + t.toFixed(2) + " s";
  }
  
  // 动画循环
  function animate(timestamp) {
    if (!isRunning) return;
    if (lastTimestamp === null) lastTimestamp = timestamp;
    const dt = (timestamp - lastTimestamp) / 1000 * TIME_SCALE;
    lastTimestamp = timestamp;
    simTime += dt;
    draw(simTime);
    animationId = requestAnimationFrame(animate);
  }
  
  function toggleAnimation() {
    if (isRunning) {
      isRunning = false;
      cancelAnimationFrame(animationId);
      document.getElementById("status").textContent = "状态: 已暂停";
      document.getElementById("status").className = "status stopped";
      document.getElementById("playBtn").textContent = "▶ 继续";
    } else {
      isRunning = true;
      lastTimestamp = null;
      document.getElementById("status").textContent = "状态: 运行中";
      document.getElementById("status").className = "status running";
      document.getElementById("playBtn").textContent = "⏸ 暂停";
      animationId = requestAnimationFrame(animate);
    }
  }
  
  function resetAnimation() {
    isRunning = false;
    cancelAnimationFrame(animationId);
    simTime = 0;
    lastTimestamp = null;
    draw(0);
    document.getElementById("status").textContent = "状态: 已停止";
    document.getElementById("status").className = "status stopped";
    document.getElementById("playBtn").textContent = "▶ 开始仿真";
  }
  
  // 初始绘制
  draw(0);
""")
    
    html_parts.append('</script>')
    html_parts.append('</body></html>')
    return '\n'.join(html_parts)


# ==================== 实时动画：量子谐振子（厄米函数） ====================

def generate_realtime_hermite_html(n_mode=3, n_max=8, time_scale=1.0):
    """
    生成量子谐振子波包随时间演化的实时Canvas动画HTML
    
    物理原理：
        量子谐振子波函数（归一化厄米函数）：
            ψ_n(x) = (1/√(2^n n! √π)) * H_n(x) * e^{-x²/2}
        含时演化：
            ψ_n(x,t) = ψ_n(x) * e^{-i(n+1/2)t}  (取 ħ=ω=1)
        
        可显示多个本征态的叠加（波包）随时间的演化。
    
    参数：
        n_mode: 初始主要本征态 (中心的量子数)
        n_max: 叠加的最高量子数
        time_scale: 时间演化速度
    
    返回：完整HTML字符串
    """
    canvas_w = 720
    canvas_h = 460
    x_min = -6.0
    x_max = 6.0
    
    html_parts = []
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html><head><meta charset="UTF-8">')
    html_parts.append('<style>')
    html_parts.append('* { box-sizing: border-box; }')
    html_parts.append('body { margin:0; padding:10px; background:#ffffff; font-family:"Segoe UI",sans-serif; }')
    html_parts.append('.container { width:100%; max-width:800px; margin:0 auto; padding:10px; }')
    html_parts.append('.title { font-size:22px; font-weight:bold; text-align:center; margin:10px 0; color:#333; }')
    html_parts.append('.subtitle { font-size:13px; text-align:center; color:#666; margin-bottom:8px; font-family:"Cambria Math",serif; }')
    html_parts.append('.info-bar { text-align:center; color:#333; font-size:13px; margin:5px 0; font-family:monospace; }')
    html_parts.append('.time-display { font-size:15px; text-align:center; color:#444; margin:8px 0; font-family:monospace; }')
    html_parts.append('.controls { text-align:center; margin:10px 0; }')
    html_parts.append('.controls button { padding:10px 30px; font-size:14px; margin:0 10px; border:none; border-radius:20px; cursor:pointer; font-weight:bold; }')
    html_parts.append('.play-btn { background:#9C27B0; color:white; }')
    html_parts.append('.reset-btn { background:#2196F3; color:white; }')
    html_parts.append('.status { text-align:center; padding:6px 16px; border-radius:20px; font-size:13px; font-weight:bold; display:inline-block; margin:5px 0; }')
    html_parts.append('.status.running { background:#f3e5f5; color:#6a1b9a; }')
    html_parts.append('.status.stopped { background:#ffebee; color:#c62828; }')
    html_parts.append('.canvas-wrapper { text-align:center; margin:15px 0; }')
    html_parts.append('canvas { border:2px solid #e0e0e0; border-radius:12px; box-shadow:0 4px 20px rgba(0,0,0,0.1); max-width:100%; }')
    html_parts.append('</style></head>')
    html_parts.append('<body>')
    html_parts.append('<div class="container">')
    html_parts.append(f'  <div class="title">量子谐振子：厄米函数波包随时间演化 (n ~ {n_mode})</div>')
    html_parts.append('  <div class="subtitle">\u03c8_n(x,t) = \u03c8_n(x) \u00b7 e^{-i(n+1/2)t}</div>')
    html_parts.append(f'  <div class="info-bar">|Re[\u03c8]|²（绿实线） + |Im[\u03c8]|²（橙虚线） + 经典概率 |\u03c8|²（蓝线）</div>')
    html_parts.append(f'  <div class="time-display" id="timeDisplay">时间: t = 0.00 s</div>')
    html_parts.append('  <div class="controls">')
    html_parts.append(f'    <button class="play-btn" id="playBtn" onclick="toggleAnimation()">▶ 开始仿真</button>')
    html_parts.append(f'    <button class="reset-btn" onclick="resetAnimation()">⟳ 重置</button>')
    html_parts.append('  </div>')
    html_parts.append('  <div style="text-align:center;"><span class="status stopped" id="status">状态: 已停止</span></div>')
    html_parts.append(f'  <div class="canvas-wrapper"><canvas id="hermiteCanvas" width="{canvas_w}" height="{canvas_h}"></canvas></div>')
    html_parts.append('</div>')
    
    html_parts.append('<script>')
    html_parts.append(f'  const N_MODE = {n_mode};')
    html_parts.append(f'  const N_MAX = {n_max};')
    html_parts.append(f'  const TIME_SCALE = {time_scale:.3f};')
    html_parts.append(f'  const X_MIN = {x_min};')
    html_parts.append(f'  const X_MAX = {x_max};')
    
    html_parts.append("""
  // 计算厄米多项式 H_n(x) 的递归:
  // H_0 = 1, H_1 = 2x, H_{n+1} = 2x H_n - 2n H_{n-1}
  function hermiteH(n, x) {
    if (n === 0) return 1.0;
    if (n === 1) return 2.0 * x;
    let hPrev = 1.0;
    let hCurr = 2.0 * x;
    for (let k = 1; k < n; k++) {
      const hNext = 2.0 * x * hCurr - 2.0 * k * hPrev;
      hPrev = hCurr;
      hCurr = hNext;
    }
    return hCurr;
  }
  
  // 计算归一化谐振子波函数 ψ_n(x)
  function psiN(n, x) {
    const logSqrtPi = 0.5 * Math.log(Math.PI);
    let logNFact = 0;
    for (let k = 2; k <= n; k++) logNFact += Math.log(k);
    const logNorm = -0.5 * (n * Math.log(2) + logNFact + logSqrtPi);
    const norm = Math.exp(logNorm);
    return norm * hermiteH(n, x) * Math.exp(-0.5 * x * x);
  }
  
  // 动画状态
  let isRunning = false;
  let simTime = 0;
  let lastTimestamp = null;
  let animationId = null;
  
  const canvas = document.getElementById('hermiteCanvas');
  const ctx = canvas.getContext('2d');
  const CW = canvas.width, CH = canvas.height;
  const padL = 70, padR = 50, padT = 40, padB = 60;
  const plotW = CW - padL - padR;
  const plotH = CH - padT - padB;
  const nPts = 400;
  
  // 构建波包：高斯叠加系数
  // 以 n_mode 为中心的高斯分布系数
  function buildCoefficients() {
    const coeffs = new Array(N_MAX + 1);
    let sumAbs = 0;
    const sigma = Math.max(1.0, N_MODE * 0.3);
    for (let n = 0; n <= N_MAX; n++) {
      const diff = n - N_MODE;
      // 高斯权重
      const w = Math.exp(-0.5 * (diff / sigma) * (diff / sigma));
      // 随机相位（固定种子）
      const phase = 0.1 * n; // 固定相位，产生相干态
      coeffs[n] = { weight: w, phase: phase };
      sumAbs += w * w;
    }
    // 归一化
    const normFactor = 1.0 / Math.sqrt(sumAbs);
    for (let n = 0; n <= N_MAX; n++) {
      coeffs[n].weight *= normFactor;
    }
    return coeffs;
  }
  
  const coeffs = buildCoefficients();
  
  function draw(t) {
    // 清屏
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, CW, CH);
    
    // 绘制网格
    ctx.strokeStyle = "#f0f0f0";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    for (let i = 0; i <= 6; i++) {
      const yy = padT + i * plotH / 6;
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(padL+plotW, yy); ctx.stroke();
    }
    for (let i = 0; i <= 12; i++) {
      const xx = padL + i * plotW / 12;
      ctx.beginPath(); ctx.moveTo(xx, padT); ctx.lineTo(xx, padT+plotH); ctx.stroke();
    }
    ctx.setLineDash([]);
    
    // 坐标轴
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(padL, padT + plotH); ctx.lineTo(padL+plotW, padT+plotH); ctx.stroke();
    const xZeroY = padT + plotH * 0.7;
    ctx.beginPath(); ctx.moveTo(padL, xZeroY); ctx.lineTo(padL+plotW, xZeroY); ctx.stroke();
    
    // 标签
    ctx.font = "14px sans-serif";
    ctx.fillStyle = "#333";
    ctx.fillText("x", padL + plotW - 20, xZeroY + 30);
    ctx.fillText("|ψ(x,t)|²", padL - 60, padT + 30);
    
    // 计算概率密度 |ψ(x,t)|²
    const realVals = new Float64Array(nPts + 1);
    const imagVals = new Float64Array(nPts + 1);
    let yMax = 0;
    
    for (let i = 0; i <= nPts; i++) {
      const x = X_MIN + (X_MAX - X_MIN) * i / nPts;
      let reSum = 0, imSum = 0;
      for (let n = 0; n <= N_MAX; n++) {
        const psi = psiN(n, x);
        const energy = (n + 0.5); // E_n = (n+1/2)ħω with ħω=1
        const phase = -energy * t + coeffs[n].phase;
        reSum += coeffs[n].weight * psi * Math.cos(phase);
        imSum += coeffs[n].weight * psi * Math.sin(phase);
      }
      realVals[i] = reSum;
      imagVals[i] = imSum;
    }
    
    // 计算最大概率密度
    let maxProb = 0;
    for (let i = 0; i <= nPts; i++) {
      const p = realVals[i]*realVals[i] + imagVals[i]*imagVals[i];
      if (p > maxProb) maxProb = p;
    }
    if (maxProb < 1e-6) maxProb = 0.5;
    const yScale = (plotH * 0.5) / maxProb;
    
    // 绘制经典位置期望值（随时间振荡）
    // 对于相干态，<x(t)> = <x(0)> cos(t) + <p(0)> sin(t)
    // 这里用简化近似：<x(t)> = sqrt(2*N_MODE) * cos(t)
    const xExpect = Math.sqrt(2 * N_MODE) * Math.cos(t);
    const expectPx = padL + (xExpect - X_MIN) / (X_MAX - X_MIN) * plotW;
    if (expectPx >= padL && expectPx <= padL + plotW) {
      ctx.strokeStyle = "#FF5722";
      ctx.lineWidth = 2;
      ctx.setLineDash([6,4]);
      ctx.beginPath();
      ctx.moveTo(expectPx, padT);
      ctx.lineTo(expectPx, padT+plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#FF5722";
      ctx.font = "12px sans-serif";
      ctx.fillText("<x(t)>", expectPx + 5, padT + 15);
    }
    
    // 绘制 |ψ|² (蓝色实线)
    ctx.strokeStyle = "#1976D2";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i <= nPts; i++) {
      const x = X_MIN + (X_MAX - X_MIN) * i / nPts;
      const px = padL + (x - X_MIN) / (X_MAX - X_MIN) * plotW;
      const prob = realVals[i]*realVals[i] + imagVals[i]*imagVals[i];
      const py = xZeroY - prob * yScale;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    
    // 填充 |ψ|²
    ctx.fillStyle = "rgba(33,150,243,0.18)";
    ctx.beginPath();
    for (let i = 0; i <= nPts; i++) {
      const x = X_MIN + (X_MAX - X_MIN) * i / nPts;
      const px = padL + (x - X_MIN) / (X_MAX - X_MIN) * plotW;
      const prob = realVals[i]*realVals[i] + imagVals[i]*imagVals[i];
      const py = xZeroY - prob * yScale;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.lineTo(padL + plotW, xZeroY);
    ctx.lineTo(padL, xZeroY);
    ctx.closePath();
    ctx.fill();
    
    // 绘制 Re[ψ]（绿色）
    ctx.strokeStyle = "#4CAF50";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i <= nPts; i++) {
      const x = X_MIN + (X_MAX - X_MIN) * i / nPts;
      const px = padL + (x - X_MIN) / (X_MAX - X_MIN) * plotW;
      const py = xZeroY - realVals[i] * yScale * 0.3;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    
    // 绘制 Im[ψ]（橙色虚线）
    ctx.strokeStyle = "#FF9800";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4,3]);
    ctx.beginPath();
    for (let i = 0; i <= nPts; i++) {
      const x = X_MIN + (X_MAX - X_MIN) * i / nPts;
      const px = padL + (x - X_MIN) / (X_MAX - X_MIN) * plotW;
      const py = xZeroY - imagVals[i] * yScale * 0.3;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    
    // 图例
    ctx.font = "13px sans-serif";
    ctx.fillStyle = "#1976D2";
    ctx.fillRect(padL + plotW - 180, padT - 25, 15, 3);
    ctx.fillStyle = "#333";
    ctx.fillText("|ψ(x,t)|² 概率密度", padL + plotW - 160, padT - 20);
    
    // x轴刻度
    ctx.font = "11px sans-serif";
    ctx.fillStyle = "#666";
    for (let i = 0; i <= 6; i++) {
      const xv = X_MIN + (X_MAX - X_MIN) * i / 6;
      const xx = padL + i * plotW / 6;
      ctx.fillText(xv.toFixed(1), xx - 10, xZeroY + 18);
    }
    
    // 时间显示
    document.getElementById("timeDisplay").textContent = "时间: t = " + t.toFixed(2) + " s";
  }
  
  function animate(timestamp) {
    if (!isRunning) return;
    if (lastTimestamp === null) lastTimestamp = timestamp;
    const dt = (timestamp - lastTimestamp) / 1000 * TIME_SCALE;
    lastTimestamp = timestamp;
    simTime += dt;
    draw(simTime);
    animationId = requestAnimationFrame(animate);
  }
  
  function toggleAnimation() {
    if (isRunning) {
      isRunning = false;
      cancelAnimationFrame(animationId);
      document.getElementById("status").textContent = "状态: 已暂停";
      document.getElementById("status").className = "status stopped";
      document.getElementById("playBtn").textContent = "▶ 继续";
    } else {
      isRunning = true;
      lastTimestamp = null;
      document.getElementById("status").textContent = "状态: 运行中";
      document.getElementById("status").className = "status running";
      document.getElementById("playBtn").textContent = "⏸ 暂停";
      animationId = requestAnimationFrame(animate);
    }
  }
  
  function resetAnimation() {
    isRunning = false;
    cancelAnimationFrame(animationId);
    simTime = 0;
    lastTimestamp = null;
    draw(0);
    document.getElementById("status").textContent = "状态: 已停止";
    document.getElementById("status").className = "status stopped";
    document.getElementById("playBtn").textContent = "▶ 开始仿真";
  }
  
  draw(0);
""")
    
    html_parts.append('</script>')
    html_parts.append('</body></html>')
    return '\n'.join(html_parts)


# ==================== 实时动画：球谐函数转动 ====================

def generate_realtime_spherical_harmonics_html(l=2, m=0, time_scale=1.0):
    """
    生成球谐函数 Y_l^m(θ,φ) 随时间转动的实时动画 HTML
    
    物理原理：
        球谐函数:
            Y_l^m(θ,φ) = N_l^m * P_l^m(cosθ) * e^{imφ}
        其中 P_l^m 是连带勒让德多项式。
        
        随时间演化（绕 z 轴转动）:
            Y_l^m(θ,φ,t) = N_l^m * P_l^m(cosθ) * e^{im(φ - ωt)}
        其中 ω = m * k（与磁量子数成正比）
    
    参数：
        l: 角动量量子数 (0,1,2,...)
        m: 磁量子数 (-l <= m <= l)
        time_scale: 时间演化速度
    
    返回：完整HTML字符串
    """
    l = max(0, int(l))
    m = max(-l, min(l, int(m)))
    
    canvas_w = 600
    canvas_h = 600
    grid_size = 60
    
    html_parts = []
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html><head><meta charset="UTF-8">')
    html_parts.append('<style>')
    html_parts.append('* { box-sizing: border-box; }')
    html_parts.append('body { margin:0; padding:10px; background:#ffffff; font-family:"Segoe UI",sans-serif; }')
    html_parts.append('.container { width:100%; max-width:720px; margin:0 auto; padding:10px; }')
    html_parts.append('.title { font-size:22px; font-weight:bold; text-align:center; margin:10px 0; color:#333; }')
    html_parts.append('.subtitle { font-size:13px; text-align:center; color:#666; margin-bottom:8px; font-family:"Cambria Math",serif; }')
    html_parts.append('.time-display { font-size:15px; text-align:center; color:#444; margin:8px 0; font-family:monospace; }')
    html_parts.append('.controls { text-align:center; margin:10px 0; }')
    html_parts.append('.controls button { padding:10px 30px; font-size:14px; margin:0 10px; border:none; border-radius:20px; cursor:pointer; font-weight:bold; }')
    html_parts.append('.play-btn { background:#FF5722; color:white; }')
    html_parts.append('.reset-btn { background:#2196F3; color:white; }')
    html_parts.append('.status { text-align:center; padding:6px 16px; border-radius:20px; font-size:13px; font-weight:bold; display:inline-block; margin:5px 0; }')
    html_parts.append('.status.running { background:#fff3e0; color:#e65100; }')
    html_parts.append('.status.stopped { background:#ffebee; color:#c62828; }')
    html_parts.append('.canvas-wrapper { text-align:center; margin:15px 0; }')
    html_parts.append('canvas { border:2px solid #e0e0e0; border-radius:12px; box-shadow:0 4px 20px rgba(0,0,0,0.1); max-width:100%; }')
    html_parts.append('.mode-info { text-align:center; font-size:13px; color:#555; margin:8px 0; font-family:monospace; }')
    html_parts.append('</style></head>')
    html_parts.append('<body>')
    html_parts.append('<div class="container">')
    html_parts.append(f'  <div class="title">球谐函数 Y_{{{l}}}^{{{m}}}(θ,φ) 实时演化</div>')
    html_parts.append('  <div class="subtitle">在 (θ,φ) 截面上显示 |Y_l^m|²（随时间绕 z 轴转动）</div>')
    html_parts.append(f'  <div class="mode-info">l = {l}, m = {m}</div>')
    html_parts.append(f'  <div class="time-display" id="timeDisplay">时间: t = 0.00 s</div>')
    html_parts.append('  <div class="controls">')
    html_parts.append(f'    <button class="play-btn" id="playBtn" onclick="toggleAnimation()">▶ 开始仿真</button>')
    html_parts.append(f'    <button class="reset-btn" onclick="resetAnimation()">⟳ 重置</button>')
    html_parts.append('  </div>')
    html_parts.append('  <div style="text-align:center;"><span class="status stopped" id="status">状态: 已停止</span></div>')
    html_parts.append(f'  <div class="canvas-wrapper"><canvas id="sphericalCanvas" width="{canvas_w}" height="{canvas_h}"></canvas></div>')
    html_parts.append('</div>')
    
    html_parts.append('<script>')
    html_parts.append(f'  const L_VAL = {l};')
    html_parts.append(f'  const M_VAL = {m};')
    html_parts.append(f'  const TIME_SCALE = {time_scale:.3f};')
    html_parts.append(f'  const GRID = {grid_size};')
    
    html_parts.append("""
  // 计算连带勒让德多项式 P_l^m(x)
  // 使用递推公式
  function legendreP(l, m, x) {
    if (m < 0 || m > l) return 0.0;
    if (Math.abs(x) > 1.0) x = Math.sign(x);
    // P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^{m/2}
    let pmm = 1.0;
    if (m > 0) {
      const factor = Math.sqrt((1 - x) * (1 + x));
      for (let i = 1; i <= m; i++) {
        pmm *= (-1) * (2*i - 1) * factor;
      }
    }
    if (l === m) return pmm;
    // P_{m+1}^m(x) = x(2m+1) P_m^m(x)
    let pmmp1 = x * (2*m + 1) * pmm;
    if (l === m + 1) return pmmp1;
    // 递推: (l-m)P_l^m = x(2l-1) P_{l-1}^m - (l+m-1) P_{l-2}^m
    for (let ll = m + 2; ll <= l; ll++) {
      const pll = (x * (2*ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
      pmm = pmmp1;
      pmmp1 = pll;
    }
    return pmmp1;
  }
  
  // 球谐函数 Y_l^m(θ,φ) 的模平方（只依赖 θ）
  // |Y_l^m|^2 = |N_l^m|^2 * |P_l^m(cosθ)|^2
  function sphericalHarmAbsSq(l, m, theta) {
    const x = Math.cos(theta);
    const p = legendreP(l, Math.abs(m), x);
    // 归一化常数（近似，只用于可视化）
    // N_l^m = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
    let logFactor = Math.log((2*l + 1) / (4 * Math.PI));
    for (let k = l - Math.abs(m) + 1; k <= l + Math.abs(m); k++) {
      if (k > 0) logFactor -= Math.log(k);
    }
    for (let k = 2; k <= l - Math.abs(m); k++) {
      logFactor += Math.log(k);
    }
    const normSq = Math.exp(logFactor);
    return normSq * p * p;
  }
  
  // 动画状态
  let isRunning = false;
  let simTime = 0;
  let lastTimestamp = null;
  let animationId = null;
  
  const canvas = document.getElementById('sphericalCanvas');
  const ctx = canvas.getContext('2d');
  const CW = canvas.width, CH = canvas.height;
  const margin = 40;
  const plotSize = Math.min(CW, CH) - 2 * margin;
  const cellSize = plotSize / GRID;
  const centerX = CW / 2;
  const centerY = CH / 2;
  
  // 颜色映射（热力图）
  function colormap(v) {
    if (v < 0) v = 0;
    if (v > 1) v = 1;
    // 蓝→青→绿→黄→红
    let r, g, b;
    if (v < 0.25) {
      const t = v / 0.25;
      r = 0; g = Math.round(50 + 200 * t); b = 255;
    } else if (v < 0.5) {
      const t = (v - 0.25) / 0.25;
      r = Math.round(255 * t); g = 255; b = Math.round(255 - 255 * t);
    } else if (v < 0.75) {
      const t = (v - 0.5) / 0.25;
      r = 255; g = Math.round(255 - 155 * t); b = 0;
    } else {
      const t = (v - 0.75) / 0.25;
      r = 255; g = Math.round(100 - 100 * t); b = 0;
    }
    return "rgb(" + r + "," + g + "," + b + ")";
  }
  
  function draw(t) {
    // 清屏
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, CW, CH);
    
    // 查找最大值
    let maxVal = 0;
    const vals = new Float64Array(GRID * GRID);
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        // 映射到球坐标: x 方向 = φ, y 方向 = cos(θ)
        const phi = -Math.PI + 2 * Math.PI * j / (GRID - 1);
        const cosTheta = 1 - 2 * i / (GRID - 1);
        const theta = Math.acos(cosTheta);
        const absSq = sphericalHarmAbsSq(L_VAL, M_VAL, theta);
        // 加入时间演化：相位因子 cos(m(φ - ωt))
        const omega = (M_VAL !== 0) ? M_VAL : 1.0;
        const timeFactor = Math.cos(M_VAL * (phi - omega * t));
        vals[i*GRID + j] = absSq * (0.5 + 0.5 * timeFactor); // 0..1 范围
        if (vals[i*GRID + j] > maxVal) maxVal = vals[i*GRID + j];
      }
    }
    
    const scale = (maxVal > 1e-10) ? 1.0 / maxVal : 1.0;
    
    // 绘制
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        const v = vals[i*GRID + j] * scale;
        const px = centerX + (j - GRID/2) * cellSize;
        const py = centerY + (i - GRID/2) * cellSize;
        ctx.fillStyle = colormap(v);
        ctx.fillRect(px, py, cellSize + 0.5, cellSize + 0.5);
      }
    }
    
    // 绘制边框和标签
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 2;
    ctx.strokeRect(centerX - plotSize/2, centerY - plotSize/2, plotSize, plotSize);
    
    ctx.font = "13px sans-serif";
    ctx.fillStyle = "#333";
    ctx.fillText("φ (方位角)", centerX - 30, centerY + plotSize/2 + 25);
    ctx.save();
    ctx.translate(centerX - plotSize/2 - 35, centerY - 10);
    ctx.rotate(-Math.PI/2);
    ctx.fillText("θ (极角)", -20, 0);
    ctx.restore();
    
    // 色条
    const barX = centerX + plotSize/2 + 15;
    const barYTop = centerY - plotSize/2;
    const barH = plotSize;
    const barW = 12;
    const nSeg = 40;
    for (let k = 0; k < nSeg; k++) {
      const v = 1 - k/(nSeg - 1);
      ctx.fillStyle = colormap(v);
      ctx.fillRect(barX, barYTop + k * barH/nSeg, barW, barH/nSeg + 1);
    }
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.strokeRect(barX, barYTop, barW, barH);
    ctx.fillStyle = "#333";
    ctx.font = "12px sans-serif";
    ctx.fillText("max", barX + barW + 5, barYTop + 10);
    ctx.fillText("0", barX + barW + 5, barYTop + barH - 5);
    
    document.getElementById("timeDisplay").textContent = "时间: t = " + t.toFixed(2) + " s";
  }
  
  function animate(timestamp) {
    if (!isRunning) return;
    if (lastTimestamp === null) lastTimestamp = timestamp;
    const dt = (timestamp - lastTimestamp) / 1000 * TIME_SCALE;
    lastTimestamp = timestamp;
    simTime += dt;
    draw(simTime);
    animationId = requestAnimationFrame(animate);
  }
  
  function toggleAnimation() {
    if (isRunning) {
      isRunning = false;
      cancelAnimationFrame(animationId);
      document.getElementById("status").textContent = "状态: 已暂停";
      document.getElementById("status").className = "status stopped";
      document.getElementById("playBtn").textContent = "▶ 继续";
    } else {
      isRunning = true;
      lastTimestamp = null;
      document.getElementById("status").textContent = "状态: 运行中";
      document.getElementById("status").className = "status running";
      document.getElementById("playBtn").textContent = "⏸ 暂停";
      animationId = requestAnimationFrame(animate);
    }
  }
  
  function resetAnimation() {
    isRunning = false;
    cancelAnimationFrame(animationId);
    simTime = 0;
    lastTimestamp = null;
    draw(0);
    document.getElementById("status").textContent = "状态: 已停止";
    document.getElementById("status").className = "status stopped";
    document.getElementById("playBtn").textContent = "▶ 开始仿真";
  }
  
  draw(0);
""")
    
    html_parts.append('</script>')
    html_parts.append('</body></html>')
    return '\n'.join(html_parts)
