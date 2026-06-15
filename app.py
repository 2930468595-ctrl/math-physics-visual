import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# ===== 在任何模块 import 之前，手动加载 .env（兼容 BOM，不依赖 python-dotenv） =====
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_path):
    try:
        with open(_env_path, "r", encoding="utf-8-sig") as f:
            for _line in f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _v = _line.split("=", 1)
                _k = _k.strip().lstrip("\ufeff")
                _v = _v.strip().strip('"').strip("'")
                if _k and _v and _k not in os.environ:
                    os.environ[_k] = _v
    except Exception:
        pass

# 再尝试 python-dotenv（如果已安装）作为补充
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# AI 家教模块
try:
    from ai_tutor import render_ai_tutor_sidebar, render_ai_tutor_main
except ImportError:
    def render_ai_tutor_sidebar():
        pass
    def render_ai_tutor_main():
        pass

# 导入模块
from modules.wave_equation import (
    standing_wave,
    generate_animation_frames,
    get_nodes_antinodes,
    calculate_physical_quantities,
    generate_realtime_html_animation
)

from modules.complex_functions import (
    generate_complex_plot_data,
    AVAILABLE_FUNCTIONS,
    get_function_description,
    get_function_formula,
    get_learning_tips
)

from modules.fourier_series import (
    square_wave_fourier,
    sawtooth_wave_fourier,
    triangle_wave_fourier,
    square_wave_exact,
    sawtooth_wave_exact,
    triangle_wave_exact,
    get_wave_description,
    AVAILABLE_WAVES,
    generate_realtime_fourier_html,
)

from modules.pde_solvers import (
    solve_heat_equation_1d,
    solve_heat_equation_2d,
    solve_wave_equation_1d,
    solve_drumhead_equation,
    generate_heat_equation_frames,
    generate_wave_equation_frames,
    generate_heat_2d_frames,
    generate_drumhead_frames,
    plot_heat_equation_solution,
    plot_wave_equation_solution,
    plot_heat_2d_snapshot,
    plot_drumhead_snapshot,
    get_available_pde_types,
    HEAT_INITIAL_CONDITIONS_1D,
    HEAT_INITIAL_CONDITIONS_2D,
    WAVE_INITIAL_CONDITIONS_1D,
    DRUMHEAD_INITIAL_CONDITIONS,
    generate_realtime_heat_1d_html,
    generate_realtime_heat_2d_html,
    generate_realtime_wave_1d_html,
    generate_realtime_drumhead_html,
)

from modules.special_functions import (
    plot_bessel_functions,
    plot_bessel_functions_2d,
    plot_besselj_zero_convergence,
    plot_legendre_polynomials,
    plot_associated_legendre,
    plot_hermite_functions,
    plot_hermite_polynomials,
    plot_laguerre_polynomials,
    plot_associated_laguerre,
    plot_chebyshev_Tn,
    plot_chebyshev_Un,
    plot_gamma_function,
    plot_beta_function,
    plot_spherical_harmonics,
    get_available_special_functions,
    get_function_description,
    generate_realtime_bessel_mode_html,
    generate_realtime_hermite_html,
    generate_realtime_spherical_harmonics_html,
)

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置页面标题
st.set_page_config(page_title="数学物理方法知识可视化", page_icon="📊", layout="wide")

# 页面标题
st.title("📊 数学物理方法知识可视化")
st.markdown("---")

# ==================== AI 家教（可折叠侧边栏） ====================
render_ai_tutor_sidebar()

# ==================== 导航菜单 ====================
module = st.sidebar.selectbox(
    "选择模块",
    ["驻波可视化", "复变函数映射", "傅里叶级数", "偏微分方程", "特殊函数", "AI 助教"],
    help="选择要查看的可视化模块",
)

# ==================== 驻波可视化模块 ====================
if module == "驻波可视化":
    st.subheader("🎵 一维波动方程驻波可视化")
    
    # 物理参数控制面板
    st.sidebar.header("🎛️ 参数控制")
    
    # 驻波模式选择
    n = st.sidebar.slider(
        "谐波阶数 n",
        min_value=1, max_value=5, value=1, step=1,
        help="驻波的谐波阶数，n=1为基频模式"
    )
    
    # 振幅控制
    A = st.sidebar.slider(
        "振幅 A",
        min_value=0.1, max_value=2.0, value=1.0, step=0.1,
        help="驻波的最大位移幅度"
    )
    
    # 弦长控制
    L = st.sidebar.slider(
        "弦长 L",
        min_value=0.5, max_value=2.0, value=1.0, step=0.1,
        help="弦的长度"
    )
    
    # 波速控制
    v = st.sidebar.slider(
        "波速 v",
        min_value=0.5, max_value=3.0, value=1.0, step=0.1,
        help="波在弦上的传播速度"
    )
    
    # 阻尼系数控制
    damping = st.sidebar.slider(
        "阻尼系数 γ",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        help="能量耗散系数，0表示无阻尼"
    )
    
    # 动画速度控制（支持更慢的速度）
    animation_speed = st.sidebar.slider(
        "动画速度",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="动画播放速度倍数，值越小越慢"
    )
    
    # 物理背景说明
    st.sidebar.markdown("---")
    st.sidebar.header("📚 物理背景")
    st.sidebar.markdown(r"""
    **一维波动方程**描述了弦的振动：
    $$\frac{\partial^2 u}{\partial t^2} = v^2 \frac{\partial^2 u}{\partial x^2}$$

    **驻波解**（两端固定边界条件）：
    $$u(x,t) = A e^{-\gamma t} \sin\left(\frac{n\pi x}{L}\right) \cos(\omega_n t + \phi)$$

    其中：
    - $A$ = 振幅
    - $L$ = 弦长
    - $n$ = 谐波数（1, 2, 3...）
    - $\omega_n = \frac{n\pi v}{L}$ = 角频率
    - $\gamma$ = 阻尼系数
    - $v$ = 波速
    """)
    
    # 计算物理量
    quantities = calculate_physical_quantities(n, L, v)
    omega = quantities['omega']
    f = quantities['frequency']
    wavelength = quantities['wavelength']
    T = quantities['period']
    
    # 生成并显示实时动画
    html_content = generate_realtime_html_animation(n, A, L, v, damping, animation_speed)
    components.html(html_content, height=750, scrolling=True)
    
    # 物理量显示
    st.markdown("---")
    st.subheader("📊 当前参数与物理量")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="振幅 A", value=f"{A:.2f}")
    with col2:
        st.metric(label="弦长 L", value=f"{L:.2f}")
    with col3:
        st.metric(label="波速 v", value=f"{v:.2f}")
    with col4:
        st.metric(label="阻尼系数 γ", value=f"{damping:.2f}")
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric(label="谐波阶数 n", value=n)
    with col6:
        st.metric(label="角频率 ω", value=f"{omega:.3f}")
    with col7:
        st.metric(label="频率 f", value=f"{f:.3f}")
    with col8:
        st.metric(label="波长 λ", value=f"{wavelength:.2f}")
    
    # 驻波特性说明
    st.markdown("---")
    st.subheader(f"📖 第{n}阶驻波模式特性")
    st.markdown(rf"""
    - **模式编号**: n = {n}
    - **节点数**: {n + 1} 个（包括两端固定点）
    - **波腹数**: {n} 个
    - **波长**: λ = {wavelength:.3f} = 2L/n
    - **频率**: f = {f:.3f} = nv/(2L)
    - **周期**: T = {T:.3f} = 1/f

    **物理意义**:
    当弦以第{n}阶模式振动时，弦上有{n}个完整的半波。
    在节点处位移始终为零，在波腹处位移最大。

    **阻尼效应**:
    当阻尼系数 γ > 0 时，振幅会随时间按指数规律衰减：
    $$A(t) = A_0 e^{{-\gamma t}}$$
    这模拟了实际弦振动中的能量耗散（如空气阻力）。
    """)
    
    # 交互说明
    st.markdown("---")
    st.info("💡 **使用提示**：\n\n1. **调节参数**：使用左侧滑块调整振幅、弦长、波速、阻尼系数\n2. **播放动画**：点击绿色按钮开始观看驻波振动过程\n3. **停止动画**：点击红色按钮暂停动画\n4. **观察变化**：注意阻尼系数不为零时振幅的衰减过程")

# ==================== 复变函数可视化模块 ====================
elif module == "复变函数映射":
    st.subheader("🔷 复变函数映射可视化")
    
    # 选择复变函数
    selected_func = st.sidebar.selectbox(
        "🎯 选择复变函数",
        options=[f['name'] for f in AVAILABLE_FUNCTIONS],
        format_func=lambda x: next(f['label'] for f in AVAILABLE_FUNCTIONS if f['name'] == x),
        help="选择一个复变函数，观察它如何变换复数平面"
    )
    
    # 获取学习提示
    learning_tip = get_learning_tips(selected_func)
    
    # 输入平面范围
    st.sidebar.header("📐 输入平面范围")
    x_min = st.sidebar.slider("实轴最小值", -5.0, 0.0, -2.0, 0.5)
    x_max = st.sidebar.slider("实轴最大值", 0.0, 5.0, 2.0, 0.5)
    y_min = st.sidebar.slider("虚轴最小值", -5.0, 0.0, -2.0, 0.5)
    y_max = st.sidebar.slider("虚轴最大值", 0.0, 5.0, 2.0, 0.5)
    
    # 分辨率
    resolution = st.sidebar.slider("🖼️ 图像分辨率", 50, 200, 100, 10,
                                  help="分辨率越高图像越清晰，但计算越慢")
    
    # 学习提示
    st.sidebar.markdown("---")
    st.sidebar.header("💡 学习提示")
    st.sidebar.markdown(learning_tip)
    
    # 物理背景说明
    st.sidebar.markdown("---")
    st.sidebar.header("📚 什么是复变函数？")
    st.sidebar.markdown("""
    **复变函数** f(z) = w 是复数平面上的"变换机器"：
    
    1. **输入**：复数平面上的一个点 z = x + iy
    2. **处理**：通过函数 f(z) 进行变换
    3. **输出**：得到新的点 w = f(z)
    
    **如何学习？**
    1. 先看**左图**（输入平面）：观察原始网格
    2. 再看**右图**（输出平面）：观察网格如何被"弯曲"
    3. 找**不动点**：哪些点变换后位置不变？
    """)
    
    # 生成可视化数据
    plot_data = generate_complex_plot_data(selected_func, (x_min, x_max), (y_min, y_max), resolution)
    
    # 显示函数详细信息
    st.markdown("---")
    st.markdown(f"### 📖 {get_function_description(selected_func)}")
    st.latex(get_function_formula(selected_func))
    
    # 创建两列布局：输入平面和输出平面
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📥 输入平面 (z-plane)")
        st.caption("这是原始的复数平面，还没有经过变换")
        # 绘制输入平面的网格
        fig_input, ax_input = plt.subplots(figsize=(6, 6))
        
        # 绘制网格线
        for line in plot_data['lines']:
            color = '#3498db' if line['type'] == 'horizontal' else '#e74c3c'
            linewidth = 1.0 if (abs(line['x'][0] if line['type'] == 'horizontal' else line['y'][0]) < 0.01 or 
                               abs(line['x'][0] if line['type'] == 'horizontal' else line['y'][0]) < 0.01) else 0.8
            ax_input.plot(line['x'], line['y'], color=color, linewidth=linewidth, alpha=0.7)
        
        ax_input.set_xlabel('实部 Re(z)', fontsize=12)
        ax_input.set_ylabel('虚部 Im(z)', fontsize=12)
        ax_input.set_xlim(x_min, x_max)
        ax_input.set_ylim(y_min, y_max)
        ax_input.grid(True, linestyle='--', alpha=0.3)
        ax_input.set_aspect('equal')
        
        # 添加原点标记
        ax_input.axhline(y=0, color='black', linewidth=0.5)
        ax_input.axvline(x=0, color='black', linewidth=0.5)
        
        st.pyplot(fig_input)
    
    with col2:
        st.markdown("#### 📤 输出平面 (w-plane)")
        st.caption("这是经过函数变换后的结果，观察网格如何变化！")
        # 绘制输出平面的映射
        fig_output, ax_output = plt.subplots(figsize=(6, 6))
        
        # 构建颜色数组（r, g, b 已经是相同形状的1D数组）
        # 直接将它们组合成 (n, 3) 形状的颜色数组用于 scatter
        n_points = len(plot_data['r'])
        colors = np.zeros((n_points, 3), dtype=np.float64)
        colors[:, 0] = plot_data['r']
        colors[:, 1] = plot_data['g']
        colors[:, 2] = plot_data['b']
        
        # 绘制映射后的网格线
        for line in plot_data['mapped_lines']:
            color = 'blue' if line['type'] == 'horizontal' else 'red'
            ax_output.plot(line['x'], line['y'], color=color, linewidth=0.8, alpha=0.6)
        
        # 绘制颜色映射的散点
        ax_output.scatter(
            plot_data['w_real'],
            plot_data['w_imag'],
            c=colors,
            s=10,
            alpha=0.8
        )
        
        ax_output.set_xlabel('实部 Re(w)')
        ax_output.set_ylabel('虚部 Im(w)')
        ax_output.grid(True, linestyle='--', alpha=0.3)
        ax_output.set_aspect('equal')
        
        # 自动调整范围
        all_x = np.concatenate([line['x'] for line in plot_data['mapped_lines']])
        all_y = np.concatenate([line['y'] for line in plot_data['mapped_lines']])
        
        if len(all_x) > 0 and len(all_y) > 0:
            x_margin = (np.max(all_x) - np.min(all_x)) * 0.1 if len(all_x) > 1 else 1
            y_margin = (np.max(all_y) - np.min(all_y)) * 0.1 if len(all_y) > 1 else 1
            ax_output.set_xlim(np.min(all_x) - x_margin, np.max(all_x) + x_margin)
            ax_output.set_ylim(np.min(all_y) - y_margin, np.max(all_y) + y_margin)
        
        st.pyplot(fig_output)
    
    # 函数特性说明
    st.markdown("---")
    st.subheader("📖 函数特性")
    
    func_properties = {
        'identity': """
        **恒等映射** f(z) = z 是最简单的复变函数，
        它将每个复数映射到自身。几何上没有任何变换。
        """,
        'square': """
        **平方函数** f(z) = z² 会将复数的模平方，辐角加倍。
        - 单位圆上的点：|z|=1 → |w|=1，arg(w)=2arg(z)
        - 射线 θ=常数 映射为射线 φ=2θ
        - 这是一个二对一的映射（除了z=0）
        """,
        'cube': """
        **立方函数** f(z) = z³ 将复数的模立方，辐角变为三倍。
        - 单位圆上的点：|z|=1 → |w|=1，arg(w)=3arg(z)
        - 这是一个三对一的映射
        """,
        'sqrt': """
        **平方根函数** f(z) = √z 是平方函数的逆函数。
        - 模变为平方根：|w|=√|z|
        - 辐角减半：arg(w)=arg(z)/2
        - 这是一个多值函数，通常取主值分支
        """,
        'exp': """
        **指数函数** f(z) = e^z = e^x (cos y + i sin y)
        - 水平线（常数y）映射为从原点出发的射线
        - 垂直线（常数x）映射为圆
        - 具有周期性：e^(z + 2πi) = e^z
        """,
        'log': """
        **对数函数** f(z) = ln(z) = ln|z| + i arg(z)
        - 是指数函数的逆函数
        - 模的对数：Re(w) = ln|z|
        - 辐角：Im(w) = arg(z)
        - 多值函数，通常取主值分支 (-π < arg(z) ≤ π)
        """,
        'sin': """
        **正弦函数** f(z) = sin(z) = sin(x)cosh(y) + i cos(x)sinh(y)
        - 当z为实数时，退化为实正弦函数
        - 虚轴上：sin(iy) = i sinh(y)
        - 具有周期性：sin(z + 2π) = sin(z)
        """,
        'cos': """
        **余弦函数** f(z) = cos(z) = cos(x)cosh(y) + i sin(x)sinh(y)
        - 当z为实数时，退化为实余弦函数
        - 虚轴上：cos(iy) = cosh(y)
        - 具有周期性：cos(z + 2π) = cos(z)
        """,
        '1/z': """
        **倒数函数** f(z) = 1/z 是一个反演变换。
        - 模变为倒数：|w| = 1/|z|
        - 辐角变为相反数：arg(w) = -arg(z)
        - 单位圆保持不变
        - 圆和直线相互映射
        """
    }
    
    if selected_func in func_properties:
        st.markdown(func_properties[selected_func])
    else:
        st.markdown(f"**{selected_func}** 的特性说明。")
    
    # 交互说明
    st.markdown("---")
    st.info("💡 **使用提示**：\n\n1. **选择函数**：使用左侧下拉菜单选择不同的复变函数\n2. **调整范围**：使用滑块调整输入平面的显示范围\n3. **观察映射**：对比左右两图，观察网格线的变换\n4. **颜色含义**：颜色表示复数的辐角（相位），亮度表示模的大小")

# ==================== 傅里叶级数可视化模块 ====================
elif module == "傅里叶级数":
    st.subheader("📈 傅里叶级数展开可视化")
    
    # 选择波形类型
    selected_wave = st.sidebar.selectbox(
        "选择波形",
        options=[w['name'] for w in AVAILABLE_WAVES],
        format_func=lambda x: next(w['label'] for w in AVAILABLE_WAVES if w['name'] == x),
        help="选择要展开的周期波形"
    )
    
    # 级数项数
    n_terms = st.sidebar.slider(
        "级数项数",
        min_value=1, max_value=50, value=5, step=1,
        help="傅里叶级数的项数，越大越接近原函数"
    )
    
    # 周期
    period = st.sidebar.slider(
        "周期 T",
        min_value=1.0, max_value=6.0, value=2*np.pi, step=0.5,
        help="波形的周期"
    )
    
    # 是否显示精确波形
    show_exact = st.sidebar.checkbox("显示精确波形", value=True)
    
    # 物理背景说明
    st.sidebar.markdown("---")
    st.sidebar.header("📚 物理背景")
    st.sidebar.markdown(r"""
    **傅里叶级数**表明，任何周期函数都可以表示为正弦和余弦函数的无穷级数：
    
    $$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos(n\omega x) + b_n \sin(n\omega x))$$
    
    其中 $\omega = 2\pi/T$ 是基频角频率。
    
    **吉布斯现象**：当傅里叶级数逼近不连续函数时，在不连续点附近会出现过冲，
    且增加项数不会消除这种现象。
    """)
    
    # 获取波形描述
    wave_desc = get_wave_description(selected_wave)
    
    # 显示波形信息
    st.markdown(f"""
    **当前波形**: {wave_desc['name']}
    
    **傅里叶级数**: ${wave_desc['formula']}$
    """)
    
    # 生成数据
    x = np.linspace(-period, 2*period, 500)
    
    # 计算傅里叶级数近似
    if selected_wave == 'square':
        y_fourier = square_wave_fourier(x, n_terms, T=period)
        y_exact = square_wave_exact(x, T=period)
    elif selected_wave == 'sawtooth':
        y_fourier = sawtooth_wave_fourier(x, n_terms, T=period)
        y_exact = sawtooth_wave_exact(x, T=period)
    elif selected_wave == 'triangle':
        y_fourier = triangle_wave_fourier(x, n_terms, T=period)
        y_exact = triangle_wave_exact(x, T=period)
    else:
        y_fourier = x * 0
        y_exact = x * 0
    
    # 绘制图形
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制傅里叶级数近似
    ax.plot(x, y_fourier, 'b-', linewidth=2, label=f'傅里叶级数 ({n_terms}项)')
    
    # 绘制精确波形
    if show_exact:
        ax.plot(x, y_exact, 'r--', linewidth=2, label=f'{wave_desc["name"]}（精确）')
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'{wave_desc["name"]}的傅里叶级数展开')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)
    
    # 创建收敛动画
    st.markdown("---")
    st.subheader("🎬 实时动画（持续推进时间）")
    st.markdown("点击 **开始仿真** 后，波形会随时间持续传播，可随时暂停或更改级数项数。")
    
    # 生成实时动画HTML
    max_terms_anim = 30
    fourier_realtime_html = generate_realtime_fourier_html(
        selected_wave, max_terms=max_terms_anim, T=period, A=1.0
    )
    components.html(fourier_realtime_html, height=730, scrolling=True)
    
    # 波形特性说明
    st.markdown("---")
    st.subheader("📖 波形特性")
    st.markdown(wave_desc['description'])
    st.markdown(f"**特点**: {wave_desc['features']}")
    
    # 吉布斯现象说明
    if selected_wave == 'square' or selected_wave == 'sawtooth':
        st.markdown("""
        **吉布斯现象**:
        在不连续点附近，傅里叶级数会出现过冲，其最大值约为原函数跳变的9%，
        即使增加项数也无法消除这种现象。
        """)
    
    # 交互说明
    st.markdown("---")
    st.info("💡 **使用提示**：\n\n1. **选择波形**：使用左侧下拉菜单选择方波、锯齿波或三角波\n2. **调整项数**：拖动滑块观察级数项数对逼近效果的影响\n3. **观察收敛**：点击播放按钮观看级数收敛过程\n4. **对比精确波形**：勾选'显示精确波形'对比近似效果")


# ==================== 偏微分方程模块 ====================
elif module == "偏微分方程":
    st.subheader("📐 偏微分方程解的可视化")
    
    # 获取可用的PDE类型
    pde_types = get_available_pde_types()
    
    # 选择方程类型
    selected_pde = st.sidebar.selectbox(
        "🎯 选择方程类型",
        options=list(pde_types.keys()),
        format_func=lambda x: pde_types[x]['name'],
        help="选择要可视化的偏微分方程"
    )
    
    pde_info = pde_types[selected_pde]
    
    # 显示方程信息
    st.markdown(f"""
    ### {pde_info['name']}
    **方程**: ${pde_info['equation']}$
    
    {pde_info['description']}
    """)
    
    # 根据方程类型显示不同的控件
    if selected_pde == 'heat_1d':
        # 一维热传导方程
        st.sidebar.header("🎛️ 参数控制")
        
        # 选择初始条件
        ic_options = list(HEAT_INITIAL_CONDITIONS_1D.keys())
        selected_ic = st.sidebar.selectbox(
            "初始条件",
            options=ic_options,
            format_func=lambda x: HEAT_INITIAL_CONDITIONS_1D[x]['name'],
            help="选择初始温度分布"
        )
        
        # 扩散系数
        alpha = st.sidebar.slider(
            "热扩散系数 α",
            min_value=0.001, max_value=0.1, value=0.02, step=0.005,
            help="控制热量扩散的速度"
        )
        
        # 模拟时间
        t_max = st.sidebar.slider(
            "模拟时间",
            min_value=0.1, max_value=2.0, value=0.8, step=0.1,
            help="总模拟时间长度"
        )
        
        # 边界条件
        left_bc = st.sidebar.selectbox(
            "左边界条件",
            ['dirichlet', 'neumann'],
            format_func=lambda x: '固定温度' if x == 'dirichlet' else '绝热（绝热边界）',
            help="左边界的温度条件"
        )
        right_bc = st.sidebar.selectbox(
            "右边界条件",
            ['dirichlet', 'neumann'],
            format_func=lambda x: '固定温度' if x == 'dirichlet' else '绝热',
            help="右边界的温度条件"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.header("📚 物理背景")
        st.sidebar.markdown(r"""
        **一维热传导方程**:
        $$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$
        
        **物理含义**:
        - u(x,t) 表示位置 x、时间 t 处的温度
        - α 是热扩散系数，与材料的热导率有关
        - 该方程描述热量在杆中的扩散过程
        """)
        
        # 求解方程
        u0_func = HEAT_INITIAL_CONDITIONS_1D[selected_ic]['func']
        x, t, u = solve_heat_equation_1d(
            u0_func,
            x_min=0.0, x_max=1.0,
            n_points=100,
            alpha=alpha,
            t_max=t_max,
            n_time_steps=200,
            left_bc=left_bc,
            right_bc=right_bc
        )
        
        # 显示静态图
        st.markdown("---")
        st.subheader("📊 解的可视化")
        
        # 绘制多个时刻的曲线
        fig = plot_heat_equation_solution(x, t, u)
        st.pyplot(fig)
        
        # 生成并显示动画
        st.markdown("---")
        st.subheader("🎬 实时动画(持续推进时间)")
        st.markdown("点击 **开始仿真** 后，温度随时间持续演化。")
        heat_html = generate_realtime_heat_1d_html(init_type=selected_ic, alpha=alpha, L=1.0, nx=100)
        components.html(heat_html, height=750, scrolling=True)

        
        # 物理分析
        st.markdown("---")
        st.subheader("📖 物理分析")
        st.markdown(f"""
        **初始条件**: {HEAT_INITIAL_CONDITIONS_1D[selected_ic]['description']}
        
        **边界条件**: 
        - 左边界: {'固定温度 = 0' if left_bc == 'dirichlet' else '绝热（∂u/∂x = 0）'}
        - 右边界: {'固定温度 = 0' if right_bc == 'dirichlet' else '绝热（∂u/∂x = 0）'}
        
        **观察要点**:
        1. 热量从高温区域向低温区域扩散
        2. 随着时间推移，温度分布趋于平缓（达到平衡）
        3. 绝热边界条件下，边界处热量无法流出
        """)
    
    elif selected_pde == 'heat_2d':
        # 二维热传导方程
        st.sidebar.header("🎛️ 参数控制")
        
        ic_options = list(HEAT_INITIAL_CONDITIONS_2D.keys())
        selected_ic = st.sidebar.selectbox(
            "初始热源类型",
            options=ic_options,
            format_func=lambda x: HEAT_INITIAL_CONDITIONS_2D[x]['name'],
            help="选择初始温度分布"
        )
        
        alpha = st.sidebar.slider(
            "热扩散系数 α",
            min_value=0.001, max_value=0.05, value=0.01, step=0.005,
            help="控制热量扩散的速度"
        )
        
        t_max = st.sidebar.slider(
            "模拟时间",
            min_value=0.05, max_value=0.5, value=0.2, step=0.05,
            help="总模拟时间长度"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.header("📚 物理背景")
        st.sidebar.markdown(r"""
        **二维热传导方程**:
        $$\frac{\partial u}{\partial t} = \alpha \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$
        
        描述热量在二维平面（如金属板）上的扩散。
        
        **边界条件**: 四边固定温度 u=0（冷边界）
        """)
        
        u0_func = HEAT_INITIAL_CONDITIONS_2D[selected_ic]['func']
        x, y, X, Y, t, u = solve_heat_equation_2d(
            u0_func,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            n_points=50,
            alpha=alpha,
            t_max=t_max,
            n_time_steps=100
        )
        
        st.markdown("---")
        st.subheader("📊 二维温度分布")
        
        # 绘制快照
        fig = plot_heat_2d_snapshot(X, Y, u, time_idx=-1)
        st.pyplot(fig)
        
        # 生成 2D 实时动画：在 [0,1]x[0,1] 方形域上解二维热传导方程
        st.markdown("---")
        st.subheader("🎬 实时动画(持续推进时间)")
        st.markdown("点击 **开始仿真** 后，温度分布随时间持续向四周扩散，冷边界消耗热量。")
        heat_2d_html = generate_realtime_heat_2d_html(
            init_type=selected_ic,
            alpha=alpha,
            L=1.0,
            n=50
        )
        components.html(heat_2d_html, height=860, scrolling=True)

    
    elif selected_pde == 'wave_1d':
        # 一维波动方程
        st.sidebar.header("🎛️ 参数控制")
        
        ic_options = list(WAVE_INITIAL_CONDITIONS_1D.keys())
        selected_ic = st.sidebar.selectbox(
            "初始波形",
            options=ic_options,
            format_func=lambda x: WAVE_INITIAL_CONDITIONS_1D[x]['name'],
            help="选择初始波形"
        )
        
        c = st.sidebar.slider(
            "波速 c",
            min_value=0.5, max_value=3.0, value=1.0, step=0.1,
            help="波在介质中的传播速度"
        )
        
        t_max = st.sidebar.slider(
            "模拟时间",
            min_value=1.0, max_value=5.0, value=2.5, step=0.5,
            help="总模拟时间"
        )
        
        left_bc = st.sidebar.selectbox(
            "左边界",
            ['fixed', 'free'],
            format_func=lambda x: '固定端' if x == 'fixed' else '自由端',
            help="左边界的边界条件"
        )
        right_bc = st.sidebar.selectbox(
            "右边界",
            ['fixed', 'free'],
            format_func=lambda x: '固定端' if x == 'fixed' else '自由端',
            help="右边界的边界条件"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.header("📚 物理背景")
        st.sidebar.markdown(r"""
        **一维波动方程**:
        $$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$
        
        **物理含义**:
        - u(x,t) 表示位置 x、时间 t 处的位移
        - c 是波速，与弦的张力和密度有关
        - 该方程描述弦、杆中的波动现象
        
        **边界条件**:
        - 固定端：位移 u=0
        - 自由端：斜率 ∂u/∂x = 0
        """)
        
        ic = WAVE_INITIAL_CONDITIONS_1D[selected_ic]
        x, t, u, r = solve_wave_equation_1d(
            ic['u0'], ic['v0'],
            x_min=0.0, x_max=1.0,
            n_points=200,
            c=c,
            t_max=t_max,
            n_time_steps=400,
            left_bc=left_bc,
            right_bc=right_bc
        )
        
        st.markdown("---")
        st.subheader("📊 波传播可视化")
        
        # 绘制多个时刻的曲线
        fig = plot_wave_equation_solution(x, t, u)
        st.pyplot(fig)
        
        # 生成动画
        st.markdown("---")
        st.subheader("🎬 实时动画(持续推进时间)")
        st.markdown("点击 **开始仿真** 后，波动随时间持续演化。")
        # 简单把 init_type 映射到 realtime 函数支持的类型
        wave_init_map = {'sine': 'sine', 'gaussian': 'gaussian', 'square': 'square', 'wave_packet': 'wave_packet'}
        wave_init_type = wave_init_map.get(selected_ic, 'sine')
        wave_html = generate_realtime_wave_1d_html(init_type=wave_init_type, c=c, L=1.0, nx=100, damping=0.0)
        components.html(wave_html, height=750, scrolling=True)

        
        # 物理分析
        st.markdown("---")
        st.subheader("📖 物理分析")
        st.markdown(f"""
        **初始波形**: {ic['description']}
        
        **边界条件**: 
        - 左边界: {'固定端（位移=0）' if left_bc == 'fixed' else '自由端（斜率=0）'}
        - 右边界: {'固定端（位移=0）' if right_bc == 'fixed' else '自由端（斜率=0）'}
        
        **波速 c = {c}**，CFL数 r = {r:.3f}
        
        **观察要点**:
        1. 波在传播过程中遇到边界会发生反射
        2. 固定端反射会产生相位反转（波峰变波谷）
        3. 自由端反射不产生相位反转
        4. 当两端都是固定端时，可能形成驻波
        """)
    
    elif selected_pde == 'drumhead':
        # 二维波动方程（鼓膜）
        st.sidebar.header("🎛️ 参数控制")
        
        ic_options = list(DRUMHEAD_INITIAL_CONDITIONS.keys())
        selected_ic = st.sidebar.selectbox(
            "初始形状",
            options=ic_options,
            format_func=lambda x: DRUMHEAD_INITIAL_CONDITIONS[x]['name'],
            help="选择鼓膜的初始形状"
        )
        
        c = st.sidebar.slider(
            "波速 c",
            min_value=0.5, max_value=2.0, value=1.0, step=0.1,
            help="波在膜上的传播速度"
        )
        
        t_max = st.sidebar.slider(
            "模拟时间",
            min_value=1.0, max_value=4.0, value=2.0, step=0.5,
            help="总模拟时间"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.header("📚 物理背景")
        st.sidebar.markdown(r"""
        **二维波动方程（鼓膜振动）**:
        $$\frac{\partial^2 u}{\partial t^2} = c^2 \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$
        
        **物理意义**:
        - 描述圆形薄膜（如鼓面）的振动
        - 边界条件：固定边界 u=0（膜被绷紧固定）
        - 解的形式与贝塞尔函数相关
        
        **模式分析**:
        鼓膜的振动可以分解为一系列本征模式的叠加，
        每个模式有特定的振动图案和固有频率。
        """)
        
        ic_func = DRUMHEAD_INITIAL_CONDITIONS[selected_ic]['func']
        x, y, X, Y, t, u = solve_drumhead_equation(
            ic_func,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            n_points=50,
            c=c,
            t_max=t_max,
            n_time_steps=200
        )
        
        st.markdown("---")
        st.subheader("📊 鼓膜振动模式")
        
        # 绘制快照
        fig = plot_drumhead_snapshot(X, Y, u, time_idx=-1)
        st.pyplot(fig)
        
        # 生成动画
        st.markdown("---")
        st.subheader("🎬 实时动画(持续推进时间)")
        st.markdown("点击 **开始仿真** 后，鼓膜随时间持续振动。")
        drum_init_map = {'gaussian': 'gaussian', 'circular': 'circular', 'sine': 'sine', 'cross': 'cross'}
        drum_init_type = drum_init_map.get(selected_ic, 'gaussian')
        drum_html = generate_realtime_drumhead_html(init_type=drum_init_type, c=c, R=1.0, n=40)
        components.html(drum_html, height=860, scrolling=True)

    
    # 通用交互说明
    st.markdown("---")
    st.info("💡 **使用提示**：\n\n1. **选择方程类型**：使用左侧菜单选择热传导或波动方程\n2. **调整参数**：通过滑块改变扩散系数、波速、边界条件等\n3. **观察演化**：点击播放按钮观看解的时间演化过程\n4. **分析物理**：思考边界条件和初始条件如何影响解的形态")


# ==================== 特殊函数模块 ====================
elif module == "特殊函数":
    st.subheader("✨ 特殊函数可视化")
    
    # 获取可用的特殊函数类型
    special_funcs = get_available_special_functions()
    
    # 选择函数大类
    selected_category = st.sidebar.selectbox(
        "🎯 选择函数类别",
        options=list(special_funcs.keys()),
        format_func=lambda x: special_funcs[x]['name'],
        help="选择要可视化的特殊函数类别"
    )
    
    category_info = special_funcs[selected_category]
    
    # 选择具体函数
    selected_func = st.sidebar.selectbox(
        "📊 选择具体函数",
        options=list(category_info['functions'].keys()),
        format_func=lambda x: category_info['functions'][x],
        help="选择要绘制的具体函数"
    )
    
    # 显示函数描述
    st.markdown(f"### {category_info['functions'][selected_func]}")
    st.markdown(get_function_description(selected_category))
    
    # 根据选择绘制图像
    st.markdown("---")
    
    if selected_func == 'plot_bessel_functions':
        n_max = st.sidebar.slider("最高阶数 n", 1, 8, 4, 1)
        x_max = st.sidebar.slider("x轴最大范围", 10.0, 40.0, 25.0, 5.0)
        fig = plot_bessel_functions(n_max=n_max, x_max=x_max)
        st.pyplot(fig)
        
        st.markdown("""
        **贝塞尔函数的性质**:
        - $J_n(x)$ 在原点有限，在无穷远处衰减振荡
        - $J_n(x)$ 有无穷多个实根
        - 渐近行为：$J_n(x) \\sim \\sqrt{\\frac{2}{\\pi x}} \\cos(x - \\frac{n\\pi}{2} - \\frac{\\pi}{4})$
        """)
    
    elif selected_func == 'plot_bessel_functions_2d':
        n_max = st.sidebar.slider("径向阶数 n", 1, 4, 3, 1)
        m_max = st.sidebar.slider("角向阶数 m", 1, 4, 3, 1)
        r_max = st.sidebar.slider("径向范围", 5.0, 15.0, 10.0, 1.0)
        fig = plot_bessel_functions_2d(n_max=n_max, m_max=m_max, r_max=r_max)
        st.pyplot(fig)
        
        st.markdown("""
        **二维波动与贝塞尔函数**:
        
        圆形膜的振动模式由贝塞尔函数描述：
        $$u(r,\\theta,t) = J_n(k_{nm} r) \\cos(n\\theta) \\cos(\\omega_{nm} t)$$
        
        其中 $k_{nm}$ 是 $J_n(x)$ 的第 m 个根。
        每个模式 $(n,m)$ 对应一个特定的振动图案和固有频率。
        """)
    
    elif selected_func == 'plot_besselj_zero_convergence':
        fig = plot_besselj_zero_convergence()
        st.pyplot(fig)
        
        st.markdown("""
        **贝塞尔函数零点的重要性**:
        
        $J_n(x) = 0$ 的根决定了圆形膜固定边界条件下的本征频率。
        
        随着阶数 n 增加，零点分布更加密集，
        且相邻零点间距趋近于 $\\pi$（渐近行为）。
        """)
    
    elif selected_func == 'plot_legendre_polynomials':
        n_max = st.sidebar.slider("最高阶数 n", 1, 10, 7, 1)
        fig = plot_legendre_polynomials(n_max=n_max)
        st.pyplot(fig)
        
        st.markdown(r"""
        **勒让德多项式的性质**:
        
        - 正交性：$\int_{-1}^{1} P_n(x) P_m(x) dx = \frac{2}{2n+1} \delta_{nm}$
        - 递推关系：$(n+1)P_{n+1}(x) = (2n+1)x P_n(x) - n P_{n-1}(x)$
        - $P_n(1) = 1, P_n(-1) = (-1)^n$
        """)
    
    elif selected_func == 'plot_associated_legendre':
        n_max = st.sidebar.slider("最高阶数 n", 1, 5, 3, 1)
        m_max = st.sidebar.slider("最高次数 m", 0, 4, 3, 1)
        fig = plot_associated_legendre(n_max=n_max, m_max=m_max)
        st.pyplot(fig)
        
        st.markdown(r"""
        **连带勒让德函数**:
        
        $P_n^m(x) = (-1)^m (1-x^2)^{m/2} \frac{d^m}{dx^m} P_n(x)$
        
        与球谐函数的关系：
        $Y_l^m(\theta, \phi) \\propto P_l^m(\\cos\\theta) e^{im\phi}$
        """)
    
    elif selected_func == 'plot_hermite_functions':
        n_max = st.sidebar.slider("最高量子数 n", 1, 8, 6, 1)
        x_range = st.sidebar.slider("x轴范围", 4.0, 12.0, 8.0, 1.0)
        fig = plot_hermite_functions(n_max=n_max, x_range=(-x_range, x_range))
        st.pyplot(fig)
        
        st.markdown(r"""
        **量子谐振子波函数**:
        
        $$\psi_n(x) = \frac{1}{\sqrt{2^n n!}} H_n(x) e^{-x^2/2}$$
        
        能量本征值：$E_n = \hbar\omega(n + \frac{1}{2})$
        
        波函数满足：$\int_{-\infty}^{\infty} |\psi_n|^2 dx = 1$
        """)
    
    elif selected_func == 'plot_hermite_polynomials':
        n_max = st.sidebar.slider("最高阶数 n", 1, 8, 5, 1)
        fig = plot_hermite_polynomials(n_max=n_max)
        st.pyplot(fig)
    
    elif selected_func == 'plot_laguerre_polynomials':
        n_max = st.sidebar.slider("最高阶数 n", 1, 10, 6, 1)
        x_max = st.sidebar.slider("x轴范围", 5.0, 20.0, 12.0, 1.0)
        fig = plot_laguerre_polynomials(n_max=n_max, x_range=(0, x_max))
        st.pyplot(fig)
        
        st.markdown(r"""
        **（广义）拉盖尔多项式**:
        
        $L_n^{(\alpha)}(x) = \\frac{x^{-\\alpha} e^x}{n!} \\frac{d^n}{dx^n}(e^{-x} x^{n+\\alpha})$
        
        递推关系：
        $(n+1)L_{n+1}^{(\\alpha)}(x) = (2n+\\alpha+1-x)L_n^{(\\alpha)}(x) - (n+\\alpha)L_{n-1}^{(\\alpha)}(x)$
        """)
    
    elif selected_func == 'plot_associated_laguerre':
        n_max = st.sidebar.slider("最高阶数 n", 1, 6, 4, 1)
        k_max = st.sidebar.slider("参数 α", 0, 3, 2, 1)
        fig = plot_associated_laguerre(n_max=n_max, k_max=k_max)
        st.pyplot(fig)
    
    elif selected_func == 'plot_chebyshev_Tn':
        n_max = st.sidebar.slider("最高阶数 n", 1, 10, 7, 1)
        fig = plot_chebyshev_Tn(n_max=n_max)
        st.pyplot(fig)
        
        st.markdown(r"""
        **第一类切比雪夫多项式**:
        
        $T_n(x) = \cos(n\arccos x)$
        
        性质：
        - $|T_n(x)| \leq 1$ for $x \in [-1,1]$
        - $T_n(x)$ 在 $[-1,1]$ 上有 $n$ 个零点
        - 首项系数为 $2^{n-1}$
        """)
    
    elif selected_func == 'plot_chebyshev_Un':
        n_max = st.sidebar.slider("最高阶数 n", 1, 8, 6, 1)
        fig = plot_chebyshev_Un(n_max=n_max)
        st.pyplot(fig)
    
    elif selected_func == 'plot_gamma_function':
        x_range = st.sidebar.slider("x轴范围", 2.0, 10.0, 5.0, 1.0)
        fig = plot_gamma_function(x_range=(-x_range, x_range))
        st.pyplot(fig)
        
        st.markdown(r"""
        **伽马函数的性质**:
        
        - $\Gamma(n) = (n-1)!$ for positive integers
        - $\Gamma(z+1) = z\Gamma(z)$ （泛函方程）
        - $\Gamma(x)\Gamma(1-x) = \frac{\pi}{\sin(\pi x)}$ （反射公式）
        - 在 $x = 0, -1, -2, ...$ 处有单极点
        """)
    
    elif selected_func == 'plot_beta_function':
        fig = plot_beta_function()
        st.pyplot(fig)
        
        st.markdown(r"""
        **贝塔函数的性质**:
        
        $B(a,b) = \int_0^1 t^{a-1}(1-t)^{b-1} dt = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$
        
        特殊值：
        - $B(1,1) = 1$
        - $B(\\frac{1}{2}, \\frac{1}{2}) = \pi$
        """)
    
    elif selected_func == 'plot_spherical_harmonics':
        l_max = st.sidebar.slider("最大角动量量子数 l", 1, 4, 3, 1)
        fig = plot_spherical_harmonics(l_max=l_max)
        st.pyplot(fig)
        
        st.markdown(r"""
        **球谐函数**:
        
        $Y_l^m(\theta, \phi) = (-1)^m \sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}} P_l^m(\cos\theta) e^{im\phi}$
        
        量子数：
        - $l = 0, 1, 2, ...$ 角动量量子数
        - $m = -l, ..., l$ 磁量子数
        
        性质：
        - 正交归一：$\int Y_l^{m*} Y_{l'}^{m'} d\Omega = \delta_{ll'}\delta_{mm'}$
        - 完备性：任意平方可积函数可展开为球谐函数的级数
        """)
    
    # 动态展示
    st.markdown("---")
    st.subheader("🎬 动态展示（实时动画）")

    col_bessel, col_hermite, col_spherical = st.columns(3)

    with col_bessel:
        st.markdown("**🥁 贝塞尔函数：鼓面振动**")
        bessel_n = st.slider("角向 n", 0, 3, 1, 1, key="bessel_n_dyn")
        bessel_m = st.slider("径向 m", 1, 3, 1, 1, key="bessel_m_dyn")

    with col_hermite:
        st.markdown("**⚛ 量子谐振子：厄米函数**")
        hermite_n = st.slider("中心量子数 n", 0, 6, 3, 1, key="hermite_n_dyn")
        hermite_nmax = st.slider("最大量子数", hermite_n + 2, 10, 8, 1, key="hermite_nmax_dyn")

    with col_spherical:
        st.markdown("**🌐 球谐函数：转动**")
        sh_l = st.slider("角量子数 l", 0, 4, 2, 1, key="sh_l_dyn")
        sh_m = st.slider("磁量子数 m", -sh_l, sh_l, 0, 1, key="sh_m_dyn")

    # 生成并展示三个动画
    c1, c2 = st.columns(2)
    with c1:
        bessel_html = generate_realtime_bessel_mode_html(n=bessel_n, m=bessel_m, R=1.0, grid_size=50, time_scale=1.0)
        components.html(bessel_html, height=760, scrolling=True)

    with c2:
        hermite_html = generate_realtime_hermite_html(n_mode=hermite_n, n_max=hermite_nmax, time_scale=1.0)
        components.html(hermite_html, height=730, scrolling=True)

    st.markdown("---")
    spherical_html = generate_realtime_spherical_harmonics_html(l=sh_l, m=sh_m, time_scale=1.0)
    components.html(spherical_html, height=820, scrolling=True)

    # 通用说明
    st.markdown("---")
    st.info("💡 **使用提示**：\n\n1. **选择函数类别**：使用左侧菜单选择贝塞尔函数、勒让德多项式等\n2. **调整参数**：通过滑块改变最高阶数、显示范围等\n3. **理解性质**：观察函数的零点、正交性、渐近行为等特性\n4. **联系物理**：思考每种特殊函数对应的物理应用场景")

# ==================== AI 助教模块 ====================
elif module == "AI 助教":
    render_ai_tutor_main()

