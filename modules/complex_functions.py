"""
复变函数可视化模块

【教学目标】
帮助初学者理解复变函数的映射概念：
1. 什么是复数平面？
2. 复变函数如何"变换"复数平面？
3. 不同函数有什么特点？

【物理意义】
复变函数可以看作是一种"变换"或"映射"：
- 输入：复数平面上的一个点 z = x + iy
- 输出：经过函数变换后的新位置 w = f(z)

【学习方法】
1. 先看"原始网格"：了解输入平面的样子
2. 再看"变换结果"：观察网格如何被扭曲
3. 理解"不动点"：哪些点变换后位置不变？
"""

import numpy as np
import cmath


def complex_grid(x_min, x_max, y_min, y_max, nx=100, ny=100):
    """
    生成复数平面网格
    
    【给初学者的话】
    复数平面就像一张地图：
    - 横轴（x轴）代表复数的"实部"（real part）
    - 纵轴（y轴）代表复数的"虚部"（imaginary part）
    - 平面上的每个点代表一个复数 z = x + iy
    
    【为什么要生成网格？】
    如果只有一个点，很难理解变换的效果。
    但是如果有一张网格（像渔网一样），变换后网格会变形，
    我们就能直观地看到函数如何"弯曲"整个平面。
    
    参数说明：
    x_min, x_max : 实部范围，比如 -2 到 2
    y_min, y_max : 虚部范围，比如 -2 到 2
    nx, ny       : 网格的密集程度，越大越平滑但计算越慢
    
    示例：
    >>> z = complex_grid(-2, 2, -2, 2, nx=50, ny=50)
    >>> z.shape  # 将是 (50, 50)
    >>> z[25, 25]  # 应该是 0+0j（平面的中心点）
    """
    # 第一步：创建一维坐标数组
    # np.linspace(-2, 2, 50) 会生成50个数，从-2到2均匀分布
    x = np.linspace(x_min, x_max, nx)  # 实部坐标：[x_min, x_min+dx, ..., x_max]
    y = np.linspace(y_min, y_max, ny)  # 虚部坐标：[y_min, y_min+dy, ..., y_max]
    
    # 第二步：创建二维坐标网格
    # np.meshgrid 会创建两个二维数组：
    # X 的每一行都是 x，Y 的每一列都是 y
    X, Y = np.meshgrid(x, y)
    
    # 第三步：组合成复数
    # z = X + iY
    # X[i,j] 是第i行第j列点的实部
    # Y[i,j] 是第i行第j列点的虚部
    z = X + 1j * Y
    
    return z


def apply_function(z, func_name):
    """
    对复数数组应用指定的复变函数
    
    【核心概念：什么是复变函数？】
    复变函数 f(z) 就像一个"复数加工机"：
    - 输入一个复数 z
    - 输出一个复数 w = f(z)
    
    【常见复变函数】
    1. f(z) = z      ：恒等映射，什么都不变
    2. f(z) = z²    ：平方，角度翻倍，模平方
    3. f(z) = √z    ：开方，角度减半，模开方（注意！有两个解）
    4. f(z) = e^z   ：指数，把直角坐标变成极坐标
    5. f(z) = ln(z) ：对数，是指数的反函数
    6. f(z) = sin(z)：正弦，双曲正弦的推广
    7. f(z) = 1/z   ：倒数，复平面上的"反演"
    
    【数学小贴士】
    如果 z = re^(iθ)，那么：
    - z² = r²e^(i2θ)        （角度翻倍！）
    - √z = √r e^(iθ/2)      （角度减半，有两个分支）
    - 1/z = (1/r)e^(-iθ)    （角度取反，模取倒数）
    
    参数：
    z         : 输入的复数数组（可以是2D数组）
    func_name : 函数名称，可选：
        - 'identity' : f(z) = z（最简单，什么都不变）
        - 'square'   : f(z) = z²（最经典的例子）
        - 'cube'     : f(z) = z³
        - 'sqrt'     : f(z) = √z（注意：这是主值分支）
        - 'exp'      : f(z) = e^z（指数函数）
        - 'log'      : f(z) = ln(z)（对数，主值）
        - 'sin'      : f(z) = sin(z)（三角函数）
        - 'cos'      : f(z) = cos(z)
        - '1/z'      : f(z) = 1/z（倒数，复反演）
    
    返回：
    w : 变换后的复数数组
    
    示例：
    >>> z = np.array([1+1j, 2+2j])
    >>> apply_function(z, 'square')
    array([0.+2.j, 0.+8.j])
    """
    # eps 是一个很小的数，用来避免除以零的错误
    # 当 z=0 时，1/z 是未定义的，加 eps = 1e-10 可以避免数值计算错误
    eps = 1e-10
    
    if func_name == 'identity':
        # 【恒等映射】f(z) = z
        # 最简单的函数：输入什么，输出就是什么
        # 复数平面上的每个点都"不动"
        w = z
        
    elif func_name == 'square':
        # 【平方函数】f(z) = z²
        # 这是复变函数中最经典的例子！
        # 
        # 如果 z = re^(iθ)：
        #   z² = r²e^(i2θ)
        # 
        # 效果：
        #   - 模（到原点的距离）变成原来的平方：r → r²
        #   - 角度（与正实轴的夹角）翻倍：θ → 2θ
        #
        # 【直观理解】
        # 想象平面上有一个箭头指向45度方向（角度=π/4）
        # 平方后，这个箭头会转到90度方向（角度=π/2）！
        w = z ** 2
        
    elif func_name == 'cube':
        # 【立方函数】f(z) = z³
        # 类似平方，但角度变成3倍
        # z³ = r³e^(i3θ)
        w = z ** 3
        
    elif func_name == 'sqrt':
        # 【平方根函数】f(z) = √z
        # 
        # 【重要概念：多值函数】
        # 与实数不同，复数的平方根通常有两个值！
        # 例如，√(1) = ±1，但在复数中还有更多分支
        #
        # 极坐标形式：√z = √r e^(iθ/2)
        # 但实际上还有另一个解：√r e^(i(θ/2 + π))
        #
        # 【本代码使用主值分支】
        # 我们取角度在 (-π, π] 范围内的那个解
        w = np.sqrt(z + eps)  # 加 eps 避免 sqrt(0) 的数值问题
        
    elif func_name == 'exp':
        # 【指数函数】f(z) = e^z
        #
        # 【复指数的神奇性质】
        # 如果 z = x + iy：
        #   e^z = e^(x+iy) = e^x · e^(iy) = e^x(cos y + i sin y)
        #
        # 效果：
        #   - 实部 x 控制"大小"（模）：变成 e^x
        #   - 虚部 y 控制"旋转"（角度）：变成 y
        #
        # 【特别提醒】
        # 指数函数是"周期函数"！
        # e^(z + 2πi) = e^z，因为 e^(2πi) = 1
        w = np.exp(z)
        
    elif func_name == 'log':
        # 【对数函数】f(z) = ln(z)
        #
        # 对数是指数的反函数！
        # 如果 w = ln(z)，那么 e^w = z
        #
        # 【极坐标表示】
        # 如果 z = re^(iθ)（θ是主值，在(-π, π]范围内）
        # 那么 ln(z) = ln(r) + iθ
        #
        # 【重要性质】
        # 对数函数也是"多值"的！
        # ln(z) = ln(r) + i(θ + 2kπ)，k = 0, ±1, ±2, ...
        # 本代码取主值（k=0）
        #
        # 【应用】
        # 对数函数可以将"乘法"变成"加法"
        # ln(ab) = ln(a) + ln(b)
        w = np.log(z + eps)
        
    elif func_name == 'sin':
        # 【正弦函数】f(z) = sin(z)
        #
        # 【定义】
        # sin(z) = (e^(iz) - e^(-iz)) / 2i
        #
        # 【与实数正弦的区别】
        # 实数 sin(x) 的值永远在 [-1, 1] 之间
        # 但复数 sin(z) 可以任意大！
        #
        # 如果 z = x + iy：
        # sin(z) = sin(x)cosh(y) + i cos(x)sinh(y)
        #
        # 【可视化意义】
        # 沿着虚轴移动（z = iy），sin(iy) = i·sinh(y)
        # 当 y 很大时，sin(iy) 的模也会变得很大！
        w = np.sin(z)
        
    elif func_name == 'cos':
        # 【余弦函数】f(z) = cos(z)
        #
        # 【定义】
        # cos(z) = (e^(iz) + e^(-iz)) / 2
        #
        # 类似正弦，但值域不受限制
        w = np.cos(z)
        
    elif func_name == '1/z':
        # 【倒数函数】f(z) = 1/z
        #
        # 【极坐标形式】
        # 如果 z = re^(iθ)
        # 那么 1/z = (1/r)e^(-iθ)
        #
        # 效果：
        #   - 模变成倒数：r → 1/r
        #   - 角度取反：θ → -θ
        #
        # 【几何意义】
        # 这叫做"复反演"或"倒数变换"
        # 单位圆上的点（|z|=1）变换后还在单位圆上
        # 单位圆内的点会跑到圆外，反之亦然
        w = 1 / (z + eps)
        
    elif func_name.startswith('z^'):
        # 【一般幂函数】f(z) = z^n
        # 从字符串中提取指数 n
        n = int(func_name.split('^')[1])
        w = z ** n
        
    else:
        # 默认情况：返回原输入
        w = z
    
    return w


def get_function_description(func_name):
    """
    获取函数的描述信息（中文）
    
    【用途】
    这个函数用于在界面上显示函数的简短描述，
    帮助初学者快速了解每个函数的特点。
    
    返回：
    description : 中文描述字符串
    """
    descriptions = {
        'identity': '【恒等映射】f(z) = z\n'
                   '最简单的函数：输入即输出，所有点都"不动"',
        
        'square': '【平方函数】f(z) = z²\n'
                 '经典案例！角度翻倍（θ → 2θ），模平方（r → r²）',
        
        'cube': '【立方函数】f(z) = z³\n'
               '角度变成3倍（θ → 3θ），模立方（r → r³）',
        
        'sqrt': '【平方根函数】f(z) = √z\n'
               '角度减半（θ → θ/2），模开方（r → √r）\n'
               '注意：这里取主值分支',
        
        'exp': '【指数函数】f(z) = e^z\n'
              '实部决定大小，虚部决定旋转\n'
              '特别性质：e^(z+2πi) = e^z（周期为2πi）',
        
        'log': '【对数函数】f(z) = ln(z)\n'
              '指数函数的反函数\n'
              'ln(z) = ln(r) + iθ，r是模，θ是辐角（主值）',
        
        'sin': '【正弦函数】f(z) = sin(z)\n'
              '与实数类似，但值可以任意大！\n'
              'sin(iy) = i·sinh(y)，当y很大时模也很大',
        
        'cos': '【余弦函数】f(z) = cos(z)\n'
              '类似正弦，cos(iy) = cosh(y)',
        
        '1/z': '【倒数函数】f(z) = 1/z\n'
              '复反演：模取倒数（r → 1/r），角度取反（θ → -θ）\n'
              '单位圆上的点还在单位圆上'
    }
    return descriptions.get(func_name, f'幂函数 {func_name}')


def get_function_formula(func_name):
    """
    获取函数的LaTeX公式
    
    【用途】
    用于在界面上用数学公式显示函数表达式。
    """
    formulas = {
        'identity': r'f(z) = z',
        'square': r'f(z) = z^2',
        'cube': r'f(z) = z^3',
        'sqrt': r'f(z) = \sqrt{z}',
        'exp': r'f(z) = e^z',
        'log': r'f(z) = \ln(z)',
        'sin': r'f(z) = \sin(z)',
        'cos': r'f(z) = \cos(z)',
        '1/z': r'f(z) = \frac{1}{z}'
    }
    return formulas.get(func_name, rf'f(z) = {func_name}')


def get_learning_tips(func_name):
    """
    获取学习提示（给初学者的建议）
    
    【设计理念】
    每个函数都有一些值得注意的点，
    这些提示帮助初学者加深理解。
    """
    tips = {
        'identity': 
            '💡 恒等映射是"基准"：其他变换的效果都与它对比\n'
            '   试着理解为什么它不改变任何东西！',
        
        'square': 
            '💡 重点观察角度变化！\n'
            '   - 原点附近的点几乎不动（因为模很小）\n'
            '   - 角度为45°的点会转到90°\n'
            '   - 单位圆（|z|=1）映射为单位圆',
        
        'cube': 
            '💡 类比平方函数，但角度变化更明显\n'
            '   - 30°的点会转到90°\n'
            '   - 120°的点会转到360°（回到起点）',
        
        'sqrt': 
            '💡 这是"多值函数"的典型例子！\n'
            '   同一个输入有多个可能的输出\n'
            '   代码中取的是"主值分支"\n'
            '   尝试理解：为什么会出现多值？',
        
        'exp': 
            '💡 指数函数有"周期性"！\n'
            '   e^(z + 2πi) = e^z\n'
            '   沿着虚轴移动2π，虚部不变\n'
            '   这是复分析与实分析最大的区别之一',
        
        'log': 
            '💡 对数也是多值函数！\n'
            '   ln(z) = ln(r) + i(θ + 2kπ)，k=0,±1,±2,...\n'
            '   复数域中，负数也有对数：ln(-1) = iπ',
        
        'sin': 
            '💡 实数sin(x)永远在[-1,1]，但复数sin(z)可以很大！\n'
            '   这与sinh(y)有关：sin(iy) = i·sinh(y)\n'
            '   当y→∞时，sinh(y)→∞，所以sin(iy)→∞',
        
        'cos': 
            '💡 类似正弦，cos(iy) = cosh(y)\n'
            '   cosh(y)永远大于1（当y≠0时）',
        
        '1/z': 
            '💡 "复反演"是一种保角变换\n'
            '   - 单位圆内→单位圆外\n'
            '   - 右半平面→右半平面\n'
            '   在复变函数的几何理论中很重要'
    }
    return tips.get(func_name, '💡 多练习，多画图！')


def generate_grid_lines(x_min, x_max, y_min, y_max, n_lines=10):
    """
    生成网格线数据
    
    【为什么需要网格线？】
    如果我们只变换一个点，很难理解变换的全貌。
    但如果我们变换一整条线（比如x=1这条垂直线），
    就能看到这条线被"弯曲"成什么形状。
    
    【网格线的意义】
    - 水平线（y=常数）：虚部固定，只改变实部
    - 垂直线（x=常数）：实部固定，只改变虚部
    
    参数：
    x_min, x_max : 实部范围
    y_min, y_max : 虚部范围  
    n_lines      : 每种方向有多少条线
    
    返回：
    lines : 包含所有网格线的列表
            每条线是一个字典：{'x': [...], 'y': [...], 'type': 'horizontal'/'vertical'}
    """
    lines = []
    
    # 【生成水平线】
    # y_values = [-2, -1.5, -1, ..., 2]  （假设范围是-2到2）
    y_values = np.linspace(y_min, y_max, n_lines)
    
    for y in y_values:
        # 对于每一条水平线，x从x_min到x_max变化
        # y值保持不变
        x = np.linspace(x_min, x_max, 100)
        lines.append({
            'x': x, 
            'y': np.full_like(x, y),  # np.full_like创建全为y的数组
            'type': 'horizontal'
        })
    
    # 【生成垂直线】
    # x_values = [-2, -1.5, -1, ..., 2]
    x_values = np.linspace(x_min, x_max, n_lines)
    
    for x in x_values:
        # 对于每一条垂直线，y从y_min到y_max变化
        # x值保持不变
        y = np.linspace(y_min, y_max, 100)
        lines.append({
            'x': np.full_like(y, x),
            'y': y,
            'type': 'vertical'
        })
    
    return lines


def apply_function_to_lines(lines, func_name):
    """
    对网格线应用复变函数变换
    
    【工作流程】
    1. 取一条线（比如 x=1 的垂直线）
    2. 把线上的每个点都看成复数 z = x + iy
    3. 计算 w = f(z)
    4. 画出 w 在新的复平面上的位置
    
    【结果】
    一条"直线"可能会变成"曲线"！
    这就是复变函数"弯曲"平面的直观表现。
    """
    mapped_lines = []
    
    for line in lines:
        # 把 (x, y) 坐标对组合成复数 z = x + iy
        z = line['x'] + 1j * line['y']
        
        # 应用复变函数变换
        w = apply_function(z, func_name)
        
        # 把结果拆分成 x 和 y 坐标
        mapped_lines.append({
            'x': np.real(w),
            'y': np.imag(w),
            'type': line['type']  # 保留线条类型（水平/垂直）
        })
    
    return mapped_lines


def calculate_color_values(z):
    """
    计算颜色值用于可视化（直接返回RGB数组）
    
    【颜色编码的原理】
    我们用颜色来表示复数的"辐角"（arg z = θ）
    - 0°（正实轴）    → 红色
    - 90°（正虚轴）   → 绿色  
    - 180°（负实轴）  → 青色
    - 270°（负虚轴）  → 蓝色
    - 360°（回到正实轴）→ 红色
    
    同时，用颜色的"明暗"表示复数的"模"（|z|）
    
    注意：此函数内部使用防御性编程，确保输出始终为有效的RGB数组。
    """
    # 1. 将输入转换为复数数组（确保形状一致）
    z = np.asarray(z, dtype=np.complex128)
    original_shape = z.shape
    total_elements = int(np.prod(original_shape))
    
    # 2. 计算辐角并归一化到 [0, 1]
    angle = np.angle(z)
    hue = (angle + np.pi) / (2 * np.pi)  # 归一化到 [0, 1]
    
    # 3. 计算模值并归一化作为亮度
    modulus = np.abs(z)
    brightness = np.log(1 + modulus) / np.log(11)
    
    # 4. 将 hue 和 brightness 展平为1D数组进行颜色计算
    h_flat = np.asarray(hue).reshape(total_elements)
    v_flat = np.asarray(brightness).reshape(total_elements)
    
    # 5. 标准 HSV 到 RGB 转换算法（saturation=0.8，固定饱和度）
    saturation = 0.8
    h_scaled = h_flat * 6  # 将 [0,1] 映射到 [0,6]
    i = np.floor(h_scaled).astype(int)  # 整数部分 0..5
    f = h_scaled - i  # 小数部分
    
    # 预先计算所有可能的颜色通道值
    p = v_flat * (1 - saturation)
    q = v_flat * (1 - saturation * f)
    t = v_flat * (1 - saturation * (1 - f))
    
    # 初始化RGB数组
    r_flat = np.zeros(total_elements, dtype=np.float64)
    g_flat = np.zeros(total_elements, dtype=np.float64)
    b_flat = np.zeros(total_elements, dtype=np.float64)
    
    # 根据区间索引赋值
    # i=0: 红→黄   (R,G,B) = (v, t, p)
    mask = (i == 0)
    r_flat[mask] = v_flat[mask]
    g_flat[mask] = t[mask]
    b_flat[mask] = p[mask]
    
    # i=1: 黄→绿   (R,G,B) = (q, v, p)
    mask = (i == 1)
    r_flat[mask] = q[mask]
    g_flat[mask] = v_flat[mask]
    b_flat[mask] = p[mask]
    
    # i=2: 绿→青   (R,G,B) = (p, v, t)
    mask = (i == 2)
    r_flat[mask] = p[mask]
    g_flat[mask] = v_flat[mask]
    b_flat[mask] = t[mask]
    
    # i=3: 青→蓝   (R,G,B) = (p, q, v)
    mask = (i == 3)
    r_flat[mask] = p[mask]
    g_flat[mask] = q[mask]
    b_flat[mask] = v_flat[mask]
    
    # i=4: 蓝→品红 (R,G,B) = (t, p, v)
    mask = (i == 4)
    r_flat[mask] = t[mask]
    g_flat[mask] = p[mask]
    b_flat[mask] = v_flat[mask]
    
    # i=5: 品红→红 (R,G,B) = (v, p, q)
    mask = (i == 5)
    r_flat[mask] = v_flat[mask]
    g_flat[mask] = p[mask]
    b_flat[mask] = q[mask]
    
    # 6. 将颜色值限制在 [0, 1] 范围（防御性）
    r_flat = np.clip(r_flat, 0.0, 1.0)
    g_flat = np.clip(g_flat, 0.0, 1.0)
    b_flat = np.clip(b_flat, 0.0, 1.0)
    
    # 7. 恢复为原始形状
    r = r_flat.reshape(original_shape)
    g = g_flat.reshape(original_shape)
    b = b_flat.reshape(original_shape)
    
    return r, g, b


# 【定义可用的函数列表】
# 这是我们将在界面上展示的所有复变函数
AVAILABLE_FUNCTIONS = [
    {
        'name': 'identity',
        'label': '恒等映射 f(z)=z',
        'difficulty': 1,  # 难度等级：1最简单
        'category': 'basic'  # 类别：基础函数
    },
    {
        'name': 'square', 
        'label': '平方函数 f(z)=z²',
        'difficulty': 2,
        'category': 'power'  # 幂函数
    },
    {
        'name': 'cube',
        'label': '立方函数 f(z)=z³', 
        'difficulty': 3,
        'category': 'power'
    },
    {
        'name': 'sqrt',
        'label': '平方根 f(z)=√z',
        'difficulty': 4,
        'category': 'power',
        'note': '主值分支'  # 特别提示
    },
    {
        'name': 'exp',
        'label': '指数函数 f(z)=e^z',
        'difficulty': 3,
        'category': 'transcendental'  # 超越函数
    },
    {
        'name': 'log',
        'label': '对数函数 f(z)=ln(z)',
        'difficulty': 4,
        'category': 'transcendental',
        'note': '主值分支'
    },
    {
        'name': 'sin',
        'label': '正弦函数 f(z)=sin(z)',
        'difficulty': 3,
        'category': 'trig'  # 三角函数
    },
    {
        'name': 'cos',
        'label': '余弦函数 f(z)=cos(z)',
        'difficulty': 3,
        'category': 'trig'
    },
    {
        'name': '1/z',
        'label': '倒数函数 f(z)=1/z',
        'difficulty': 3,
        'category': 'rational'  # 有理函数
    }
]


# 【可用的复数平面范围选项】
# 用户可以选择不同的缩放级别来观察函数的不同特点
PLANE_RANGES = [
    {'label': '局部 (-1, 1)', 'value': (-1, 1, -1, 1)},
    {'label': '中等 (-2, 2)', 'value': (-2, 2, -2, 2)},
    {'label': '全局 (-3, 3)', 'value': (-3, 3, -3, 3)},
    {'label': '宽广 (-5, 5)', 'value': (-5, 5, -5, 5)},
]


def generate_complex_plot_data(func_name, x_range, y_range, resolution=100):
    """
    生成复变函数可视化的数据
    
    【功能】
    这个函数生成用于绘图的所有数据：
    1. 原始网格线数据
    2. 变换后的网格线数据
    3. 颜色映射数据（基于辐角和模）
    
    参数：
    func_name  : 函数名称
    x_range    : 实部范围 (x_min, x_max)
    y_range    : 虚部范围 (y_min, y_max)
    resolution : 网格分辨率
    
    返回：
    dict 包含：
        - 'lines': 原始网格线
        - 'mapped_lines': 变换后的网格线
        - 'w_real', 'w_imag': 变换后的复数点坐标
        - 'r', 'g', 'b': 颜色值（已展平为1D数组）
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # 生成复数平面网格
    z = complex_grid(x_min, x_max, y_min, y_max, nx=resolution, ny=resolution)
    
    # 应用复变函数
    w = apply_function(z, func_name)
    
    # 计算颜色值（直接返回RGB，形状与 w 相同）
    r, g, b = calculate_color_values(w)
    
    # 生成网格线数据
    n_lines = 10
    lines = generate_grid_lines(x_min, x_max, y_min, y_max, n_lines)
    mapped_lines = apply_function_to_lines(lines, func_name)
    
    # 返回可视化数据 - 所有颜色通道都展平为1D数组
    return {
        'lines': lines,
        'mapped_lines': mapped_lines,
        'w_real': np.real(w).flatten(),
        'w_imag': np.imag(w).flatten(),
        'r': r.flatten(),
        'g': g.flatten(),
        'b': b.flatten()
    }

