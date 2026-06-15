"""
傅里叶级数展开可视化模块

本模块实现了周期函数的傅里叶级数展开可视化，包括：
1. 方波的傅里叶级数展开
2. 三角波的傅里叶级数展开
3. 锯齿波的傅里叶级数展开
4. 动态展示级数收敛过程
"""

import numpy as np


def square_wave_fourier(x, n_terms, A=1, T=2*np.pi):
    """
    方波的傅里叶级数展开
    
    物理意义：
    方波可以表示为一系列正弦波的叠加，只有奇次谐波。
    
    傅里叶级数：
    f(x) = (4A/π) * Σ_{k=0}^{∞} sin((2k+1)ωx) / (2k+1)
    其中 ω = 2π/T
    
    参数：
    x: 位置坐标数组
    n_terms: 级数项数
    A: 方波幅度
    T: 周期
    
    返回：
    y: 傅里叶级数近似值
    """
    y = np.zeros_like(x)
    omega = 2 * np.pi / T
    
    for k in range(n_terms):
        n = 2 * k + 1  # 只取奇次谐波
        y += (4 * A / (np.pi * n)) * np.sin(n * omega * x)
    
    return y


def sawtooth_wave_fourier(x, n_terms, A=1, T=2*np.pi):
    """
    锯齿波的傅里叶级数展开
    
    物理意义：
    锯齿波包含所有谐波分量。
    
    傅里叶级数：
    f(x) = (2A/π) * Σ_{n=1}^{∞} (-1)^{n+1} sin(nωx) / n
    其中 ω = 2π/T
    
    参数：
    x: 位置坐标数组
    n_terms: 级数项数
    A: 幅度
    T: 周期
    
    返回：
    y: 傅里叶级数近似值
    """
    y = np.zeros_like(x)
    omega = 2 * np.pi / T
    
    for n in range(1, n_terms + 1):
        y += (2 * A / (np.pi * n)) * ((-1) ** (n + 1)) * np.sin(n * omega * x)
    
    return y


def triangle_wave_fourier(x, n_terms, A=1, T=2*np.pi):
    """
    三角波的傅里叶级数展开
    
    物理意义：
    三角波只包含奇次谐波，且振幅按1/n²衰减。
    
    傅里叶级数：
    f(x) = (8A/π²) * Σ_{k=0}^{∞} (-1)^k sin((2k+1)ωx) / (2k+1)²
    其中 ω = 2π/T
    
    参数：
    x: 位置坐标数组
    n_terms: 级数项数
    A: 幅度
    T: 周期
    
    返回：
    y: 傅里叶级数近似值
    """
    y = np.zeros_like(x)
    omega = 2 * np.pi / T
    
    for k in range(n_terms):
        n = 2 * k + 1  # 只取奇次谐波
        y += (8 * A / (np.pi ** 2 * n ** 2)) * ((-1) ** k) * np.sin(n * omega * x)
    
    return y


def square_wave_exact(x, A=1, T=2*np.pi):
    """
    方波的精确值（用于对比）
    
    参数：
    x: 位置坐标数组
    A: 幅度
    T: 周期
    
    返回：
    y: 方波精确值
    """
    # 归一化到一个周期内
    x_normalized = np.mod(x, T)
    # 在 [0, T/2) 区间为 A，在 [T/2, T) 区间为 -A
    y = np.where(x_normalized < T/2, A, -A)
    return y


def sawtooth_wave_exact(x, A=1, T=2*np.pi):
    """
    锯齿波的精确值（用于对比）
    
    参数：
    x: 位置坐标数组
    A: 幅度
    T: 周期
    
    返回：
    y: 锯齿波精确值
    """
    # 归一化到一个周期内
    x_normalized = np.mod(x, T)
    # 线性增长从 -A 到 A
    y = (2 * A / T) * x_normalized - A
    return y


def triangle_wave_exact(x, A=1, T=2*np.pi):
    """
    三角波的精确值（用于对比）
    
    参数：
    x: 位置坐标数组
    A: 幅度
    T: 周期
    
    返回：
    y: 三角波精确值
    """
    # 归一化到一个周期内
    x_normalized = np.mod(x, T)
    
    # 前半周期从 -A 线性增长到 A
    # 后半周期从 A 线性下降到 -A
    half_T = T / 2
    y = np.where(
        x_normalized < half_T,
        (4 * A / T) * x_normalized - A,
        (-4 * A / T) * x_normalized + 3 * A
    )
    return y


def generate_fourier_animation_frames(wave_type, max_terms=20, num_frames=50, T=2*np.pi):
    """
    生成傅里叶级数收敛动画的帧数据
    
    参数：
    wave_type: 波形类型 ('square', 'sawtooth', 'triangle')
    max_terms: 最大项数
    num_frames: 帧数
    T: 周期
    
    返回：
    frames_data: 包含每帧数据的列表
    """
    # 生成x坐标
    x = np.linspace(-T, 2*T, 500)
    
    frames_data = []
    
    for frame in range(num_frames):
        # 计算当前帧对应的项数（从1到max_terms）
        n_terms = int(1 + (max_terms - 1) * (frame / (num_frames - 1)))
        
        # 根据波形类型计算傅里叶级数
        if wave_type == 'square':
            y_fourier = square_wave_fourier(x, n_terms, T=T)
            y_exact = square_wave_exact(x, T=T)
        elif wave_type == 'sawtooth':
            y_fourier = sawtooth_wave_fourier(x, n_terms, T=T)
            y_exact = sawtooth_wave_exact(x, T=T)
        elif wave_type == 'triangle':
            y_fourier = triangle_wave_fourier(x, n_terms, T=T)
            y_exact = triangle_wave_exact(x, T=T)
        else:
            y_fourier = x * 0
            y_exact = x * 0
        
        frames_data.append({
            'x': x.tolist(),
            'y_fourier': y_fourier.tolist(),
            'y_exact': y_exact.tolist(),
            'n_terms': n_terms
        })
    
    return frames_data


def get_wave_description(wave_type):
    """
    获取波形的描述信息
    
    参数：
    wave_type: 波形类型
    
    返回：
    description: 描述字符串
    """
    descriptions = {
        'square': {
            'name': '方波',
            'formula': r'f(x) = \frac{4A}{\pi} \sum_{k=0}^{\infty} \frac{\sin((2k+1)\omega x)}{2k+1}',
            'description': '方波由一系列奇次正弦谐波叠加而成，幅度按 1/n 衰减',
            'features': '只包含奇次谐波，在不连续点有吉布斯现象'
        },
        'sawtooth': {
            'name': '锯齿波',
            'formula': r'f(x) = \frac{2A}{\pi} \sum_{n=1}^{\infty} (-1)^{n+1} \frac{\sin(n\omega x)}{n}',
            'description': '锯齿波包含所有谐波分量，幅度按 1/n 衰减',
            'features': '包含所有谐波，收敛较慢'
        },
        'triangle': {
            'name': '三角波',
            'formula': r'f(x) = \frac{8A}{\pi^2} \sum_{k=0}^{\infty} (-1)^k \frac{\sin((2k+1)\omega x)}{(2k+1)^2}',
            'description': '三角波只包含奇次谐波，幅度按 1/n² 衰减',
            'features': '只包含奇次谐波，收敛较快，无明显吉布斯现象'
        }
    }
    return descriptions.get(wave_type, {'name': '未知', 'formula': '', 'description': '', 'features': ''})


# 可用波形列表
AVAILABLE_WAVES = [
    {'name': 'square', 'label': '方波', 'description': '奇次谐波叠加'},
    {'name': 'sawtooth', 'label': '锯齿波', 'description': '包含所有谐波'},
    {'name': 'triangle', 'label': '三角波', 'description': '奇次谐波，1/n²衰减'}
]


def generate_realtime_fourier_html(wave_type, max_terms=30, T=2*np.pi, A=1.0):
    """
    生成傅里叶级数的实时Canvas动画HTML
    
    此函数将物理计算完全移到JavaScript中，
    使用 requestAnimationFrame 持续推进时间，直到用户点击暂停。
    
    参数：
        wave_type: 波形类型 ('square', 'sawtooth', 'triangle')
        max_terms: 最大级数项数
        T: 周期
        A: 幅度
    
    返回：
        html_content: 完整的HTML字符串
    """
    # 波形元数据（供JavaScript使用）
    wave_info = {
        'square': {
            'name': '方波',
            'type': 'odd_only',  # 只含奇次谐波
            'amplitude_factor': 4 * A / np.pi,
            'denominator_power': 1,  # 1/n
            'alternating': False,
            'description': '奇次谐波叠加，幅度按 1/n 衰减'
        },
        'sawtooth': {
            'name': '锯齿波',
            'type': 'all_harmonics',  # 含所有谐波
            'amplitude_factor': 2 * A / np.pi,
            'denominator_power': 1,
            'alternating': True,  # (-1)^(n+1)
            'description': '含所有谐波，幅度按 1/n 衰减'
        },
        'triangle': {
            'name': '三角波',
            'type': 'odd_only',
            'amplitude_factor': 8 * A / (np.pi ** 2),
            'denominator_power': 2,  # 1/n²
            'alternating': True,  # (-1)^k
            'description': '奇次谐波，幅度按 1/n² 衰减，收敛最快'
        }
    }
    
    info = wave_info[wave_type]
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            margin: 0;
            padding: 10px;
            background-color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }}
        .container {{
            width: 100%;
            max-width: 820px;
            margin: 0 auto;
            padding: 10px;
        }}
        .title {{
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            margin: 5px 0;
            color: #333;
        }}
        .info-bar {{
            display: flex;
            justify-content: space-around;
            margin: 10px 0;
            font-size: 13px;
        }}
        .info-box {{
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: bold;
        }}
        .info-time {{ background: #e3f2fd; color: #1565c0; }}
        .info-terms {{ background: #fff3e0; color: #e65100; }}
        .info-error {{ background: #fce4ec; color: #c2185b; }}
        .controls {{
            text-align: center;
            margin: 12px 0;
        }}
        .controls button {{
            padding: 10px 28px;
            font-size: 14px;
            margin: 0 8px;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-weight: bold;
        }}
        .play-btn {{ background-color: #4CAF50; color: white; }}
        .reset-btn {{ background-color: #2196F3; color: white; }}
        .term-slider {{
            margin: 12px auto;
            text-align: center;
        }}
        .term-slider input {{
            width: 300px;
            vertical-align: middle;
        }}
        canvas {{
            display: block;
            margin: 10px auto;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            max-width: 100%;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="title">傅里叶级数：{info['name']}（实时计算）</div>
        
        <div class="info-bar">
            <div class="info-box info-time" id="timeInfo">时间: t = 0.00 s</div>
            <div class="info-box info-terms" id="termsInfo">级数项数: {min(10, max_terms)}</div>
            <div class="info-box info-error" id="errorInfo">最大误差: 0.000</div>
        </div>
        
        <div class="controls">
            <button class="play-btn" id="playBtn" onclick="toggleAnimation()">▶️ 开始仿真</button>
            <button class="reset-btn" onclick="resetAnimation()">🔄 重置</button>
        </div>
        
        <div class="term-slider">
            <label>级数项数: <span id="termValue">{min(10, max_terms)}</span> / {max_terms}</label>
            <br>
            <input type="range" id="termSlider" min="1" max="{max_terms}" value="{min(10, max_terms)}" 
                   oninput="onTermChange(this.value)">
        </div>
        
        <canvas id="fourierCanvas" width="780" height="380"></canvas>
    </div>

    <script>
        // ==== 参数配置（从Python传入）====
        const waveType = '{wave_type}';
        const T = {T:.4f};
        const A = {A:.2f};
        const MAX_TERMS = {max_terms};
        const omega = 2 * Math.PI / T;
        
        // 波形计算配置
        const waveConfig = {{
            'square': {{
                type: 'odd',          // 只含奇次谐波
                factor: {info['amplitude_factor']:.6f},
                power: 1,
                alternating: false
            }},
            'sawtooth': {{
                type: 'all',          // 含所有谐波
                factor: {info['amplitude_factor']:.6f},
                power: 1,
                alternating: true
            }},
            'triangle': {{
                type: 'odd',
                factor: {info['amplitude_factor']:.6f},
                power: 2,
                alternating: true
            }}
        }};
        
        // ==== 动画状态 ====
        let isRunning = false;
        let startTime = 0;
        let elapsedTime = 0;
        let animationId = null;
        let currentTerms = {min(10, max_terms)};
        
        // ==== DOM元素 ====
        const canvas = document.getElementById('fourierCanvas');
        const ctx = canvas.getContext('2d');
        const timeInfo = document.getElementById('timeInfo');
        const termsInfo = document.getElementById('termsInfo');
        const errorInfo = document.getElementById('errorInfo');
        const termValue = document.getElementById('termValue');
        const playBtn = document.getElementById('playBtn');
        
        // ==== 计算精确波形 ====
        function exactWave(x, type) {{
            const xn = ((x % T) + T) % T;  // 归一化到 [0, T)
            if (type === 'square') {{
                return xn < T/2 ? A : -A;
            }} else if (type === 'sawtooth') {{
                return (2*A/T) * xn - A;
            }} else {{ // triangle
                const half = T/2;
                if (xn < half) return (4*A/T)*xn - A;
                else return (-4*A/T)*xn + 3*A;
            }}
        }}
        
        // ==== 计算傅里叶级数部分和 ====
        // u(x, t) = 精确位置不变，但我们让正弦波有时间相关性
        // 更直观地：f(x, t) = Σ A_n sin(nω(x + vt)) 模拟行波
        function fourierPartialSum(x, t, nTerms, type) {{
            const cfg = waveConfig[type];
            let sum = 0;
            
            if (cfg.type === 'odd') {{
                // 奇次谐波: (2k+1)，k=0,...,nTerms-1
                for (let k = 0; k < nTerms; k++) {{
                    const n = 2*k + 1;
                    let coeff = cfg.factor;
                    if (cfg.alternating) coeff *= (k % 2 === 0 ? 1 : -1);
                    const denom = Math.pow(n, cfg.power);
                    // 加入时间依赖：sin(nωx - nωvt)，这里取 v=1
                    sum += (coeff / denom) * Math.sin(n * omega * x - n * omega * t);
                }}
            }} else {{
                // 所有谐波: n=1,...,nTerms
                for (let n = 1; n <= nTerms; n++) {{
                    let coeff = cfg.factor;
                    if (cfg.alternating) coeff *= ((n+1) % 2 === 0 ? 1 : -1);
                    const denom = Math.pow(n, cfg.power);
                    sum += (coeff / denom) * Math.sin(n * omega * x - n * omega * t);
                }}
            }}
            return sum;
        }}
        
        // ==== 主绘制函数 ====
        function draw(t) {{
            // 1. 清屏
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // 2. 绘制网格
            ctx.strokeStyle = '#f0f0f0';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            for (let i = 0; i <= 10; i++) {{
                // 横线
                const y = 30 + i * 32;
                ctx.beginPath(); ctx.moveTo(50, y); ctx.lineTo(750, y); ctx.stroke();
                // 竖线
                const x = 50 + i * 70;
                ctx.beginPath(); ctx.moveTo(x, 30); ctx.lineTo(x, 350); ctx.stroke();
            }}
            ctx.setLineDash([]);
            
            // 3. 绘制坐标轴
            ctx.strokeStyle = '#555';
            ctx.lineWidth = 2;
            // x轴
            ctx.beginPath(); ctx.moveTo(50, 190); ctx.lineTo(750, 190); ctx.stroke();
            // y轴
            ctx.beginPath(); ctx.moveTo(50, 30); ctx.lineTo(50, 350); ctx.stroke();
            
            // 标签
            ctx.font = '13px sans-serif';
            ctx.fillStyle = '#333';
            ctx.fillText('x', 735, 210);
            ctx.fillText('f(x,t)', 55, 45);
            ctx.fillText('0', 55, 210);
            ctx.fillText('A', 55, 50);
            ctx.fillText('-A', 55, 335);
            
            // 绘制范围
            const xMin = -T;
            const xMax = 2*T;
            const yMin = -A * 1.3;
            const yMax = A * 1.3;
            const xScale = 700 / (xMax - xMin);
            const yScale = 160 / (yMax - yMin);
            
            // 4. 计算数据点
            const numPoints = 500;
            let maxError = 0;
            const exactPts = [];
            const fourierPts = [];
            
            for (let i = 0; i <= numPoints; i++) {{
                const x = xMin + (xMax - xMin) * i / numPoints;
                const exactVal = exactWave(x - t, waveType);  // 精确波形也随时间平移
                const fourierVal = fourierPartialSum(x, t, currentTerms, waveType);
                exactPts.push(exactVal);
                fourierPts.push(fourierVal);
                const err = Math.abs(exactVal - fourierVal);
                if (err > maxError) maxError = err;
            }}
            
            // 5. 绘制精确波形（虚线红色）
            ctx.strokeStyle = '#e74c3c';
            ctx.lineWidth = 2;
            ctx.setLineDash([8, 4]);
            ctx.beginPath();
            for (let i = 0; i <= numPoints; i++) {{
                const px = 50 + (xMin + (xMax-xMin)*i/numPoints - xMin) * xScale;
                const py = 190 - exactPts[i] * yScale;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }}
            ctx.stroke();
            ctx.setLineDash([]);
            
            // 6. 绘制傅里叶近似（实线蓝色）
            ctx.strokeStyle = '#2196F3';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            for (let i = 0; i <= numPoints; i++) {{
                const px = 50 + (xMin + (xMax-xMin)*i/numPoints - xMin) * xScale;
                const py = 190 - fourierPts[i] * yScale;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }}
            ctx.stroke();
            
            // 7. 绘制个谐波分量（淡色，供参考）
            if (currentTerms <= 10) {{  // 项数太多时不显示
                const cfg = waveConfig[waveType];
                ctx.lineWidth = 1;
                const harmonics = [];
                let count = 0;
                if (cfg.type === 'odd') {{
                    for (let k = 0; k < currentTerms && count < 5; k++) {{
                        const n = 2*k + 1;
                        let coeff = cfg.factor;
                        if (cfg.alternating) coeff *= (k % 2 === 0 ? 1 : -1);
                        const denom = Math.pow(n, cfg.power);
                        harmonics.push({{n: n, c: coeff/denom}});
                        count++;
                    }}
                }} else {{
                    for (let n = 1; n <= currentTerms && n <= 5; n++) {{
                        let coeff = cfg.factor;
                        if (cfg.alternating) coeff *= ((n+1) % 2 === 0 ? 1 : -1);
                        const denom = Math.pow(n, cfg.power);
                        harmonics.push({{n: n, c: coeff/denom}});
                    }}
                }}
                
                const colors = ['#9c27b0', '#ff9800', '#00bcd4', '#4caf50', '#ff5722'];
                ctx.globalAlpha = 0.35;
                for (let h = 0; h < harmonics.length; h++) {{
                    const harm = harmonics[h];
                    ctx.strokeStyle = colors[h % colors.length];
                    ctx.beginPath();
                    for (let i = 0; i <= numPoints; i++) {{
                        const x = xMin + (xMax-xMin)*i/numPoints;
                        const val = harm.c * Math.sin(harm.n * omega * x - harm.n * omega * t);
                        const px = 50 + (x - xMin) * xScale;
                        const py = 190 - val * yScale;
                        if (i === 0) ctx.moveTo(px, py);
                        else ctx.lineTo(px, py);
                    }}
                    ctx.stroke();
                }}
                ctx.globalAlpha = 1.0;
            }}
            
            // 8. 图例
            ctx.fillStyle = '#e74c3c';
            ctx.fillRect(600, 360, 20, 3);
            ctx.fillStyle = '#333';
            ctx.font = '12px sans-serif';
            ctx.fillText('精确波形', 625, 365);
            
            ctx.fillStyle = '#2196F3';
            ctx.fillRect(600, 375, 20, 3);
            ctx.fillStyle = '#333';
            ctx.fillText('傅里叶近似', 625, 380);
            
            // 9. 更新信息栏
            timeInfo.textContent = `时间: t = ${{t.toFixed(2)}} s`;
            termsInfo.textContent = `级数项数: ${{currentTerms}}`;
            errorInfo.textContent = `最大误差: ${{maxError.toFixed(4)}}`;
        }}
        
        // ==== 动画循环 ====
        function animate(timestamp) {{
            if (!isRunning) return;
            elapsedTime = (timestamp - startTime) / 1000;
            draw(elapsedTime);
            animationId = requestAnimationFrame(animate);
        }}
        
        // ==== 控制函数 ====
        function toggleAnimation() {{
            if (isRunning) {{
                isRunning = false;
                cancelAnimationFrame(animationId);
                playBtn.textContent = '▶️ 继续';
            }} else {{
                if (elapsedTime === 0) {{
                    startTime = performance.now();
                }} else {{
                    startTime = performance.now() - elapsedTime * 1000;
                }}
                isRunning = true;
                playBtn.textContent = '⏸️ 暂停';
                animate(performance.now());
            }}
        }}
        
        function resetAnimation() {{
            isRunning = false;
            cancelAnimationFrame(animationId);
            elapsedTime = 0;
            draw(0);
            playBtn.textContent = '▶️ 开始仿真';
        }}
        
        function onTermChange(val) {{
            currentTerms = parseInt(val);
            termValue.textContent = currentTerms;
            draw(elapsedTime);
        }}
        
        // 初始绘制
        draw(0);
    </script>
</body>
</html>
"""
    return html_content

