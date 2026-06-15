"""
一维波动方程驻波可视化模块

本模块实现了驻波的计算和可视化功能，包括：
1. 驻波函数计算
2. 实时动画渲染（真正的时间推进，不是循环）
3. 节点和波腹位置计算
4. 物理量计算（频率、波长、周期等）
"""

import json
import numpy as np


def standing_wave(x, t, n, A, L, v, damping):
    """
    计算驻波在位置x和时间t的位移
    
    物理意义：
    驻波是由两列振幅、频率相同但传播方向相反的波叠加形成的。
    在两端固定的弦上，满足边界条件 u(0,t)=0 和 u(L,t)=0。
    
    参数：
    x: 位置坐标数组（numpy array）
    t: 时间（float）
    n: 谐波阶数（int），n=1为基频模式
    A: 振幅（float），驻波的最大位移幅度
    L: 弦长（float），弦的长度
    v: 波速（float），波在弦上的传播速度
    damping: 阻尼系数（float），能量耗散系数
    
    返回：
    u: 位移数组（numpy array），与x同形状
    """
    # 角频率 ω = nπv/L
    omega = n * np.pi * v / L
    
    # 驻波公式：u(x,t) = A * exp(-γt) * sin(nπx/L) * cos(ωt)
    # exp(-γt) 表示阻尼衰减，sin(nπx/L) 是空间分布，cos(ωt) 是时间变化
    return A * np.exp(-damping * t) * np.sin(n * np.pi * x / L) * np.cos(omega * t)


def generate_realtime_html_animation(n, A, L, v, damping, animation_speed=2):
    """
    生成真正的实时动画HTML组件
    
    与旧版本不同，这个版本会在JavaScript中实时计算每一帧的波形，
    时间会一直前进直到点击停止，阻尼会真正衰减到零。
    
    参数：
    n: 谐波阶数（int）
    A: 振幅（float）
    L: 弦长（float）
    v: 波速（float）
    damping: 阻尼系数（float）
    animation_speed: 动画速度倍数（int），默认2倍
    
    返回：
    html_content: HTML字符串，包含完整的动画组件
    """
    # 生成节点和波腹数据
    nodes = np.linspace(0, L, n+1).tolist()
    antinodes = []
    for i in range(n):
        antinodes.append((nodes[i] + nodes[i+1]) / 2)
    
    # 创建HTML内容
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            padding: 10px;
            background-color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 500px;
        }}
        .container {{
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
            padding: 10px;
        }}
        .title {{
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
            color: #333333;
        }}
        .time-display {{
            font-size: 16px;
            text-align: center;
            color: #666;
            margin-bottom: 15px;
            font-family: monospace;
        }}
        .controls {{
            text-align: center;
            margin: 15px 0;
        }}
        .controls button {{
            padding: 12px 35px;
            font-size: 15px;
            margin: 0 10px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
        }}
        .play-btn {{
            background-color: #4CAF50;
            color: white;
        }}
        .reset-btn {{
            background-color: #2196F3;
            color: white;
        }}
        .status {{
            text-align: center;
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
            display: inline-block;
            margin: 10px 0;
        }}
        .status.running {{
            background-color: #e8f5e9;
            color: #2e7d32;
        }}
        .status.stopped {{
            background-color: #ffebee;
            color: #c62828;
        }}
        .canvas-wrapper {{
            text-align: center;
            margin: 15px 0;
        }}
        canvas {{
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            max-width: 100%;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="title">第{n}阶驻波模式（实时仿真）</div>
        <div class="time-display" id="timeDisplay">时间: t = 0.00 s</div>
        <div class="controls">
            <button class="play-btn" id="playBtn" onclick="toggleAnimation()">▶️ 开始仿真</button>
            <button class="reset-btn" onclick="resetAnimation()">🔄 重置</button>
        </div>
        <div style="text-align: center;">
            <span class="status stopped" id="status">状态: 已停止</span>
        </div>
        <div class="canvas-wrapper">
            <canvas id="waveCanvas" width="750" height="400"></canvas>
        </div>
    </div>

    <script>
        // 物理参数（从Python传入）
        const n = {n};
        const L = {L};
        const A0 = {A};
        const v = {v};
        const damping = {damping};
        const speed = {animation_speed};
        
        // 节点和波腹位置
        const nodes = {json.dumps(nodes)};
        const antinodes = {json.dumps(antinodes)};
        
        // 动画状态
        let isRunning = false;
        let startTime = 0;
        let elapsedTime = 0;
        let animationId = null;
        
        // 获取元素
        const canvas = document.getElementById('waveCanvas');
        const ctx = canvas.getContext('2d');
        const timeDisplay = document.getElementById('timeDisplay');
        const statusDisplay = document.getElementById('status');
        const playBtn = document.getElementById('playBtn');
        
        // 实时计算驻波波形
        function calculateWave(t) {{
            const x = [];
            const y = [];
            const numPoints = 200;
            
            for (let i = 0; i <= numPoints; i++) {{
                const xi = (i / numPoints) * L;
                const omega = n * Math.PI * v / L;
                const Ai = A0 * Math.exp(-damping * t);
                const yi = Ai * Math.sin(n * Math.PI * xi / L) * Math.cos(omega * t);
                x.push(xi);
                y.push(yi);
            }}
            
            return {{ x, y }};
        }}
        
        // 绘制函数
        function draw(t) {{
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const wave = calculateWave(t);
            timeDisplay.textContent = `时间: t = ${{t.toFixed(2)}} s`;
            
            // 绘制网格
            ctx.strokeStyle = '#f0f0f0';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            
            for (let i = 0; i <= 6; i++) {{
                const y = 50 + (i * 350 / 6);
                ctx.beginPath();
                ctx.moveTo(60, y);
                ctx.lineTo(690, y);
                ctx.stroke();
            }}
            
            const xSteps = L <= 2 ? 10 : Math.ceil(L);
            for (let i = 0; i <= xSteps; i++) {{
                const x = 60 + (i * 630 / xSteps);
                ctx.beginPath();
                ctx.moveTo(x, 50);
                ctx.lineTo(x, 350);
                ctx.stroke();
            }}
            
            ctx.setLineDash([]);
            
            // 坐标轴
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(60, 200);
            ctx.lineTo(690, 200);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(60, 50);
            ctx.lineTo(60, 350);
            ctx.stroke();
            
            // 标签
            ctx.font = '14px sans-serif';
            ctx.fillStyle = '#333';
            ctx.fillText('位置 x', 650, 220);
            
            ctx.save();
            ctx.translate(30, 200);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('位移 u(x,t)', -150, 0);
            ctx.restore();
            
            // 刻度
            ctx.font = '12px sans-serif';
            for (let i = 0; i <= xSteps; i++) {{
                const xi = (i / xSteps) * L;
                const x = 60 + (i * 630 / xSteps);
                ctx.fillText(xi.toFixed(1), x - 10, 220);
            }}
            
            const currentA = A0 * Math.exp(-damping * t);
            const yMax = Math.max(currentA * 1.3, 1);
            ctx.fillText(yMax.toFixed(1), 25, 65);
            ctx.fillText((yMax/2).toFixed(1), 25, 135);
            ctx.fillText('0', 50, 205);
            ctx.fillText((-yMax/2).toFixed(1), 25, 275);
            ctx.fillText((-yMax).toFixed(1), 25, 345);
            
            // 缩放
            const xScale = 630 / L;
            const yScale = 150 / yMax;
            
            // 波形
            ctx.strokeStyle = '#2196F3';
            ctx.lineWidth = 3;
            ctx.beginPath();
            
            for (let i = 0; i < wave.x.length; i++) {{
                const x = 60 + wave.x[i] * xScale;
                const y = 200 - wave.y[i] * yScale;
                
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();
            
            // 节点
            ctx.fillStyle = '#f44336';
            nodes.forEach(nodeX => {{
                const x = 60 + nodeX * xScale;
                ctx.beginPath();
                ctx.arc(x, 200, 8, 0, Math.PI * 2);
                ctx.fill();
            }});
            
            // 波腹
            ctx.fillStyle = '#4CAF50';
            antinodes.forEach(antinodeX => {{
                const x = 60 + antinodeX * xScale;
                const idx = Math.round(antinodeX / L * 200);
                const yPos = wave.y[idx] * yScale;
                ctx.beginPath();
                ctx.arc(x, 200 - yPos, 8, 0, Math.PI * 2);
                ctx.fill();
            }});
            
            // 图例
            ctx.font = '12px sans-serif';
            ctx.fillStyle = '#f44336';
            ctx.beginPath();
            ctx.arc(620, 70, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillText('节点', 635, 75);
            
            ctx.fillStyle = '#4CAF50';
            ctx.beginPath();
            ctx.arc(620, 95, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillText('波腹', 635, 100);
            
            // 阻尼信息
            if (damping > 0) {{
                ctx.fillStyle = '#ff9800';
                ctx.fillText('初始振幅: ' + A0.toFixed(2), 620, 125);
                ctx.fillText('当前振幅: ' + currentA.toFixed(3), 620, 145);
            }}
        }}
        
        // 动画循环
        function animate(timestamp) {{
            if (!isRunning) return;
            elapsedTime = (timestamp - startTime) / 1000 * speed;
            draw(elapsedTime);
            animationId = requestAnimationFrame(animate);
        }}
        
        // 切换动画
        function toggleAnimation() {{
            if (isRunning) {{
                isRunning = false;
                cancelAnimationFrame(animationId);
                statusDisplay.textContent = '状态: 已暂停';
                statusDisplay.className = 'status stopped';
                playBtn.textContent = '▶️ 继续';
            }} else {{
                if (elapsedTime === 0) {{
                    startTime = performance.now();
                }} else {{
                    startTime = performance.now() - elapsedTime * 1000 / speed;
                }}
                isRunning = true;
                statusDisplay.textContent = '状态: 运行中';
                statusDisplay.className = 'status running';
                playBtn.textContent = '⏸️ 暂停';
                animate(performance.now());
            }}
        }}
        
        // 重置
        function resetAnimation() {{
            isRunning = false;
            cancelAnimationFrame(animationId);
            elapsedTime = 0;
            draw(0);
            statusDisplay.textContent = '状态: 已停止';
            statusDisplay.className = 'status stopped';
            playBtn.textContent = '▶️ 开始仿真';
        }}
        
        // 初始绘制
        draw(0);
    </script>
</body>
</html>
"""
    
    return html_content


# 保持向后兼容的旧函数
def generate_html_animation(n, A, L, v, damping, animation_speed=2):
    """
    生成动画HTML组件（向后兼容）
    
    注意：新代码应使用 generate_realtime_html_animation
    """
    return generate_realtime_html_animation(n, A, L, v, damping, animation_speed)


def generate_animation_frames(n, A, L, v, damping, num_frames=100):
    """
    预先生成所有动画帧的数据
    
    参数：
    n: 谐波阶数（int）
    A: 振幅（float）
    L: 弦长（float）
    v: 波速（float）
    damping: 阻尼系数（float）
    num_frames: 帧数（int），默认100帧
    
    返回：
    frames_data: 包含所有帧数据的列表
    """
    x = np.linspace(0, L, 100).tolist()
    frames_data = []
    for frame in range(num_frames):
        t = frame * 0.02
        y = standing_wave(np.array(x), t, n, A, L, v, damping).tolist()
        frames_data.append({"x": x, "y": y})
    return frames_data


def get_nodes_antinodes(n, L):
    """
    获取节点和波腹位置
    
    参数：
    n: 谐波阶数（int）
    L: 弦长（float）
    
    返回：
    nodes: 节点位置列表
    antinodes: 波腹位置列表
    """
    nodes = np.linspace(0, L, n+1).tolist()
    antinodes = []
    for i in range(n):
        antinodes.append((nodes[i] + nodes[i+1]) / 2)
    return nodes, antinodes


def calculate_physical_quantities(n, L, v):
    """
    计算驻波的关键物理量
    
    参数：
    n: 谐波阶数（int）
    L: 弦长（float）
    v: 波速（float）
    
    返回：
    quantities: 包含各物理量的字典
    """
    # 角频率：ω = nπv/L
    omega = n * np.pi * v / L
    
    # 频率：f = ω/(2π) = nv/(2L)
    f = omega / (2 * np.pi)
    
    # 波长：λ = 2L/n
    wavelength = 2 * L / n
    
    # 周期：T = 1/f
    T = 1 / f if f != 0 else float('inf')
    
    return {
        'omega': omega,      # 角频率
        'frequency': f,      # 频率
        'wavelength': wavelength,  # 波长
        'period': T          # 周期
    }
