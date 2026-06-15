"""实时 JavaScript Canvas 动画模块。

提供三个函数：
- generate_realtime_heat_1d_html(init_type, alpha, L, nx)
- generate_realtime_wave_1d_html(init_type, c, L, nx, damping)
- generate_realtime_drumhead_html(init_type, c, R, n)

每个函数返回一个完整的 HTML 字符串，里面包含 CSS 样式、
Canvas 元素以及真正做数值推进的 JavaScript 代码，
动画通过 requestAnimationFrame 循环。
"""

import math


# ================================================================
# 一维热传导方程
# ================================================================
def _heat_init_expr(init_type, L):
    if init_type == 'sine':
        return 'Math.sin(Math.PI * xi / {:.17g})'.format(L)
    if init_type == 'gaussian':
        return 'Math.exp(-50 * (xi - {:.17g}) * (xi - {:.17g}))'.format(L/2, L/2)
    if init_type == 'step':
        return '(xi < {:.17g}) ? 1.0 : 0.0'.format(L/2)
    if init_type == 'double_peak':
        return ('Math.exp(-50*(xi-{:.17g})*(xi-{:.17g})) + '
                'Math.exp(-50*(xi-{:.17g})*(xi-{:.17g}))').format(
                    L*0.25, L*0.25, L*0.75, L*0.75)
    return 'Math.sin(Math.PI * xi / {:.17g})'.format(L)


def _heat_title(init_type):
    return {
        'sine': '正弦初始分布',
        'gaussian': '高斯脉冲热源',
        'step': '阶梯初始分布',
        'double_peak': '双峰初始分布',
    }.get(init_type, '正弦初始分布')


def _common_css():
    return (
        '* { box-sizing: border-box; }\n'
        'body { margin:0; padding:5px; background:#ffffff;'
        ' font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif;'
        ' min-height:400px; }\n'
        '.container { width:100%; max-width:800px; margin:0 auto; padding:5px; }\n'
        '.title { font-size:18px; font-weight:bold; text-align:center;'
        ' margin:4px 0; color:#333333; }\n'
        '.subtitle { font-size:12px; text-align:center; color:#888;'
        ' margin-bottom:4px; font-family:"Cambria Math",serif; }\n'
        '.time-display { font-size:14px; text-align:center; color:#666;'
        ' margin-bottom:8px; font-family:monospace; }\n'
        '.controls { text-align:center; margin:8px 0; }\n'
        '.controls button { padding:8px 24px; font-size:13px; margin:0 6px;'
        ' border:none; border-radius:18px; cursor:pointer; font-weight:bold; }\n'
        '.play-btn { background:#4CAF50; color:white; }\n'
        '.reset-btn { background:#2196F3; color:white; }\n'
        '.status { text-align:center; padding:4px 12px; border-radius:12px;'
        ' font-size:12px; font-weight:bold; display:inline-block; margin:4px 0; }\n'
        '.status.running { background:#e8f5e9; color:#2e7d32; }\n'
        '.status.stopped { background:#ffebee; color:#c62828; }\n'
        '.info-bar { text-align:center; color:#555; font-size:12px;'
        ' margin:2px 0; font-family:monospace; }\n'
        '.canvas-wrapper { text-align:center; margin:8px 0; }\n'
        'canvas { border:2px solid #e0e0e0; border-radius:8px;'
        ' box-shadow:0 2px 8px rgba(0,0,0,0.1); max-width:100%; }\n'
    )


def _heat_js_body(nx, L, alpha, dt, steps_per_frame, init_expr, plot_color, fill_color):
    return (
        '  const L = {:.17g};\n'.format(L) +
        '  const alpha = {:.17g};\n'.format(alpha) +
        '  const nx = {};\n'.format(nx) +
        '  const dx = L / (nx - 1);\n'
        '  const dt = {:.17g};\n'.format(dt) +
        '  const r = alpha * dt / (dx * dx);\n'
        '  const stepsPerFrame = {};\n'.format(steps_per_frame) +
        '\n'
        '  let u = new Array(nx);\n'
        '  let uNew = new Array(nx);\n'
        '  let isRunning = false;\n'
        '  let simTime = 0;\n'
        '  let animationId = null;\n'
        '  let uMax = 1.0;\n'
        '\n'
        '  function initialize() {\n'
        '    for (let i = 0; i < nx; i++) {\n'
        '      const xi = i * dx;\n'
        '      u[i] = ' + init_expr + ';\n'
        '    }\n'
        '    u[0] = 0; u[nx-1] = 0;\n'
        '    let m = 1e-6;\n'
        '    for (let i = 0; i < nx; i++) if (Math.abs(u[i]) > m) m = Math.abs(u[i]);\n'
        '    uMax = m;\n'
        '    if (uMax < 1e-6) uMax = 1.0;\n'
        '  }\n'
        '\n'
        '  function step() {\n'
        '    for (let i = 1; i < nx-1; i++)\n'
        '      uNew[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1]);\n'
        '    uNew[0] = 0; uNew[nx-1] = 0;\n'
        '    const tmp = u; u = uNew; uNew = tmp;\n'
        '    simTime += dt;\n'
        '  }\n'
        '\n'
        '  function draw() {\n'
        '    const canvas = document.getElementById("heatCanvas");\n'
        '    const ctx = canvas.getContext("2d");\n'
        '    const W = canvas.width, H = canvas.height;\n'
        '    ctx.fillStyle = "#ffffff"; ctx.fillRect(0,0,W,H);\n'
        '    const padL=70, padR=50, padT=40, padB=60;\n'
        '    const plotW = W - padL - padR;\n'
        '    const plotH = H - padT - padB;\n'
        '    let curMax = 1e-6;\n'
        '    for (let i = 0; i < nx; i++)\n'
        '      if (Math.abs(u[i]) > curMax) curMax = Math.abs(u[i]);\n'
        '    const yMax = Math.max(curMax * 1.3, uMax * 0.05);\n'
        '\n'
        '    ctx.strokeStyle = "#f0f0f0"; ctx.lineWidth = 1; ctx.setLineDash([5,5]);\n'
        '    for (let i = 0; i <= 5; i++) {\n'
        '      const yy = padT + i*plotH/5;\n'
        '      ctx.beginPath(); ctx.moveTo(padL,yy); ctx.lineTo(padL+plotW,yy); ctx.stroke();\n'
        '    }\n'
        '    for (let i = 0; i <= 10; i++) {\n'
        '      const xx = padL + i*plotW/10;\n'
        '      ctx.beginPath(); ctx.moveTo(xx,padT); ctx.lineTo(xx,padT+plotH); ctx.stroke();\n'
        '    }\n'
        '    ctx.setLineDash([]);\n'
        '    ctx.strokeStyle = "#333"; ctx.lineWidth = 2;\n'
        '    ctx.beginPath(); ctx.moveTo(padL, padT+plotH); ctx.lineTo(padL+plotW, padT+plotH); ctx.stroke();\n'
        '    ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT+plotH); ctx.stroke();\n'
        '\n'
        '    ctx.font = "14px sans-serif"; ctx.fillStyle = "#333";\n'
        '    ctx.fillText("x", padL+plotW-30, padT+plotH+40);\n'
        '    ctx.save(); ctx.translate(padL-50, padT+plotH/2);\n'
        '    ctx.rotate(-Math.PI/2); ctx.fillText("u(x,t)", 0, 0); ctx.restore();\n'
        '\n'
        '    ctx.font = "12px sans-serif";\n'
        '    for (let i = 0; i <= 10; i++) {\n'
        '      const xv = (i/10)*L;\n'
        '      const xx = padL + i*plotW/10;\n'
        '      ctx.fillText(xv.toFixed(1), xx-8, padT+plotH+18);\n'
        '    }\n'
        '    for (let i = 0; i <= 5; i++) {\n'
        '      const yv = (1-i/5)*yMax;\n'
        '      const yy = padT + i*plotH/5;\n'
        '      ctx.fillText(yv.toFixed(2), padL-50, yy+4);\n'
        '    }\n'
        '\n'
        '    ctx.strokeStyle = "' + plot_color + '"; ctx.lineWidth = 2.5;\n'
        '    ctx.beginPath();\n'
        '    for (let i = 0; i < nx; i++) {\n'
        '      const xx = padL + (i/(nx-1))*plotW;\n'
        '      const yy = padT + plotH - (u[i]/yMax)*plotH;\n'
        '      if (i===0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);\n'
        '    }\n'
        '    ctx.stroke();\n'
        '\n'
        '    ctx.fillStyle = "' + fill_color + '";\n'
        '    ctx.beginPath();\n'
        '    for (let i = 0; i < nx; i++) {\n'
        '      const xx = padL + (i/(nx-1))*plotW;\n'
        '      const yy = padT + plotH - (u[i]/yMax)*plotH;\n'
        '      if (i===0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);\n'
        '    }\n'
        '    ctx.lineTo(padL+plotW, padT+plotH); ctx.lineTo(padL, padT+plotH);\n'
        '    ctx.closePath(); ctx.fill();\n'
        '\n'
        '    ctx.font = "12px sans-serif"; ctx.fillStyle = "#333";\n'
        '    ctx.fillText("r = alpha*dt/dx^2 = " + r.toFixed(3), padL+plotW-180, padT-15);\n'
        '    document.getElementById("timeDisplay").textContent =\n'
        '        "时间: t = " + simTime.toFixed(3) + " s";\n'
        '  }\n'
    )


def _animate_loop_js(canvas_name_init_fn):
    """返回通用动画循环 + 切换/重置函数的 JS 字符串。"""
    return (
        '  function animate() {\n'
        '    if (!isRunning) return;\n'
        '    for (let s = 0; s < stepsPerFrame; s++) step();\n'
        '    draw();\n'
        '    animationId = requestAnimationFrame(animate);\n'
        '  }\n'
        '\n'
        '  function toggleAnimation() {\n'
        '    if (isRunning) {\n'
        '      isRunning = false; cancelAnimationFrame(animationId);\n'
        '      document.getElementById("status").textContent = "状态: 已暂停";\n'
        '      document.getElementById("status").className = "status stopped";\n'
        '      document.getElementById("playBtn").textContent = "继续";\n'
        '    } else {\n'
        '      isRunning = true;\n'
        '      document.getElementById("status").textContent = "状态: 运行中";\n'
        '      document.getElementById("status").className = "status running";\n'
        '      document.getElementById("playBtn").textContent = "暂停";\n'
        '      animationId = requestAnimationFrame(animate);\n'
        '    }\n'
        '  }\n'
        '\n'
        '  function resetAnimation() {\n'
        '    isRunning = false; cancelAnimationFrame(animationId);\n'
        '    simTime = 0;\n'
        '    ' + canvas_name_init_fn + '();\n'
        '    draw();\n'
        '    document.getElementById("status").textContent = "状态: 已停止";\n'
        '    document.getElementById("status").className = "status stopped";\n'
        '    document.getElementById("playBtn").textContent = "开始仿真";\n'
        '  }\n'
        '\n'
    )


def generate_realtime_heat_1d_html(init_type='sine', alpha=0.1, L=1.0, nx=100):
    """生成实时一维热传导方程动画 HTML。

    使用 Forward Euler 显式有限差分推进：
        u_i^{n+1} = u_i^n + r * (u_{i+1}^n - 2*u_i^n + u_{i-1}^n)
    其中 r = alpha * dt / dx^2。稳定条件 r < 0.5。
    """
    dx = L / (nx - 1)
    r_stable = 0.2
    dt = r_stable * dx * dx / alpha
    steps_per_frame = max(1, int(1.0 / 30.0 / dt))

    title_name = _heat_title(init_type)
    init_expr = _heat_init_expr(init_type, L)

    html_parts = []
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html><head><meta charset="UTF-8">')
    html_parts.append('<style>')
    html_parts.append(_common_css())
    html_parts.append('</style></head>')
    html_parts.append('<body>')
    html_parts.append('<div class="container">')
    html_parts.append('  <div class="title">一维热传导方程（实时仿真）</div>')
    # 标题中用 Unicode 字符表示方程
    html_parts.append('  <div class="subtitle">\u2202u/\u2202t = '
                      '\u03b1 \u2202\u00b2u/\u2202x\u00b2'
                      '&nbsp;&nbsp;' + title_name + '</div>')
    html_parts.append(
        '  <div class="info-bar">\u03b1 = {:.3g}, '
        '&nbsp; nx = {}, '
        '&nbsp; dt = {:.6f} s</div>'.format(alpha, nx, dt))
    html_parts.append(
        '  <div class="time-display" id="timeDisplay">时间: t = 0.00 s</div>')
    html_parts.append('  <div class="controls">')
    html_parts.append(
        '    <button class="play-btn" id="playBtn" onclick="toggleAnimation()">开始仿真</button>')
    html_parts.append(
        '    <button class="reset-btn" onclick="resetAnimation()">重置</button>')
    html_parts.append('  </div>')
    html_parts.append(
        '  <div style="text-align:center;">'
        '<span class="status stopped" id="status">状态: 已停止</span></div>')
    html_parts.append(
        '  <div class="canvas-wrapper">'
        '<canvas id="heatCanvas" width="750" height="400"></canvas></div>')
    html_parts.append('</div>')

    html_parts.append('<script>')
    html_parts.append(_heat_js_body(nx, L, alpha, dt, steps_per_frame,
                                    init_expr, '#FF5722', 'rgba(255,87,34,0.12)'))
    html_parts.append(_animate_loop_js('initialize'))
    html_parts.append('  initialize(); draw();')
    html_parts.append('</script>')
    html_parts.append('</body></html>')
    return '\n'.join(html_parts)


# ================================================================
# 一维波动方程
# ================================================================
def _wave_init_expr(init_type, L):
    if init_type == 'sine':
        return 'Math.sin(Math.PI * xi / {:.17g})'.format(L)
    if init_type == 'gaussian':
        return 'Math.exp(-50 * (xi - {:.17g}) * (xi - {:.17g}))'.format(L/2, L/2)
    if init_type == 'square':
        return '((xi > {:.17g}) && (xi < {:.17g})) ? 1.0 : 0.0'.format(L*0.3, L*0.7)
    if init_type == 'wave_packet':
        return ('Math.exp(-50*(xi-{:.17g})*(xi-{:.17g}))'
                '*Math.sin(20*Math.PI*xi/{:.17g})').format(L/2, L/2, L)
    return 'Math.sin(Math.PI * xi / {:.17g})'.format(L)


def _wave_title(init_type):
    return {
        'sine': '正弦初始位移',
        'gaussian': '高斯波包',
        'square': '方波脉冲',
        'wave_packet': '行波波包',
    }.get(init_type, '正弦初始位移')


def _wave_js_body(nx, L, c, damping, dt, steps_per_frame, u0_expr,
                  plot_color, fill_color):
    return (
        '  const L = {:.17g};\n'.format(L) +
        '  const c = {:.17g};\n'.format(c) +
        '  const damping = {:.17g};\n'.format(damping) +
        '  const nx = {};\n'.format(nx) +
        '  const dx = L / (nx - 1);\n'
        '  const dt = {:.17g};\n'.format(dt) +
        '  const cfl = c * dt / dx;\n'
        '  const cfl2 = cfl * cfl;\n'
        '  const stepsPerFrame = {};\n'.format(steps_per_frame) +
        '\n'
        '  let uPrev = new Array(nx);\n'
        '  let uCurr = new Array(nx);\n'
        '  let uNext = new Array(nx);\n'
        '  let isRunning = false;\n'
        '  let simTime = 0;\n'
        '  let animationId = null;\n'
        '  let u0Max = 1.0;\n'
        '\n'
        '  function setU0() {\n'
        '    for (let i = 0; i < nx; i++) {\n'
        '      const xi = i * dx;\n'
        '      uCurr[i] = ' + u0_expr + ';\n'
        '    }\n'
        '    uCurr[0] = 0; uCurr[nx-1] = 0;\n'
        '  }\n'
        '\n'
        '  function initialize() {\n'
        '    setU0();\n'
        '    let m = 1e-6;\n'
        '    for (let i = 0; i < nx; i++)\n'
        '      if (Math.abs(uCurr[i]) > m) m = Math.abs(uCurr[i]);\n'
        '    u0Max = m; if (u0Max < 1e-6) u0Max = 1.0;\n'
        '    for (let i = 1; i < nx-1; i++) {\n'
        '      const lap = uCurr[i+1] - 2*uCurr[i] + uCurr[i-1];\n'
        '      uNext[i] = uCurr[i] + 0.5 * cfl2 * lap;\n'
        '    }\n'
        '    uNext[0] = 0; uNext[nx-1] = 0;\n'
        '    const tmp1 = uPrev; uPrev = uCurr; uCurr = uNext; uNext = tmp1;\n'
        '    simTime = dt;\n'
        '  }\n'
        '\n'
        '  function step() {\n'
        '    for (let i = 1; i < nx-1; i++) {\n'
        '      const lap = uCurr[i+1] - 2*uCurr[i] + uCurr[i-1];\n'
        '      let val = 2*uCurr[i] - uPrev[i] + cfl2*lap;\n'
        '      if (damping > 0) val = val - damping * dt * (uCurr[i] - uPrev[i]);\n'
        '      uNext[i] = val;\n'
        '    }\n'
        '    uNext[0] = 0; uNext[nx-1] = 0;\n'
        '    const tmp = uPrev; uPrev = uCurr; uCurr = uNext; uNext = tmp;\n'
        '    simTime += dt;\n'
        '  }\n'
        '\n'
        '  function draw() {\n'
        '    const canvas = document.getElementById("waveCanvas");\n'
        '    const ctx = canvas.getContext("2d");\n'
        '    const W = canvas.width, H = canvas.height;\n'
        '    ctx.fillStyle = "#ffffff"; ctx.fillRect(0,0,W,H);\n'
        '    const padL=70, padR=50, padT=40, padB=60;\n'
        '    const plotW = W - padL - padR;\n'
        '    const plotH = H - padT - padB;\n'
        '    const yCenter = padT + plotH/2;\n'
        '    let curMax = 1e-6;\n'
        '    for (let i = 0; i < nx; i++)\n'
        '      if (Math.abs(uCurr[i]) > curMax) curMax = Math.abs(uCurr[i]);\n'
        '    const yMax = Math.max(curMax * 1.3, u0Max * 0.05);\n'
        '\n'
        '    ctx.strokeStyle = "#f0f0f0"; ctx.lineWidth = 1; ctx.setLineDash([5,5]);\n'
        '    for (let i = 0; i <= 6; i++) {\n'
        '      const yy = padT + i*plotH/6;\n'
        '      ctx.beginPath(); ctx.moveTo(padL,yy); ctx.lineTo(padL+plotW,yy); ctx.stroke();\n'
        '    }\n'
        '    for (let i = 0; i <= 10; i++) {\n'
        '      const xx = padL + i*plotW/10;\n'
        '      ctx.beginPath(); ctx.moveTo(xx,padT); ctx.lineTo(xx,padT+plotH); ctx.stroke();\n'
        '    }\n'
        '    ctx.setLineDash([]);\n'
        '    ctx.strokeStyle = "#bbb"; ctx.lineWidth = 1;\n'
        '    ctx.beginPath(); ctx.moveTo(padL, yCenter); ctx.lineTo(padL+plotW, yCenter); ctx.stroke();\n'
        '    ctx.strokeStyle = "#333"; ctx.lineWidth = 2;\n'
        '    ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT+plotH); ctx.stroke();\n'
        '\n'
        '    ctx.font = "14px sans-serif"; ctx.fillStyle = "#333";\n'
        '    ctx.fillText("x", padL+plotW-30, yCenter+40);\n'
        '    ctx.save(); ctx.translate(padL-50, yCenter);\n'
        '    ctx.rotate(-Math.PI/2); ctx.fillText("u(x,t)", 0, 0); ctx.restore();\n'
        '\n'
        '    ctx.font = "12px sans-serif";\n'
        '    for (let i = 0; i <= 10; i++) {\n'
        '      const xv = (i/10)*L;\n'
        '      const xx = padL + i*plotW/10;\n'
        '      ctx.fillText(xv.toFixed(1), xx-8, yCenter+18);\n'
        '    }\n'
        '    for (let i = 0; i <= 6; i++) {\n'
        '      const yv = (0.5-i/6)*2*yMax;\n'
        '      const yy = padT + i*plotH/6;\n'
        '      ctx.fillText(yv.toFixed(2), padL-50, yy+4);\n'
        '    }\n'
        '\n'
        '    ctx.strokeStyle = "' + plot_color + '"; ctx.lineWidth = 2.5;\n'
        '    ctx.beginPath();\n'
        '    for (let i = 0; i < nx; i++) {\n'
        '      const xx = padL + (i/(nx-1))*plotW;\n'
        '      const yy = yCenter - (uCurr[i]/yMax)*(plotH/2);\n'
        '      if (i===0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);\n'
        '    }\n'
        '    ctx.stroke();\n'
        '\n'
        '    ctx.fillStyle = "' + fill_color + '";\n'
        '    ctx.beginPath();\n'
        '    for (let i = 0; i < nx; i++) {\n'
        '      const xx = padL + (i/(nx-1))*plotW;\n'
        '      const yy = yCenter - (uCurr[i]/yMax)*(plotH/2);\n'
        '      if (i===0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);\n'
        '    }\n'
        '    ctx.lineTo(padL+plotW, yCenter); ctx.lineTo(padL, yCenter);\n'
        '    ctx.closePath(); ctx.fill();\n'
        '\n'
        '    ctx.font = "12px sans-serif"; ctx.fillStyle = "#333";\n'
        '    ctx.fillText("CFL = c*dt/dx = " + cfl.toFixed(3), padL+plotW-180, padT-15);\n'
        '    document.getElementById("timeDisplay").textContent =\n'
        '        "时间: t = " + simTime.toFixed(3) + " s";\n'
        '  }\n'
    )


def generate_realtime_wave_1d_html(init_type='sine', c=1.0, L=1.0, nx=100, damping=0.0):
    """生成实时一维波动方程动画 HTML。

    使用蛙跳法 (leapfrog) 有限差分推进：
        u_i^{n+1} = 2*u_i^n - u_i^{n-1}
                  + (c*dt/dx)^2 * (u_{i+1}^n - 2*u_i^n + u_{i-1}^n)
    可加入粘滞阻尼使振动衰减。稳定条件 CFL < 1。
    """
    dx = L / (nx - 1)
    cfl_val = 0.8
    dt = cfl_val * dx / max(c, 1e-9)
    steps_per_frame = max(1, int(1.0 / 60.0 / dt))

    title_name = _wave_title(init_type)
    u0_expr = _wave_init_expr(init_type, L)

    html_parts = []
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html><head><meta charset="UTF-8">')
    html_parts.append('<style>')
    html_parts.append(_common_css())
    html_parts.append('</style></head>')
    html_parts.append('<body>')
    html_parts.append('<div class="container">')
    html_parts.append('  <div class="title">一维波动方程（实时仿真）</div>')
    html_parts.append('  <div class="subtitle">'
                      '\u2202\u00b2u/\u2202t\u00b2 = c\u00b2 '
                      '\u2202\u00b2u/\u2202x\u00b2'
                      '&nbsp;&nbsp;' + title_name + '</div>')
    html_parts.append(
        '  <div class="info-bar">c = {:.3g}, &nbsp; '
        '\u03b3 = {:.3g}, &nbsp; nx = {}, '
        '&nbsp; dt = {:.6f} s</div>'.format(c, damping, nx, dt))
    html_parts.append(
        '  <div class="time-display" id="timeDisplay">时间: t = 0.00 s</div>')
    html_parts.append('  <div class="controls">')
    html_parts.append(
        '    <button class="play-btn" id="playBtn" onclick="toggleAnimation()">开始仿真</button>')
    html_parts.append(
        '    <button class="reset-btn" onclick="resetAnimation()">重置</button>')
    html_parts.append('  </div>')
    html_parts.append(
        '  <div style="text-align:center;">'
        '<span class="status stopped" id="status">状态: 已停止</span></div>')
    html_parts.append(
        '  <div class="canvas-wrapper">'
        '<canvas id="waveCanvas" width="750" height="400"></canvas></div>')
    html_parts.append('</div>')

    html_parts.append('<script>')
    html_parts.append(_wave_js_body(nx, L, c, damping, dt, steps_per_frame,
                                    u0_expr, '#2196F3', 'rgba(33,150,243,0.10)'))
    html_parts.append(_animate_loop_js('setU0'))
    html_parts.append('  initialize();')
    html_parts.append('  setU0(); simTime = 0; draw();')
    html_parts.append('</script>')
    html_parts.append('</body></html>')
    return '\n'.join(html_parts)


# ================================================================
# 二维波动方程（鼓膜）
# ================================================================
def _drum_head_u0_expr(init_type, R):
    if init_type == 'circular':
        return '(r < {:.17g}) ? Math.cos(Math.PI * r / {:.17g}) : 0.0'.format(0.3, 0.3)
    if init_type == 'sine':
        return 'Math.sin(Math.PI * x / {:.17g}) * Math.sin(Math.PI * y / {:.17g})'.format(R, R)
    if init_type == 'cross':
        return ('(Math.abs(x) < {:.17g} || Math.abs(y) < {:.17g}) '
                '? 1.0 : 0.0').format(R*0.1, R*0.1)
    if init_type == 'gaussian':
        return 'Math.exp(-(x*x + y*y) / {:.17g})'.format(R*R*0.08)
    return 'Math.exp(-(x*x + y*y) / {:.17g})'.format(R*R*0.08)


def _drum_head_title(init_type):
    return {
        'circular': '圆形凸起',
        'sine': '正弦模式',
        'cross': '十字形',
        'gaussian': '高斯凸起',
    }.get(init_type, '高斯凸起')


def _drum_js_body(n, R, c, dt, steps_per_frame, u0_expr):
    return (
        '  const R = {:.17g};\n'.format(R) +
        '  const c = {:.17g};\n'.format(c) +
        '  const N = {};\n'.format(n) +
        '  const dx = (2*R) / (N - 1);\n'
        '  const dt = {:.17g};\n'.format(dt) +
        '  const cfl = c * dt / dx;\n'
        '  const cfl2 = cfl * cfl;\n'
        '  const stepsPerFrame = {};\n'.format(steps_per_frame) +
        '\n'
        '  function makeField() { return new Float64Array(N*N); }\n'
        '  let uPrev = makeField();\n'
        '  let uCurr = makeField();\n'
        '  let uNext = makeField();\n'
        '  let isRunning = false;\n'
        '  let simTime = 0;\n'
        '  let animationId = null;\n'
        '  let u0Max = 1.0;\n'
        '  const boundary = new Uint8Array(N*N);\n'
        '  const idx = (i,j) => i*N + j;\n'
        '\n'
        '  function setInitial(f) {\n'
        '    for (let i = 0; i < N; i++) {\n'
        '      for (let j = 0; j < N; j++) {\n'
        '        const x = -R + j*dx;\n'
        '        const y = -R + i*dx;\n'
        '        const r = Math.sqrt(x*x + y*y);\n'
        '        if (boundary[idx(i,j)]) f[idx(i,j)] = 0;\n'
        '        else f[idx(i,j)] = ' + u0_expr + ';\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '\n'
        '  function initialize() {\n'
        '    for (let i = 0; i < N; i++) {\n'
        '      for (let j = 0; j < N; j++) {\n'
        '        const x = -R + j*dx;\n'
        '        const y = -R + i*dx;\n'
        '        const r = Math.sqrt(x*x + y*y);\n'
        '        boundary[idx(i,j)] = (r > R) ? 1 : 0;\n'
        '      }\n'
        '    }\n'
        '    setInitial(uCurr);\n'
        '    let m = 1e-6;\n'
        '    for (let k = 0; k < N*N; k++) {\n'
        '      if (boundary[k]) continue;\n'
        '      if (Math.abs(uCurr[k]) > m) m = Math.abs(uCurr[k]);\n'
        '    }\n'
        '    u0Max = m; if (u0Max < 1e-6) u0Max = 1.0;\n'
        '    for (let i = 1; i < N-1; i++) {\n'
        '      for (let j = 1; j < N-1; j++) {\n'
        '        const k = idx(i,j);\n'
        '        if (boundary[k]) { uNext[k] = 0; continue; }\n'
        '        const lap = (uCurr[idx(i,j+1)] + uCurr[idx(i,j-1)]'
        '                   + uCurr[idx(i+1,j)] + uCurr[idx(i-1,j)]'
        '                   - 4*uCurr[k]);\n'
        '        uNext[k] = uCurr[k] + 0.5*cfl2*lap;\n'
        '      }\n'
        '    }\n'
        '    for (let k = 0; k < N*N; k++) if (boundary[k]) uNext[k] = 0;\n'
        '    const tmp1 = uPrev; uPrev = uCurr; uCurr = uNext; uNext = tmp1;\n'
        '    simTime = dt;\n'
        '  }\n'
        '\n'
        '  function step() {\n'
        '    for (let i = 1; i < N-1; i++) {\n'
        '      for (let j = 1; j < N-1; j++) {\n'
        '        const k = idx(i,j);\n'
        '        if (boundary[k]) { uNext[k] = 0; continue; }\n'
        '        const lap = (uCurr[idx(i,j+1)] + uCurr[idx(i,j-1)]'
        '                   + uCurr[idx(i+1,j)] + uCurr[idx(i-1,j)]'
        '                   - 4*uCurr[k]);\n'
        '        uNext[k] = 2*uCurr[k] - uPrev[k] + cfl2*lap;\n'
        '      }\n'
        '    }\n'
        '    for (let k = 0; k < N*N; k++) if (boundary[k]) uNext[k] = 0;\n'
        '    const tmp = uPrev; uPrev = uCurr; uCurr = uNext; uNext = tmp;\n'
        '    simTime += dt;\n'
        '  }\n'
        '\n'
        '  function colormap(v) {\n'
        '    if (v > 1) v = 1; if (v < -1) v = -1;\n'
        '    let r,g,b;\n'
        '    if (v >= 0) { r = 255; g = Math.round(255 - v*200); b = Math.round(255 - v*230); }\n'
        '    else { r = Math.round(255 - (-v)*220); g = Math.round(255 - (-v)*180); b = 255; }\n'
        '    return "rgb(" + r + "," + g + "," + b + ")";\n'
        '  }\n'
        '\n'
        '  function draw() {\n'
        '    const canvas = document.getElementById("drumCanvas");\n'
        '    const ctx = canvas.getContext("2d");\n'
        '    const W = canvas.width, H = canvas.height;\n'
        '    ctx.fillStyle = "#ffffff"; ctx.fillRect(0,0,W,H);\n'
        '    let m = 1e-6;\n'
        '    for (let k = 0; k < N*N; k++) {\n'
        '      if (boundary[k]) continue;\n'
        '      if (Math.abs(uCurr[k]) > m) m = Math.abs(uCurr[k]);\n'
        '    }\n'
        '    const curMax = Math.max(m, u0Max*0.05);\n'
        '    const margin = 30;\n'
        '    const plotSize = Math.min(W,H) - 2*margin;\n'
        '    const cellSize = plotSize / N;\n'
        '    const offsetX = (W - plotSize)/2;\n'
        '    const offsetY = (H - plotSize)/2;\n'
        '    for (let i = 0; i < N; i++) {\n'
        '      for (let j = 0; j < N; j++) {\n'
        '        const k = idx(i,j);\n'
        '        if (boundary[k]) ctx.fillStyle = "#e8e8e8";\n'
        '        else ctx.fillStyle = colormap(uCurr[k]/curMax);\n'
        '        ctx.fillRect(offsetX + j*cellSize, offsetY + i*cellSize,'
        '                     cellSize+0.5, cellSize+0.5);\n'
        '      }\n'
        '    }\n'
        '    ctx.strokeStyle = "#333"; ctx.lineWidth = 2.5;\n'
        '    ctx.beginPath();\n'
        '    ctx.arc(offsetX + plotSize/2, offsetY + plotSize/2, plotSize/2, 0, Math.PI*2);\n'
        '    ctx.stroke();\n'
        '\n'
        '    const barX = W - 20; const barYTop = margin; const barH = plotSize;\n'
        '    const barW = 12; const nSeg = 40;\n'
        '    for (let k = 0; k < nSeg; k++) {\n'
        '      const t = 1 - k/(nSeg-1);\n'
        '      const v = 2*t - 1;\n'
        '      ctx.fillStyle = colormap(v);\n'
        '      ctx.fillRect(barX, barYTop + k*barH/nSeg, barW, barH/nSeg+1);\n'
        '    }\n'
        '    ctx.strokeStyle = "#333"; ctx.lineWidth = 1;\n'
        '    ctx.strokeRect(barX, barYTop, barW, barH);\n'
        '    ctx.font = "12px sans-serif"; ctx.fillStyle = "#333";\n'
        '    ctx.fillText("+max", barX-50, barYTop+10);\n'
        '    ctx.fillText("0", barX-20, barYTop + barH/2);\n'
        '    ctx.fillText("-max", barX-50, barYTop + barH - 5);\n'
        '    ctx.font = "12px sans-serif"; ctx.fillStyle = "#333";\n'
        '    ctx.fillText("CFL = c*dt/dx = " + cfl.toFixed(3), offsetX, offsetY-10);\n'
        '    document.getElementById("timeDisplay").textContent =\n'
        '        "时间: t = " + simTime.toFixed(3) + " s";\n'
        '  }\n'
    )


def generate_realtime_drumhead_html(init_type='gaussian', c=1.0, R=1.0, n=40):
    """生成实时二维波动方程（圆形鼓膜）动画 HTML。

    使用二维蛙跳法有限差分：
        u_{ij}^{n+1} = 2*u_{ij}^n - u_{ij}^{n-1}
                     + (c*dt/dx)^2 * (u_{i+1,j}^n + u_{i-1,j}^n
                                    + u_{i,j+1}^n + u_{i,j-1}^n
                                    - 4*u_{ij}^n)
    圆形边界固定为 0。以 RdBu 风格热力图画振幅。
    """
    dx = (2 * R) / (n - 1)
    cfl_val = 0.5
    dt = cfl_val * dx / (max(c, 1e-9) * math.sqrt(2))
    steps_per_frame = max(1, int(1.0 / 60.0 / dt))

    title_name = _drum_head_title(init_type)
    u0_expr = _drum_head_u0_expr(init_type, R)

    canvas_w = 560
    canvas_h = 560

    html_parts = []
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html><head><meta charset="UTF-8">')
    html_parts.append('<style>')
    html_parts.append(_common_css())
    html_parts.append('</style></head>')
    html_parts.append('<body>')
    html_parts.append('<div class="container">')
    html_parts.append('  <div class="title">二维波动方程（鼓膜）实时仿真</div>')
    html_parts.append('  <div class="subtitle">'
                      '\u2202\u00b2u/\u2202t\u00b2 = c\u00b2 '
                      '(\u2202\u00b2u/\u2202x\u00b2 + '
                      '\u2202\u00b2u/\u2202y\u00b2)'
                      '&nbsp;&nbsp;' + title_name + '</div>')
    html_parts.append(
        '  <div class="info-bar">c = {:.3g}, &nbsp; '
        'R = {:.3g}, &nbsp; grid {}\u00d7{}, '
        '&nbsp; dt = {:.6f} s</div>'.format(c, R, n, n, dt))
    html_parts.append(
        '  <div class="time-display" id="timeDisplay">时间: t = 0.00 s</div>')
    html_parts.append('  <div class="controls">')
    html_parts.append(
        '    <button class="play-btn" id="playBtn" onclick="toggleAnimation()">开始仿真</button>')
    html_parts.append(
        '    <button class="reset-btn" onclick="resetAnimation()">重置</button>')
    html_parts.append('  </div>')
    html_parts.append(
        '  <div style="text-align:center;">'
        '<span class="status stopped" id="status">状态: 已停止</span></div>')
    html_parts.append(
        '  <div class="canvas-wrapper">'
        '<canvas id="drumCanvas" width="{}" height="{}"></canvas></div>'.format(
            canvas_w, canvas_h))
    html_parts.append('</div>')

    html_parts.append('<script>')
    html_parts.append(_drum_js_body(n, R, c, dt, steps_per_frame, u0_expr))
    html_parts.append(_animate_loop_js('setInitial'))
    html_parts.append('  initialize();')
    html_parts.append('  setInitial(uCurr); simTime = 0; draw();')
    html_parts.append('</script>')
    html_parts.append('</body></html>')
    return '\n'.join(html_parts)


# ================================================================
# 二维热传导方程
# ================================================================
def _heat_2d_u0_expr(init_type, L):
    if init_type == 'point':
        return 'Math.exp(-100 * ((x - {0:.17g})*(x - {0:.17g}) + ' \
               '(y - {0:.17g})*(y - {0:.17g})))'.format(L/2)
    if init_type == 'line':
        return 'Math.exp(-100 * (x - {:.17g})*(x - {:.17g}))'.format(L/2, L/2)
    if init_type == 'ring':
        center = L / 2
        return ('(Math.exp(-100 * (Math.sqrt((x - {0:.17g})*(x - {0:.17g})'
                '+ (y - {0:.17g})*(y - {0:.17g})) '
                '- {1:.17g})*(Math.sqrt((x - {0:.17g})*(x - {0:.17g})'
                '+ (y - {0:.17g})*(y - {0:.17g})) '
                '- {1:.17g})))').format(center, 0.3)
    return 'Math.exp(-100 * ((x - {0:.17g})*(x - {0:.17g}) + ' \
           '(y - {0:.17g})*(y - {0:.17g})))'.format(L/2)


def _heat_2d_title(init_type):
    return {
        'point': '点热源',
        'line': '线热源',
        'ring': '环形热源',
    }.get(init_type, '点热源')


def _heat_2d_js_body(n, L, alpha, dt, steps_per_frame, u0_expr):
    return (
        '  const L = {:.17g};\n'.format(L) +
        '  const alpha = {:.17g};\n'.format(alpha) +
        '  const N = {};\n'.format(n) +
        '  const dx = L / (N - 1);\n' +
        '  const dt = {:.17g};\n'.format(dt) +
        '  const r = alpha * dt / (dx * dx);\n'
        '  const r2 = r / 2.0;\n'
        '  const stepsPerFrame = {};\n'.format(steps_per_frame) +
        '\n'
        '  function makeField() { return new Float64Array(N*N); }\n'
        '  let uCurr = makeField();\n'
        '  let uNext = makeField();\n'
        '  let isRunning = false;\n'
        '  let simTime = 0;\n'
        '  let animationId = null;\n'
        '  let u0Max = 1.0;\n'
        '  const idx = (i,j) => i*N + j;\n'
        '\n'
        '  function setInitial(f) {\n'
        '    for (let i = 0; i < N; i++) {\n'
        '      for (let j = 0; j < N; j++) {\n'
        '        const x = j*dx;\n'
        '        const y = (N-1-i)*dx;\n'
        '        f[idx(i,j)] = ' + u0_expr + ';\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '\n'
        '  function initialize() {\n'
        '    setInitial(uCurr);\n'
        '    let m = 1e-6;\n'
        '    for (let k = 0; k < N*N; k++)\n'
        '      if (Math.abs(uCurr[k]) > m) m = Math.abs(uCurr[k]);\n'
        '    u0Max = m;\n'
        '    if (u0Max < 1e-6) u0Max = 1.0;\n'
        '    simTime = 0;\n'
        '  }\n'
        '\n'
        '  function step() {\n'
        '    for (let i = 1; i < N-1; i++) {\n'
        '      for (let j = 1; j < N-1; j++) {\n'
        '        const k = idx(i,j);\n'
        '        const lap = (uCurr[idx(i,j+1)] + uCurr[idx(i,j-1)]'
        '                   + uCurr[idx(i+1,j)] + uCurr[idx(i-1,j)]'
        '                   - 4*uCurr[k]);\n'
        '        uNext[k] = uCurr[k] + r2*lap;\n'
        '      }\n'
        '    }\n'
        '    for (let i = 0; i < N; i++) { uNext[idx(0,i)] = 0; uNext[idx(N-1,i)] = 0; uNext[idx(i,0)] = 0; uNext[idx(i,N-1)] = 0; }\n'
        '    const tmp = uCurr; uCurr = uNext; uNext = tmp;\n'
        '    simTime += dt;\n'
        '  }\n'
        '\n'
        '  function colormap(v, vmax) {\n'
        '    if (vmax < 1e-9) vmax = 1e-9;\n'
        '    let t = v / vmax;\n'
        '    if (t < 0) t = 0; if (t > 1) t = 1;\n'
        '    let r,g,b;\n'
        '    if (t < 0.17) { let s = t/0.17; r = Math.round(s*50); g = Math.round(s*50); b = Math.round(80 + s*120); }\n'
        '    else if (t < 0.33) { let s = (t-0.17)/0.16; r = 50; g = Math.round(50 + s*100); b = Math.round(200 - s*50); }\n'
        '    else if (t < 0.5) { let s = (t-0.33)/0.17; r = Math.round(50 + s*50); g = Math.round(150 + s*50); b = Math.round(150 - s*150); }\n'
        '    else if (t < 0.67) { let s = (t-0.5)/0.17; r = Math.round(100 + s*100); g = 200; b = 0; }\n'
        '    else if (t < 0.83) { let s = (t-0.67)/0.16; r = Math.round(200 + s*50); g = Math.round(200 - s*80); b = 0; }\n'
        '    else { r = 255; g = Math.round(120 - (t-0.83)/0.17*120); b = 0; }\n'
        '    return "rgb(" + r + "," + g + "," + b + ")";\n'
        '  }\n'
        '\n'
        '  function draw() {\n'
        '    const canvas = document.getElementById("heat2dCanvas");\n'
        '    const ctx = canvas.getContext("2d");\n'
        '    const W = canvas.width, H = canvas.height;\n'
        '    ctx.fillStyle = "#ffffff"; ctx.fillRect(0,0,W,H);\n'
        '    let curMax = 1e-9;\n'
        '    for (let k = 0; k < N*N; k++)\n'
        '      if (uCurr[k] > curMax) curMax = uCurr[k];\n'
        '    const displayMax = Math.max(curMax * 1.2, u0Max*0.02);\n'
        '    const margin = 40;\n'
        '    const plotSize = Math.min(W,H) - 2*margin;\n'
        '    const cellSize = plotSize / N;\n'
        '    const offsetX = (W - plotSize)/2;\n'
        '    const offsetY = (H - plotSize)/2;\n'
        '    for (let i = 0; i < N; i++) {\n'
        '      for (let j = 0; j < N; j++) {\n'
        '        const k = idx(i,j);\n'
        '        ctx.fillStyle = colormap(uCurr[k], displayMax);\n'
        '        ctx.fillRect(offsetX + j*cellSize, offsetY + i*cellSize,\n'
        '                     cellSize+0.5, cellSize+0.5);\n'
        '      }\n'
        '    }\n'
        '    ctx.strokeStyle = "#333"; ctx.lineWidth = 2;\n'
        '    ctx.strokeRect(offsetX, offsetY, plotSize, plotSize);\n'
        '    const barX = offsetX + plotSize + 30; const barYTop = offsetY;\n'
        '    const barW = 18; const barH = plotSize; const nSeg = 50;\n'
        '    for (let k = 0; k < nSeg; k++) {\n'
        '      const t = 1 - k/(nSeg-1);\n'
        '      ctx.fillStyle = colormap(t*displayMax, displayMax);\n'
        '      ctx.fillRect(barX, barYTop + k*barH/nSeg, barW, barH/nSeg+1);\n'
        '    }\n'
        '    ctx.strokeStyle = "#333"; ctx.lineWidth = 1;\n'
        '    ctx.strokeRect(barX, barYTop, barW, barH);\n'
        '    ctx.font = "12px sans-serif"; ctx.fillStyle = "#333";\n'
        '    ctx.fillText("u_max", barX - 70, barYTop + 12);\n'
        '    ctx.fillText(displayMax.toExponential(2), barX - 70, barYTop + 25);\n'
        '    ctx.fillText("0", barX - 30, barYTop + barH - 5);\n'
        '    ctx.font = "12px sans-serif"; ctx.fillStyle = "#333";\n'
        '    ctx.fillText("r = alpha*dt/dx^2 = " + r2.toFixed(3), offsetX, offsetY - 10);\n'
        '    document.getElementById("timeDisplay").textContent =\n'
        '        "时间: t = " + simTime.toFixed(3) + " s";\n'
        '  }\n'
    )


def generate_realtime_heat_2d_html(init_type='point', alpha=0.01, L=1.0, n=50):
    """生成实时二维热传导方程动画 HTML。

    在方形域 [0,1]x[0,1] 上解二维热传导方程
        ∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)
    四边固定为 0（冷边界）。

    使用 Forward Euler 显式有限差分推进：
        u_{ij}^{n+1} = u_{ij}^n + r2 * (∇²u)_{ij}
    其中 r2 = α*dt/dx²，稳定条件 r2 < 0.25（2D）。
    """
    dx = L / (n - 1)
    r_stable = 0.2
    dt = r_stable * dx * dx / alpha
    steps_per_frame = max(1, int(1.0 / 30.0 / dt))

    title_name = _heat_2d_title(init_type)
    u0_expr = _heat_2d_u0_expr(init_type, L)

    canvas_w = 560
    canvas_h = 560

    html_parts = []
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html><head><meta charset="UTF-8">')
    html_parts.append('<style>')
    html_parts.append(_common_css())
    html_parts.append('</style></head>')
    html_parts.append('<body>')
    html_parts.append('<div class="container">')
    html_parts.append('  <div class="title">二维热传导方程（实时仿真）</div>')
    html_parts.append('  <div class="subtitle">'
                      '\u2202u/\u2202t = \u03b1 '
                      '(\u2202\u00b2u/\u2202x\u00b2 + '
                      '\u2202\u00b2u/\u2202y\u00b2)'
                      '&nbsp;&nbsp;' + title_name + '</div>')
    html_parts.append(
        '  <div class="info-bar">\u03b1 = {:.3g}, '
        '&nbsp; grid {}\u00d7{}, '
        '&nbsp; dt = {:.6f} s</div>'.format(alpha, n, n, dt))
    html_parts.append(
        '  <div class="time-display" id="timeDisplay">时间: t = 0.00 s</div>')
    html_parts.append('  <div class="controls">')
    html_parts.append(
        '    <button class="play-btn" id="playBtn" onclick="toggleAnimation()">开始仿真</button>')
    html_parts.append(
        '    <button class="reset-btn" onclick="resetAnimation()">重置</button>')
    html_parts.append('  </div>')
    html_parts.append(
        '  <div style="text-align:center;">'
        '<span class="status stopped" id="status">状态: 已停止</span></div>')
    html_parts.append(
        '  <div class="canvas-wrapper">'
        '<canvas id="heat2dCanvas" width="{}" height="{}"></canvas></div>'.format(
            canvas_w, canvas_h))
    html_parts.append('</div>')

    html_parts.append('<script>')
    html_parts.append(_heat_2d_js_body(n, L, alpha, dt, steps_per_frame, u0_expr))
    html_parts.append(_animate_loop_js('setInitial'))
    html_parts.append('  initialize();')
    html_parts.append('  setInitial(uCurr); simTime = 0; draw();')
    html_parts.append('</script>')
    html_parts.append('</body></html>')
    return '\n'.join(html_parts)
