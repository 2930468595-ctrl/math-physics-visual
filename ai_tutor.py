# -*- coding: utf-8 -*-
"""
AI 家教模块 (ai_tutor.py)
====================================================

本模块封装数学物理方法课程的 AI 教学功能，包括：

1. 课程大纲与章节数据结构
2. 教学流程状态机 (引导提问 -> 讲解 -> 出题 -> 批改 -> 纠错 -> 总结)
3. DeepSeek API 调用 (OpenAI 兼容格式)
4. Streamlit 界面集成

作者：项目自动生成
"""

# ===================== 标准库 =====================
import os
import json
import time
from typing import Dict, List, Optional, Any


# ===================== 第三方库 =====================
try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        pass


# ===================== 多平台 LLM 服务配置 =====================
# 用户可以在界面上选择不同的服务提供商。
# 每个平台都兼容 OpenAI 的 chat.completions 接口，只需切换 base_url 和模型名即可。
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "env_key": "DEEPSEEK_API_KEY",
        "description": "DeepSeek 官方 API，推理速度快，教学效果佳",
        "free_note": "新注册用户有免费额度，额度用完需充值",
    },
    "SiliconFlow (硅基流动)": {
        "base_url": "https://api.siliconflow.cn/v1",
        "models": [
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
            "THUDM/glm-4-9b-chat",
        ],
        "env_key": "SILICONFLOW_API_KEY",
        "description": "国内服务，每日免费 tokens，中文模型丰富",
        "free_note": "每日免费 tokens，注册即可获取",
    },
    "智谱 AI (GLM)": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4-flash", "glm-4", "glm-4-plus", "glm-4-air"],
        "env_key": "ZHIPU_API_KEY",
        "description": "智谱官方，国内可访问，支持 GLM 系列模型",
        "free_note": "新用户赠送免费额度",
    },
    "阿里云百炼 (Qwen)": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": [
            "qwen-plus",
            "qwen-turbo",
            "qwen-max",
            "qwen2.5-72b-instruct",
            "qwen2.5-32b-instruct",
        ],
        "env_key": "DASHSCOPE_API_KEY",
        "description": "阿里云官方，Qwen 系列模型，国内速度快",
        "free_note": "新用户有免费额度",
    },
    "自定义 (任意 OpenAI 兼容)": {
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-chat"],
        "env_key": "CUSTOM_API_KEY",
        "description": "自行填入 base_url、模型名、API Key",
        "free_note": "完全自定义",
    },
}


# ==================================================================
# 1. 课程大纲数据
# ==================================================================

# 完整课程大纲 —— 包含复变函数 + 数学物理方程两大模块
COURSE_OUTLINE: List[Dict[str, Any]] = [
    # -------- 模块一：复变函数论 (第 1-5 章) --------
    {
        "chapter_id": 1,
        "title": "复数与复变函数",
        "module": "复变函数论",
        "sections": [
            {"section_id": "1.1", "title": "复数的定义与表示",
             "key_points": ["复数定义 z = x + iy", "实部与虚部", "共轭复数"]},
            {"section_id": "1.2", "title": "复数的代数运算与几何意义",
             "key_points": ["四则运算", "三角形式与指数形式", "乘积与商的几何意义"]},
            {"section_id": "1.3", "title": "复平面与区域",
             "key_points": ["邻域、开集、闭集", "单连通与多连通区域", "无穷远点"]},
            {"section_id": "1.4", "title": "复变函数的概念",
             "key_points": ["映射与变换", "极限与连续性"]},
        ],
    },
    {
        "chapter_id": 2,
        "title": "解析函数",
        "module": "复变函数论",
        "sections": [
            {"section_id": "2.1", "title": "导数与解析性",
             "key_points": ["复变函数的导数", "解析函数的定义"]},
            {"section_id": "2.2", "title": "柯西-黎曼条件",
             "key_points": ["C-R 方程推导", "解析性判定定理"]},
            {"section_id": "2.3", "title": "初等解析函数",
             "key_points": ["指数函数 e^z", "三角函数 sin z, cos z", "对数函数 ln z"]},
            {"section_id": "2.4", "title": "调和函数与共轭调和函数",
             "key_points": ["调和函数定义", "与解析函数的关系"]},
        ],
    },
    {
        "chapter_id": 3,
        "title": "复积分",
        "module": "复变函数论",
        "sections": [
            {"section_id": "3.1", "title": "复变函数积分的定义与性质",
             "key_points": ["曲线积分定义", "基本性质"]},
            {"section_id": "3.2", "title": "柯西积分定理",
             "key_points": ["单连通区域柯西定理", "多连通区域推广"]},
            {"section_id": "3.3", "title": "柯西积分公式",
             "key_points": ["基本公式", "解析函数值由边界值确定"]},
            {"section_id": "3.4", "title": "高阶导数公式",
             "key_points": ["解析函数的各阶导数存在", "柯西不等式", "刘维尔定理"]},
        ],
    },
    {
        "chapter_id": 4,
        "title": "级数展开",
        "module": "复变函数论",
        "sections": [
            {"section_id": "4.1", "title": "复级数的收敛性",
             "key_points": ["收敛、绝对收敛、一致收敛", "幂级数的收敛半径"]},
            {"section_id": "4.2", "title": "泰勒级数",
             "key_points": ["泰勒展开定理", "常见函数的泰勒级数"]},
            {"section_id": "4.3", "title": "洛朗级数",
             "key_points": ["双边级数展开", "圆环域内的解析函数展开"]},
            {"section_id": "4.4", "title": "孤立奇点的分类",
             "key_points": ["可去奇点", "极点", "本性奇点"]},
        ],
    },
    {
        "chapter_id": 5,
        "title": "留数定理",
        "module": "复变函数论",
        "sections": [
            {"section_id": "5.1", "title": "留数的定义与计算",
             "key_points": ["留数定义", "极点处留数公式"]},
            {"section_id": "5.2", "title": "留数基本定理",
             "key_points": ["定理内容", "围道积分转化为留数和"]},
            {"section_id": "5.3", "title": "用留数计算实积分",
             "key_points": ["三角函数积分", "无穷积分", "多值函数积分"]},
        ],
    },

    # -------- 模块二：数学物理方程 (第 6-10 章) --------
    {
        "chapter_id": 6,
        "title": "数学物理方程的建立与分类",
        "module": "数学物理方程",
        "sections": [
            {"section_id": "6.1", "title": "典型方程的建立",
             "key_points": ["弦振动方程 (波动方程)", "热传导方程", "拉普拉斯方程"]},
            {"section_id": "6.2", "title": "定解条件",
             "key_points": ["初始条件", "第一、二、三类边界条件"]},
            {"section_id": "6.3", "title": "二阶线性偏微分方程的分类",
             "key_points": ["椭圆型、抛物型、双曲型", "特征线与标准形式"]},
        ],
    },
    {
        "chapter_id": 7,
        "title": "分离变量法",
        "module": "数学物理方程",
        "sections": [
            {"section_id": "7.1", "title": "直角坐标系下的分离变量法",
             "key_points": ["一维波动方程", "一维热传导方程", "二维拉普拉斯方程"]},
            {"section_id": "7.2", "title": "圆域与极坐标系下的分离变量法",
             "key_points": ["极坐标下的拉普拉斯方程", "傅里叶-贝塞尔级数"]},
            {"section_id": "7.3", "title": "非齐次方程与非齐次边界条件",
             "key_points": ["齐次化原理 (Duhamel 原理)", "边界条件齐次化"]},
            {"section_id": "7.4", "title": "本征值问题",
             "key_points": ["斯特姆-刘维尔型问题", "本征函数的正交性"]},
        ],
    },
    {
        "chapter_id": 8,
        "title": "特殊函数",
        "module": "数学物理方程",
        "sections": [
            {"section_id": "8.1", "title": "贝塞尔函数",
             "key_points": ["贝塞尔方程", "第一类贝塞尔函数 J_n(x)", "递推公式"]},
            {"section_id": "8.2", "title": "勒让德多项式",
             "key_points": ["勒让德方程", "勒让德多项式 P_n(x)", "生成函数"]},
            {"section_id": "8.3", "title": "连带勒让德函数与球谐函数",
             "key_points": ["连带勒让德函数 P_l^m(x)", "球谐函数 Y_l^m"]},
            {"section_id": "8.4", "title": "其他特殊函数",
             "key_points": ["厄米多项式", "拉盖尔多项式", "切比雪夫多项式"]},
        ],
    },
    {
        "chapter_id": 9,
        "title": "积分变换法",
        "module": "数学物理方程",
        "sections": [
            {"section_id": "9.1", "title": "傅里叶积分与傅里叶变换",
             "key_points": ["傅里叶积分定理", "傅里叶变换性质"]},
            {"section_id": "9.2", "title": "傅里叶变换求解偏微分方程",
             "key_points": ["无界杆热传导方程", "无界弦波动方程"]},
            {"section_id": "9.3", "title": "拉普拉斯变换",
             "key_points": ["定义与基本性质", "逆变换 (卷积定理)"]},
            {"section_id": "9.4", "title": "拉普拉斯变换求解偏微分方程",
             "key_points": ["初值问题", "含时间变量的方程"]},
        ],
    },
    {
        "chapter_id": 10,
        "title": "格林函数法与定解问题",
        "module": "数学物理方程",
        "sections": [
            {"section_id": "10.1", "title": "δ 函数与基本解",
             "key_points": ["δ 函数概念与性质", "拉普拉斯方程的基本解"]},
            {"section_id": "10.2", "title": "格林公式",
             "key_points": ["格林第一、第二公式", "格林第三公式"]},
            {"section_id": "10.3", "title": "格林函数",
             "key_points": ["格林函数定义", "镜像法构造格林函数"]},
            {"section_id": "10.4", "title": "格林函数求解边值问题",
             "key_points": ["第一边值问题", "狄利克雷问题"]},
        ],
    },
]


def get_chapter_titles() -> List[str]:
    """获取所有章节标题列表，用于下拉菜单。"""
    return [f"第{ch['chapter_id']}章：{ch['title']}" for ch in COURSE_OUTLINE]


def get_chapter(chapter_idx: int) -> Optional[Dict[str, Any]]:
    """根据索引获取章节内容。"""
    if 0 <= chapter_idx < len(COURSE_OUTLINE):
        return COURSE_OUTLINE[chapter_idx]
    return None


# ==================================================================
# 2. LLM API 封装（通用多平台 OpenAI 兼容客户端）
# ==================================================================


class LLMClient:
    """通用 LLM API 客户端 —— 支持 DeepSeek、SiliconFlow、智谱、百炼、自定义等多平台。"""

    DEFAULTS = {
        "provider": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "temperature": 0.7,
        "max_tokens": 2000,
    }

    def __init__(self, api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 provider: Optional[str] = None):
        """
        参数
        ----
        api_key  : API 密钥；若为 None，则根据 provider 从对应环境变量读取
        base_url : API 基础地址；若为 None 则根据 provider 使用默认值
        model    : 使用的模型名；若为 None 则根据 provider 使用默认值
        provider : 服务提供商名称（PROVIDERS 字典的 key）
        """
        # 1) 加载 .env（优先尝试 python-dotenv）
        try:
            load_dotenv()
        except Exception:
            pass

        # 2) 兜底：手动解析当前目录/脚本目录的 .env（兼容 BOM，不依赖 python-dotenv）
        #    会解析多个可能的 env key（DEEPSEEK_API_KEY、SILICONFLOW_API_KEY 等）
        known_env_keys = [p["env_key"] for p in PROVIDERS.values()]
        already_has_key = any(os.environ.get(k, "") for k in known_env_keys)
        if not already_has_key:
            env_candidates = [os.path.join(os.getcwd(), ".env")]
            try:
                env_candidates.append(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
                )
            except NameError:
                pass
            for env_path in env_candidates:
                if os.path.isfile(env_path):
                    try:
                        with open(env_path, "r", encoding="utf-8-sig") as f:
                            for line in f:
                                line = line.strip()
                                if not line or line.startswith("#") or "=" not in line:
                                    continue
                                k, v = line.split("=", 1)
                                k = k.strip().lstrip("\ufeff")
                                v = v.strip().strip('"').strip("'")
                                if k and v and k not in os.environ:
                                    os.environ[k] = v
                        break
                    except Exception:
                        continue

        # 3) 根据 provider 确定 base_url / model / env_key 的默认值
        provider = provider or os.environ.get("LLM_PROVIDER", self.DEFAULTS["provider"])
        if provider not in PROVIDERS:
            provider = self.DEFAULTS["provider"]
        self.provider = provider
        provider_info = PROVIDERS[provider]
        env_key = provider_info["env_key"]

        # 4) 确定最终的 api_key / base_url / model
        self.api_key = api_key or os.environ.get(env_key, "")
        self.base_url = base_url or os.environ.get(
            f"{provider_info['env_key'].replace('API_KEY', 'BASE_URL')}",
            provider_info["base_url"],
        )
        self.model = model or os.environ.get(
            f"{provider_info['env_key'].replace('API_KEY', 'MODEL')}",
            provider_info["models"][0],
        )
        self.temperature = float(
            os.environ.get("LLM_TEMPERATURE", self.DEFAULTS["temperature"])
        )
        self.max_tokens = int(
            os.environ.get("LLM_MAX_TOKENS", self.DEFAULTS["max_tokens"])
        )

        # 5) 初始化 openai 客户端
        self._client = None
        if OpenAI is not None and self.api_key:
            try:
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception:
                self._client = None

    @property
    def is_ready(self) -> bool:
        """是否可以进行 API 调用（密钥已配置且库已安装）。"""
        return self._client is not None

    def update(self, api_key: Optional[str] = None,
               base_url: Optional[str] = None,
               model: Optional[str] = None,
               provider: Optional[str] = None):
        """允许在运行时动态更新配置并重新初始化客户端。"""
        if provider is not None:
            self.provider = provider
        if api_key is not None:
            self.api_key = api_key
        if base_url is not None:
            self.base_url = base_url
        if model is not None:
            self.model = model

        self._client = None
        if OpenAI is not None and self.api_key:
            try:
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception:
                self._client = None

    def chat(self, messages: List[Dict[str, str]],
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             timeout: int = 60) -> Dict[str, Any]:
        """
        向 LLM 发起对话请求。

        参数
        ----
        messages    : OpenAI 格式的消息列表 [{"role":"system"/"user"/"assistant","content":...}]
        temperature : 采样温度 (0-2)
        max_tokens  : 最大生成 tokens

        返回
        ----
        dict : { 'success': bool, 'content': str, 'error': str|None }
        """
        if not self.is_ready:
            return {
                "success": False,
                "content": "",
                "error": "API 未就绪：请在上方配置面板输入 API Key 并点击「测试连接」确认。",
            }

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                timeout=timeout,
            )
            content = response.choices[0].message.content or ""
            return {"success": True, "content": content, "error": None}
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            # 根据错误码给出更友好的中文提示
            if "401" in msg or "Unauthorized" in msg:
                friendly = (
                    f"🔒 API 密钥无效或已过期（错误码 401）。\n\n"
                    f"当前平台：**{self.provider}**，请确认 API Key 是否正确。"
                )
            elif "402" in msg or "Insufficient Balance" in msg:
                friendly = (
                    f"💸 账户余额不足（错误码 402）。\n\n"
                    f"当前平台「{self.provider}」的额度已用完。您可以：\n\n"
                    f"1. 在下方「服务提供商」切换到其他免费平台（如 SiliconFlow 每日免费 tokens）\n"
                    f"2. 充值当前平台\n"
                    f"3. 更换新的 API Key"
                )
            elif "403" in msg or "Forbidden" in msg:
                friendly = f"🚫 访问被拒绝（错误码 403）。\n\n请检查「{self.provider}」的 API Key 是否正确、是否被封禁。"
            elif "404" in msg:
                friendly = f"❌ 模型不存在或地址错误（错误码 404）。\n\n平台「{self.provider}」不支持模型「{self.model}」，请切换其他模型。"
            elif "429" in msg or "rate limit" in msg.lower():
                friendly = f"⏳ 调用频率超限（错误码 429）。\n\n请稍后再试。"
            elif "timeout" in msg.lower():
                friendly = f"⏱️ 请求超时。\n\n请检查网络连接，或稍后再试。"
            else:
                friendly = f"❌ API 调用失败：\n\n```\n{msg}\n```"
            return {"success": False, "content": "", "error": friendly}


# 兼容：DeepSeekClient 作为 LLMClient 的别名（默认 provider=DeepSeek）
class DeepSeekClient(LLMClient):
    """兼容旧调用 —— 等价于 LLMClient(provider='DeepSeek')。"""

    def __init__(self, api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url, model=model, provider="DeepSeek")


# ==================================================================
# 3. 教学流程系统提示词
# ==================================================================

SYSTEM_PROMPT_TEMPLATE = """你是一位专业的数学物理方法助教老师。当前正在讲授：
课程：《数学物理方法》
章节：{chapter_title} ({section_id})
本节重点：{key_points}

【核心教学风格与要求】
1. 将内容拆分为若干小部分，每讲解完一小部分后设置"互动检查点"。
2. 在检查点中通过苏格拉底式提问确认学生是否理解，不要直接给出答案。
   - 先问引导性问题，让学生自己思考
   - 学生答错时，指出思考方向而非直接讲答案
3. 公式推导必须一步一步展示，不跳过关键步骤。
   - 每步之间留出互动检查，询问学生是否理解当前步骤
4. 语言通俗易懂，结合直观物理或几何意义解释。
5. 数学公式使用 LaTeX 格式 (用 $...$ 或 $$...$$)。
6. 用中文教学，语气友善、耐心。
7. 当学生提出疑问时：
   - 先用反问启发思考
   - 根据回答逐步引导
   - 最终确认学生理解后返回主教学流程
8. 每完成一个章节，生成"知识小结"和"阶段测验"(3-5个题目)。

【当前章节上下文】
{section_context}

现在，从本节第1个要点开始教学。"""


def build_system_prompt(chapter: Dict[str, Any], section: Dict[str, Any]) -> str:
    """为当前章节构建系统提示词。"""
    section_context = (
        f"章节：第{chapter['chapter_id']}章《{chapter['title']}》\n"
        f"所属模块：{chapter['module']}\n"
        f"本章包含 {len(chapter['sections'])} 个小节"
    )
    key_points_str = "、".join(section["key_points"])
    return SYSTEM_PROMPT_TEMPLATE.format(
        chapter_title=chapter["title"],
        section_id=section["section_id"],
        key_points=key_points_str,
        section_context=section_context,
    )


# ==================================================================
# 4. 教学流程状态机
# ==================================================================

class TutorSession:
    """AI 教学会话 —— 管理教学进度、对话历史和用户交互。

    教学流程：
        [选择章节] -> [开始讲解] -> [互动检查点/提问] -> [小节完成]
        -> [继续下一小节] -> ... -> [章节完成 -> 小结+测验]
    """

    STATE_START = "start"
    STATE_TEACHING = "teaching"     # 正在讲解要点
    STATE_CHECKPOINT = "checkpoint"  # 互动检查点 (等待学生回答)
    STATE_QUIZ = "quiz"             # 章节末尾测验
    STATE_FINISHED = "finished"     # 章节结束

    def __init__(self):
        # 教学位置
        self.chapter_idx: Optional[int] = None
        self.section_idx: int = 0
        self.key_point_idx: int = 0

        # 对话状态
        self.state: str = self.STATE_START
        self.messages: List[Dict[str, str]] = []  # 传给 AI 的消息列表
        self.conversation_log: List[Dict[str, str]] = []  # 显示给用户的对话

        # 进度信息
        self.current_progress: str = "尚未开始"

    # -------------------- 会话控制 --------------------
    def start_chapter(self, chapter_idx: int, client: DeepSeekClient) -> Dict[str, Any]:
        """开始教学某一章节，返回 AI 的开场讲解。"""
        self.chapter_idx = chapter_idx
        self.section_idx = 0
        self.key_point_idx = 0
        self.state = self.STATE_TEACHING
        self.messages = []
        self.conversation_log = []

        chapter = get_chapter(chapter_idx)
        if chapter is None:
            return {"success": False, "reply": "无效的章节编号。"}

        # 从本章第 1 节第 1 个要点开始
        section = chapter["sections"][0]
        system_prompt = build_system_prompt(chapter, section)
        self.messages = [{"role": "system", "content": system_prompt}]
        self.current_progress = f"{section['section_id']} {section['title']}"

        # 加入开场提示
        opening = (
            f"我们现在开始学习：第{chapter['chapter_id']}章《{chapter['title']}》。\n"
            f"本章包含 {len(chapter['sections'])} 个小节：\n"
            + "\n".join([f"  - {s['section_id']} {s['title']}" for s in chapter["sections"]])
            + f"\n\n我们从 {section['section_id']} {section['title']} 开始。"
              f"请你先讲一讲这一节的核心概念，并用苏格拉底式提问检查我是否理解。"
        )
        self.messages.append({"role": "user", "content": opening})
        self.conversation_log.append({"role": "ai", "content": opening, "type": "intro"})

        # 调用 AI
        result = client.chat(self.messages)
        if result["success"]:
            self.messages.append({"role": "assistant", "content": result["content"]})
            self.conversation_log.append({"role": "ai", "content": result["content"], "type": "teach"})
            self.state = self.STATE_CHECKPOINT
            return {"success": True, "reply": result["content"]}

        return {"success": False, "reply": result["error"]}

    # -------------------- 用户消息处理 --------------------
    def send_user_message(self, user_text: str, client: DeepSeekClient) -> Dict[str, Any]:
        """用户回复 (回答检查点问题 / 提问)。"""
        if self.chapter_idx is None:
            return {"success": False, "reply": "请先选择要学习的章节。"}

        if not user_text.strip():
            return {"success": False, "reply": "请输入您的回答或问题。"}

        self.messages.append({"role": "user", "content": user_text})
        self.conversation_log.append({"role": "user", "content": user_text, "type": "reply"})

        result = client.chat(self.messages)
        if result["success"]:
            self.messages.append({"role": "assistant", "content": result["content"]})
            self.conversation_log.append({"role": "ai", "content": result["content"], "type": "reply"})
            self.state = self.STATE_CHECKPOINT
            return {"success": True, "reply": result["content"]}

        return {"success": False, "reply": result["error"]}

    # -------------------- 进入下一小节 --------------------
    def next_section(self, client: DeepSeekClient) -> Dict[str, Any]:
        """确认学生掌握当前小节后，推进到下一小节。"""
        if self.chapter_idx is None:
            return {"success": False, "reply": "请先选择章节。"}

        chapter = get_chapter(self.chapter_idx)
        self.section_idx += 1

        # 本章所有小节已完成 → 进入章节小结与测验
        if self.section_idx >= len(chapter["sections"]):
            return self._finish_chapter(client)

        # 切换到下一小节，重新设置系统提示词
        section = chapter["sections"][self.section_idx]
        self.current_progress = f"{section['section_id']} {section['title']}"

        transition = (
            f"很好！现在进入下一小节：{section['section_id']} {section['title']}。\n"
            f"本节要点：{''.join(section['key_points'])}。\n"
            f"请继续按照苏格拉底风格教学。"
        )
        self.messages.append({"role": "user", "content": transition})
        self.conversation_log.append({"role": "ai", "content": transition, "type": "intro"})

        result = client.chat(self.messages)
        if result["success"]:
            self.messages.append({"role": "assistant", "content": result["content"]})
            self.conversation_log.append({"role": "ai", "content": result["content"], "type": "teach"})
            return {"success": True, "reply": result["content"]}

        return {"success": False, "reply": result["error"]}

    # -------------------- 完成章节：生成小结+测验 --------------------
    def _finish_chapter(self, client: DeepSeekClient) -> Dict[str, Any]:
        chapter = get_chapter(self.chapter_idx)
        summary_req = (
            f"本章《{chapter['title']}》教学已完成。\n"
            f"请您：\n"
            f"1) 生成 300 字以内的知识小结，包含本章核心概念、关键公式、典型解题思路\n"
            f"2) 生成 3-5 道阶段测验题 (包含选择题、计算题、证明题)\n"
            f"3) 每道题目给出参考答案\n"
            f"格式要求：先用【本章小结】开头，再用【阶段测验】列出题目，最后用【参考答案】给出答案。"
        )
        self.messages.append({"role": "user", "content": summary_req})
        self.conversation_log.append({"role": "ai", "content": summary_req, "type": "summary"})

        result = client.chat(self.messages, max_tokens=2500)
        if result["success"]:
            self.messages.append({"role": "assistant", "content": result["content"]})
            self.conversation_log.append({"role": "ai", "content": result["content"], "type": "summary"})
            self.state = self.STATE_FINISHED
            self.current_progress = f"第{chapter['chapter_id']}章 已完成 [OK]"
            return {"success": True, "reply": result["content"]}
        return {"success": False, "reply": result["error"]}

    # -------------------- 重置会话 --------------------
    def reset(self):
        self.chapter_idx = None
        self.section_idx = 0
        self.key_point_idx = 0
        self.state = self.STATE_START
        self.messages = []
        self.conversation_log = []
        self.current_progress = "尚未开始"


# ==================================================================
# 5. Streamlit 界面集成
# ==================================================================

class StreamlitAITutorUI:
    """将 AI 家教界面嵌入 Streamlit。"""

    _STATE_KEY = "ai_tutor_session"
    _CLIENT_KEY = "ai_tutor_client"

    def __init__(self):
        if st is None:
            raise RuntimeError("需要安装 streamlit 才能使用 StreamlitAITutorUI")
        self._init_session()

    # -------------------- 初始化 --------------------
    def _init_session(self):
        if self._STATE_KEY not in st.session_state:
            st.session_state[self._STATE_KEY] = TutorSession()
        if self._CLIENT_KEY not in st.session_state:
            # 使用通用 LLMClient；启动时尝试从 .env 读取任意平台的 key
            st.session_state[self._CLIENT_KEY] = LLMClient()

    @property
    def session(self) -> TutorSession:
        return st.session_state[self._STATE_KEY]

    @property
    def client(self) -> LLMClient:
        return st.session_state[self._CLIENT_KEY]

    # -------------------- 主渲染入口 --------------------
    def render_in_sidebar(self):
        """在侧边栏中折叠显示的简化入口（作为快捷入口）。"""
        with st.sidebar:
            with st.expander("📖 AI 助教 · 快捷入口", expanded=False):
                st.markdown(
                    "在上方「选择模块」下拉菜单中选择 **「AI 助教」**，即可进入完整教学界面。\n\n"
                    f"当前 API 状态：{'**✅ 已就绪**' if self.client.is_ready else '**⚠️ 未就绪**'}"
                )

    def render_in_main(self):
        """主教学界面：顶部 API 配置，中间课程选择，下方大对话区域。"""
        st.title("📖 数学物理方法课程 · AI 助教")
        st.markdown(
            "苏格拉底式教学 | 十大章节 | 随问随答 | 章节小结与阶段测验 | 多平台 LLM 自由切换"
        )
        st.markdown("---")

        # =============== 1) 顶部：API 配置面板 ===============
        self._render_api_config_panel()

        st.markdown("---")

        # =============== 2) 中间：课程选择与对话 ===============
        titles = get_chapter_titles()
        if not self.client.is_ready:
            st.warning(
                "⚠️ API 尚未就绪，无法开始教学。\n\n"
                "请在上方配置面板：\n\n"
                "1. 选择您想使用的服务提供商\n"
                "2. 输入对应的 API Key（或在 .env 文件中配置）\n"
                "3. 点击「测试连接」确认可用后再开始学习。"
            )
        else:
            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.subheader("📚 课程设置")

                chapter_idx = st.selectbox(
                    "选择要学习的章节",
                    range(len(titles)),
                    format_func=lambda i: titles[i],
                    key="ai_tutor_chapter_select",
                )

                st.markdown(f"**当前进度**：{self.session.current_progress}")

                c1, c2 = st.columns(2)
                with c1:
                    start_clicked = st.button(
                        "🚀 开始学习本章", key="ai_tutor_start_btn"
                    )
                with c2:
                    if st.button("🔄 重置会话", key="ai_tutor_reset_btn"):
                        self.session.reset()
                        st.rerun()

                if start_clicked:
                    with st.spinner("AI 助教正在准备课程..."):
                        result = self.session.start_chapter(chapter_idx, self.client)
                        if not result["success"]:
                            st.error(result["reply"])

                with st.container(border=True):
                    st.markdown("**教学模式说明**")
                    st.markdown(
                        "- AI 助教会分小节讲解本章内容，并通过提问检查您是否理解\n"
                        "- 如果理解了，请在对话后点击「✅ 我已理解，进入下一小节」\n"
                        "- 如果有疑问，直接在对话区域输入即可\n"
                        "- 整章完成后会生成知识小结和阶段测验"
                    )

            with col_right:
                self._render_conversation()

    # -------------------- API 配置面板 --------------------
    def _render_api_config_panel(self):
        """顶部 API 配置面板：提供商、API Key、模型、测试连接。"""
        with st.container(border=True):
            st.markdown("### ⚙️ API 配置")
            provider_names = list(PROVIDERS.keys())

            # 选择默认值：如果 client 当前 provider 在列表中则使用它
            default_provider_idx = 0
            if self.client.provider in provider_names:
                default_provider_idx = provider_names.index(self.client.provider)

            col_p, col_info = st.columns([1, 2])
            with col_p:
                provider = st.selectbox(
                    "服务提供商",
                    provider_names,
                    index=default_provider_idx,
                    key="ai_tutor_provider_select",
                )
            with col_info:
                info = PROVIDERS[provider]
                st.markdown(
                    f"**说明**：{info['description']}\n\n**免费策略**：{info['free_note']}"
                )

            provider_info = PROVIDERS[provider]

            # 第二行：API Key + 模型
            col_k, col_m = st.columns(2)
            with col_k:
                key_label = f"{provider_info['env_key']}（可直接输入，或写入 .env 文件）"
                api_key = st.text_input(
                    "API Key",
                    value=self.client.api_key or "",
                    type="password",
                    placeholder="sk-...  或通过 .env 配置",
                    key="ai_tutor_api_key",
                    help=key_label,
                )
            with col_m:
                # 让用户可以从预设列表中选择，也可以自行输入
                model_choices = provider_info["models"]
                current_model = self.client.model
                if current_model not in model_choices:
                    model_choices = list(model_choices) + [current_model]
                model = st.selectbox(
                    "使用的模型",
                    model_choices,
                    index=(
                        model_choices.index(current_model)
                        if current_model in model_choices
                        else 0
                    ),
                    key="ai_tutor_model_select",
                )

            # 自定义模式：显示 base_url 输入框
            if provider == "自定义 (任意 OpenAI 兼容)":
                base_url = st.text_input(
                    "自定义 Base URL",
                    value=self.client.base_url or provider_info["base_url"],
                    placeholder="https://api.example.com/v1",
                    key="ai_tutor_base_url",
                )
            else:
                base_url = provider_info["base_url"]

            # 测试连接 + 状态显示
            col_btn, col_status, _ = st.columns([1, 2, 2])
            with col_btn:
                test_clicked = st.button(
                    "🧪 测试连接", key="ai_tutor_test_btn", type="primary"
                )

            # 如果用户在界面上修改了值 → 更新客户端
            if (
                api_key != self.client.api_key
                or base_url != self.client.base_url
                or model != self.client.model
                or provider != self.client.provider
            ):
                self.client.update(
                    api_key=api_key or None,
                    base_url=base_url or None,
                    model=model or None,
                    provider=provider,
                )

            if test_clicked:
                with st.spinner("正在测试连接..."):
                    result = self.client.chat(
                        [{"role": "user", "content": "请用一句话回答：你好。"}],
                        max_tokens=30,
                        timeout=30,
                    )
                if result["success"]:
                    st.success(
                        f"✅ 连接成功！平台「{self.client.provider}」模型「{self.client.model}」正常工作。"
                    )
                else:
                    st.error(result["error"])
            else:
                # 静态状态显示
                if self.client.is_ready:
                    st.success(
                        f"✅ API 已就绪 · 平台：{self.client.provider} · 模型：{self.client.model}"
                    )
                else:
                    st.warning(
                        f"⚠️ API 未就绪 · 当前平台「{self.client.provider}」需要有效的 API Key。"
                    )

    # -------------------- 对话区域 --------------------
    def _render_conversation(self):
        st.subheader("💬 对话区域")

        if not self.client.is_ready:
            st.info("请先在上方配置 API 并点击「测试连接」确认可用。")
            return

        if not self.session.conversation_log:
            st.info("请在左侧选择章节后点击「🚀 开始学习本章」开启教学。")
        else:
            for msg in self.session.conversation_log:
                if msg["role"] == "ai":
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("user"):
                        st.markdown(msg["content"])

            # 进入下一小节按钮
            if (
                self.session.chapter_idx is not None
                and self.session.state != TutorSession.STATE_FINISHED
            ):
                btn_key = f"ai_tutor_next_section_{len(self.session.conversation_log)}"
                if st.button("✅ 我已理解，进入下一小节", key=btn_key):
                    with st.spinner("准备下一小节..."):
                        result = self.session.next_section(self.client)
                        if not result["success"]:
                            st.error(result["reply"])
                        else:
                            st.rerun()

            # 用户输入
            user_input = st.chat_input(
                "请输入您的回答或问题...", key="ai_tutor_user_input"
            )
            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.spinner("AI 助教思考中..."):
                    result = self.session.send_user_message(user_input, self.client)
                if result["success"]:
                    with st.chat_message("assistant"):
                        st.markdown(result["reply"])
                else:
                    st.error(result["reply"])


# ==================================================================
# 6. 便捷函数 —— 给 app.py 使用
# ==================================================================


def create_ai_tutor_ui() -> StreamlitAITutorUI:
    """创建并返回 AI 助教 UI 对象。"""
    return StreamlitAITutorUI()


def render_ai_tutor_sidebar():
    """在 Streamlit 侧边栏中渲染 AI 助教快捷入口。"""
    ui = create_ai_tutor_ui()
    ui.render_in_sidebar()


def render_ai_tutor_main():
    """在 Streamlit 主内容区域渲染 AI 助教（完整大区域）。"""
    ui = create_ai_tutor_ui()
    ui.render_in_main()


# ==================================================================
# 7. 独立测试（直接运行本文件时可用）
# ==================================================================

if __name__ == "__main__":
    # 简单自检：打印课程大纲、测试客户端初始化
    print("=" * 60)
    print("数学物理方法 AI 助教 - 模块自检")
    print("=" * 60)

    print("\n【课程大纲】")
    for ch in COURSE_OUTLINE:
        print(f"\n第{ch['chapter_id']}章：{ch['title']} (模块: {ch['module']})")
        for s in ch["sections"]:
            print(f"  - {s['section_id']} {s['title']}")

    print("\n【客户端初始化】")
    client = DeepSeekClient()
    print(f"  API 就绪: {client.is_ready}")
    print(f"  使用模型: {client.model}")

    if not client.api_key:
        print("  提示: 未配置 DEEPSEEK_API_KEY，请创建 .env 文件")
    print("\n模块自检完成 [OK]")
