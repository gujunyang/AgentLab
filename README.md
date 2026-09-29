# AgentLab

一个基于 **ReAct（Reasoning + Acting）** 模式的智能旅行助手机器人。Agent 会通过大语言模型自主思考、调用工具（查询天气、搜索景点），并在多轮循环中逐步完成用户的旅行咨询请求。

## 功能特性

- **ReAct 智能体循环**：模型每一轮输出一对 `Thought` / `Action`，程序解析并执行工具，再把 `Observation` 反馈给模型，直到模型给出 `Finish[...]` 最终答案。
- **实时天气查询**：通过 [wttr.in](https://wttr.in) API 获取指定城市的真实天气。
- **景点智能推荐**：通过 [Tavily](https://tavily.com) 搜索 API，结合城市与天气生成景点推荐。
- **OpenAI 兼容客户端**：可对接任何兼容 OpenAI 接口的大模型服务（示例使用 DeepSeek）。

## 工作原理

```
用户请求
   │
   ▼
┌─────────────────────────────────────┐
│  构建 Prompt（历史对话 + 观察结果）    │
└─────────────────────────────────────┘
   │
   ▼
调用 LLM 生成 Thought / Action
   │
   ▼
解析 Action ──► 调用工具（get_weather / get_attraction）
   │                     │
   │                     ▼
   │              返回 Observation
   │                     │
   └────── 追加到 Prompt ─┘
   │
   ▼
Action: Finish[最终答案] ──► 结束
```

## 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)（推荐的包管理器）

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件（参考下方模板），填入你的密钥：

```dotenv
# 大语言模型配置（示例使用 DeepSeek）
LLM_MODEL_ID="deepseek-flash"
LLM_API_KEY="your-llm-api-key"
LLM_BASE_URL="https://api.deepseek.com"

# Tavily 搜索 API 密钥
TAVILY_API_KEY="your-tavily-api-key"
```

| 变量名 | 说明 |
| --- | --- |
| `LLM_MODEL_ID` | 大模型名称（如 `deepseek-flash`） |
| `LLM_API_KEY` | 大模型服务的 API Key |
| `LLM_BASE_URL` | 大模型服务的接口地址 |
| `TAVILY_API_KEY` | Tavily 搜索服务 API Key，可在 [tavily.com](https://tavily.com) 申请 |

> `.env` 已被 `.gitignore` 忽略，请勿将密钥提交到版本库。

### 3. 运行

```bash
uv run travel_agent.py
```

默认示例问题：

> 你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。

运行后会打印每一轮的 `Thought`、`Action` 和 `Observation`，最终输出 `Finish[...]` 中的答案。你可以修改 `travel_agent.py` 底部的 `user_prompt` 变量来更换提问。

## 项目结构

| 文件 | 说明 |
| --- | --- |
| `travel_agent.py` | 主程序：工具定义、LLM 客户端与 ReAct 主循环 |
| `pyproject.toml` | 项目元数据与依赖声明 |
| `.env` | 本地环境变量（不纳入版本控制） |
| `README.md` | 项目说明文档 |

## 核心模块

- `get_weather(city)`：调用 wttr.in 返回天气描述与气温。
- `get_attraction(city, weather)`：调用 Tavily 根据城市和天气搜索景点推荐。
- `OpenAICompatibleClient`：封装兼容 OpenAI 接口的对话调用。
- `AGENT_SYSTEM_PROMPT`：约束模型的输出格式（`Thought` / `Action`）。

## 扩展自定义工具

在 `travel_agent.py` 中新增函数，并注册到 `available_tools` 字典即可：

```python
def get_hotel(city: str) -> str:
    ...

available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
    "get_hotel": get_hotel,
}
```

工具函数建议接收字符串参数并返回字符串结果，同时在系统提示词的「可用工具」列表中补充说明。

## 注意事项

- 主循环默认最多执行 **5 轮**（`travel_agent.py` 中的 `for i in range(5)`），可按需调整。
- 需要联网访问 wttr.in、Tavily 以及大模型服务接口。
- 请妥善保管 `.env` 中的 API 密钥。
