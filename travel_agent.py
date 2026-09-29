"""一个基于 ReAct（Reasoning + Acting）模式的智能旅行助手 Agent。"""

from __future__ import annotations

import os
import re

import requests
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式,包含一对Thought和Action:

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一:
1. 调用工具:function_name(arg_name="arg_value")
2. 结束任务:Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行,不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

WEATHER_API_URL = "https://wttr.in/{city}?format=j1"
REQUEST_TIMEOUT = 10
MAX_ITERATIONS = 5


def get_weather(city: str) -> str:
    """通过调用 wttr.in API 查询真实的天气信息。"""
    url = WEATHER_API_URL.format(city=city)
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
    except requests.exceptions.RequestException as e:
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """根据城市和天气，使用 Tavily Search API 搜索并返回景点推荐。"""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    tavily = TavilyClient(api_key=api_key)
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        if response.get("answer"):
            return response["answer"]

        results = response.get("results", [])
        if not results:
            return "抱歉，没有找到相关的旅游景点推荐。"

        formatted_results = [
            f"- {result['title']}: {result['content']}" for result in results
        ]
        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"


available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}


class OpenAICompatibleClient:
    """一个用于调用任何兼容 OpenAI 接口的客户端。"""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用 LLM API 来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"


def truncate_thought_action(llm_output: str) -> str:
    """截断模型可能输出的多余 Thought-Action 对。"""
    match = re.search(
        r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
        llm_output,
        re.DOTALL,
    )
    if not match:
        return llm_output

    truncated = match.group(1).strip()
    if truncated != llm_output.strip():
        print("已截断多余的 Thought-Action 对")
    return truncated


def parse_action(llm_output: str) -> str | None:
    """从模型输出中解析 Action 字段。"""
    match = re.search(r"Action:\s*(.*)", llm_output, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_finish(action_str: str) -> str | None:
    """如果 Action 是 Finish[...] 格式，则返回其中的最终答案。"""
    match = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_tool_call(action_str: str) -> tuple[str, dict[str, str]]:
    """解析工具调用，返回 (工具名, 参数字典)。"""
    func_match = re.match(r"(\w+)\(", action_str)
    args_match = re.search(r"\((.*)\)", action_str, re.DOTALL)
    if not func_match or not args_match:
        raise ValueError(f"无法解析的 Action 格式: {action_str}")

    tool_name = func_match.group(1)
    kwargs = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', args_match.group(1)))
    return tool_name, kwargs


def build_client() -> OpenAICompatibleClient:
    """从环境变量构建 LLM 客户端，并校验必要的配置。"""
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    model_id = os.environ.get("LLM_MODEL_ID")

    missing = [
        name
        for name, value in (
            ("LLM_API_KEY", api_key),
            ("LLM_BASE_URL", base_url),
            ("LLM_MODEL_ID", model_id),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"缺少必要的环境变量: {', '.join(missing)}")

    return OpenAICompatibleClient(model=model_id, api_key=api_key, base_url=base_url)


def run_agent(
    user_prompt: str,
    llm: OpenAICompatibleClient,
    max_iterations: int = MAX_ITERATIONS,
) -> str | None:
    """运行 ReAct 主循环，返回最终答案（未完成时返回 None）。"""
    prompt_history = [f"用户请求: {user_prompt}"]
    print(f"用户输入: {user_prompt}\n" + "=" * 40)

    for i in range(max_iterations):
        print(f"--- 循环 {i + 1} ---\n")

        full_prompt = "\n".join(prompt_history)
        llm_output = truncate_thought_action(
            llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        )
        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        action_str = parse_action(llm_output)
        if action_str is None:
            observation_str = (
                "Observation: 错误: 未能解析到 Action 字段。"
                "请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
            )
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        final_answer = parse_finish(action_str)
        if final_answer is not None:
            print(f"任务完成，最终答案: {final_answer}")
            return final_answer

        try:
            tool_name, kwargs = parse_tool_call(action_str)
        except ValueError as e:
            observation = f"错误:{e}"
        else:
            if tool_name in available_tools:
                observation = available_tools[tool_name](**kwargs)
            else:
                observation = f"错误:未定义的工具 '{tool_name}'"

        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)

    print("已达到最大循环次数，未能得到最终答案。")
    return None


def main() -> None:
    """程序入口：构建客户端并运行 Agent。"""
    llm = build_client()
    user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    run_agent(user_prompt, llm)


if __name__ == "__main__":
    main()
