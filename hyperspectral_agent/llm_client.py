"""
LLM API 客户端
==============
支持 Anthropic 协议和 OpenAI 协议的多模型调用。

DeepSeek V4 Pro 使用 Anthropic 兼容协议：
  POST https://api.deepseek.com/anthropic/v1/messages
"""

import json
import re
from typing import Any, Optional

import requests

from .config import AppConfig


class LLMClient:
    """统一的 LLM 调用客户端"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.api_key = config.resolve_api_key()

    # ── 公共接口 ──────────────────────────────────────────

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        发送请求并返回结构化响应。
        返回格式: {"thought": str, "tool_name": str|None, "tool_params": dict|None,
                    "result": str|None, "raw": dict}
        """
        provider = self.config.provider

        if provider == "anthropic":
            return self._chat_anthropic(system_prompt, user_message, tools, temperature, max_tokens)
        else:
            return self._chat_openai(system_prompt, user_message, tools, temperature, max_tokens)

    # ── Anthropic 协议 ────────────────────────────────────

    def _chat_anthropic(
        self,
        system_prompt: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Anthropic Messages API 调用"""
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # 构建工具定义（Anthropic 格式）
        anthropic_tools = None
        if tools:
            anthropic_tools = []
            for tool in tools:
                anthropic_tools.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": {
                        "type": "object",
                        "properties": tool.get("parameters", {}).get("properties", {}),
                        "required": tool.get("parameters", {}).get("required", []),
                    },
                })

        body: dict[str, Any] = {
            "model": self.config.model_id,
            "max_tokens": max_tok,
            "temperature": temp,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        if anthropic_tools:
            body["tools"] = anthropic_tools

        # 构建 URL
        url = self.config.base_url.rstrip("/") + "/v1/messages"

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return {
                "thought": f"API 请求失败: {str(e)}",
                "tool_name": None,
                "tool_params": None,
                "result": None,
                "raw": {},
            }

        return self._parse_anthropic_response(data)

    def _parse_anthropic_response(self, data: dict) -> dict:
        """解析 Anthropic 响应为统一格式"""
        thought = ""
        tool_name = None
        tool_params = None
        result = None

        for block in data.get("content", []):
            if block["type"] == "text":
                thought += block.get("text", "")
            elif block["type"] == "tool_use":
                tool_name = block.get("name", "")
                tool_params = block.get("input", {})

        # 如果没有 tool_use，视为 final_answer
        if not tool_name:
            tool_name = "final_answer"
            tool_params = {"result": thought.strip()}
            result = thought.strip()

        return {
            "thought": thought.strip(),
            "tool_name": tool_name,
            "tool_params": tool_params or {},
            "result": result,
            "raw": data,
        }

    # ── OpenAI 协议 ────────────────────────────────────────

    def _chat_openai(
        self,
        system_prompt: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """OpenAI Chat Completions API 调用"""
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        body: dict[str, Any] = {
            "model": self.model_id if hasattr(self, 'model_id') else self.config.model_id,
            "messages": messages,
            "max_tokens": max_tok,
            "temperature": temp,
        }

        # OpenAI function calling 格式
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
                for t in tools
            ]

        url = self.config.base_url.rstrip("/") + "/chat/completions"

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return {
                "thought": f"API 请求失败: {str(e)}",
                "tool_name": None,
                "tool_params": None,
                "result": None,
                "raw": {},
            }

        return self._parse_openai_response(data)

    def _parse_openai_response(self, data: dict) -> dict:
        """解析 OpenAI 响应为统一格式"""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        thought = message.get("content", "") or ""

        tool_name = None
        tool_params = None
        result = None

        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            try:
                tool_params = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                # 尝试从文本中提取 JSON
                args_str = func.get("arguments", "{}")
                tool_params = _extract_json(args_str)

        if not tool_name:
            tool_name = "final_answer"
            tool_params = {"result": thought.strip()}
            result = thought.strip()

        return {
            "thought": thought.strip(),
            "tool_name": tool_name,
            "tool_params": tool_params or {},
            "result": result,
            "raw": data,
        }


def _extract_json(text: str) -> dict:
    """从文本中提取 JSON 对象（容错解析）"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试匹配 {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


# ── 工具定义生成 ────────────────────────────────────────────

def build_tool_definitions(tool_registry: dict, tool_descriptions: dict) -> list[dict]:
    """
    根据工具箱注册表生成 LLM 工具定义。
    返回 OpenAI 兼容格式，Anthropic 调用时会自动转换。
    """
    tools = []
    for name, desc in tool_descriptions.items():
        if name not in tool_registry:
            continue

        # 从描述中提取参数信息
        params_info = _extract_params_from_desc(desc)
        params_def = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for pname, pdesc in params_info:
            params_def["properties"][pname] = {
                "type": "string",
                "description": pdesc,
            }

        # 特殊类型处理
        if name == "sg_smooth":
            params_def["properties"]["window_length"] = {"type": "integer", "description": "窗口大小"}
            params_def["properties"]["polyorder"] = {"type": "integer", "description": "多项式阶数"}
        if name == "render_heatmap":
            params_def["properties"]["band_index"] = {"type": "integer", "description": "波段索引(0-based)"}
        if name == "render_spectrum":
            params_def["properties"]["row"] = {"type": "integer", "description": "行号"}
            params_def["properties"]["col"] = {"type": "integer", "description": "列号"}
            params_def["properties"]["compare"] = {"type": "boolean", "description": "是否对比所有土层"}

        tools.append({
            "name": name,
            "description": desc,
            "parameters": params_def,
        })
    return tools


def _extract_params_from_desc(desc: str) -> list:
    """从工具描述文本中提取参数名和说明"""
    params = []
    # 匹配 "参数: xxx (说明), yyy (说明)" 模式
    param_match = re.search(r"参数[：:]\s*(.+)", desc)
    if param_match:
        param_text = param_match.group(1)
        # 匹配 "name (desc)" 或 "name"
        for match in re.finditer(r"(\w+)\s*(?:\(([^)]*)\))?", param_text):
            pname = match.group(1)
            pdesc = match.group(2) if match.group(2) else match.group(1)
            params.append((pname, pdesc))
    return params
