"""
配置模块：管理 API 密钥、模型选择、应用设置。
支持 DeepSeek / OpenAI 兼容 API 等多种模型。
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── 预设模型列表 ──────────────────────────────────────────────
PRESET_MODELS = {
    "deepseek-v4-pro": {
        "name": "DeepSeek V4 Pro",
        "base_url": "https://api.deepseek.com/anthropic",
        "api_key_env": "DEEPSEEK_API_KEY",
        "provider": "anthropic",
        "description": "DeepSeek 最新旗舰模型（兼容 Anthropic 协议）",
    },
    "deepseek-chat": {
        "name": "DeepSeek Chat (OpenAI)",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "provider": "openai",
        "description": "DeepSeek Chat 模型（兼容 OpenAI 协议）",
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "provider": "openai",
        "description": "OpenAI GPT-4o 多模态模型",
    },
    "claude-sonnet-4-20250514": {
        "name": "Claude Sonnet 4",
        "base_url": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY",
        "provider": "anthropic",
        "description": "Anthropic Claude Sonnet 4 模型",
    },
}


@dataclass
class AppConfig:
    """应用全局配置"""

    # ── 模型配置 ──
    model_id: str = "deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com/anthropic"
    api_key: str = ""
    provider: str = "anthropic"  # "anthropic" | "openai"

    # ── 自定义模型（用户通过 UI 添加）──
    custom_models: dict = field(default_factory=dict)

    # ── Agent 配置 ──
    max_turns: int = 10
    temperature: float = 0.7
    max_tokens: int = 4096

    # ── 路径 ──
    data_dir: str = "data"
    output_dir: str = "output"
    meta_file_path: str = ""  # 分层元数据文件路径（持久化）

    @classmethod
    def from_file(cls, path: str = "config.json") -> "AppConfig":
        """从 JSON 文件加载配置"""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()

    def save(self, path: str = "config.json") -> None:
        """保存配置到 JSON 文件"""
        data = {k: v for k, v in self.__dict__.items() if k in self.__dataclass_fields__}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def resolve_api_key(self) -> str:
        """解析 API 密钥（优先使用直接设置的，其次从环境变量读取）"""
        if self.api_key:
            return self.api_key

        # 从预设模型获取环境变量名
        preset = PRESET_MODELS.get(self.model_id, {})
        env_var = preset.get("api_key_env", "")
        if env_var:
            return os.environ.get(env_var, "")

        return ""

    def get_all_models(self) -> dict:
        """获取所有可用模型（预设 + 自定义）"""
        all_models = dict(PRESET_MODELS)
        all_models.update(self.custom_models)
        return all_models

    def set_custom_model(self, model_id: str, name: str, base_url: str, provider: str, api_key_env: str = "") -> None:
        """添加自定义模型"""
        self.custom_models[model_id] = {
            "name": name,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "provider": provider,
            "description": f"自定义模型: {name}",
        }

    def switch_model(self, model_id: str) -> bool:
        """切换当前使用的模型"""
        all_models = self.get_all_models()
        if model_id in all_models:
            model_info = all_models[model_id]
            self.model_id = model_id
            self.base_url = model_info["base_url"]
            self.provider = model_info["provider"]
            self.api_key = ""  # 清除直接设置的 key，从环境变量重新读取
            return True
        return False
