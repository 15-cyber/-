"""
高光谱智能分析 Agent 核心
==========================
负责 LLM 对话循环、工具路由、自主思考与执行。
"""

import traceback
from typing import Any, Callable, Optional

from .config import AppConfig
from .llm_client import LLMClient, build_tool_definitions
from .toolbox import TOOL_REGISTRY, TOOL_DESCRIPTIONS, get_data


# ═══════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个高光谱遥感图像分析专家 Agent。

## 你的能力
你可以操作高光谱土壤剖面 TIFF 图像，执行分层数据清洗与光谱分析。
所有操作通过工具调用完成，你会收到每步的 Observation 反馈。

## 工具使用规则
1. **第一步**：始终先用 load_tiff 加载 TIFF 文件和分层元数据。
2. **信息确认**：调用 get_layer_info 了解土层结构。
3. **分析流程**：根据用户需求，依次执行 extract / clean / transform / render 等操作。
4. **最终回答**：分析完成后，用 final_answer 总结结果。

## 高光谱分析最佳实践
- 清洗噪声后再做光谱变换 (MSC/SNV/SG)
- MSC 和 SNV 二选一即可，不要重复
- SG 平滑时 window_length 应为奇数，且大于 polyorder
- 先看统计 (compute_layer_stats) 再渲染图表

## 响应格式
- thought: 你的思考过程
- 调用工具: 指定 tool_name 和 tool_params
- 完成时: tool_name = "final_answer"，tool_params.result = 你的最终回答
"""


# ═══════════════════════════════════════════════════════════════
# Agent 类
# ═══════════════════════════════════════════════════════════════

class HyperSpectralAgent:
    """高光谱分析 Agent"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = LLMClient(config)
        self.memory: list[dict] = []
        self.max_turns = config.max_turns

        # 构建工具定义
        self.tool_definitions = build_tool_definitions(TOOL_REGISTRY, TOOL_DESCRIPTIONS)
        self.tools: dict[str, Callable] = dict(TOOL_REGISTRY)

        # 回调（供 UI 使用）
        self.on_thought: Optional[Callable[[str], None]] = None
        self.on_tool_call: Optional[Callable[[str, dict], None]] = None
        self.on_observation: Optional[Callable[[str], None]] = None
        self.on_final: Optional[Callable[[str], None]] = None

    def run(self, user_prompt: str, tif_path: str = "", meta_path: str = "") -> str:
        """
        运行 Agent 对话循环。

        参数:
            user_prompt : 用户指令
            tif_path    : TIFF 文件路径
            meta_path   : 分层元数据路径

        返回:
            最终分析结果
        """
        # 构建初始 prompt
        context = ""
        if tif_path:
            context += f"\nTIFF 文件路径: {tif_path}"
        if meta_path:
            context += f"\n分层元数据路径: {meta_path}"

        current_prompt = f"用户任务: {user_prompt}{context}\n\n请开始分析。"

        self.memory = []

        for turn in range(self.max_turns):
            # ── 调用 LLM ──
            response = self.llm.chat(
                system_prompt=SYSTEM_PROMPT,
                user_message=current_prompt,
                tools=self.tool_definitions,
            )

            thought = response.get("thought", "")
            tool_name = response.get("tool_name", "")
            tool_params = response.get("tool_params", {})

            # 回调
            if self.on_thought:
                self.on_thought(thought)

            # 记录到记忆
            self.memory.append({
                "turn": turn,
                "thought": thought,
                "tool_name": tool_name,
                "tool_params": tool_params,
            })

            # ── final_answer ──
            if tool_name == "final_answer":
                result = tool_params.get("result", "分析完成。")
                if self.on_final:
                    self.on_final(result)
                return result

            # ── 执行工具 ──
            if self.on_tool_call:
                self.on_tool_call(tool_name, tool_params)

            observation = self._execute_tool(tool_name, tool_params)

            if self.on_observation:
                self.on_observation(observation)

            current_prompt = (
                f"工具 {tool_name} 的执行结果:\n{observation}\n\n"
                f"请根据结果决定下一步操作（继续调用工具，或用 final_answer 结束）。"
            )

        return "分析达到最大轮次限制，请检查任务是否过于复杂。"

    def _execute_tool(self, tool_name: str, tool_params: dict) -> str:
        """执行工具并返回 Observation"""
        if tool_name not in self.tools:
            return f"错误: 未知工具 '{tool_name}'。可用工具: {list(self.tools.keys())}"

        try:
            func = self.tools[tool_name]
            # 过滤掉不在函数签名中的参数
            result = func(**tool_params)
            return str(result)
        except TypeError as e:
            return f"参数错误: {str(e)}。请检查参数名和类型。"
        except Exception as e:
            return f"工具执行异常: {str(e)}\n{traceback.format_exc()}"

    def run_manual(self, tool_name: str, tool_params: dict) -> str:
        """手动执行单个工具（不走 LLM）"""
        return self._execute_tool(tool_name, tool_params)

    def get_memory_summary(self) -> str:
        """返回对话记忆摘要"""
        lines = []
        for m in self.memory:
            lines.append(f"[Turn {m['turn']}] {m['tool_name']}: {m['thought'][:100]}...")
        return "\n".join(lines) if lines else "无对话记录"
