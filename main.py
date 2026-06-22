#!/usr/bin/env python
"""
高光谱图像分层清洗与智能分析 Agent - CLI 入口
==============================================

用法:
    # 交互模式
    python main.py

    # 指定模型
    python main.py --model deepseek-v4-pro

    # 列出可用模型
    python main.py --list-models

    # 切换模型
    python main.py --switch gpt-4o

    # 设置 API Key
    python main.py --set-key YOUR_API_KEY

    # 指定数据文件直接分析
    python main.py --tif path/to/image.tif --meta path/to/layers.xlsx --prompt "分析各层光谱差异"

    # 手动使用工具箱（不走 LLM）
    python main.py --tool load_tiff --tif path/to/image.tif --meta path/to/layers.xlsx
    python main.py --tool compute_layer_stats
"""

import argparse
import os
import sys

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hyperspectral_agent.config import AppConfig, PRESET_MODELS
from hyperspectral_agent.agent import HyperSpectralAgent
from hyperspectral_agent.toolbox import TOOL_REGISTRY, get_data


def print_banner():
    print(r"""
+======================================================+
|      高光谱图像分层清洗与智能分析 Agent            |
|     Hyperspectral Soil Profile Analysis Agent        |
+======================================================+
""")


def list_models(config: AppConfig):
    """列出所有可用模型"""
    print("\n[List] 可用模型:\n")
    print(f"{'模型ID':<30} {'名称':<20} {'协议':<12}")
    print("-" * 62)
    all_models = config.get_all_models()
    for mid, info in all_models.items():
        marker = " →" if mid == config.model_id else "  "
        print(f"{marker} {mid:<28} {info['name']:<20} {info['provider']:<12}")
    print(f"\n当前使用: {config.model_id}")


def interactive_mode(config: AppConfig):
    """交互式对话模式"""
    print_banner()
    print(f"当前模型: {config.model_id} ({config.provider})")
    print(f"API 端点: {config.base_url}")

    api_key = config.resolve_api_key()
    if not api_key:
        print("\n[WARN]  未设置 API Key。请使用 --set-key 设置，或设置环境变量。")
        print(f"   预设模型推荐环境变量: {PRESET_MODELS.get(config.model_id, {}).get('api_key_env', '')}")

    agent = HyperSpectralAgent(config)

    # 注册回调
    agent.on_thought = lambda t: print(f"\n[Think] 思考: {t}\n")
    agent.on_tool_call = lambda n, p: print(f"[Tool] 调用工具: {n}({p})")
    agent.on_observation = lambda o: print(f"[Result] 结果:\n{o}\n")

    print("\n命令: folder=选文件夹 | meta=设分层表 | key=设Key | switch=切模型 | tools/models/quit\n")

    current_tif = ""
    current_meta = ""
    current_folder = ""

    while True:
        try:
            user_input = input("You 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("再见！")
            break

        if user_input.lower() == "models":
            list_models(config)
            continue

        if user_input.lower() == "tools":
            from hyperspectral_agent.toolbox import TOOL_DESCRIPTIONS
            print("\n[Tool] 可用工具:")
            for name, desc in TOOL_DESCRIPTIONS.items():
                print(f"  {name}: {desc}")
            continue

        if user_input.lower().startswith("switch "):
            model_id = user_input[7:].strip()
            if config.switch_model(model_id):
                config.save()
                agent = HyperSpectralAgent(config)
                print(f"[OK] 已切换到模型: {model_id}")
            else:
                print(f"[FAIL] 未知模型: {model_id}")
            continue

        if user_input.lower().startswith("tif "):
            current_tif = user_input[4:].strip()
            print(f"[OK] TIFF 路径已设置: {current_tif}")
            continue

        if user_input.lower().startswith("meta "):
            current_meta = user_input[5:].strip()
            print(f"[OK] 元数据路径已设置: {current_meta}")
            continue

        if user_input.lower().startswith("folder "):
            current_folder = user_input[7:].strip()
            print(f"[OK] 剖面文件夹: {current_folder}")
            continue

        if user_input.lower().startswith("key "):
            config.api_key = user_input[4:].strip()
            config.save()
            agent = HyperSpectralAgent(config)
            print("[OK] API Key 已保存")
            continue

        # ── 运行 Agent ──
        # 如果设置了文件夹和元数据，自动提示 Agent 用 load_from_folder
        if current_folder and current_meta:
            auto_prompt = (
                f"{user_input}\n\n【文件信息】\n"
                f"剖面文件夹: {current_folder}\n"
                f"分层元数据: {current_meta}\n"
                f"请先用 load_from_folder 加载文件夹 '{current_folder}'，再用 set_meta_file 设置元数据。"
            )
            # 先在本地自动设置 meta 和加载
            from hyperspectral_agent.toolbox import set_meta_file, load_from_folder
            set_meta_file(current_meta)
            load_result = load_from_folder(current_folder)
            print(f"[Folder] 自动加载结果:\n{load_result}\n")
            prompt = f"{user_input}\n\n数据已加载，请直接开始分析。"
        elif current_folder:
            prompt = f"{user_input}\n\n请用 load_from_folder 加载文件夹: {current_folder}"
        else:
            prompt = user_input

        print("\nAgent Agent 开始分析...\n")
        result = agent.run(
            user_prompt=prompt,
            tif_path=current_tif,
            meta_path=current_meta,
        )
        print(f"\n[OK] 最终结果:\n{result}\n")

        # 询问是否需要图表
        if get_data().data is not None:
            ask = input("是否生成热力图/光谱图? (y/n): ").strip().lower()
            if ask == "y":
                agent.run_manual("render_heatmap", {"band_index": 0})
                agent.run_manual("render_spectrum", {"compare": True})
                print("图表已保存到 output/ 目录。")


def main():
    # Windows console UTF-8 support
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(
        description="高光谱图像分层清洗与智能分析 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                        # 交互模式
  python main.py --list-models                          # 列出模型
  python main.py --model deepseek-chat                  # 指定模型
  python main.py --switch gpt-4o                        # 切换默认模型
  python main.py --set-key sk-xxx                       # 设置 API Key
  python main.py --tool load_tiff --tif ./image.tif --meta ./layers.xlsx
  python main.py --tool compute_layer_stats
  python main.py --tif ./image.tif --meta ./layers.xlsx --prompt "分析光谱差异"
        """,
    )

    parser.add_argument("--model", "-m", help="使用的模型 ID")
    parser.add_argument("--list-models", action="store_true", help="列出所有可用模型")
    parser.add_argument("--switch", help="切换默认模型并保存配置")
    parser.add_argument("--set-key", help="设置 API Key 并保存到配置")
    parser.add_argument("--tif", help="TIFF 文件路径")
    parser.add_argument("--meta", help="分层元数据文件路径 (.xlsx/.json/.txt)")
    parser.add_argument("--folder", "-f", help="剖面文件夹路径（文件夹名=剖面编号，自动找TIF）")
    parser.add_argument("--prompt", "-p", help="分析提示 (非交互模式)")
    parser.add_argument("--gui", "-g", action="store_true", help="启动图形界面")
    parser.add_argument("--tool", "-t", help="手动调用工具箱中的工具 (不走 LLM)")
    parser.add_argument("--tool-params", default="{}", help="工具参数 (JSON 格式)")

    args = parser.parse_args()

    # ── 加载配置 ──
    config = AppConfig.from_file()

    # ── 处理命令行操作 ──
    if args.list_models:
        list_models(config)
        return

    if args.switch:
        if config.switch_model(args.switch):
            config.save()
            print(f"[OK] 已切换到模型: {args.switch}")
            print(f"   API 端点: {config.base_url}")
        else:
            print(f"[FAIL] 未知模型: {args.switch}")
            print("使用 --list-models 查看可用模型。")
        return

    if args.set_key:
        config.api_key = args.set_key
        config.save()
        print("[OK] API Key 已保存")
        return

    if args.model:
        if config.switch_model(args.model):
            print(f"使用模型: {args.model}")
        else:
            print(f"[WARN]  未知模型 '{args.model}'，使用默认配置。")

    # ── 自动加载已持久化的元数据文件 ──
    auto_meta = config.meta_file_path or args.meta
    if auto_meta and os.path.exists(auto_meta):
        from hyperspectral_agent.toolbox import set_meta_file
        set_meta_file(auto_meta)

    # ── 手动工具模式 ──
    if args.tool:
        if args.tool not in TOOL_REGISTRY:
            print(f"[FAIL] 未知工具: {args.tool}")
            print(f"可用: {list(TOOL_REGISTRY.keys())}")
            return

        import json
        try:
            params = json.loads(args.tool_params)
        except json.JSONDecodeError:
            params = {}

        # 自动注入路径参数
        if args.tool == "load_tiff":
            params.setdefault("tif_path", args.tif or "")
            params.setdefault("meta_path", args.meta or "")
        if args.tool == "load_from_folder":
            params.setdefault("folder_path", args.folder or "")
            # 如果有 --meta，先设置元数据文件
            if args.meta:
                from hyperspectral_agent.toolbox import set_meta_file
                set_meta_file(args.meta)
        if args.tool == "set_meta_file":
            params.setdefault("meta_path", args.meta or "")

        func = TOOL_REGISTRY[args.tool]
        result = func(**params)
        print(result)
        return

    # ── GUI 模式 ──
    if args.gui:
        from PyQt6.QtWidgets import QApplication
        app = QApplication(sys.argv)
        from hyperspectral_agent.ui.main_window import MainWindow
        window = MainWindow()
        window.show()
        sys.exit(app.exec())

    # ── 单次分析模式 ──
    if args.prompt:
        if not args.tif and not args.folder:
            print("[WARN]  单次分析模式建议指定 --tif 或 --folder。")

        # 如果指定了 --folder，先让 Agent 知道用 load_from_folder
        if args.folder:
            prompt = args.prompt + f"\n请先用 load_from_folder 加载文件夹: {args.folder}"
        else:
            prompt = args.prompt

        agent = HyperSpectralAgent(config)
        agent.on_thought = lambda t: print(f"[Think] {t}")
        agent.on_tool_call = lambda n, p: print(f"[Tool] {n}({p})")
        agent.on_observation = lambda o: print(f"[Result] {o}")
        result = agent.run(
            user_prompt=prompt,
            tif_path=args.tif or "",
            meta_path=args.meta or "",
        )
        print(f"\n[OK] {result}")
        return

    # ── 交互模式 ──
    interactive_mode(config)


if __name__ == "__main__":
    main()
