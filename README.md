# 🌈 高光谱图像分层清洗与智能分析 Agent

基于大语言模型（LLM）驱动的桌面端高光谱遥感图像智能分析系统。针对土壤剖面高光谱影像，实现**自主思考 → 工具路由 → 分层清洗 → 光谱分析**的全自动工作流。

---

## 功能特性

- **🧠 LLM Agent 自主分析** — 接入 DeepSeek / GPT / Claude 等模型，Agent 自主规划工具调用链
- **📂 智能文件夹识别** — 选文件夹即可，自动用文件夹名匹配剖面编号、发现 TIF 文件
- **🔬 11 个专业工具** — TIFF 解析、分层提取、MSC/SNV 散射校正、SG 光谱平滑、热力图/光谱图渲染
- **🖥️ PyQt6 桌面界面** — 文件浏览、对话日志、图表实时展示，开箱即用
- **🔧 CLI 双模式** — 命令行工具模式 + 交互对话模式，无需 GUI 也能跑
- **🔌 多模型支持** — 预设 DeepSeek / GPT-4o / Claude，可自定义任意兼容 API

## 快速开始

### 环境要求

- Python 3.11+
- Windows / Linux / macOS

### 安装

```bash
git clone <仓库地址>
cd 高光谱agent开发

# 创建虚拟环境
python -m venv venv

# 激活 (Windows)
venv\Scripts\activate
# 激活 (Linux/Mac)
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 启动 GUI

```bash
python main.py --gui
```

### 命令行工具模式

```bash
# 第一步：设置分层元数据文件（只需一次）
python main.py --meta "标注标签文件.xlsx" --tool set_meta_file

# 第二步：加载剖面数据
python main.py --folder "4203810101100034" --tool load_from_folder

# 查看分层统计
python main.py --tool compute_layer_stats
```

### 交互对话模式

```bash
python main.py

> meta 标注标签文件.xlsx          # 设置分层表
> folder 4203810101100034          # 选择剖面文件夹
> 分析各土层光谱差异并绘制对比图    # 自然语言指令
```

## 项目结构

```
├── main.py                          # CLI / GUI 入口
├── requirements.txt                 # Python 依赖
├── README.md
├── 标注标签文件.xlsx                 # 分层元数据（示例）
├── hyperspectral_agent/
│   ├── config.py                    # 多模型配置管理
│   ├── toolbox.py                   # 高光谱专业工具箱（11 个工具）
│   ├── llm_client.py                # LLM API 客户端（Anthropic/OpenAI 双协议）
│   ├── agent.py                     # Agent 核心对话循环
│   └── ui/
│       └── main_window.py           # PyQt6 桌面界面
└── output/                          # 图表输出目录
```

## 工具箱一览

| 工具 | 功能 |
|------|------|
| `load_tiff` | 加载 TIFF + 分层元数据（.xlsx/.json/.txt） |
| `load_from_folder` | 选文件夹自动发现 TIF + 匹配分层 |
| `set_meta_file` | 设置全局分层元数据文件 |
| `get_layer_info` | 查看当前剖面所有分层信息 |
| `extract_layer` | 提取指定土层的像素矩阵 |
| `extract_spectrum` | 提取指定像元的全波段光谱曲线 |
| `clean_layer_noise` | 分层噪声清洗（中值滤波 / 均值插值） |
| `msc_transform` | 多元散射校正（MSC） |
| `snv_transform` | 标准正态变量变换（SNV） |
| `sg_smooth` | Savitzky-Golay 光谱平滑 |
| `render_heatmap` | 空间丰度热力图渲染 |
| `render_spectrum` | 光谱响应曲线绘制（单像元 / 分层对比） |
| `compute_layer_stats` | 分层统计特征（均值、标准差、变异系数） |

## 支持的模型

| 模型 | 协议 | 说明 |
|------|------|------|
| DeepSeek V4 Pro | Anthropic | DeepSeek 最新旗舰 |
| DeepSeek Chat | OpenAI | DeepSeek 通用对话 |
| GPT-4o | OpenAI | OpenAI 多模态 |
| Claude Sonnet 4 | Anthropic | Anthropic 旗舰 |

支持通过配置文件添加任意 OpenAI/Anthropic 兼容 API 的自定义模型。

## 配置

首次运行后自动生成 `config.json`，可手动编辑或通过 CLI 管理：

```bash
python main.py --list-models          # 查看可用模型
python main.py --switch gpt-4o        # 切换模型
python main.py --set-key YOUR_KEY     # 设置 API Key
```

也可设置环境变量：`DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`

## 数据格式

### 分层元数据 (.xlsx)

| 序号 | 编号 | 层次 | 发生层 | 发生层名称 | 发生层顶部(cm) | 发生层底部(cm) |
|------|------|------|--------|-----------|---------------|---------------|
| 1 | 4203810101100034 | 1 | Ap1 | 耕作层 | 0 | 12 |

- **编号** = 剖面唯一标识 = 文件夹名
- 同一编号的多行为该剖面的不同土层

### 剖面文件夹

```
4203810101100034/          ← 文件夹名即剖面编号
├── zuizhong.tif           ← 高光谱影像（任意文件名）
├── zuizhong.tfw
└── ...
```

## License

MIT
