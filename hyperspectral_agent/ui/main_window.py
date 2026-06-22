"""
高光谱 Agent 主窗口 (PyQt6)
===========================
布局：顶部工具栏 | 左：对话日志 | 右：图片展示 | 底部：输入区
"""

import os
import sys
import json
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QTextEdit, QLabel,
    QComboBox, QFileDialog, QSplitter, QStatusBar,
    QScrollArea, QGroupBox, QMessageBox, QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QPixmap, QFont, QColor, QTextCursor, QIcon

from ..config import AppConfig, PRESET_MODELS
from ..agent import HyperSpectralAgent
from ..toolbox import set_meta_file, load_from_folder, get_data


# ═══════════════════════════════════════════════════════════════
# 后台工作线程
# ═══════════════════════════════════════════════════════════════

class AgentWorker(QThread):
    """在后台线程运行 Agent，避免 UI 卡顿"""
    thought_signal = pyqtSignal(str)
    tool_call_signal = pyqtSignal(str, str)
    observation_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, config: AppConfig, prompt: str, tif_path: str, meta_path: str):
        super().__init__()
        self.config = config
        self.prompt = prompt
        self.tif_path = tif_path
        self.meta_path = meta_path

    def run(self):
        try:
            agent = HyperSpectralAgent(self.config)
            agent.on_thought = lambda t: self.thought_signal.emit(t)
            agent.on_tool_call = lambda n, p: self.tool_call_signal.emit(n, str(p))
            agent.on_observation = lambda o: self.observation_signal.emit(o)
            result = agent.run(self.prompt, self.tif_path, self.meta_path)
            self.finished_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))


# ═══════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = AppConfig.from_file()
        self._worker: AgentWorker | None = None
        self._current_image: str = ""
        self._folder_loaded = False

        self.setWindowTitle("高光谱图像分层清洗与智能分析 Agent")
        self.setMinimumSize(1200, 750)

        self._setup_ui()
        self._apply_style()
        self._load_config_to_ui()

    # ── UI 构建 ──────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── 顶部工具栏 ──
        toolbar = QHBoxLayout()

        # 文件夹选择
        toolbar.addWidget(QLabel("剖面文件夹:"))
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("选择剖面文件夹（文件夹名=编号）...")
        self.folder_edit.setMinimumWidth(250)
        toolbar.addWidget(self.folder_edit)
        btn_folder = QPushButton("浏览...")
        btn_folder.clicked.connect(self._select_folder)
        toolbar.addWidget(btn_folder)

        toolbar.addSpacing(12)

        # 分层表选择
        toolbar.addWidget(QLabel("分层表:"))
        self.meta_edit = QLineEdit()
        self.meta_edit.setPlaceholderText("选择分层元数据文件 (.xlsx)...")
        self.meta_edit.setMinimumWidth(200)
        toolbar.addWidget(self.meta_edit)
        btn_meta = QPushButton("浏览...")
        btn_meta.clicked.connect(self._select_meta)
        toolbar.addWidget(btn_meta)

        toolbar.addSpacing(12)

        # 模型选择
        toolbar.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(160)
        all_models = self.config.get_all_models()
        for mid, info in all_models.items():
            self.model_combo.addItem(f"{info['name']} ({mid})", mid)
        idx = self.model_combo.findData(self.config.model_id)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        toolbar.addWidget(self.model_combo)

        toolbar.addStretch()

        # 加载按钮
        btn_load = QPushButton("📂 加载数据")
        btn_load.setMinimumHeight(30)
        btn_load.clicked.connect(self._load_data)
        toolbar.addWidget(btn_load)

        root.addLayout(toolbar)

        # ── 中间分割区 ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧：对话日志
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("思考与日志"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 10))
        left_layout.addWidget(self.log_view)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        left_layout.addWidget(self.progress_bar)

        splitter.addWidget(left_panel)

        # 右侧：图片展示
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("图片展示"))
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333;")
        self.image_label = QLabel("加载数据后可在此查看热力图 / 光谱图")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("color: #888; font-size: 14px;")
        self.image_scroll.setWidget(self.image_label)
        right_layout.addWidget(self.image_scroll, 1)

        # 图片切换按钮
        img_btns = QHBoxLayout()
        self.btn_heatmap = QPushButton("热力图")
        self.btn_heatmap.clicked.connect(lambda: self._show_image("heatmap"))
        self.btn_heatmap.setEnabled(False)
        img_btns.addWidget(self.btn_heatmap)
        self.btn_spectrum = QPushButton("光谱对比")
        self.btn_spectrum.clicked.connect(lambda: self._show_image("spectrum"))
        self.btn_spectrum.setEnabled(False)
        img_btns.addWidget(self.btn_spectrum)
        img_btns.addStretch()
        right_layout.addLayout(img_btns)

        splitter.addWidget(right_panel)
        splitter.setSizes([600, 550])
        root.addWidget(splitter, 1)

        # ── 底部输入区 ──
        input_layout = QHBoxLayout()
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("输入分析指令，例如：分析各土层光谱差异，绘制热力图...")
        self.prompt_edit.setMinimumHeight(32)
        self.prompt_edit.returnPressed.connect(self._send_prompt)
        input_layout.addWidget(self.prompt_edit, 1)

        self.btn_send = QPushButton("发送")
        self.btn_send.setMinimumHeight(32)
        self.btn_send.setMinimumWidth(80)
        self.btn_send.clicked.connect(self._send_prompt)
        self.btn_send.setEnabled(False)
        input_layout.addWidget(self.btn_send)

        root.addLayout(input_layout)

        # ── 状态栏 ──
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 — 请先加载数据")

    # ── 样式 ────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QWidget { color: #ddd; font-size: 13px; }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #3c3c3c; border: 1px solid #555;
                border-radius: 4px; padding: 4px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c; selection-background-color: #505050;
            }
            QPushButton {
                background-color: #0e639c; border: none;
                border-radius: 4px; padding: 6px 14px; color: white;
            }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:pressed { background-color: #094771; }
            QPushButton:disabled { background-color: #444; color: #888; }
            QProgressBar { border: 1px solid #555; border-radius: 3px; }
            QProgressBar::chunk { background-color: #0e639c; }
            QSplitter::handle { background-color: #555; width: 2px; }
            QScrollArea { border: none; }
            QLabel { color: #ccc; }
        """)

    # ── 配置加载 ────────────────────────────────────────

    def _load_config_to_ui(self):
        if self.config.meta_file_path:
            self.meta_edit.setText(self.config.meta_file_path)

    # ── 槽函数 ──────────────────────────────────────────

    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择剖面文件夹")
        if folder:
            self.folder_edit.setText(folder)

    def _select_meta(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择分层元数据文件",
            "", "Excel/JSON/Text (*.xlsx *.xls *.json *.txt);;All (*.*)"
        )
        if path:
            self.meta_edit.setText(path)

    def _on_model_changed(self):
        model_id = self.model_combo.currentData()
        if model_id and model_id != self.config.model_id:
            self.config.switch_model(model_id)
            self.config.save()
            self._log(f"[系统] 已切换模型: {model_id}")

    def _load_data(self):
        folder = self.folder_edit.text().strip()
        meta = self.meta_edit.text().strip()

        if not folder:
            QMessageBox.warning(self, "提示", "请先选择剖面文件夹")
            return
        if not meta:
            QMessageBox.warning(self, "提示", "请先选择分层元数据文件")
            return
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "提示", f"文件夹不存在: {folder}")
            return
        if not os.path.exists(meta):
            QMessageBox.warning(self, "提示", f"元数据文件不存在: {meta}")
            return

        # 设置元数据
        set_meta_file(meta)
        self.config.meta_file_path = meta
        try:
            self.config.save()
        except Exception:
            pass

        # 加载文件夹
        result = load_from_folder(folder)
        self._log(f"━━━ 数据加载 ━━━\n{result}\n")

        if "加载成功" in result:
            self._folder_loaded = True
            self.btn_send.setEnabled(True)
            self.btn_heatmap.setEnabled(True)
            self.btn_spectrum.setEnabled(True)
            self.status_bar.showMessage(f"数据已加载 — {os.path.basename(folder)}")

            # 自动生成初始图表
            from ..toolbox import render_heatmap, render_spectrum
            os.makedirs("output", exist_ok=True)
            render_heatmap(band_index=0, output_path="output/gui_heatmap.png")
            render_spectrum(compare=True, output_path="output/gui_spectrum.png")
            self._show_image("heatmap")
        else:
            self._folder_loaded = False
            self.status_bar.showMessage("加载失败，请检查文件夹和分层表")

    def _send_prompt(self):
        prompt = self.prompt_edit.text().strip()
        if not prompt:
            return
        if not self._folder_loaded:
            QMessageBox.warning(self, "提示", "请先加载数据")
            return

        self.prompt_edit.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度

        self._log(f"\n{'='*50}\n[用户] {prompt}\n{'='*50}\n")

        # 启动后台线程
        tif_path = ""  # 数据已加载，不需要重复传
        meta_path = ""
        self._worker = AgentWorker(self.config, prompt, tif_path, meta_path)
        self._worker.thought_signal.connect(self._on_thought)
        self._worker.tool_call_signal.connect(self._on_tool_call)
        self._worker.observation_signal.connect(self._on_observation)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.error_signal.connect(self._on_error)
        self._worker.start()

    def _on_thought(self, text: str):
        self._log(f"[思考] {text}")

    def _on_tool_call(self, name: str, params: str):
        self._log(f"[工具调用] {name}({params})")

    def _on_observation(self, text: str):
        self._log(f"[结果]\n{text}")

        # 如果生成了图片，自动刷新
        if "已保存" in text and self._folder_loaded:
            # 提取图片路径
            for line in text.split("\n"):
                if "已保存:" in line or "已保存：" in line:
                    parts = line.split(":") if ":" in line else line.split("：")
                    if len(parts) >= 2:
                        img_path = parts[-1].strip()
                        if os.path.exists(img_path):
                            self._current_image = img_path
                            self._display_image(img_path)

    def _on_finished(self, result: str):
        self._log(f"\n{'='*50}\n[最终结果]\n{result}\n{'='*50}\n")
        self._reset_ui()
        self.status_bar.showMessage("分析完成")

        # 自动刷新图表
        if self._folder_loaded:
            from ..toolbox import render_heatmap, render_spectrum
            os.makedirs("output", exist_ok=True)
            render_heatmap(band_index=0, output_path="output/gui_heatmap.png")
            render_spectrum(compare=True, output_path="output/gui_spectrum.png")
            self._show_image("heatmap")

    def _on_error(self, error: str):
        self._log(f"[错误] {error}")
        self._reset_ui()
        self.status_bar.showMessage("分析出错")

    def _reset_ui(self):
        self.prompt_edit.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.prompt_edit.clear()
        self._worker = None

    # ── 日志与图片 ─────────────────────────────────────

    def _log(self, text: str):
        self.log_view.append(text)
        # 自动滚动到底部
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_view.setTextCursor(cursor)

    def _show_image(self, kind: str):
        if kind == "heatmap":
            path = "output/gui_heatmap.png"
        else:
            path = "output/gui_spectrum.png"
        if os.path.exists(path):
            self._display_image(path)

    def _display_image(self, path: str):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        # 缩放到显示区域
        scaled = pixmap.scaled(
            self.image_scroll.width() - 20,
            self.image_scroll.height() - 20,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
