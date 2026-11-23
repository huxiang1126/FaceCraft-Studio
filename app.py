"""FaceCraft Studio (换脸工坊) — Creator-First macOS Face Swap Studio.

Features:
- Multi-character library with multi-photo identity fusion.
- Single-pass multi-character timeline swapping (swap 3+ people in one render).
- Real-time A/B preview (hold to compare original vs swapped) before rendering.
- Non-blocking QThread rendering with live preview, ETA, cancel, and Finder reveal.
- Built-in live webcam preview with real macOS camera enumeration and instant face switch.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from engine import (
    CharacterProfile,
    FaceCraftEngine,
    RenderSettings,
    TemporalKeypointSmoother,
    TimelineSegment,
    imread_unicode,
    imwrite_unicode,
    list_macos_cameras,
    probe_video_info,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(APP_DIR, "outputs")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #14161a;
    color: #e8ecf2;
    font-family: "PingFang SC", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #262a33;
    border-radius: 10px;
    background-color: #181b21;
    top: -1px;
}
QTabBar::tab {
    background-color: #1c2028;
    color: #9aa3b2;
    padding: 9px 20px;
    margin-right: 6px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #2563eb;
    color: #ffffff;
}
QFrame#card {
    background-color: #1d2129;
    border: 1px solid #2a303c;
    border-radius: 10px;
}
QLabel#cardTitle {
    color: #93c5fd;
    font-size: 13px;
    font-weight: 700;
}
QLabel#muted {
    color: #8b95a5;
    font-size: 12px;
}
QLabel#previewBox {
    background-color: #0e1014;
    border: 1px solid #2a303c;
    border-radius: 10px;
    color: #6b7280;
}
QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 7px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover { background-color: #3b82f6; }
QPushButton:pressed { background-color: #1d4ed8; }
QPushButton:disabled { background-color: #2f3542; color: #6c7585; }
QPushButton#secondary {
    background-color: #282e3a;
    color: #e2e8f0;
}
QPushButton#secondary:hover { background-color: #353d4d; }
QPushButton#danger {
    background-color: #dc2626;
}
QPushButton#danger:hover { background-color: #ef4444; }
QPushButton#success {
    background-color: #16a34a;
}
QPushButton#success:hover { background-color: #22c55e; }
QPushButton#preset {
    background-color: #232834;
    border: 1px solid #333b4a;
    color: #cbd5e1;
    padding: 6px 10px;
}
QPushButton#preset:hover {
    border-color: #3b82f6;
    color: #ffffff;
}
QComboBox {
    background-color: #232834;
    border: 1px solid #353d4d;
    border-radius: 6px;
    padding: 5px 10px;
    min-height: 22px;
}
QListWidget {
    background-color: #15181f;
    border: 1px solid #2a303c;
    border-radius: 7px;
    padding: 4px;
    outline: none;
}
QListWidget::item {
    padding: 6px 8px;
    border-radius: 6px;
    margin-bottom: 3px;
}
QListWidget::item:selected {
    background-color: #2563eb;
    color: #ffffff;
}
QProgressBar {
    background-color: #14161a;
    border: 1px solid #2a303c;
    border-radius: 6px;
    height: 16px;
    text-align: center;
    font-size: 11px;
}
QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 5px;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #2f3542;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f8fafc;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #2563eb;
    border-radius: 3px;
}
"""


def bgr_to_qpixmap(bgr: np.ndarray, max_w: int = 0, max_h: int = 0) -> QPixmap:
    if bgr is None or bgr.size == 0:
        return QPixmap()
    h, w = bgr.shape[:2]
    if max_w > 0 and max_h > 0:
        scale = min(max_w / max(w, 1), max_h / max(h, 1))
        if scale < 1.0 or scale > 1.0:
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:04.1f}"


class _RenderWorker(QThread):
    progress = Signal(int, int, float, object, str)
    finished = Signal(bool, str, str)

    def __init__(
        self,
        engine: FaceCraftEngine,
        target_path: str,
        is_image_target: bool,
        output_path: str,
        characters: Dict[str, CharacterProfile],
        segments: List[TimelineSegment],
        default_char_id: Optional[str],
        settings: RenderSettings,
    ):
        super().__init__()
        self.engine = engine
        self.target_path = target_path
        self.is_image_target = is_image_target
        self.output_path = output_path
        self.characters = characters
        self.segments = segments
        self.default_char_id = default_char_id
        self.settings = settings
        self.stop_event = threading.Event()

    def cancel(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        try:
            if self.is_image_target:
                img = imread_unicode(self.target_path)
                if img is None:
                    self.finished.emit(False, "无法读取目标图片", "")
                    return
                char = self.characters.get(self.default_char_id) if self.default_char_id else None
                if char is None or char.embedding is None:
                    self.finished.emit(False, "请先添加并选择一个角色人脸", "")
                    return
                out = self.engine.swap_frame(img, char.embedding, self.settings)
                ok = imwrite_unicode(self.output_path, out)
                if ok:
                    self.progress.emit(1, 1, 1.0, out, char.name)
                    self.finished.emit(True, "图片换脸导出完成", self.output_path)
                else:
                    self.finished.emit(False, "保存图片失败", "")
            else:
                def _cb(cur, tot, fps, prev, cname):
                    self.progress.emit(cur, tot, fps, prev, cname)

                ok, msg = self.engine.render_video_timeline(
                    video_path=self.target_path,
                    output_path=self.output_path,
                    characters=self.characters,
                    segments=self.segments,
                    default_char_id=self.default_char_id,
                    settings=self.settings,
                    stop_event=self.stop_event,
                    progress_cb=_cb,
                )
                self.finished.emit(ok, msg, self.output_path if ok else "")
        except Exception as exc:
            self.finished.emit(False, f"渲染出错: {exc}", "")


class _WebcamWorker(QThread):
    frameReady = Signal(object, float)
    error = Signal(str)

    def __init__(self, engine: FaceCraftEngine, camera_index: int, width: int, height: int):
        super().__init__()
        self.engine = engine
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._embedding: Optional[np.ndarray] = None
        self._async_mode: bool = True
        self._color_match: float = 0.25
        self._mirror: bool = True
        self._enhancer: str = "None"
        self._enhance_strength: float = 0.70

    def set_embedding(self, emb: Optional[np.ndarray]) -> None:
        with self._lock:
            self._embedding = emb

    def set_options(
        self,
        async_mode: bool,
        color_match: float,
        mirror: bool,
        enhancer: str = "None",
        enhance_strength: float = 0.70,
    ) -> None:
        with self._lock:
            self._async_mode = async_mode
            self._color_match = color_match
            self._mirror = mirror
            self._enhancer = enhancer
            self._enhance_strength = enhance_strength

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error.emit(f"无法打开摄像头 Camera {self.camera_index}，请检查系统摄像头权限")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        settings = RenderSettings(
            enhancer="None",
            enhance_strength=0.70,
            color_match=0.25,
            temporal_smooth=True,
            det_size=320,
        )
        smoother = TemporalKeypointSmoother(min_alpha=0.55, max_alpha=0.95)

        # Background thread for async mode
        swap_q: queue.Queue = queue.Queue(maxsize=1)
        latest_swapped = [None]
        swap_stop = threading.Event()

        def _bg_swap():
            while not swap_stop.is_set():
                try:
                    f, emb, cm, enh, enh_s = swap_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if emb is None:
                    latest_swapped[0] = None
                    continue
                settings.color_match = cm
                settings.enhancer = enh
                settings.enhance_strength = enh_s
                try:
                    latest_swapped[0] = self.engine.swap_frame(f, emb, settings, smoother=smoother)
                except Exception:
                    pass

        bg_thread = threading.Thread(target=_bg_swap, daemon=True)
        bg_thread.start()

        t_prev = time.perf_counter()
        cnt = 0
        fps = 0.0
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                with self._lock:
                    emb = self._embedding
                    async_mode = self._async_mode
                    cm = self._color_match
                    mirror = self._mirror
                    enh = self._enhancer
                    enh_s = self._enhance_strength
                if mirror:
                    frame = cv2.flip(frame, 1)

                if emb is None:
                    out = frame
                    latest_swapped[0] = None
                elif async_mode:
                    try:
                        if swap_q.full():
                            swap_q.get_nowait()
                        swap_q.put_nowait((frame.copy(), emb, cm, enh, enh_s))
                    except Exception:
                        pass
                    out = latest_swapped[0] if latest_swapped[0] is not None else frame
                else:
                    settings.color_match = cm
                    settings.enhancer = enh
                    settings.enhance_strength = enh_s
                    try:
                        out = self.engine.swap_frame(frame, emb, settings, smoother=smoother)
                    except Exception:
                        out = frame

                cnt += 1
                now = time.perf_counter()
                if now - t_prev >= 0.5:
                    fps = cnt / (now - t_prev)
                    cnt = 0
                    t_prev = now
                self.frameReady.emit(out, fps)
        finally:
            swap_stop.set()
            bg_thread.join(timeout=2)
            cap.release()


class FaceCraftWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("换脸工坊 · FaceCraft Studio (Apple Silicon)")
        self.resize(1220, 820)
        self.setMinimumSize(1040, 720)

        self.engine = FaceCraftEngine()
        self.characters: Dict[str, CharacterProfile] = {}
        self.char_order: List[str] = []
        self.segments: List[TimelineSegment] = []

        self.target_path: Optional[str] = None
        self.target_is_image: bool = False
        self.video_info: Dict[str, float] = {}
        self.current_raw_frame: Optional[np.ndarray] = None
        self.current_swapped_frame: Optional[np.ndarray] = None
        self.output_dir: str = DEFAULT_OUTPUT_DIR
        self.last_output_path: Optional[str] = None

        self.render_worker: Optional[_RenderWorker] = None
        self.webcam_worker: Optional[_WebcamWorker] = None

        self._build_ui()
        self._apply_preset("balanced")

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("card")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        lbl = QLabel(title)
        lbl.setObjectName("cardTitle")
        lay.addWidget(lbl)
        return frame, lay

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("换脸工坊 · FaceCraft Studio")
        title.setStyleSheet("font-size: 18px; font-weight: 800; color: #f8fafc;")
        sub = QLabel("Apple Silicon GPU 加速 · 多图高还原融合 · 一段口播分镜多人一遍出片")
        sub.setObjectName("muted")
        hdr.addWidget(title)
        hdr.addSpacing(12)
        hdr.addWidget(sub)
        hdr.addStretch(1)
        root.addLayout(hdr)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        self.tabs.addTab(self._build_studio_tab(), "🎬 视频 / 图片换脸（出片推荐）")
        self.tabs.addTab(self._build_webcam_tab(), "📷 实时摄像头预览")

    # ── Tab 1: Creator Video/Image Studio ────────────────────────────────

    def _build_studio_tab(self) -> QWidget:
        page = QWidget()
        main_lay = QHBoxLayout(page)
        main_lay.setContentsMargins(10, 10, 10, 10)
        main_lay.setSpacing(12)

        # Left sidebar (380px): Characters + Quality settings
        left = QVBoxLayout()
        left.setSpacing(10)

        # Card 1: Character library
        c_char, l_char = self._card("① 角色库（我想变成谁）")
        hint = QLabel("支持添加多个角色（分镜切脸用）；同一角色可加多张不同角度照片提升还原度")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        l_char.addWidget(hint)

        self.list_chars = QListWidget()
        self.list_chars.setMinimumHeight(145)
        self.list_chars.currentRowChanged.connect(self._on_char_selected)
        l_char.addWidget(self.list_chars)

        char_btns = QHBoxLayout()
        btn_add_char = QPushButton("+ 添加角色")
        btn_add_char.clicked.connect(lambda: self._on_add_character(False))
        btn_add_photo = QPushButton("+ 加参考图")
        btn_add_photo.setObjectName("secondary")
        btn_add_photo.setToolTip("给选中的角色追加不同角度的照片，融合特征让转头更像")
        btn_add_photo.clicked.connect(self._on_add_ref_photo)
        btn_del_char = QPushButton("删除")
        btn_del_char.setObjectName("secondary")
        btn_del_char.clicked.connect(self._on_delete_character)
        char_btns.addWidget(btn_add_char)
        char_btns.addWidget(btn_add_photo)
        char_btns.addWidget(btn_del_char)
        l_char.addLayout(char_btns)
        left.addWidget(c_char)

        # Card 2: Quality & Likeness controls
        c_set, l_set = self._card("② 画质与还原度调节")
        preset_row = QHBoxLayout()
        b_bal = QPushButton("高还原·推荐")
        b_bal.setObjectName("preset")
        b_bal.clicked.connect(lambda: self._apply_preset("balanced"))
        b_hq = QPushButton("极致高清")
        b_hq.setObjectName("preset")
        b_hq.clicked.connect(lambda: self._apply_preset("hq"))
        b_fast = QPushButton("极速模式")
        b_fast.setObjectName("preset")
        b_fast.clicked.connect(lambda: self._apply_preset("fast"))
        preset_row.addWidget(b_bal)
        preset_row.addWidget(b_hq)
        preset_row.addWidget(b_fast)
        l_set.addLayout(preset_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(7)

        grid.addWidget(QLabel("高清修复模型"), 0, 0)
        self.cb_enhancer = QComboBox()
        self.cb_enhancer.addItems(self.engine.available_enhancers())
        grid.addWidget(self.cb_enhancer, 0, 1, 1, 2)

        self.s_enh, self.lbl_enh = self._add_slider_row(grid, 1, "清晰度融合", 0, 100, 72, "%",
                                                        "70% 左右最平衡：既高清又不改变原图五官神态")
        self.s_color, self.lbl_color = self._add_slider_row(grid, 2, "肤色光照匹配", 0, 100, 35, "%",
                                                            "让换上的脸自动贴合原视频的光线和肤色")
        self.s_mouth, self.lbl_mouth = self._add_slider_row(grid, 3, "保留原嘴型", 0, 100, 0, "%",
                                                            "口播说话多时可拉到 15–25% 防止糊嘴")
        self.s_sharp, self.lbl_sharp = self._add_slider_row(grid, 4, "细节锐化", 0, 100, 0, "%",
                                                            "轻度增强五官锐度")
        l_set.addLayout(grid)

        self.chk_smooth = QCheckBox("人脸防抖平滑（减少视频帧间微抖）")
        self.chk_smooth.setChecked(True)
        self.chk_cap30 = QCheckBox("自动 30fps 导出（60fps 素材耗时减半，推荐）")
        self.chk_cap30.setChecked(True)
        self.chk_audio = QCheckBox("保留原视频声音")
        self.chk_audio.setChecked(True)
        l_set.addWidget(self.chk_smooth)
        l_set.addWidget(self.chk_cap30)
        l_set.addWidget(self.chk_audio)
        left.addWidget(c_set, 1)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(385)
        main_lay.addWidget(left_widget)

        # Right stage: Preview + Multi-face Timeline + Export
        right = QVBoxLayout()
        right.setSpacing(10)

        # Card 3: Source media & Preview
        c_prev, l_prev = self._card("③ 素材与实时效果预览（渲染前先看效果）")
        top_bar = QHBoxLayout()
        self.btn_pick_target = QPushButton("选择视频或图片")
        self.btn_pick_target.clicked.connect(self._on_pick_target)
        self.lbl_target_info = QLabel("尚未选择素材")
        self.lbl_target_info.setObjectName("muted")
        self.btn_refresh_prev = QPushButton("刷新预览")
        self.btn_refresh_prev.setObjectName("secondary")
        self.btn_refresh_prev.clicked.connect(self._refresh_preview)
        self.btn_compare = QPushButton("按住对比原图")
        self.btn_compare.setObjectName("secondary")
        self.btn_compare.pressed.connect(self._show_original_preview)
        self.btn_compare.released.connect(self._show_swapped_preview)
        top_bar.addWidget(self.btn_pick_target)
        top_bar.addWidget(self.lbl_target_info, 1)
        top_bar.addWidget(self.btn_refresh_prev)
        top_bar.addWidget(self.btn_compare)
        l_prev.addLayout(top_bar)

        self.preview_label = QLabel("先在左侧添加角色照片，再点上方「选择视频或图片」")
        self.preview_label.setObjectName("previewBox")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(290)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        l_prev.addWidget(self.preview_label, 1)

        scrub_row = QHBoxLayout()
        self.lbl_cur_time = QLabel("00:00.0")
        self.s_timeline = QSlider(Qt.Orientation.Horizontal)
        self.s_timeline.setRange(0, 0)
        self.s_timeline.setEnabled(False)
        self.s_timeline.sliderReleased.connect(self._on_scrub_released)
        self.s_timeline.valueChanged.connect(self._on_scrub_moved)
        self.lbl_tot_time = QLabel("00:00.0")
        scrub_row.addWidget(self.lbl_cur_time)
        scrub_row.addWidget(self.s_timeline, 1)
        scrub_row.addWidget(self.lbl_tot_time)
        l_prev.addLayout(scrub_row)
        right.addWidget(c_prev, 3)

        # Card 4: Multi-character timeline segments
        c_tl, l_tl = self._card("④ 分镜换脸时间轴（可选：一段口播换多人，一遍出片）")
        tl_bar = QHBoxLayout()
        self.cb_seg_char = QComboBox()
        self.cb_seg_char.addItem("原脸（不换脸）", None)
        btn_add_seg = QPushButton("+ 从当前时间点切入该角色")
        btn_add_seg.clicked.connect(self._on_add_segment)
        btn_del_seg = QPushButton("删除选中")
        btn_del_seg.setObjectName("secondary")
        btn_del_seg.clicked.connect(self._on_delete_segment)
        btn_clear_seg = QPushButton("清空（全程单人）")
        btn_clear_seg.setObjectName("secondary")
        btn_clear_seg.clicked.connect(self._on_clear_segments)
        tl_bar.addWidget(self.cb_seg_char, 1)
        tl_bar.addWidget(btn_add_seg)
        tl_bar.addWidget(btn_del_seg)
        tl_bar.addWidget(btn_clear_seg)
        l_tl.addLayout(tl_bar)

        self.list_segments = QListWidget()
        self.list_segments.setMaximumHeight(95)
        l_tl.addWidget(self.list_segments)
        right.addWidget(c_tl)

        # Card 5: Export
        c_exp, l_exp = self._card("⑤ 导出成片")
        dir_row = QHBoxLayout()
        self.lbl_out_dir = QLabel(f"保存位置：{self.output_dir}")
        self.lbl_out_dir.setObjectName("muted")
        btn_chg_dir = QPushButton("更改目录")
        btn_chg_dir.setObjectName("secondary")
        btn_chg_dir.clicked.connect(self._on_change_out_dir)
        dir_row.addWidget(self.lbl_out_dir, 1)
        dir_row.addWidget(btn_chg_dir)
        l_exp.addLayout(dir_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        l_exp.addWidget(self.progress_bar)

        self.lbl_status = QLabel("就绪")
        self.lbl_status.setObjectName("muted")
        l_exp.addWidget(self.lbl_status)

        act_row = QHBoxLayout()
        self.btn_render = QPushButton("开始渲染导出")
        self.btn_render.setMinimumHeight(38)
        self.btn_render.clicked.connect(self._on_start_render)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setMinimumHeight(38)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel_render)
        self.btn_reveal = QPushButton("在访达中显示")
        self.btn_reveal.setObjectName("success")
        self.btn_reveal.setMinimumHeight(38)
        self.btn_reveal.setEnabled(False)
        self.btn_reveal.clicked.connect(self._on_reveal_output)
        self.btn_play = QPushButton("立即播放")
        self.btn_play.setObjectName("secondary")
        self.btn_play.setMinimumHeight(38)
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._on_play_output)
        act_row.addWidget(self.btn_render, 2)
        act_row.addWidget(self.btn_cancel, 1)
        act_row.addWidget(self.btn_reveal, 1)
        act_row.addWidget(self.btn_play, 1)
        l_exp.addLayout(act_row)
        right.addWidget(c_exp)

        main_lay.addLayout(right, 1)
        return page

    def _add_slider_row(self, grid: QGridLayout, row: int, label: str, lo: int, hi: int, val: int, unit: str, tip: str):
        lbl = QLabel(label)
        lbl.setToolTip(tip)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.setToolTip(tip)
        v_lbl = QLabel(f"{val}{unit}")
        v_lbl.setFixedWidth(40)
        v_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        s.valueChanged.connect(lambda v, l=v_lbl, u=unit: l.setText(f"{v}{u}"))
        grid.addWidget(lbl, row, 0)
        grid.addWidget(s, row, 1)
        grid.addWidget(v_lbl, row, 2)
        return s, v_lbl

    # ── Tab 2: Live Webcam ───────────────────────────────────────────────

    def _build_webcam_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # Row 1: Camera, Character selector, Add Character, Toggle
        bar1 = QHBoxLayout()
        bar1.addWidget(QLabel("摄像头："))
        self.cam_indices, self.cam_names = list_macos_cameras()
        self.cb_cam = QComboBox()
        self.cb_cam.addItems(self.cam_names)
        bar1.addWidget(self.cb_cam, 1)

        bar1.addWidget(QLabel("实时切脸："))
        self.cb_live_char = QComboBox()
        self.cb_live_char.addItem("原脸（不换脸）", None)
        self.cb_live_char.currentIndexChanged.connect(self._on_live_char_changed)
        bar1.addWidget(self.cb_live_char, 1)

        btn_live_add_char = QPushButton("+ 添加角色")
        btn_live_add_char.setObjectName("secondary")
        btn_live_add_char.setToolTip("直接添加人脸照片并立即切换到该角色（与左侧角色库同步）")
        btn_live_add_char.clicked.connect(lambda: self._on_add_character(True))
        bar1.addWidget(btn_live_add_char)

        self.btn_cam_toggle = QPushButton("开启摄像头")
        self.btn_cam_toggle.clicked.connect(self._on_toggle_webcam)
        bar1.addWidget(self.btn_cam_toggle)
        lay.addLayout(bar1)

        # Row 2: Live clarity / enhancer + blend strength + async/mirror
        bar2 = QHBoxLayout()
        bar2.addWidget(QLabel("清晰度："))
        self.cb_live_enhancer = QComboBox()
        enh_labels = {
            "None": "流畅（关闭修复）",
            "GPEN-512": "高清（GPEN-512）",
            "GFPGAN-1024": "极致（GFPGAN-1024）",
        }
        for key in self.engine.available_enhancers():
            self.cb_live_enhancer.addItem(enh_labels.get(key, key), key)
        self.cb_live_enhancer.setToolTip("开启高清修复可显著提升画质，但会降低实时帧率")
        self.cb_live_enhancer.currentIndexChanged.connect(self._sync_live_options)
        bar2.addWidget(self.cb_live_enhancer)

        bar2.addSpacing(8)
        bar2.addWidget(QLabel("修复融合："))
        self.s_live_enh = QSlider(Qt.Orientation.Horizontal)
        self.s_live_enh.setRange(0, 100)
        self.s_live_enh.setValue(70)
        self.s_live_enh.setFixedWidth(130)
        self.s_live_enh.setEnabled(False)
        self.s_live_enh.setToolTip("70% 左右最平衡：清晰同时保留原图神态")
        self.lbl_live_enh = QLabel("70%")
        self.lbl_live_enh.setFixedWidth(36)
        self.s_live_enh.valueChanged.connect(self._on_live_enh_changed)
        bar2.addWidget(self.s_live_enh)
        bar2.addWidget(self.lbl_live_enh)

        bar2.addStretch(1)
        self.chk_live_async = QCheckBox("流畅优先（异步）")
        self.chk_live_async.setChecked(True)
        self.chk_live_async.setToolTip("勾选：画面流畅但晃动略滞后；取消：五官紧贴但帧率较低")
        self.chk_live_async.toggled.connect(self._sync_live_options)
        bar2.addWidget(self.chk_live_async)

        self.chk_live_mirror = QCheckBox("镜像")
        self.chk_live_mirror.setChecked(True)
        self.chk_live_mirror.toggled.connect(self._sync_live_options)
        bar2.addWidget(self.chk_live_mirror)
        lay.addLayout(bar2)

        self.live_preview = QLabel("点击右上角「开启摄像头」开始实时预览，过程中可随时添加或切换角色、调节清晰度")
        self.live_preview.setObjectName("previewBox")
        self.live_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self.live_preview, 1)

        self.lbl_live_fps = QLabel("未启动")
        self.lbl_live_fps.setObjectName("muted")
        lay.addWidget(self.lbl_live_fps)
        return page

    # ── Settings & Presets ───────────────────────────────────────────────

    def _apply_preset(self, mode: str) -> None:
        avail = [self.cb_enhancer.itemText(i) for i in range(self.cb_enhancer.count())]
        if mode == "balanced":
            if "GPEN-512" in avail:
                self.cb_enhancer.setCurrentText("GPEN-512")
            elif "GFPGAN-1024" in avail:
                self.cb_enhancer.setCurrentText("GFPGAN-1024")
            self.s_enh.setValue(72)
            self.s_color.setValue(35)
            self.s_mouth.setValue(0)
            self.s_sharp.setValue(0)
            self.chk_smooth.setChecked(True)
            self.chk_cap30.setChecked(True)
        elif mode == "hq":
            if "GFPGAN-1024" in avail:
                self.cb_enhancer.setCurrentText("GFPGAN-1024")
            elif "GPEN-512" in avail:
                self.cb_enhancer.setCurrentText("GPEN-512")
            self.s_enh.setValue(78)
            self.s_color.setValue(35)
            self.s_sharp.setValue(10)
            self.chk_smooth.setChecked(True)
        elif mode == "fast":
            self.cb_enhancer.setCurrentText("None")
            self.s_enh.setValue(0)
            self.s_color.setValue(25)
            self.s_mouth.setValue(0)
            self.chk_cap30.setChecked(True)

    def _collect_settings(self) -> RenderSettings:
        return RenderSettings(
            enhancer=self.cb_enhancer.currentText(),
            enhance_strength=self.s_enh.value() / 100.0,
            color_match=self.s_color.value() / 100.0,
            mouth_mask=self.s_mouth.value() / 100.0,
            sharpness=self.s_sharp.value() / 100.0,
            temporal_smooth=self.chk_smooth.isChecked(),
            cap_30fps=self.chk_cap30.isChecked(),
            det_size=640,
            keep_audio=self.chk_audio.isChecked(),
        )

    # ── Character Library Handlers ───────────────────────────────────────

    def _sync_character_dropdowns(self) -> None:
        cur_seg = self.cb_seg_char.currentData()
        cur_live = self.cb_live_char.currentData()
        for cb in (self.cb_seg_char, self.cb_live_char):
            cb.blockSignals(True)
            cb.clear()
            for cid in self.char_order:
                c = self.characters[cid]
                cb.addItem(f"{c.name} ({len(c.image_paths)}张图)", cid)
            cb.addItem("原脸（不换脸）", None)
            cb.blockSignals(False)
        # Restore selections
        idx_s = self.cb_seg_char.findData(cur_seg)
        if idx_s >= 0:
            self.cb_seg_char.setCurrentIndex(idx_s)
        idx_l = self.cb_live_char.findData(cur_live)
        if idx_l >= 0:
            self.cb_live_char.setCurrentIndex(idx_l)
        self._on_live_char_changed()

    def _refresh_char_list(self) -> None:
        cur_row = self.list_chars.currentRow()
        self.list_chars.clear()
        for cid in self.char_order:
            c = self.characters[cid]
            item = QListWidgetItem(f"{c.name}   ·   {len(c.image_paths)} 张参考图")
            if c.thumb_bgr is not None:
                item.setIcon(bgr_to_qpixmap(c.thumb_bgr, 48, 48))
            item.setData(Qt.ItemDataRole.UserRole, cid)
            self.list_chars.addItem(item)
        if self.char_order:
            self.list_chars.setCurrentRow(max(0, min(cur_row, len(self.char_order) - 1)))
        self._sync_character_dropdowns()

    def _selected_char_id(self) -> Optional[str]:
        item = self.list_chars.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else (self.char_order[0] if self.char_order else None)

    def _on_add_character(self, select_for_live: bool = False) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择角色人脸照片（可一次多选同一人的不同角度）", "",
            "图片 (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not paths:
            return
        default_name = os.path.splitext(os.path.basename(paths[0]))[0]
        name, ok = QInputDialog.getText(self, "角色名称", "给这个角色起个名字（如 彭于晏 / 黄仁勋）：", text=default_name)
        if not ok or not name.strip():
            return
        self.lbl_status.setText("正在提取人脸特征...")
        QApplication.processEvents()
        emb, thumb = self.engine.build_character_embedding(paths)
        if emb is None:
            QMessageBox.warning(self, "未检测到人脸", "所选照片中没有检测到清晰的人脸，请换一张正脸照。")
            self.lbl_status.setText("就绪")
            return
        char = CharacterProfile(name=name.strip(), image_paths=list(paths), embedding=emb, thumb_bgr=thumb)
        self.characters[char.id] = char
        self.char_order.append(char.id)
        self._refresh_char_list()
        self.list_chars.setCurrentRow(len(self.char_order) - 1)
        if select_for_live:
            idx_l = self.cb_live_char.findData(char.id)
            if idx_l >= 0:
                self.cb_live_char.setCurrentIndex(idx_l)
        self.lbl_status.setText(f"已添加角色：{char.name}（{len(paths)} 张参考图融合）")
        self._refresh_preview()

    def _on_add_ref_photo(self) -> None:
        cid = self._selected_char_id()
        if not cid or cid not in self.characters:
            QMessageBox.information(self, "提示", "请先添加一个角色，再给它追加参考图。")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "追加同一角色的其他角度照片", "",
            "图片 (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not paths:
            return
        c = self.characters[cid]
        merged = c.image_paths + [p for p in paths if p not in c.image_paths]
        self.lbl_status.setText(f"正在融合 {c.name} 的 {len(merged)} 张参考图特征...")
        QApplication.processEvents()
        emb, thumb = self.engine.build_character_embedding(merged)
        if emb is None:
            QMessageBox.warning(self, "提示", "追加的照片未能提取出有效人脸。")
            return
        c.image_paths = merged
        c.embedding = emb
        if thumb is not None:
            c.thumb_bgr = thumb
        self._refresh_char_list()
        self.lbl_status.setText(f"已更新 {c.name}，当前融合 {len(merged)} 张参考图")
        self._refresh_preview()

    def _on_delete_character(self) -> None:
        cid = self._selected_char_id()
        if not cid:
            return
        self.characters.pop(cid, None)
        if cid in self.char_order:
            self.char_order.remove(cid)
        self.segments = [s for s in self.segments if s.character_id != cid]
        self._refresh_char_list()
        self._refresh_segment_list()

    def _on_char_selected(self, _row: int) -> None:
        if self.target_path and not self.segments:
            self._refresh_preview()

    # ── Target Media & Scrubber ──────────────────────────────────────────

    def _on_pick_target(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择你的原视频或图片", "",
            "视频与图片 (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.png *.jpg *.jpeg *.webp)",
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        self.target_path = path
        self.target_is_image = ext in IMAGE_EXTS
        self.segments.clear()
        self._refresh_segment_list()

        if self.target_is_image:
            self.current_raw_frame = imread_unicode(path)
            h, w = self.current_raw_frame.shape[:2] if self.current_raw_frame is not None else (0, 0)
            self.lbl_target_info.setText(f"{os.path.basename(path)}  ·  图片 {w}×{h}")
            self.s_timeline.setEnabled(False)
            self.s_timeline.setRange(0, 0)
            self.lbl_cur_time.setText("00:00.0")
            self.lbl_tot_time.setText("00:00.0")
        else:
            self.video_info = probe_video_info(path)
            w = int(self.video_info.get("width", 0))
            h = int(self.video_info.get("height", 0))
            fps = self.video_info.get("fps", 30.0)
            dur = self.video_info.get("duration", 0.0)
            self.lbl_target_info.setText(f"{os.path.basename(path)}  ·  {w}×{h} · {fps:.0f}fps · {fmt_time(dur)}")
            self.s_timeline.setEnabled(True)
            self.s_timeline.setRange(0, max(1, int(dur * 10)))  # 0.1s resolution
            self.s_timeline.setValue(0)
            self.lbl_cur_time.setText("00:00.0")
            self.lbl_tot_time.setText(fmt_time(dur))
            self._load_video_frame_at(0.0)

        self._refresh_preview()

    def _load_video_frame_at(self, sec: float) -> None:
        if not self.target_path or self.target_is_image:
            return
        cap = cv2.VideoCapture(self.target_path)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, sec * 1000.0))
            ok, frame = cap.read()
            if ok and frame is not None:
                self.current_raw_frame = frame
            cap.release()

    def _on_scrub_moved(self, val: int) -> None:
        self.lbl_cur_time.setText(fmt_time(val / 10.0))

    def _on_scrub_released(self) -> None:
        sec = self.s_timeline.value() / 10.0
        self._load_video_frame_at(sec)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if self.current_raw_frame is None:
            return
        t_sec = self.s_timeline.value() / 10.0 if not self.target_is_image else 0.0
        cid = self.engine.resolve_character_at_time(t_sec, self.segments, self._selected_char_id())
        char = self.characters.get(cid) if cid else None
        if char is not None and char.embedding is not None:
            self.lbl_status.setText(f"正在生成预览（{char.name}）...")
            QApplication.processEvents()
            try:
                self.current_swapped_frame = self.engine.swap_frame(
                    self.current_raw_frame, char.embedding, self._collect_settings()
                )
                self._show_swapped_preview()
                self.lbl_status.setText(f"预览已更新 · 当前角色：{char.name}（按住「按住对比原图」可对比）")
            except Exception as exc:
                self.lbl_status.setText(f"预览出错：{exc}")
        else:
            self.current_swapped_frame = self.current_raw_frame
            self._show_original_preview()
            self.lbl_status.setText("当前时间点：原脸（不换脸）")

    def _show_original_preview(self) -> None:
        if self.current_raw_frame is not None:
            pm = bgr_to_qpixmap(self.current_raw_frame, self.preview_label.width() - 8, self.preview_label.height() - 8)
            self.preview_label.setPixmap(pm)

    def _show_swapped_preview(self) -> None:
        img = self.current_swapped_frame if self.current_swapped_frame is not None else self.current_raw_frame
        if img is not None:
            pm = bgr_to_qpixmap(img, self.preview_label.width() - 8, self.preview_label.height() - 8)
            self.preview_label.setPixmap(pm)

    # ── Multi-Character Timeline Segments ────────────────────────────────

    def _normalize_segments(self) -> None:
        self.segments.sort(key=lambda s: s.start_sec)
        # Deduplicate same start_sec (keep latest)
        dedup: List[TimelineSegment] = []
        for s in self.segments:
            if dedup and abs(dedup[-1].start_sec - s.start_sec) < 0.05:
                dedup[-1] = s
            else:
                dedup.append(s)
        for i in range(len(dedup)):
            dedup[i].end_sec = dedup[i + 1].start_sec if i + 1 < len(dedup) else 0.0
        self.segments = dedup

    def _refresh_segment_list(self) -> None:
        self._normalize_segments()
        self.list_segments.clear()
        if not self.segments:
            self.list_segments.addItem("（未设置分镜：将全程使用左侧选中的单个角色）")
            return
        for seg in self.segments:
            cname = self.characters[seg.character_id].name if (seg.character_id and seg.character_id in self.characters) else "原脸（不换脸）"
            end_str = fmt_time(seg.end_sec) if seg.end_sec > 0 else "结尾"
            self.list_segments.addItem(f"{fmt_time(seg.start_sec)}  →  {end_str}    |    {cname}")

    def _on_add_segment(self) -> None:
        if not self.target_path or self.target_is_image:
            QMessageBox.information(self, "提示", "分镜时间轴仅用于视频，请先选择一个视频素材。")
            return
        sec = self.s_timeline.value() / 10.0
        cid = self.cb_seg_char.currentData()
        self.segments.append(TimelineSegment(start_sec=sec, end_sec=0.0, character_id=cid))
        self._refresh_segment_list()
        self._refresh_preview()

    def _on_delete_segment(self) -> None:
        row = self.list_segments.currentRow()
        if 0 <= row < len(self.segments):
            self.segments.pop(row)
            self._refresh_segment_list()
            self._refresh_preview()

    def _on_clear_segments(self) -> None:
        self.segments.clear()
        self._refresh_segment_list()
        self._refresh_preview()

    # ── Rendering ────────────────────────────────────────────────────────

    def _on_change_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择成片保存目录", self.output_dir)
        if d:
            self.output_dir = d
            self.lbl_out_dir.setText(f"保存位置：{self.output_dir}")

    def _on_start_render(self) -> None:
        if self.render_worker is not None and self.render_worker.isRunning():
            return
        if not self.target_path:
            QMessageBox.warning(self, "缺少素材", "请先点击「选择视频或图片」加载素材。")
            return
        if not self.segments and not self._selected_char_id():
            QMessageBox.warning(self, "缺少角色", "请先在左侧添加至少一个角色照片。")
            return

        # Stop webcam if active so GPU/CoreML isn't contended
        if self.webcam_worker is not None and self.webcam_worker.isRunning():
            self._on_toggle_webcam()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".png" if self.target_is_image else ".mp4"
        out_path = os.path.join(self.output_dir, f"换脸成片_{ts}{ext}")

        self.btn_render.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_reveal.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("正在初始化模型并启动渲染...")

        self.render_worker = _RenderWorker(
            engine=self.engine,
            target_path=self.target_path,
            is_image_target=self.target_is_image,
            output_path=out_path,
            characters=dict(self.characters),
            segments=list(self.segments),
            default_char_id=self._selected_char_id(),
            settings=self._collect_settings(),
        )
        self.render_worker.progress.connect(self._on_render_progress)
        self.render_worker.finished.connect(self._on_render_finished)
        self.render_worker.start()

    def _on_cancel_render(self) -> None:
        if self.render_worker is not None and self.render_worker.isRunning():
            self.render_worker.cancel()
            self.lbl_status.setText("正在安全取消渲染...")

    def _on_render_progress(self, cur: int, tot: int, fps: float, preview, active_name: str) -> None:
        pct = int(min(100, cur * 100 / max(tot, 1)))
        self.progress_bar.setValue(pct)
        rem = max(0, tot - cur) / max(fps, 0.1)
        self.lbl_status.setText(
            f"渲染中 {cur}/{tot} 帧 ({pct}%) · {fps:.1f} 帧/秒 · 约剩 {fmt_time(rem)} · 当前：{active_name}"
        )
        if preview is not None:
            pm = bgr_to_qpixmap(preview, self.preview_label.width() - 8, self.preview_label.height() - 8)
            self.preview_label.setPixmap(pm)

    def _on_render_finished(self, ok: bool, msg: str, out_path: str) -> None:
        self.btn_render.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if ok and out_path:
            self.last_output_path = out_path
            self.progress_bar.setValue(100)
            self.btn_reveal.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.lbl_status.setText(f"✅ {msg} → {out_path}")
        else:
            self.lbl_status.setText(f"⚠️ {msg}")

    def _on_reveal_output(self) -> None:
        if self.last_output_path and os.path.exists(self.last_output_path):
            subprocess.Popen(["open", "-R", self.last_output_path])

    def _on_play_output(self) -> None:
        if self.last_output_path and os.path.exists(self.last_output_path):
            subprocess.Popen(["open", self.last_output_path])

    # ── Webcam Handlers ──────────────────────────────────────────────────

    def _on_toggle_webcam(self) -> None:
        if self.webcam_worker is not None and self.webcam_worker.isRunning():
            self.webcam_worker.stop()
            self.webcam_worker.wait(3000)
            self.webcam_worker = None
            self.btn_cam_toggle.setText("开启摄像头")
            self.btn_cam_toggle.setObjectName("")
            self.btn_cam_toggle.style().polish(self.btn_cam_toggle)
            self.lbl_live_fps.setText("已停止")
            return

        idx = self.cb_cam.currentIndex()
        cam_id = self.cam_indices[idx] if 0 <= idx < len(self.cam_indices) else 0
        self.webcam_worker = _WebcamWorker(self.engine, cam_id, 640, 480)
        self._on_live_char_changed()
        self._sync_live_options()
        self.webcam_worker.frameReady.connect(self._on_live_frame)
        self.webcam_worker.error.connect(self._on_live_error)
        self.webcam_worker.start()
        self.btn_cam_toggle.setText("停止摄像头")
        self.btn_cam_toggle.setObjectName("danger")
        self.btn_cam_toggle.style().polish(self.btn_cam_toggle)

    def _on_live_char_changed(self) -> None:
        if self.webcam_worker is None:
            return
        cid = self.cb_live_char.currentData()
        char = self.characters.get(cid) if cid else None
        self.webcam_worker.set_embedding(char.embedding if char else None)

    def _on_live_enh_changed(self, val: int) -> None:
        self.lbl_live_enh.setText(f"{val}%")
        self._sync_live_options()

    def _sync_live_options(self) -> None:
        enh = self.cb_live_enhancer.currentData() or "None"
        self.s_live_enh.setEnabled(enh != "None")
        if self.webcam_worker is None:
            return
        self.webcam_worker.set_options(
            async_mode=self.chk_live_async.isChecked(),
            color_match=0.25,
            mirror=self.chk_live_mirror.isChecked(),
            enhancer=enh,
            enhance_strength=self.s_live_enh.value() / 100.0,
        )

    def _on_live_frame(self, frame, fps: float) -> None:
        pm = bgr_to_qpixmap(frame, self.live_preview.width() - 8, self.live_preview.height() - 8)
        self.live_preview.setPixmap(pm)
        self.lbl_live_fps.setText(f"实时预览运行中 · {fps:.1f} FPS")

    def _on_live_error(self, msg: str) -> None:
        self.webcam_worker = None
        self.btn_cam_toggle.setText("开启摄像头")
        self.btn_cam_toggle.setObjectName("")
        self.btn_cam_toggle.style().polish(self.btn_cam_toggle)
        self.lbl_live_fps.setText(f"⚠️ {msg}")
        QMessageBox.warning(self, "摄像头提示", msg)

    def closeEvent(self, event) -> None:
        if self.webcam_worker is not None and self.webcam_worker.isRunning():
            self.webcam_worker.stop()
            self.webcam_worker.wait(2000)
        if self.render_worker is not None and self.render_worker.isRunning():
            self.render_worker.cancel()
            self.render_worker.wait(3000)
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = FaceCraftWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
