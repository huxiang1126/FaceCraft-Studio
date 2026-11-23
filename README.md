# FaceCraft Studio（换脸工坊）

面向视频创作者的 **Apple Silicon（macOS）** 本地换脸工作台，基于 PySide6 与 ONNX Runtime / CoreML 构建。

## 来源与致谢（Attribution）

本项目在交互架构、稳定性与创作者工作流上受到开源项目 **[hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)** 的启发并兼容其模型权重，底层推理依赖开源人脸分析与修复模型生态（InsightFace `buffalo_l`、`inswapper_128_fp16`、`GPEN-BFR-512`、`GFPGAN-1024`）。

感谢上游开源社区与作者的贡献。本仓库仅包含独立编写的桌面应用与推理封装代码，**不包含任何模型权重文件**。

## 核心改进

1. **自适应五官关键点防抖（不再发飘）**  
   内置 `TemporalKeypointSmoother`，按两眼间距动态衡量头部运动速度：静态或正常口播时自动平滑滤除检测器微抖；快速转头或切镜时即时解除平滑跟上动作，兼顾稳定性与低延迟。
2. **多图身份特征融合（转头更像）**  
   同一个角色支持添加 2–4 张不同角度、光线的参考照，对 512 维人脸特征向量做均值融合与 L2 归一化，在不增加任何渲染耗时的前提下显著提升侧脸与大表情还原度。
3. **单次渲染多角色分镜时间轴（一遍出片）**  
   支持在角色库管理多个人物，并在视频时间轴上标记切换时间点，一段口播换多个人的脸只需渲染导出一次，同时自动保留原视频音轨。
4. **更可控的画质与融合细节**  
   - 脸部内椭圆区域 **LAB 肤色与光照匹配**，降低边缘色差与假面感。
   - 高清修复模型（GPEN-512 / GFPGAN-1024）支持 **0%–100% 融合强度调节**（默认 72%），在清晰度与原图神态之间取得平衡。
   - 支持嘴部区域保留（减少口播糊嘴）与 A/B 按住对比原图。
5. **Apple Silicon 专项优化与实时摄像头预览**  
   - 将动态输入尺寸的人脸检测（`buffalo_l`）固定在稳定的 `CPUExecutionProvider`，换脸与高清修复走 `CoreMLExecutionProvider`（`CPUAndGPU` + 持久化编译缓存），彻底规避 macOS 下 CoreML 动态 shape 报错与 ANE 卡死问题。
   - 视频导出默认使用 macOS `h264_videotoolbox` 硬件编码。
   - 实时摄像头标签页支持真实 macOS AVFoundation 摄像头枚举、直接添加/切换角色、实时切换清晰度与修复融合权重。

## 环境要求

- macOS（推荐 Apple Silicon M1/M2/M3/M4）
- Python 3.10+
- FFmpeg（`brew install ffmpeg`）

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 准备模型文件

在项目根目录创建 `models/` 文件夹（或通过环境变量 `FACECRAFT_MODELS_DIR` 指定已有模型目录），下载并放入以下模型文件：

- **必选换脸模型**：[inswapper_128_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx)
- **可选高清修复模型**：[gfpgan-1024.onnx](https://huggingface.co/hacksider/deep-live-cam/blob/main/gfpgan-1024.onnx) 或 `GPEN-BFR-512.onnx`

（`buffalo_l` 检测与特征提取模型会在首次运行时由 `insightface` 自动下载到 `~/.insightface/models/buffalo_l/`。）

### 3. 启动应用

```bash
python app.py
```

## 合规与伦理声明

本项目仅供学习研究、个人创意视频制作与技术交流使用。使用者须遵守所在地法律法规，在使用他人肖像前应获得合法授权，严禁用于欺诈、冒充、诽谤或任何违法违规用途。
