#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境自检
========

**坐下实测之前先跑这个**，一次把四件事验完：

    1. 依赖装齐没有（opencv / mediapipe / numpy / matplotlib）
    2. 手部模型能不能加载
    3. 摄像头能不能打开、给到多少分辨率
    4. 手部推理耗时（判断这台机器跑不跑得动）

    python tools\\check_env.py
    python tools\\check_env.py --camera 1     # 指定摄像头索引

⚠️ 为什么需要这个工具
   实测踩到过：mediapipe 1.0.1 在 **import 阶段**就硬依赖 matplotlib
   （`tasks/python/vision/drawing_utils.py` 直接 `import matplotlib.pyplot`）。
   少了它，报错是 `ModuleNotFoundError: No module named 'matplotlib'`，
   但**要翻到 mediapipe 内部的 traceback 才看得出来**，看着像是模型或代码的问题。
   把这几项一次性验完，省得在机位前才发现环境缺东西。
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

OK = "✅"
BAD = "❌"
WARN = "⚠️"


def main():
    ap = argparse.ArgumentParser(description="握笔项目环境自检")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-camera", action="store_true", help="跳过摄像头检查")
    args = ap.parse_args()

    print("=" * 60)
    print("握笔项目 · 环境自检")
    print("=" * 60)

    # ---- 1. 依赖 ----
    print("\n【1】依赖包")
    missing = []
    for mod, label in (("cv2", "opencv-python"), ("mediapipe", "mediapipe"),
                       ("numpy", "numpy"), ("matplotlib", "matplotlib")):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  {OK} {label:16s} {ver}")
        except Exception as e:
            print(f"  {BAD} {label:16s} 导入失败：{type(e).__name__}: {e}")
            missing.append(label)
    if missing:
        print(f"\n  缺这些，装一下：")
        print("    envs\\pg\\Scripts\\python.exe -m pip install -i "
              "https://pypi.tuna.tsinghua.edu.cn/simple " + " ".join(missing))
        print("\n  ⚠️ mediapipe 用 --no-deps 装（避开 opencv-contrib 冲突）时，"
              "matplotlib 必须单独补上")
        return 1

    # ---- 2. 模型 ----
    print("\n【2】手部模型")
    import pg_utils as pg
    try:
        model = pg.find_model("hand_landmarker.task")
        print(f"  {OK} {model}")
        print(f"     {os.path.getsize(model) / 1024 / 1024:.1f} MB")
    except SystemExit as e:
        print(f"  {BAD} 找不到模型\n{e}")
        return 1

    # ---- 3. 模型加载 + 推理耗时 ----
    print("\n【3】模型加载与推理")
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    t0 = time.perf_counter()
    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO, num_hands=1))
    print(f"  {OK} 加载成功（{time.perf_counter() - t0:.1f}s）")

    tick, ts = 0, []
    for _ in range(12):
        tick += 33
        img = np.full((720, 1280, 3), 210, dtype=np.uint8)
        t0 = time.perf_counter()
        lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=img), tick)
        ts.append((time.perf_counter() - t0) * 1000)
    avg = sum(ts) / len(ts)
    print(f"  {OK} 单帧推理 {avg:.1f}ms（空图，是成本下限；有手会多一点）")
    print("     采样按 5FPS 算，每帧预算 200ms —— 余量充足")
    lm.close()

    if args.no_camera:
        print("\n（跳过了摄像头检查）")
        return 0

    # ---- 4. 摄像头 ----
    print(f"\n【4】摄像头（索引 {args.camera}）")
    cap = pg.open_camera(args.camera)
    fr = pg.grab_frame(cap)
    if fr is None:
        print(f"  {BAD} 打开成功但读不到画面")
        cap.release()
        return 1
    h, w = fr.shape[:2]
    print(f"  {OK} 可用，分辨率 {w}x{h}")
    cap.release()

    print("\n" + "=" * 60)
    if w < 1280:
        print(f"{WARN} 分辨率只有 {w}x{h}。握笔要把手拍大，"
              f"低分辨率会让手部像素不够 —— 机位得拉得更近。")
    print(f"{OK} 环境没问题。下一步：")
    print("     envs\\pg\\Scripts\\python.exe hand_probe.py --camera "
          f"{args.camera}")
    print("   或直接双击「实测握笔.bat」")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
