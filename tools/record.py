#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
录制握笔视频（为"盖子前压"这种机位准备）
=========================================

⚠️ 为什么需要单独一个"录制"工具，而不是直接用实时预览
----------------------------------------------------
笔记本要实现俯视，只能**把盖子朝键盘方向压下去**（约 55~65°），
让屏幕顶部那颗摄像头朝下看桌面。**代价是屏幕根本没法看** ——
所以"一边看预览一边按键录样本"这条路走不通。

这个工具就是补这个缺：**盲录一段**，录完把盖子抬回来再看。
（录完用 `hand_probe.py --video 文件` 分析。）

⚠️ 两个要点
   1. **用蜂鸣声报时**。屏幕看不见，只能靠听：开始/结束各响一声。
   2. **同时存一张快照 jpg**。录完抬起盖子先看这张图确认手在不在画面里，
      比翻视频快。

用法
----
    python tools\\record.py                    # 录 30 秒
    python tools\\record.py --seconds 60       # 录 60 秒
    python tools\\record.py --camera 1         # 指定摄像头
    python tools\\record.py --no-beep          # 关掉蜂鸣
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import pg_utils as pg   # noqa: E402

OUT_DIR = os.path.join(ROOT, "hand_probe")


def beep(freq=880, ms=250):
    """屏幕看不见时靠声音报时。Windows 上用 winsound；其他平台静默跳过。"""
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="录一段握笔视频")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-beep", action="store_true")
    args = ap.parse_args()

    beep_on = not args.no_beep
    # 分辨率/索引都从探测配置读（tools/probe_camera.py 写）——
    # 保证「手宽 ≥200px」这条门槛在录制的视频里含义一致
    cap = pg.open_camera_auto(idx=args.camera)
    fr = pg.grab_frame(cap)
    if fr is None:
        sys.exit("摄像头打开成功但读不到画面")
    h, w = fr.shape[:2]
    print(f"[录制] 采集分辨率 {w}x{h}（来自探测配置）")

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    vpath = os.path.join(OUT_DIR, f"record_{stamp}.mp4")
    jpath = os.path.join(OUT_DIR, f"record_{stamp}_first.jpg")

    import cv2
    vw = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    if not vw.isOpened():
        sys.exit("打不开视频写入器（mp4v 编码不可用？）")

    print("=" * 56)
    print(f"录制 {args.seconds:.0f} 秒   分辨率 {w}x{h}")
    print(f"输出：{vpath}")
    print("=" * 56)
    print("现在把笔记本盖子压到约 55~65°（摄像头朝下看桌面），摆好位置。")
    print("听到**两声蜂鸣**就开始录，录完再响两声。")
    print("（屏幕这时候看不见，正常 —— 录完把盖子抬回来再看结果。）")
    print()
    for i in (3, 2, 1):
        print(f"  {i} …", flush=True)
        if beep_on:
            beep(660, 150)
        time.sleep(1.0)

    if beep_on:
        beep(1200, 400)
    print("▶ 开始录制", flush=True)

    t0 = time.perf_counter()
    n = 0
    first_saved = False
    while time.perf_counter() - t0 < args.seconds:
        f = pg.grab_frame(cap)
        if f is None:
            continue
        if f.shape[0] != h or f.shape[1] != w:
            f = cv2.resize(f, (w, h))
        vw.write(f)
        if not first_saved:
            cv2.imwrite(jpath, f)
            first_saved = True
        n += 1
        done = time.perf_counter() - t0
        barlen = 28
        filled = int(barlen * done / args.seconds)
        print(f"\r  [{'#' * filled}{'.' * (barlen - filled)}] "
              f"{done:4.1f}/{args.seconds:.0f}s  {n} 帧", end="", flush=True)
    print()

    vw.release()
    cap.release()
    if beep_on:
        beep(1200, 200)
        time.sleep(0.15)
        beep(1200, 400)
    print("■ 录制结束")
    print()
    print(f"  视频 ：{vpath}   （{n} 帧，约 {n / max(1e-9, args.seconds):.1f} fps）")
    print(f"  快照 ：{jpath}")
    print()
    print("现在把盖子抬回来：")
    print(f"  1. 先看快照 {os.path.basename(jpath)} —— 确认手在画面里、没糊")
    print("  2. 再分析录像：")
    print(f"     envs\\pg\\Scripts\\python.exe hand_probe.py --video \"{vpath}\"")
    print()
    print("⚠️ 如果快照里手是糊的：说明摄像头离手太近（固定焦点的摄像头")
    print("   在 20cm 左右会失焦）。把笔记本往外挪一点、盖子别压那么低。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
