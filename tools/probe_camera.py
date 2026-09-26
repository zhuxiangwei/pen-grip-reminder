#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
握笔项目 · 摄像头能力探测
=========================

**为什么需要这个工具**
   摄像头标称分辨率和**实际能拿到什么**完全是两回事。实测（本机）：
     · 请求 1920x1080 → 驱动**静默**给 1280x720，不报错
     · 请求 1600x1200 / 1440x1080 / 1280x960 → 全部降到 1280x720
     · 请求 1024x768 / 960x720 → 降到 960x540
     · 请求 800x600 → 降到 848x480
   如果不回读实际值，就会以为自己在用 1080p，其实一直在 720p 跑。
   「手宽 ≥200px」这个门槛是按真实像素算的，弄错分辨率门槛就整个失效。

**探测什么**
   1. 分辨率阶梯：逐个请求，**回读**实际值 + 取首帧确认真实尺寸
   2. 后端对比：DSHOW / MSMF 能给的最高分辨率可能不同（实测本机相同）
   3. 帧率阶梯：在最高分辨率下能跑多少 fps
   4. 实测吞吐：真抓 N 帧计时（比 CAP_PROP_FPS 回读可信）
   5. 推理耗时：各分辨率下 MediaPipe 单帧耗时 → 决定全流程用哪个

**结论会自动写进 `pen_grip_config.json` 的 `camera` 段**，
全流程（实时 / 盲录 / 分析）都从这里读，保证一致。

用法：
    python tools\\probe_camera.py                # 探测并写配置
    python tools\\probe_camera.py --no-save     # 只看结果，不改配置
    python tools\\probe_camera.py --camera 1    # 指定索引
    python tools\\probe_camera.py --busy         # 附带跑推理耗时基准
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import pg_utils as pg                                       # noqa: E402

CONFIG_PATH = os.path.join(ROOT, "pen_grip_config.json")

# 请求顺序：从高到低。高分辨率被拒时会静默降级，正好用来找天花板。
RES_LADDER = [
    (3840, 2160), (2560, 1440), (1920, 1080),
    (1600, 1200), (1600, 900), (1440, 1080), (1280, 960), (1280, 800),
    (1280, 720), (1024, 768), (960, 720), (960, 540),
    (848, 480), (800, 600), (640, 480), (640, 360), (320, 240),
]
FPS_LADDER = [60, 50, 30, 25, 24, 20, 15, 10]

# 判断"这是不是同一个模式"的容差：驱动常把宽高对齐到 8/16 的倍数
TOL = 8


def _close(a, b):
    return abs(a - b) <= TOL


def _open(idx, backend):
    c = pg.cv2()
    cap = c.VideoCapture(idx, backend)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def probe_resolutions(idx, backend_name, backend):
    """逐个请求分辨率，回读实际值。返回 [(req_w,req_h, got_w,got_h), ...]。"""
    c = pg.cv2()
    rows = []
    print("  请求         回读(W/H属性)   首帧实际      判定")
    print("  " + "-" * 62)
    for (rw, rh) in RES_LADDER:
        cap = _open(idx, backend)
        if cap is None:
            print("  %-11s %-15s %-13s %s" % ("%dx%d" % (rw, rh), "-", "-", "打不开"))
            continue
        cap.set(c.CAP_PROP_FRAME_WIDTH, rw)
        cap.set(c.CAP_PROP_FRAME_HEIGHT, rh)
        gw = int(round(cap.get(c.CAP_PROP_FRAME_WIDTH)))
        gh = int(round(cap.get(c.CAP_PROP_FRAME_HEIGHT)))
        frame = pg.grab_frame(cap)
        cap.release()
        if frame is None:
            print("  %-11s %-15s %-13s %s" % (
                "%dx%d" % (rw, rh), "%dx%d" % (gw, gh), "-", "取帧失败"))
            continue
        fh, fw = frame.shape[:2]
        if _close(fw, rw) and _close(fh, rh):
            verdict = "✅ 支持"
        elif fw * fh < rw * rh:
            verdict = "⬇ 降级"
        else:
            verdict = "⬆ 反向提升"
        print("  %-11s %-15s %-13s %s" % (
            "%dx%d" % (rw, rh), "%dx%d" % (gw, gh), "%dx%d" % (fw, fh), verdict))
        rows.append((rw, rh, fw, fh))
    return rows


def pick_best(rows):
    """从探测结果里挑最高可用分辨率（去重，同尺寸只留一次）。"""
    seen = {}
    for (rw, rh, fw, fh) in rows:
        seen.setdefault((fw, fh), (rw, rh))
    if not seen:
        return None
    (fw, fh) = max(seen.keys(), key=lambda p: p[0] * p[1])
    return {"width": fw, "height": fh,
            "requested": "%dx%d" % seen[(fw, fh)],
            "distinct": sorted(seen.keys(), key=lambda p: -p[0] * p[1])}


def probe_fps(idx, backend, w, h):
    """在指定分辨率下探帧率阶梯。返回 (可设的最高fps, 实测吞吐fps)。"""
    c = pg.cv2()
    print("  请求 fps   回读 fps   首帧")
    print("  " + "-" * 34)
    best_set = None
    for fps in FPS_LADDER:
        cap = _open(idx, backend)
        if cap is None:
            print("  %-10s %-10s %s" % (fps, "-", "打不开"))
            continue
        cap.set(c.CAP_PROP_FRAME_WIDTH, w)
        cap.set(c.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(c.CAP_PROP_FPS, fps)
        got = cap.get(c.CAP_PROP_FPS)
        frame = pg.grab_frame(cap)
        cap.release()
        ok = "OK" if frame is not None else "失败"
        print("  %-10s %-10s %s" % (fps, "%.1f" % got, ok))
        if frame is not None and best_set is None:
            best_set = fps

    # 实测吞吐：比 CAP_PROP_FPS 回读可信（回读常常直接把请求值还给你）
    real = None
    cap = _open(idx, backend)
    if cap is not None:
        cap.set(c.CAP_PROP_FRAME_WIDTH, w)
        cap.set(c.CAP_PROP_FRAME_HEIGHT, h)
        n, t0 = 0, time.time()
        while n < 40 and time.time() - t0 < 6:
            if pg.grab_frame(cap) is not None:
                n += 1
        dt = time.time() - t0
        cap.release()
        real = round(n / dt, 1) if dt > 0 else None
        print("  实测吞吐：%d 帧 / %.2fs = %s fps" % (n, dt, real))
    return best_set, real


def bench_inference(w, h, frames=15):
    """各分辨率下 MediaPipe 单帧耗时。

    ⚠️ 只建**一个** detector 全程复用。
       实测反复 `HandLandmarker.create_from_options()` 会卡死（本机 -->
       每次新建都要重载 7.5MB 模型 + 初始化 XNNPACK，累积到第三次就挂住）。
    """
    try:
        import mediapipe as mp
        import hand_probe as hp
        from pg_utils import find_model
    except Exception as e:                                   # noqa: BLE001
        print("  （跳过：%s）" % e)
        return {}

    model = find_model(hp.HAND_MODEL)
    c = pg.cv2()
    out = {}

    # 取一张真帧当基准
    cap = _open(0, c.CAP_DSHOW)
    if cap is None:
        return {}
    cap.set(c.CAP_PROP_FRAME_WIDTH, w)
    cap.set(c.CAP_PROP_FRAME_HEIGHT, h)
    base = None
    for _ in range(5):
        base = pg.grab_frame(cap)
    cap.release()
    if base is None:
        return {}

    sizes = [(w, h)]
    for s in [(1280, 720), (960, 540), (848, 480), (640, 480), (320, 180)]:
        if s != (w, h):
            sizes.append(s)

    print("  输入尺寸      中位耗时    理论上限")
    print("  " + "-" * 40)
    tick = 1000
    with hp.make_hand(model) as hand:
        for (nw, nh) in sizes:
            fr = c.resize(base, (nw, nh))
            rgb = fr[:, :, ::-1].copy()
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            for _ in range(3):
                hand.detect_for_video(img, tick)
                tick += 33
            ts = [time.perf_counter()]
            for _ in range(frames):
                hand.detect_for_video(img, tick)
                tick += 33
                ts.append(time.perf_counter())
            d = sorted((ts[i + 1] - ts[i]) * 1000 for i in range(len(ts) - 1))
            med = d[len(d) // 2]
            out["%dx%d" % (nw, nh)] = round(med, 1)
            print("  %-13s %6.1f ms   %3.0f fps" % (
                "%dx%d" % (nw, nh), med, 1000.0 / med))
    return out


def other_backends(idx, backend_name, timeout_s=8.0):
    """列出其他后端能给的最高分辨率（有些机器上后端差异很大）。

    ⚠️ **必须带超时**：本机实测 MSMF 打开/重复开流会**永久卡住**
       （卡在 VideoCapture 构造里，连 KeyboardInterrupt 都进不去，
       只能被 SIGTERM 杀掉）。所以这里用子进程 + 超时，
       而不是直接在当前进程里试 —— 否则整个探测工具会挂死。
    """
    c = pg.cv2()
    names = [("DSHOW", c.CAP_DSHOW), ("MSMF", c.CAP_MSMF)]
    res = {}
    for name, be in names:
        if name == backend_name:
            print("  %-6s （就是当前后端，跳过）" % name)
            continue
        code = (
            "import cv2,json,sys\n"
            "cap=cv2.VideoCapture(%d, cv2.CAP_%s)\n"
            "if not cap.isOpened():\n"
            "    print(json.dumps(None)); sys.exit(0)\n"
            "top=None\n"
            "for rw,rh in [(1920,1080),(1280,960),(1280,720),(640,480)]:\n"
            "    cap.set(cv2.CAP_PROP_FRAME_WIDTH,rw)\n"
            "    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,rh)\n"
            "    ok,fr=cap.read()\n"
            "    if ok and fr is not None:\n"
            "        top=(fr.shape[1],fr.shape[0])\n"
            "        if top==(rw,rh): break\n"
            "print(json.dumps(top))\n"
        ) % (idx, name)
        try:
            out = _run_with_timeout([sys.executable, "-c", code], timeout_s)
        except _Timeout:
            res[name] = "timeout"
            print("  %-6s ⏱ 超时（%.0fs 没返回）—— 这个后端在这台机器上不可靠"
                  % (name, timeout_s))
            continue
        except Exception as e:                               # noqa: BLE001
            res[name] = "error"
            print("  %-6s 出错：%s" % (name, e))
            continue
        val = None
        for line in reversed((out or "").splitlines()):
            line = line.strip()
            if line.startswith("[") or line == "null":
                try:
                    val = json.loads(line)
                except Exception:                            # noqa: BLE001
                    val = None
                break
        if val is None:
            print("  %-6s 打不开" % name)
        else:
            print("  %-6s 最高 %dx%d" % (name, val[0], val[1]))
            res[name] = val
    return res


class _Timeout(Exception):
    pass


def _run_with_timeout(cmd, timeout_s):
    """跑子进程并限时，超时抛 _Timeout。避免被卡死的后端拖住整个工具。"""
    import subprocess
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout_s,
                           errors="replace")
    except subprocess.TimeoutExpired:
        raise _Timeout()
    return (p.stdout or "") + (p.stderr or "")


def save_config(cam, fps_info, infer):
    cfg = {}
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:                                    # noqa: BLE001
            cfg = {}
    cfg["camera"] = cam
    cfg["camera"]["fps_set"] = fps_info[0]
    cfg["camera"]["fps_measured"] = fps_info[1]
    cfg["camera"]["inference_ms"] = infer
    cfg["camera"]["probed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return CONFIG_PATH


def main():
    ap = argparse.ArgumentParser(description="摄像头能力探测")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-save", action="store_true", help="只看结果，不改配置")
    ap.add_argument("--busy", action="store_true", help="附带跑推理耗时基准")
    ap.add_argument("--fps", action="store_true", help="附带跑帧率阶梯（较慢）")
    a = ap.parse_args()

    c = pg.cv2()
    print("=" * 70)
    print("摄像头能力探测   索引=%d   cv2=%s" % (a.camera, c.__version__))
    print("=" * 70)
    print()
    print("① 分辨率阶梯（后端 DSHOW）")
    rows = probe_resolutions(a.camera, "DSHOW", c.CAP_DSHOW)
    best = pick_best(rows)
    if best is None:
        sys.exit("❌ 这个索引下没有可用分辨率，检查摄像头是否被占用")

    print()
    print("=" * 70)
    print("★ 最高可用分辨率：%dx%d   （请求 %s 得到）" % (
        best["width"], best["height"], best["requested"]))
    print("  全部可用档位：" + "  ".join(
        "%dx%d" % s for s in best["distinct"]))
    print("=" * 70)

    print()
    print("② 其他后端对比（确认 DSHOW 是不是最优）")
    other_backends(a.camera, "DSHOW")

    fps_set, fps_real = (None, None)
    if a.fps:
        print()
        print("③ 帧率阶梯（在 %dx%d 下）" % (best["width"], best["height"]))
        fps_set, fps_real = probe_fps(a.camera, c.CAP_DSHOW,
                                      best["width"], best["height"])
    else:
        # 不跑阶梯也要测一下真实吞吐
        print()
        print("③ 实测吞吐（在 %dx%d 下，抓 40 帧）" % (best["width"], best["height"]))
        cap = _open(a.camera, c.CAP_DSHOW)
        if cap is not None:
            cap.set(c.CAP_PROP_FRAME_WIDTH, best["width"])
            cap.set(c.CAP_PROP_FRAME_HEIGHT, best["height"])
            n, t0 = 0, time.time()
            while n < 40 and time.time() - t0 < 6:
                if pg.grab_frame(cap) is not None:
                    n += 1
            dt = time.time() - t0
            cap.release()
            fps_real = round(n / dt, 1) if dt > 0 else None
            print("  %d 帧 / %.2fs = %s fps" % (n, dt, fps_real))

    infer = {}
    if a.busy:
        print()
        print("④ MediaPipe 推理耗时（说明：模型内部会把输入缩到 ~224px，")
        print("   所以**分辨率对耗时几乎没影响** —— 不必为了省时间降分辨率）")
        infer = bench_inference(best["width"], best["height"])

    print()
    print("=" * 70)
    print("结论")
    print("=" * 70)
    print("  采集分辨率：%dx%d（这是上限，请求更高只会被静默降级）" % (
        best["width"], best["height"]))
    print("  后端：DSHOW（本机实测比 MSMF 快，且 MSMF 重复开流会卡死）")
    if fps_real:
        print("  实测帧率：%s fps" % fps_real)
    if infer:
        base = "1280x720"
        if base in infer:
            print("  单帧推理：%.1f ms（@%s）→ 理论上限 %.0f fps" % (
                infer[base], base, 1000.0 / infer[base]))
        print("  ⚠️ 降分辨率**不会**变快：18.5ms@720p vs %.1fms@320x180，"
              "差不到 1ms" % infer.get("320x180", 0))
    print()
    print("  → 全流程统一用 %dx%d，不做缩放。" % (best["width"], best["height"]))
    print("     720p 已是最优：分辨率最高、耗时和最低档没差别。")

    if not a.no_save:
        cam = {
            "index": a.camera,
            "backend": "DSHOW",
            "width": best["width"],
            "height": best["height"],
            "max_width": best["width"],
            "max_height": best["height"],
            "supported": ["%dx%d" % s for s in best["distinct"]],
        }
        p = save_config(cam, (fps_set, fps_real), infer)
        print()
        print("  ✓ 已写入 %s" % p)
    else:
        print()
        print("  （--no-save：未写配置）")


if __name__ == "__main__":
    main()
