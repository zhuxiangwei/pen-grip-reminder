#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
握笔项目 · 基础工具（自包含）
=============================

相机打开、模型查找、阈值建议 —— 都用得上的小工具。

⚠️ **为什么是复制而不是 import 坐姿项目的代码**
   握笔是个**独立项目**（独立目录、独立仓库、独立摄像头）。
   如果 import 隔壁项目的 `posture_monitor`，那这个项目单独克隆下来就跑不起来 ——
   两个仓库之间会有个看不见的硬依赖。宁可复这 100 来行。
   两边共享的**设计结论**写在各自的文档里，代码不共享。

⚠️ 从坐姿项目复制时修掉的一个真 bug
   `posture_monitor.read_frame()` 返回的是 **`(ok, frame)` 二元组**，
   而调用方很容易写成 `frame = read_frame(cap)` 然后 `frame.shape` ——
   摄像头一开就 AttributeError。这个 bug 在两处工具里真实存在过，
   因为只测了 `--selftest` / `--image`、**从没跑过实时摄像头**所以没暴露。

   本模块提供的 `grab_frame()` **只返回帧本身**（失败返回 None），
   从签名上就不给人写错的机会。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# ⚠️ 本文件就在项目根目录下，所以 ROOT 就是 HERE。
#    别写成 os.path.dirname(HERE) —— 那样会指到上层目录，
#    模型查找会去找 Downloads\bench\models 这种不存在的地方。
ROOT = HERE

# 模型查找目录（本项目的，跟坐姿项目无关）
MODEL_DIRS = [
    os.path.join(ROOT, "bench", "models"),
    os.path.join(ROOT, "models"),
    HERE,
]

DEFAULT_CAM_W = 1280
DEFAULT_CAM_H = 720

_cv2 = None
_cv2_tried = False


def cv2():
    """延迟导入 cv2（这样纯算法测试不需要装 opencv）。"""
    global _cv2, _cv2_tried
    if not _cv2_tried:
        _cv2_tried = True
        try:
            import cv2 as _m
            _cv2 = _m
        except ImportError:
            _cv2 = None
    if _cv2 is None:
        sys.exit("需要 opencv： pip install opencv-python")
    return _cv2


def find_model(name):
    """在 MODEL_DIRS 里找模型。找不到时说清楚怎么弄。"""
    for d in MODEL_DIRS:
        p = os.path.normpath(os.path.join(d, name))
        if os.path.isfile(p):
            return p
    sys.exit(
        f"找不到模型 {name}\n"
        f"下载： python tools\\fetch_models.py\n"
        f"或手动放到任一下面：\n  "
        + "\n  ".join(os.path.normpath(d) for d in MODEL_DIRS)
        + "\n\n⚠️ 手部模型的地址是：\n"
        "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task\n"
        "  （注意 hand_landmarker 在路径里出现两次）"
    )


BACKENDS = []


def _build_backends():
    """后端必须挨个试，不能写死。

    实测（2026-09-22 本机）：同一个摄像头，DSHOW 和 MSMF 能看到的东西完全不同，
    写死单一后端会得到"这台机器没摄像头"的假结论。
    """
    global BACKENDS
    if not BACKENDS:
        c = cv2()
        BACKENDS = [("DSHOW", c.CAP_DSHOW), ("MSMF", c.CAP_MSMF), ("ANY", c.CAP_ANY)]
    return BACKENDS


def grab_frame(cap):
    """安全读一帧，**只返回帧本身**（失败返回 None）。

    ⚠️ 必须包 try：驱动异常时 `cap.read()` 会直接抛 cv2.error，
       而不是规规矩矩返回 (False, None)。
       实测报错：(-215:Assertion failed) _step >= minstep in function 'cv::Mat::Mat'
    """
    try:
        ok, frame = cap.read()
    except Exception:
        return None
    if not ok or frame is None:
        return None
    return frame


def open_camera(idx=None, backend=None, width=None, height=None):
    """逐个组合尝试打开摄像头：索引 × 后端 × 分辨率。

    三者都不可靠：
      · 索引：接多个摄像头时 0 未必是想要的那颗
      · 后端：DSHOW / MSMF 能看到的设备集合可能完全不同
      · 分辨率：不设时 OpenCV 常只给 640x480；但**有些后端设了反而打不开流**
    所以先按请求分辨率试一轮，全失败再退到"不设分辨率"兜底。
    """
    c = cv2()
    bts = _build_backends()
    table = dict(bts)
    bt = [(backend, table[backend])] if backend and backend in table else bts
    indices = [idx] if idx is not None else [0, 1, 2, 3]
    w_req = width or DEFAULT_CAM_W
    h_req = height or DEFAULT_CAM_H

    for res in ((w_req, h_req), (None, None)):
        for name, be in bt:
            for i in indices:
                cap = c.VideoCapture(i, be)
                if not cap.isOpened():
                    cap.release()
                    continue
                if res[0]:
                    cap.set(c.CAP_PROP_FRAME_WIDTH, res[0])
                    cap.set(c.CAP_PROP_FRAME_HEIGHT, res[1])
                frame = grab_frame(cap)
                if frame is not None:
                    h, w = frame.shape[:2]
                    note = ""
                    if res[0] and (w, h) != (w_req, h_req):
                        note = f"（不支持 {w_req}x{h_req}，用了实际值）"
                    elif res[0] is None:
                        note = "（设分辨率会打不开流，已退回默认）"
                    print(f"[摄像头] 后端={name} 索引={i} 分辨率={w}x{h}{note}")
                    return cap
                cap.release()
    sys.exit(
        "打不开任何摄像头。排查顺序：\n"
        "  1) 是否被别的程序占用（相机 App / 会议软件 / 浏览器标签页）\n"
        "  2) 隐私设置是否禁止桌面应用访问相机\n"
        "     Windows 设置 > 隐私和安全性 > 相机 > 允许桌面应用访问\n"
        "  3) USB 线接触不良 / 供电不足（换口试，优先主板直连口）\n"
        "  4) 用 --list-cameras 看看系统到底认到了什么"
    )


def list_cameras(width=None, height=None):
    """枚举：后端 × 索引，并试着把分辨率拉上去。"""
    c = cv2()
    print("枚举摄像头（后端 × 索引）…\n")
    found = []
    for name, be in _build_backends():
        for i in range(4):
            cap = c.VideoCapture(i, be)
            if not cap.isOpened():
                cap.release()
                continue
            f0 = grab_frame(cap)
            got = ""
            if f0 is not None:
                h0, w0 = f0.shape[:2]
                cap.set(c.CAP_PROP_FRAME_WIDTH, width or DEFAULT_CAM_W)
                cap.set(c.CAP_PROP_FRAME_HEIGHT, height or DEFAULT_CAM_H)
                f1 = grab_frame(cap)
                if f1 is not None:
                    h1, w1 = f1.shape[:2]
                    got = f"默认 {w0}x{h0} -> 提升后 {w1}x{h1}"
                else:
                    got = f"默认 {w0}x{h0}（改分辨率后打不开流，属正常）"
            print(f"  ✅ 后端={name:<6} 索引={i}  {got}")
            found.append((name, i))
            cap.release()
    if not found:
        print("  ❌ 一个都没打开")
    return found


# ---------------------------------------------------------------- 阈值建议

def suggest_threshold(values_should, values_shouldnt, current=None, fp_weight=1.5):
    """在两组观测值之间找最佳阈值。规则：`值 > 阈值` 即判定为「该报警」。

    values_should   : 真实情况**应该**报警的样本值
    values_shouldnt : 真实情况**不该**报警的样本值
    fp_weight       : 误报权重，默认 1.5 —— **误报比漏报更烦人**：
                      叫得太勤，孩子几天就不用了。

    返回 dict；两类各需至少 1 条，否则返回 None。

    实现与坐姿项目的同名函数一致（那边的结论这边照用）。
    """
    a = [float(x) for x in values_should if x is not None]
    b = [float(x) for x in values_shouldnt if x is not None]
    if not a or not b:
        return None

    def cost(t):
        fp = sum(1 for v in b if v > t)      # 不该报却报了
        fn = sum(1 for v in a if v <= t)     # 该报却没报
        return fp * fp_weight + fn, fp, fn

    vals = sorted(set(a + b))
    cands = [vals[0] - 1e-9]
    cands += [(vals[i] + vals[i + 1]) / 2.0 for i in range(len(vals) - 1)]
    cands += [vals[-1] + 1e-9]

    best = min(cands, key=lambda t: (cost(t)[0], cost(t)[1]))   # 同分优先少误报
    _, fp_new, fn_new = cost(best)
    return {
        "suggested": best,
        "current": current,
        "n_should": len(a),
        "n_shouldnt": len(b),
        "fp_at_suggested": fp_new,
        "fn_at_suggested": fn_new,
        "fp_at_current": None if current is None else sum(1 for v in b if v > float(current)),
    }
