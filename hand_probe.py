#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
握笔对照实验工具
================

**这个工具决定这个项目还要不要往下做。**

可行性评估（docs/04）里有两个叫停点，这个工具就是用来回答它们的：

  第 0 关：**看得见吗？** —— 手在画面里够大吗（≥200px）、跟得住吗（检出率）
  第 1 关：**分得开吗？** —— 把「正确握笔」「拇指包食指」「拳握」分别录一组，
                        看哪个指标真的能把它们分开。

⚠️ 为什么必须先做这一步：文献实测 MediaPipe 手部关节角误差 **22.5°**，
   而要分辨的握笔差异只有 10~30° —— 误差和信号同一个量级。
   所以"某个指标看起来合理"完全不能说明它有用，**必须实测**。
   如果实测下来三组数据全部重叠，那就是**该停**的信号（评估文档 §8 写了退路）。

用法
----
    # ① 实时对照实验（主用法）
    python hand_probe.py --camera 0

    # ② 拿一张照片先看手够不够大
    python hand_probe.py --image D:\\握笔照片.jpg

    # ③ 不带摄像头，只验证代码能跑
    python hand_probe.py --selftest

实时按键
--------
    1  把当前这一帧记成「正确握笔」
    2  把当前这一帧记成「拇指包食指」
    3  把当前这一帧记成「拳握」
    r  清空已记录的数据，重来
    p  打印对照表（这一步才是重点）
    w  把已录的样本存成 json（方便发给别人复核）
    s  存一张当前画面
    q / ESC  退出

⚠️ 只在「视角合格」时才允许记录（手太小/抖动大会被拒绝并提示），
   因为在小样本上录出来的数据本身就没意义。

实现上避开的坑
------------
1. **CV 的 putText 写不了中文**（字体不含 CJK，会变问号）→ 画面上的字全用英文，
   中文结论走控制台。
2. **detect_for_video 的时间戳必须全生命周期单调递增**（本项目踩过两次）
   → 用全局计数器，绝不重置。
3. **抓拍不能指望 cv2 的键盘回调去刷新画面** —— 记录的是"按键那一刻最近一帧
   的指标"，所以每帧都把指标缓存下来。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE                      # 独立项目：根目录就是这里
sys.path.insert(0, HERE)

import grip_metrics as gm        # noqa: E402
import pg_utils as pg            # noqa: E402  （相机、模型查找、阈值建议）

HAND_MODEL = "hand_landmarker.task"

# 三组标签。键要能用键盘按，名要给家长看得懂。
CLASSES = {
    "1": ("正确握笔", (60, 200, 60)),
    "2": ("拇指包食指", (60, 60, 230)),
    "3": ("拳握", (230, 160, 40)),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def make_hand(model_path):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.3,   # 宁可检测到再筛，别漏
        min_tracking_confidence=0.3))


class Recorder:
    """按类别攒指标样本，并算「分不分得开」。"""

    def __init__(self, window=45):
        self.data = {k: [] for k in CLASSES}
        self.window = window      # 抖动统计的滑动窗口
        self.centers = []         # 最近若干帧的掌心位置，用来算抖动
        self.note = ""            # 画面上临时提示
        self.last = None

    def add_center(self, m, lm, w, h):
        if lm is None:
            return
        p = lm[gm.INDEX_MCP]
        self.centers.append((p.x * w, p.y * h))
        del self.centers[:-self.window]

    @property
    def jitter(self):
        if len(self.centers) < 3:
            return None
        a = np.array(self.centers)
        return float(np.sqrt(a[:, 0].std() ** 2 + a[:, 1].std() ** 2))

    def record(self, key, m):
        self.data[key].append(dict(m))

    def counts(self):
        return {k: len(v) for k, v in self.data.items()}

    def _col(self, key, metric):
        return [s[metric] for s in self.data[key]
                if s.get(metric) is not None and not _isnan(s[metric])]

    def compare(self, min_n=5):
        """核心：三组数据在每个指标上分不分得开。

        判据（刻意保守）：把「正确」当一类、「两个错误」合并成另一类
        —— 因为提醒逻辑要回答的就是"是不是正确"，不需要区分错误类型。
        对每个指标算两类分布的范围重叠情况；再做一次真实阈值扫描
        （复用坐姿项目的 suggest_threshold）给出误报/漏报数。

        返回 (文本报告, 可分性字典)
        """
        need = min_n
        lines = []
        counts = self.counts()
        lines.append("样本数：" + "  ".join(
            f"{CLASSES[k][0]}={counts[k]}" for k in CLASSES))
        if counts["1"] < need or (counts["2"] + counts["3"]) < need:
            lines.append(f"⚠️ 样本不够（「正确」至少 {need} 条、两个错误合计至少 {need} 条）")
            lines.append("   多录一些再按 p —— 特别是每个类别要覆盖不同的手型和角度。")
            return "\n".join(lines), {}

        metrics = [k for k in gm.METRIC_INFO
                   if k not in ("px_w", "px_h", "palm_w", "hand_ratio")]
        sep = {}
        lines.append("")
        lines.append(f"{'指标':<20}{'正确':>12}{'错误(合并)':>14}   可分性")
        lines.append("-" * 68)
        for mt in metrics:
            a = self._col("1", mt)
            b = self._col("2", mt) + self._col("3", mt)
            if len(a) < 2 or len(b) < 2:
                continue
            ma, sa = float(np.mean(a)), float(np.std(a))
            mb, sb = float(np.mean(b)), float(np.std(b))
            # 类间距离 / 类内离散：越大越分得开。用两类的合并标准差做尺度。
            pooled = max(1e-9, ((sa ** 2 + sb ** 2) / 2) ** 0.5)
            margin = abs(ma - mb) / pooled
            flag = "✅ 分得开" if margin >= 1.5 else (
                "⚠️ 勉强" if margin >= 0.8 else "❌ 重叠")
            sep[mt] = margin
            lines.append(f"{gm.METRIC_INFO[mt][0]:<20}"
                         f"{ma:>8.3f}±{sa:.3f}{mb:>9.3f}±{sb:.3f}   "
                         f"margin={margin:4.2f} {flag}")

        # 对最好的那个指标做一次真实阈值扫描
        best = max(sep.items(), key=lambda kv: kv[1]) if sep else None
        if best and best[1] >= 1.5:
            mt = best[0]
            should = self._col("2", mt) + self._col("3", mt)   # 应当报警
            shouldnt = self._col("1", mt)                       # 不该报警
            # 方向：如果"错误"的值比"正确"小，就取负号让语义统一成"越大越该报"
            if np.mean(should) < np.mean(shouldnt):
                should, shouldnt = [-x for x in should], [-x for x in shouldnt]
            r = pg.suggest_threshold(should, shouldnt)
            if r:
                lines.append("")
                lines.append(f"→ 最有希望的是「{gm.METRIC_INFO[mt][0]}」"
                             f"（margin={best[1]:.2f}）")
                lines.append(f"   建议阈值 {r['suggested']:.3f}："
                             f"误报 {r['fp_at_suggested']} / 漏报 {r['fn_at_suggested']}"
                             f"（共 {r['n_should'] + r['n_shouldnt']} 条样本）")
                if r["fp_at_suggested"] + r["fn_at_suggested"] > 0:
                    lines.append("   ⚠️ 还有分错的样本 —— 真实使用里这个指标单独用不够，"
                                 "需要多指标组合，或者干脆放弃。")
        else:
            lines.append("")
            lines.append("→ ⚠️ 没有任何指标 margin ≥ 1.5。")
            lines.append("   这**不一定**是代码问题 —— 也可能就是"
                         "「手部关键点分辨不了握笔姿势」（文献预期如此）。")
            lines.append("   建议：再多录些样本（换角度/换光照）确认一遍；"
                         "若仍然重叠，就按 docs/04 §8 退回方案 E（握笔器）。")
        return "\n".join(lines), sep


def _isnan(v):
    try:
        return v != v
    except Exception:
        return False


def draw(frame, lm, m, quality, counts, note=""):
    """画手部特写 + 指标。⚠️ 只能用 ASCII（OpenCV 写不了中文）。"""
    import cv2
    h, w = frame.shape[:2]
    out = frame.copy()

    if lm is not None:
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, pts[a], pts[b], (0, 220, 0), 2)
        for i, p in enumerate(pts):
            cv2.circle(out, p, 4 if i in (4, 8, 12, 16, 20) else 3,
                       (255, 255, 255), -1)

    # 左侧：视角质量（第 0 关）
    qcolor = {"ok": (0, 200, 0), "marginal": (0, 180, 255), "bad": (0, 0, 255)}
    lines = []
    if m:
        lines.append(("hand width : %6.0f px" % m["px_w"], qcolor[quality]))
    else:
        lines.append(("hand: not detected", qcolor["bad"]))
    lines.append((("view: " + quality.upper()), qcolor[quality]))
    lines.append(("--- metric candidates ---", (200, 200, 200)))
    # 右侧：候选指标（第 1 关要比的就是这些数）
    right = []
    if m:
        for k in ("fist", "curl_index", "tip_close", "spread",
                  "thumb_index_gap", "thumb_side", "thumb_index_angle"):
            v = m.get(k)
            label = gm.METRIC_INFO[k][0]
            if v is None:
                right.append(("%-14s   n/a" % label, (160, 160, 160)))
            else:
                col = (120, 220, 255)
                if k == "thumb_side":
                    col = (80, 80, 255) if v > 0.25 else col
                right.append(("%-14s %7.3f" % (label, v), col))

    for i, (t, c) in enumerate(lines):
        cv2.putText(out, t, (12, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, c, 2)
    for i, (t, c) in enumerate(right):
        cv2.putText(out, t, (w - 300, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, c, 1)

    # 底部：已录样本数 + 提示
    cnt = "  ".join(f"[{k}]{CLASSES[k][0][:2]}={counts[k]}" for k in CLASSES)
    cv2.putText(out, cnt, (12, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    cv2.putText(out, "1/2/3 record   p compare   r reset   w save   q quit",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1)
    if note:
        cv2.putText(out, note, (12, h - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 200, 255), 2)
    return out


def run_live(cam_idx, backend, out_dir):
    import cv2
    import mediapipe as mp

    model = pg.find_model(HAND_MODEL)
    hand = make_hand(model)
    cap = pg.open_camera(cam_idx, backend)
    rec = Recorder()

    print(f"模型：{os.path.basename(model)}")
    print("摆好机位后：先看左边 hand width 是否 ≥200px、view 是否 OK。")
    print("然后让孩子分别摆出「正确握笔」「拇指包食指」「拳握」，每种按对应数字键多录几帧。")
    print("录完按 p 看对照表 —— 那才是这个项目能不能做下去的依据。\n")

    tick = 0
    last_note = 0.0
    while True:
        fr = pg.grab_frame(cap)
        if fr is None:
            continue
        h, w = fr.shape[:2]
        rgb = fr[:, :, ::-1].copy()
        tick += 33
        r = hand.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), tick)

        lm = r.hand_landmarks[0] if r.hand_landmarks else None
        m = gm.compute_grip_metrics(lm, w, h) if lm is not None else None
        rec.add_center(m, lm, w, h)
        quality, qmsg = gm.view_quality(m, rec.jitter)
        rec.last = (m, quality)

        note = rec.note if time.time() - last_note < 2.5 else ""
        cv2.imshow("pen grip probe  (1/2/3 record  p compare  q quit)",
                   draw(fr, lm, m, quality, rec.counts(), note))
        k = cv2.waitKey(1) & 0xFF
        key = chr(k) if 0 <= k < 128 else ""

        if k in (ord("q"), 27):
            break
        elif key in CLASSES:
            if quality != "ok":
                # ⚠️ 视角不合格就拒绝记录。在手太小/抖得厉害的帧上录下来的
                #    样本本身没有意义，收进来只会污染结论。
                rec.note = f"REFUSED ({quality})"
                print(f"  ✗ 拒绝记录（{CLASSES[key][0]}）：{qmsg}")
            else:
                rec.record(key, m)
                rec.note = f"recorded {key}"
                print(f"  ✓ 记录 {CLASSES[key][0]}  "
                      f"（共 {rec.counts()[key]} 条）  {_brief(m)}")
            last_note = time.time()
        elif key == "p":
            report, sep = rec.compare()
            print("\n" + "=" * 68)
            print(report)
            print("=" * 68 + "\n")
            rec.note = "report -> console"
            last_note = time.time()
        elif key == "r":
            rec.data = {x: [] for x in CLASSES}
            print("  已清空样本")
            rec.note = "reset"
            last_note = time.time()
        elif key == "w":
            p = save_samples(rec, out_dir)
            print(f"  已保存 {p}")
            rec.note = "saved"
            last_note = time.time()
        elif key == "s":
            os.makedirs(out_dir, exist_ok=True)
            p = os.path.join(out_dir, time.strftime("probe_%Y%m%d-%H%M%S.jpg"))
            cv2.imwrite(p, draw(fr, lm, m, quality, rec.counts(), note))
            print(f"  已存 {p}")

    cap.release()
    cv2.destroyAllWindows()
    hand.close()
    cnt = rec.counts()
    print("\n=== 退出小结 ===")
    print("  样本数：" + "  ".join(f"{CLASSES[k][0]}={cnt[k]}" for k in CLASSES))
    if sum(cnt.values()) >= 15:
        print("  跑一次完整对照：python repos\\p1-pen-grip\\hand_probe.py "
              "--replay 上面用 w 存的文件")
    return rec


def _brief(m):
    return (f"fist={m['fist']:.2f} thumb_side={m['thumb_side']:+.2f} "
            f"gap={m['thumb_index_gap']:.2f} angle={m['thumb_index_angle']:.0f}°")


def save_samples(rec, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, time.strftime("samples_%Y%m%d-%H%M%S.json"))
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"classes": {k: v[0] for k, v in CLASSES.items()},
                   "data": rec.data}, f, ensure_ascii=False, indent=2)
    return p


def replay(path):
    """回放已保存的样本，重新出对照表（不用再占着摄像头）。"""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    rec = Recorder()
    rec.data = obj["data"]
    report, _ = rec.compare()
    print("=" * 68)
    print(f"回放：{os.path.basename(path)}")
    print(report)
    print("=" * 68)
    return 0


def run_image(path):
    import cv2
    import mediapipe as mp
    fr = cv2.imread(path)
    if fr is None:
        sys.exit(f"读不了这张图：{path}")
    h, w = fr.shape[:2]
    hand = make_hand(pg.find_model(HAND_MODEL))
    r = hand.detect_for_video(mp.Image(
        image_format=mp.ImageFormat.SRGB, data=fr[:, :, ::-1].copy()), 33)
    lm = r.hand_landmarks[0] if r.hand_landmarks else None
    m = gm.compute_grip_metrics(lm, w, h) if lm is not None else None
    quality, qmsg = gm.view_quality(m)
    hand.close()

    print(f"图片 {w}x{h}")
    if lm is None:
        print("❌ 画面里没检测到手 —— 先解决「手进不了画面」。")
        return 1
    print(f"手部像素宽：{m['px_w']:.0f}px   掌宽：{m['palm_w']:.0f}px")
    print(f"视角判定：{quality} —— {qmsg}")
    print("\n候选指标：")
    for k in ("fist", "curl_index", "tip_close", "spread",
              "thumb_index_gap", "thumb_side", "thumb_index_angle"):
        v = m.get(k)
        print(f"  {gm.METRIC_INFO[k][0]:<16} {'n/a' if v is None else f'{v:.3f}'}")
    out = os.path.splitext(path)[0] + "_probe.jpg"
    cv2.imwrite(out, draw(fr, lm, m, quality, {k: 0 for k in CLASSES}))
    print(f"\n标注图已存：{out}")
    print("\n⚠️ 单张图只能说「看得见吗」。要判断「分得开吗」，"
          "得给三种握法各录一组，跑 --camera 模式。")
    return 0 if quality == "ok" else 1


def selftest():
    print("=== 自检（不需要摄像头）===")
    model = pg.find_model(HAND_MODEL)
    if not os.path.isfile(model):
        print(f"❌ 找不到 {HAND_MODEL}。先跑："
              f"python tools\\fetch_models.py")
        return 1
    print(f"模型：{os.path.basename(model)}  "
          f"{os.path.getsize(model) / 1024 / 1024:.1f} MB")

    hand = make_hand(model)
    import mediapipe as mp
    tick, ts = 0, []
    for _ in range(12):
        tick += 33
        img = np.full((720, 1280, 3), 210, dtype=np.uint8)
        t0 = time.perf_counter()
        hand.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=img), tick)
        ts.append((time.perf_counter() - t0) * 1000)
    hand.close()
    print(f"手部推理：{sum(ts) / len(ts):.1f} ms/帧（空图，成本下限）")

    # 用合成手走一遍「记录 -> 对照表」全流程，确认逻辑不崩
    rec = Recorder()
    for _ in range(6):
        rec.record("1", gm.compute_grip_metrics(*gm.synthetic_landmarks(
            curl=(0.2, 0.6, 0.7, 0.75), thumb_side=-0.3)))
        rec.record("2", gm.compute_grip_metrics(*gm.synthetic_landmarks(
            curl=(0.2, 0.6, 0.7, 0.75), thumb_side=+0.6)))
        rec.record("3", gm.compute_grip_metrics(*gm.synthetic_landmarks(
            curl=(1.0, 1.0, 1.0, 1.0), thumb_side=0.0)))
    report, sep = rec.compare()
    print(report)
    print("\n✅ 自检通过（模型可用、记录/对照流程可跑）")
    print("⚠️ 上面的数字来自**合成手**，只证明流程能跑，"
          "不代表真实数据也这样 —— 真实结论必须用摄像头录。")
    return 0


def main():
    ap = argparse.ArgumentParser(description="握笔对照实验工具")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--replay", default=None, help="回放之前用 w 存下的样本文件")
    ap.add_argument("--backend", default=None, help="强制 cv2 后端，如 dshow")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "hand_probe"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.replay:
        return replay(args.replay)
    if args.image:
        return run_image(args.image)
    if args.camera is None:
        ap.error("给 --camera N / --image 路径 / --replay 文件 / --selftest 之一")
    run_live(args.camera, args.backend, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
