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
    2  把当前这一帧记成「**拇指包食指**」（主要要抓的问题）
    3  把当前这一帧记成「其他错误」（拳握等，留着免得污染「正确」那一类）
    r  清空已记录的数据，重来
    p  打印对照表（这一步才是重点）
    w  把已录的样本存成 json（方便发给别人复核）
    z  开关「拇指-食指放大镜」（右下角）—— 判断包食指全靠那一块，务必盯它
    s  存一张当前画面
    q / ESC  退出

⚠️ 只在「视角合格」时才允许记录（手太小/抖动大会被拒绝并提示），
   因为在小样本上录出来的数据本身就没意义。

判据说明（「拇指包住食指」到底看什么）
------------------------------------
不是看单一指标，而是两个合成：

    thumb_side        拇指尖相对「食指根→小指根」横轴的偏移
                      偏正 = 越到食指外侧 = 疑似包住
    thumb_index_dist  拇指尖到**食指折线**的最短距离
                      小 = 真的搭在食指上（不是悬在旁边）

两个必须一起看：
  · 只看 thumb_side —— 拇指伸得很长越过食指尖、或根本没搭上，也可能偏正，
    但那是别的形态，纠正方式完全不同
  · 只看距离 —— 分不清搭在食指的哪一侧（正常捏笔时拇指也贴着食指）

⚠️ 这里原本用的是「拇指尖沿食指轴的投影」，被离线自检 C9/C10 否掉了：
   食指一弯那条轴就转向侧面，投影会变成负数（实测 -0.166），语义失效。
   改成「到折线的最近点」后天然落在 [0,1]，且对弯曲免疫。

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
#
# ⚠️ 按用户定的范围收窄过（2026-09-26）：
#    主要就是纠正「大拇指包住食指」，所以第 2 类单独拿出来，
#    对照表也重点比「正确 vs 包食指」。
#    第 3 类留着不是凑数 —— 万一孩子的主要问题是别的（拳握等），
#    有这一格才不会把那些样本误算进"正确"里污染结论。
CLASSES = {
    "1": ("正确握笔", (60, 200, 60)),
    "2": ("拇指包食指", (60, 60, 230)),      # ← 主要要抓的问题
    "3": ("其他错误", (230, 160, 40)),
}

FOCUS = ("1", "2")        # 对照表重点比较的两种；"3" 单独报

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
        """核心：看哪些指标能把「正确握笔」和「拇指包食指」分开。

        按用户定的范围收窄（2026-09-26）：主要问题就是「大拇指包住食指」，
        所以主比较是 **1 正确 vs 2 包食指**；第 3 类「其他错误」单独列出作参考
        （不参与可分性判断，但能看出孩子是不是还有别的问题）。

        返回 (文本报告, 可分性字典)
        """
        need = min_n
        lines = []
        counts = self.counts()
        lines.append("样本数：" + "  ".join(
            f"{CLASSES[k][0]}={counts[k]}" for k in CLASSES))
        miss = [k for k in FOCUS if counts[k] < need]
        if miss:
            lines.append("⚠️ 样本不够：" + "、".join(
                f"{CLASSES[k][0]}（{counts[k]}/{need}）" for k in miss))
            lines.append("   多录一些再按 p —— 每个类别都要覆盖不同的手型、角度、位置，")
            lines.append("   只在一个位置录出来的高可分性是假的。")
            return "\n".join(lines), {}

        # 重点指标排前面：判断「拇指包食指」靠这两个合成判据
        focus = list(gm.THUMB_OVER_INDEX_PAIR)
        metrics = focus + [k for k in gm.METRIC_INFO
                           if k not in ("px_w", "px_h", "palm_w", "hand_ratio")
                           and k not in focus]
        sep = {}
        has3 = counts["3"] > 0
        lines.append("")
        head = f"{'指标':<22}{'正确':>13}{'包食指':>13}"
        lines.append(head + (f"{'其他错误':>13}" if has3 else "") + "   可分性")
        lines.append("-" * (74 + (14 if has3 else 0)))
        for mt in metrics:
            a = self._col("1", mt)
            b = self._col("2", mt)
            if len(a) < 2 or len(b) < 2:
                continue
            ma, sa = float(np.mean(a)), float(np.std(a))
            mb, sb = float(np.mean(b)), float(np.std(b))
            pooled = max(1e-9, ((sa ** 2 + sb ** 2) / 2) ** 0.5)
            margin = abs(ma - mb) / pooled
            flag = "✅ 分得开" if margin >= 1.5 else (
                "⚠️ 勉强" if margin >= 0.8 else "❌ 重叠")
            sep[mt] = margin
            row = (f"{gm.METRIC_INFO[mt][0]:<22}"
                   f"{ma:>9.3f}±{sa:.3f}{mb:>9.3f}±{sb:.3f}")
            if has3:
                c = self._col("3", mt)
                row += (f"{float(np.mean(c)):>9.3f}±{float(np.std(c)):.3f}"
                        if len(c) >= 2 else f"{'—':>13}")
            lines.append(row + f"   margin={margin:4.2f} {flag}")

        best = max(sep.items(), key=lambda kv: kv[1]) if sep else None
        if best and best[1] >= 1.5:
            mt = best[0]
            shouldnt = self._col("1", mt)      # 不该报警
            should = self._col("2", mt)        # 应当报警
            if np.mean(should) < np.mean(shouldnt):   # 统一成"越大越该报"
                should, shouldnt = [-x for x in should], [-x for x in shouldnt]
            r = pg.suggest_threshold(should, shouldnt)
            if r:
                lines.append("")
                lines.append(f"→ 最有希望的是「{gm.METRIC_INFO[mt][0]}」"
                             f"（margin={best[1]:.2f}）")
                lines.append(f"   建议阈值 {r['suggested']:.3f}："
                             f"误报 {r['fp_at_suggested']} / 漏报 {r['fn_at_suggested']}"
                             f"（共 {r['n_should'] + r['n_shouldnt']} 条样本）")
                if r["fp_at_suggested"] + r["fn_at_suggested"] == 0:
                    lines.append("   ✅ 这一组样本上完全分开了 —— 但样本还少，"
                                 "换位置 / 换光照再录一轮确认再说。")
                else:
                    lines.append("   ⚠️ 还有分错的样本 —— 单独用这一个指标不够，"
                                 "要么做多指标组合，要么放弃。")
        else:
            lines.append("")
            lines.append("→ ⚠️ 没有任何指标 margin ≥ 1.5。")
            lines.append("   这**不一定**是代码问题 —— 也可能就是"
                         "「手部关键点分辨不了这个形态」。")
            lines.append("   建议：换角度 / 光照再多录一轮确认；若仍然重叠，")
            lines.append("   按 docs/01-plan.md §6 退回握笔器。")
        return "\n".join(lines), sep


def _isnan(v):
    try:
        return v != v
    except Exception:
        return False


def _zoom_thumb_index(frame, lm, size=210, pad=1.7):
    """把「拇指-食指」区域放大后贴到右下角。

    为什么要这个：判断孩子有没有「拇指包住食指」，**全靠这一小块区域的
    几个关键点**（拇指尖 + 食指三段）。如果 MediaPipe 在这里跟错了，
    看整张画面是完全看不出来的 —— 放大之后家长一眼就能确认
    关键点到底有没有贴在手指上。

    这一块是 occlusion 最容易出问题的地方（拇指压着食指），
    所以它值得单独占一块屏幕。
    """
    import cv2
    h, w = frame.shape[:2]
    if lm is None:
        return frame
    idx = (gm.THUMB_MCP, gm.THUMB_TIP, gm.INDEX_MCP, gm.INDEX_PIP, gm.INDEX_TIP)
    xs = [lm[i].x * w for i in idx]
    ys = [lm[i].y * h for i in idx]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    half = max(max(xs) - min(xs), max(ys) - min(ys)) * pad / 2.0
    half = max(half, 25.0)
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return frame

    crop = frame[y0:y1, x0:x1].copy()
    for a, b in HAND_CONNECTIONS:
        if a in idx and b in idx:
            cv2.line(crop,
                     (int(lm[a].x * w) - x0, int(lm[a].y * h) - y0),
                     (int(lm[b].x * w) - x0, int(lm[b].y * h) - y0),
                     (0, 220, 0), 2)
    for i in idx:
        px, py = int(lm[i].x * w) - x0, int(lm[i].y * h) - y0
        cv2.circle(crop, (px, py), 4, (255, 255, 255), -1)
    # 拇指尖单独标出来（它是这个判据的主角）
    tx, ty = int(lm[gm.THUMB_TIP].x * w) - x0, int(lm[gm.THUMB_TIP].y * h) - y0
    cv2.circle(crop, (tx, ty), 8, (60, 60, 255), 2)

    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(crop, (0, 0), (size - 1, size - 1), (0, 220, 0), 1)
    cv2.putText(crop, "thumb-index ZOOM", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)
    cv2.putText(crop, "check landmarks here", (6, size - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    ox, oy = w - size - 10, h - size - 78
    if ox > 0 and oy > 0:
        frame[oy:oy + size, ox:ox + size] = crop
    return frame


def draw(frame, lm, m, quality, counts, note="", zoom=True):
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
    # 右侧：候选指标。重点那两个（判断「包食指」的合成判据）放最上面并加亮。
    right = []
    if m:
        order = list(gm.THUMB_OVER_INDEX_PAIR) + [
            "thumb_index_gap", "thumb_index_angle",
            "fist", "curl_index", "tip_close", "spread"]
        for k in order:
            v = m.get(k)
            label = gm.METRIC_INFO[k][0]
            if v is None:
                right.append(("%-16s   n/a" % label, (160, 160, 160)))
                continue
            col = (120, 220, 255)
            if k in gm.THUMB_OVER_INDEX_PAIR:
                # 拇指越过量偏正 = 疑似包住食指 —— 变红提醒
                if k == "thumb_side" and v > 0.25:
                    col = (80, 80, 255)
                else:
                    col = (200, 255, 255)
            right.append(("%-16s %7.3f" % (label, v), col))

    for i, (t, c) in enumerate(lines):
        cv2.putText(out, t, (12, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, c, 2)
    for i, (t, c) in enumerate(right):
        cv2.putText(out, t, (w - 330, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, c, 1)

    # 底部：已录样本数 + 提示
    cnt = "  ".join(f"[{k}]{CLASSES[k][0][:2]}={counts[k]}" for k in CLASSES)
    cv2.putText(out, cnt, (12, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    cv2.putText(out, "1/2/3 record   p compare   r reset   w save   z zoom   q quit",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1)
    if note:
        cv2.putText(out, note, (12, h - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 200, 255), 2)
    if zoom:
        out = _zoom_thumb_index(out, lm)
    return out


def run_live(cam_idx, backend, out_dir):
    import cv2
    import mediapipe as mp

    model = pg.find_model(HAND_MODEL)
    hand = make_hand(model)
    cap = pg.open_camera(cam_idx, backend)
    rec = Recorder()

    print(f"模型：{os.path.basename(model)}")
    print("=" * 64)
    print("实测步骤（按顺序做）")
    print("=" * 64)
    print("① 摆机位：桌面斜上方俯视手部，让手占画面主体。")
    print("   先看左边三行：hand width 要 ≥200px、view 要是 OK。")
    print("   view 显示 marginal/bad 时录不进样本（会被拒绝）—— 先把机位调好。")
    print()
    print("② 盯右下角的「拇指-食指放大镜」：确认关键点确实贴在小手上。")
    print("   拇指尖画的是红圈。如果红圈飘在手指外面，说明这个角度跟不住，")
    print("   换另一侧试试（z 键可以开关放大镜）。")
    print()
    print("③ 让孩子分别摆出下面三种，每种按对应数字键录 5~10 帧：")
    print("     1 = 正确握笔     2 =【拇指包食指】（主要要抓的）   3 = 其他错误")
    print("   ⚠️ 每种都要换 2~3 个位置各录一遍（画面中间/偏左/偏右），")
    print("      只在一个位置录出来的高可分性是假的。")
    print()
    print("④ 按 p 出对照表 —— 那才是这个项目能不能做下去的依据。")
    print("   重点看「正确」和「包食指」那两列的差，以及 margin 后面的标记：")
    print("      ✅ 分得开(≥1.5)   ⚠️ 勉强(0.8~1.5)   ❌ 重叠(<0.8)")
    print()
    print("⑤ 按 w 把样本存成 json（留在 hand_probe\\ 目录，方便回看/发我复核）。")
    print("=" * 64)
    print()

    tick = 0
    last_note = 0.0
    zoom = True        # 拇指-食指放大镜，z 键切换
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
        cv2.imshow("pen grip probe  (1/2/3 record  p compare  z zoom  q quit)",
                   draw(fr, lm, m, quality, rec.counts(), note, zoom=zoom))
        k = cv2.waitKey(1) & 0xFF
        key = chr(k) if 0 <= k < 128 else ""

        if k in (ord("q"), 27):
            break
        elif key == "z":
            zoom = not zoom
            print(f"  拇指-食指放大镜：{'开' if zoom else '关'}")
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

    # 用合成手走一遍「记录 -> 对照表」全流程，确认逻辑不崩。
    # ⚠️ 每类都加了抖动 —— 不给抖动的话类内标准差是 0，margin 会被算成
    #    134 这种荒唐值，看着像"完美可分"，其实只是人造数据太干净。
    rec = Recorder()
    rng = np.random.default_rng(0)
    for _ in range(8):
        j = lambda: float(rng.normal(0, 0.06))          # noqa: E731
        rec.record("1", gm.compute_grip_metrics(*gm.synthetic_landmarks(
            curl=(0.2, 0.6, 0.7, 0.75),
            thumb_side=-0.35 + j(), thumb_along_shift=0.0 + j())))
        rec.record("2", gm.compute_grip_metrics(*gm.synthetic_landmarks(
            curl=(0.2, 0.6, 0.7, 0.75),
            thumb_side=+0.45 + j(), thumb_along_shift=0.15 + j())))
        rec.record("3", gm.compute_grip_metrics(*gm.synthetic_landmarks(
            curl=(1.0, 1.0, 1.0, 1.0), thumb_side=0.0 + j())))
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
