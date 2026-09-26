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
    [  锁定「画面左半」那只手为被测手（画面正中有分界线）
    ]  锁定「画面右半」那只手
    \\  取消锁定，回到自动（按运动量猜写字的手）
    r  清空已记录的数据，重来
    p  打印对照表（这一步才是重点）
    w  把已录的样本存成 json（方便发给别人复核）
    z  开关「拇指-食指放大镜」（右下角）—— 判断包食指全靠那一块，务必盯它
    s  存一张当前画面
    q / ESC  退出

⚠️ **关于「测哪一只手」**（用户实测踩到的坑）
------------------------------------------
孩子写字时**左手也搭在桌面上**，两只手都在画面里。
原来的实现 `num_hands=1` 只回一只，很可能就是那只**静止的左手** ——
指标全算在错的手上，看起来就像"总是误识别左手"。

⚠️ 而 MediaPipe 的 `handedness` 标签（Left/Right）**不能用来选**：
   它假设输入图像是**镜像的**（自拍视角），外接摄像头是正像，
   所以标签会**左右报反** —— 这是"总是识别成左手"的直接来源。
   本工具改用**画面位置**（左半/右半）指定，完全不碰那个标签；
   标签只在屏幕上显示出来做参考，并标注 UNRELIABLE。

选法优先级：
  1. **手动锁定**（`[` / `]`，存进 `pen_grip_config.json`，下次沿用）—— 最可靠
  2. 自动：运动量大的那只（写字的手一直在动，搭在桌上的基本不动）
  3. 分不出来时**拒绝记录**并提示锁定 —— 录到错的手上比没数据更糟

好消息：**指标本身不关心左右手**。判据是手部内禀的，镜像下完全不变
（离线自检 B2 专门锁这条），所以左右手用同一套判据、正负方向不用翻。

⚠️ 另外，只有「视角合格」时才允许记录（手太小/抖动大会被拒绝并提示），
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
import math
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
        # ⚠️ 必须是 2，不是 1。
        #    孩子写字时**左手也搭在桌面上**，两只手都在画面里。
        #    设成 1 的话模型只回一只（很可能就是那只静止的左手），
        #    指标全算在错的手上 —— 这是实测踩到的坑。
        num_hands=2,
        min_hand_detection_confidence=0.3,   # 宁可检测到再筛，别漏
        min_tracking_confidence=0.3))


# ---------------------------------------------------------------- 配置持久化

CONFIG_PATH = os.path.join(ROOT, "pen_grip_config.json")


def load_cfg():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cfg(d):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


class HandSelector:
    """决定测**哪一只手**。

    ⚠️ 为什么需要它（用户实测踩到的）
    ------------------------------
    孩子写字时左手也搭在桌面上，**两只手都在画面里**。
    原来 `num_hands=1` 只回一只，很可能就是那只静止的左手 → 指标全算错。

    ⚠️ 为什么不能用 MediaPipe 的 handedness 标签来选
    ---------------------------------------------
    它的 Left/Right **假设输入图像是镜像的**（自拍视角）。
    外接摄像头是正像，所以标签会**左右报反** —— 实测就出现了"总是误识别左手"。
    所以这里用**画面位置**（左半 / 右半）来指定，完全不碰那个标签。
    标签只显示出来做参考，并明确标注"不可信"。

    选法优先级：
      1. **手动锁定**（`[` / `]` 键指定画面左半 / 右半那只，存进 config，下次记住）
         —— 最可靠。机位固定时锁一次就不用再管。
      2. 自动：**运动量大的那只**（写字的手一直在动，搭在桌上的基本不动）
      3. 分不出来时**拒绝记录**并提示按 `[` / `]` 锁定
    """

    def __init__(self, lock=None, window=20):
        self.lock = lock          # None / "screen_left" / "screen_right"
        self.window = window
        self.prev = []
        self.motion = []
        self.note = ""

    @staticmethod
    def center(lm, w, h):
        n = len(lm)
        return (sum(p.x for p in lm) / n * w, sum(p.y for p in lm) / n * h)

    def update(self, hands, w, h):
        """hands: [(lm, handedness_label)] -> 返回选中手的下标；分不出返回 None。"""
        centers = [self.center(lm, w, h) for lm, _ in hands]

        # 运动量 = 与上一帧最近掌心的位移（EMA 平滑）
        mot = []
        for c in centers:
            if not self.prev:
                mot.append(0.0)
            else:
                mot.append(min(math.hypot(c[0] - p[0], c[1] - p[1])
                               for p in self.prev))
        self.prev = centers
        self.motion = (mot if len(self.motion) != len(mot)
                       else [0.7 * a + 0.3 * b for a, b in zip(self.motion, mot)])

        if not hands:
            self.note = "画面里没检测到手"
            return None
        if len(hands) == 1:
            self.note = "画面里只有一只手 —— 就测它"
            return 0

        # ---- 1) 手动锁定（按画面位置，不看 handedness 标签）----
        if self.lock in ("screen_left", "screen_right"):
            want_left = self.lock == "screen_left"
            cand = [i for i, c in enumerate(centers)
                    if (c[0] < w / 2) == want_left]
            side = "左半" if want_left else "右半"
            if len(cand) == 1:
                self.note = f"已锁定：画面{side}那只"
                return cand[0]
            self.note = f"锁定了画面{side}，但那一侧现在有 {len(cand)} 只手 —— 请调整"
            return None

        # ---- 2) 自动：运动量大的是写字的手 ----
        best = max(range(len(hands)), key=lambda i: self.motion[i])
        if self.motion[best] < 1.5:        # 都几乎不动，分不出来
            self.note = "两只手都没怎么动 —— 按 [ 或 ] 指定写字的那只手"
            return None
        self.note = f"自动选中：动得多的那只（{self.motion[best]:.1f}px/帧）"
        return best

    def label(self, hands, sel, w, h):
        """给画面上每只手生成一行说明（含 handedness 标签，标注不可信）。"""
        out = []
        for i, (lm, hd) in enumerate(hands):
            c = self.center(lm, w, h)
            side = "左半" if c[0] < w / 2 else "右半"
            tag = "MEASURING" if i == sel else "ignored"
            mot = self.motion[i] if i < len(self.motion) else 0.0
            out.append((f"{side} x={c[0]:.0f} mp={mot:.1f} "
                        f"[{hd}?] {tag}", i == sel))
        return out


class Recorder:
    """按类别攒指标样本，并算「分不分得开」。"""

    def __init__(self, window=45):
        self.data = {k: [] for k in CLASSES}
        self.window = window      # 抖动统计的滑动窗口
        self.centers = []         # 最近若干帧的掌心位置，用来算抖动
        self.focus_hist = []      # 最近若干帧的两个判据指标，用来算"指标稳不稳"
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

    def add_focus(self, m):
        """记下两个判据指标的滚动历史，用来判断这个**角度**跟得稳不稳。

        ⚠️ 为什么需要看指标自己的波动（不只是掌心抖动）：
           判断「拇指包食指」靠的是拇指尖和食指那几个点。
           如果这个角度下拇指挡住了食指，MediaPipe 只能**猜**被挡住的点，
           表现就是：姿势没变、但指标在帧间乱跳。
           掌心位置可能很稳（因为手没动），而这两个指标在飘 —— 只看掌心抖动发现不了。
           所以要看**指标本身**的 std。
        """
        if not m:
            return
        v = (m.get("thumb_side"), m.get("thumb_index_dist"))
        if v[0] is None or v[1] is None:
            return
        self.focus_hist.append(v)
        del self.focus_hist[:-self.window]

    @property
    def focus_std(self):
        """返回 (thumb_side_std, thumb_index_dist_std)；样本不足返回 (None, None)。"""
        if len(self.focus_hist) < 8:
            return None, None
        a = np.array(self.focus_hist, dtype=float)
        return float(a[:, 0].std()), float(a[:, 1].std())

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


def draw(frame, lm, m, quality, counts, note="", zoom=True,
         hands=None, sel=None, sel_lines=None, lock=None, focus_std=(None, None)):
    """画手部特写 + 指标。⚠️ 只能用 ASCII（OpenCV 写不了中文）。

    hands / sel 用来把**两只手都画出来**，并明确标出哪一只在被测。
    这一点很关键：孩子左手也搭在桌上，如果只画一只，
    选错了完全看不出来（用户实测踩到的就是这个问题）。

    focus_std 是两个判据指标最近若干帧的波动，用来判断**这个角度可不可信**：
    姿势没变时 std 应该很小；std 大说明模型在"猜"被挡住的点（遮挡严重）。
    """
    import cv2
    h, w = frame.shape[:2]
    out = frame.copy()

    # 画面左右分界线 —— 锁定用的就是"左半/右半"，画出来让用户有依据
    if hands is not None and len(hands) > 1:
        cv2.line(out, (w // 2, 0), (w // 2, h), (90, 90, 90), 1)
        cv2.putText(out, "left half", (12, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
        cv2.putText(out, "right half", (w // 2 + 12, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)

    # 先画**没被测**的手（灰暗），再画被测的（亮绿），避免被盖住
    if hands:
        for i, (h_lm, _hd) in enumerate(hands):
            if i == sel:
                continue
            pts = [(int(p.x * w), int(p.y * h)) for p in h_lm]
            for a, b in HAND_CONNECTIONS:
                cv2.line(out, pts[a], pts[b], (110, 110, 110), 1)
            for p in pts:
                cv2.circle(out, p, 2, (150, 150, 150), -1)
            c = (int(sum(p.x for p in h_lm) / len(h_lm) * w),
                 int(sum(p.y for p in h_lm) / len(h_lm) * h))
            cv2.putText(out, "ignored", (c[0] - 30, c[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)

    if lm is not None:
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, pts[a], pts[b], (0, 220, 0), 2)
        for i, p in enumerate(pts):
            cv2.circle(out, p, 4 if i in (4, 8, 12, 16, 20) else 3,
                       (255, 255, 255), -1)

    # 左侧：视角质量（第 0 关）+ 选手机制状态
    qcolor = {"ok": (0, 200, 0), "marginal": (0, 180, 255), "bad": (0, 0, 255)}
    lines = []
    if m:
        lines.append(("hand width : %6.0f px" % m["px_w"], qcolor[quality]))
    else:
        lines.append(("hand: not detected", qcolor["bad"]))
    lines.append((("view: " + quality.upper()), qcolor[quality]))

    # 视角角：决定「横向的拇指-食指关系看不看得见」。越接近 90° 越好。
    # ⚠️ 这条是用户实测逼出来的：原机位从拇指侧平着看，食指被挡住。
    #    "改成俯视"能解决，但"够不够俯"靠感觉说不准 —— 这个角度能算出来。
    if m is not None:
        va = m.get("palm_view_angle")
        vlv, _vmsg = gm.palm_view_verdict(m)
        if va is None:
            lines.append(("palm view: n/a", (0, 0, 255)))
        else:
            lines.append((f"palm view: {va:5.1f} deg",
                          {"ok": (0, 200, 0), "marginal": (0, 180, 255),
                           "bad": (0, 0, 255)}[vlv]))
            if vlv != "ok":
                lines.append(("  -> raise the camera!", (0, 180, 255)
                              if vlv == "marginal" else (0, 0, 255)))

    # 指标稳定性 —— 判断「这个角度到底行不行」的第二个量化依据
    # ⚠️ 姿势没变时这两个指标的 std 应该很小。std 大 = 模型在猜被挡住的点。
    #    掌心可能很稳（手没动），所以只看掌心抖动发现不了这个问题。
    s0, s1 = focus_std
    if s0 is not None:
        worst = max(s0 / 0.08, (s1 or 0.0) / 0.05)
        if worst < 1.0:
            lines.append(("metric stability: GOOD", (0, 200, 0)))
        elif worst < 2.0:
            lines.append(("metric stability: FAIR", (0, 180, 255)))
        else:
            lines.append(("metric stability: POOR!", (0, 0, 255)))
            lines.append(("  -> change camera angle", (0, 0, 255)))

    if sel_lines is not None:
        lockname = {"screen_left": "LEFT half", "screen_right": "RIGHT half"}
        lines.append(("--- which hand ---", (200, 200, 200)))
        lk = lockname.get(lock, "AUTO (motion)")
        lines.append(("lock: " + lk,
                      (0, 220, 255) if lock is None else (0, 200, 0)))
        for t, is_sel in sel_lines:
            lines.append((t, (0, 220, 0) if is_sel else (140, 140, 140)))
        lines.append(("[?]=handedness label UNRELIABLE", (120, 120, 160)))

    # 右侧：候选指标。重点那两个（判断「包食指」的合成判据）放最上面并加亮。
    right = []
    if m:
        right.append(("--- metrics (measured hand) ---", (200, 200, 200)))
        sd = {"thumb_side": focus_std[0], "thumb_index_dist": focus_std[1]}
        thr = {"thumb_side": 0.08, "thumb_index_dist": 0.05}
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
                col = (200, 255, 255)
                if k == "thumb_side" and v > 0.25:
                    col = (80, 80, 255)          # 疑似包住 —— 变红
                s = sd.get(k)
                if s is not None:
                    # 把波动一起显示：姿势稳住时应该小；大 = 这个角度在"猜"
                    mark = "ok" if s < thr[k] else "NOISY"
                    right.append(("%-16s %7.3f +-%.3f %s"
                                  % (label, v, s, mark), col))
                    continue
            right.append(("%-16s %7.3f" % (label, v), col))

    for i, (t, c) in enumerate(lines):
        cv2.putText(out, t, (12, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, c, 2)
    for i, (t, c) in enumerate(right):
        cv2.putText(out, t, (w - 340, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, c, 1)

    # 底部：已录样本数 + 提示
    cnt = "  ".join(f"[{k}]{CLASSES[k][0][:2]}={counts[k]}" for k in CLASSES)
    cv2.putText(out, cnt, (12, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    cv2.putText(out, "1/2/3 rec  [/]=lock hand  \\=auto  p cmp  r reset  "
                     "w save  z zoom  q quit",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
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
    # 分辨率从探测配置读（tools/probe_camera.py 写），保证全流程一致
    cap = pg.open_camera_auto(idx=cam_idx, backend=backend)
    rec = Recorder()

    print(f"模型：{os.path.basename(model)}")
    print("=" * 64)
    print("实测步骤（按顺序做）")
    print("=" * 64)
    print("① 摆机位：**正面俯视** —— 摄像头在孩子前方稍高处，"
          "以 45~60° 俯角往下看手。")
    print("   ⚠️ 别摆成 90° 正俯：笔杆是往后倒的，正上方会被笔杆挡住手背。")
    print("   ⚠️ 也别平着从侧面看：横向的拇指-食指关系会被压扁")
    print("      （这正是之前从拇指侧平视失败的原因）。")
    print()
    print("   先看左边这几行，**三个都要达标**，否则录不进样本（会被拒绝）：")
    print("      hand width   >= 200px      手够大")
    print("      view         =  OK          综合视角合格")
    print("      palm view    ≈  90°         够不够俯 —— 这个数直接告诉你")
    print()
    print("   `palm view` 是怎么算的：手掌平面上两条基本正交的轴")
    print("   （食指根→小指根、手腕→中指根），从上方看时接近垂直；")
    print("   在手掌平面内平视会塌向 0/180°。所以 90° ≈ 正上方俯视，越小越平。")
    print()
    print("   还有两个辅助判断：")
    print("      metric stability  GOOD=指标很稳  POOR=模型在猜被挡住的点，")
    print("                        这个角度**不可用**，换角度再录")
    print("      右下角放大镜       红圈是拇指尖，看它和食指的点有没有贴住手指")
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

    # ---- 选哪一只手（实测踩到的问题：左手也搭在桌上，会被当成被测的手）----
    cfg = load_cfg()
    sels = HandSelector(lock=cfg.get("hand_lock"))
    if sels.lock:
        side = "画面左半" if sels.lock == "screen_left" else "画面右半"
        print(f"【选手】沿用上次的锁定：{side}那只")
    else:
        print("【选手】当前是自动模式（按运动量猜写字的手）。")
        print("       ⚠️ 如果屏幕上 MEASURING 标在了错的那只手上，立刻按")
        print("          [  = 锁定画面左半那只（画面正中有分界线）")
        print("          ]  = 锁定画面右半那只")
        print("          \\  = 取消锁定，回到自动")
        print("       锁定会记住，下次直接沿用。")
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

        # 两手都收下来（含 handedness 标签 —— 只做展示，不拿它做判断）
        hands = []
        for i, h_lm in enumerate(r.hand_landmarks or []):
            hd = "?"
            if r.handedness and i < len(r.handedness) and r.handedness[i]:
                cat = r.handedness[i][0]
                hd = getattr(cat, "category_name", None) or getattr(
                    cat, "display_name", "?")
            hands.append((h_lm, hd))

        sel = sels.update(hands, w, h)
        lm = hands[sel][0] if sel is not None else None
        m = gm.compute_grip_metrics(lm, w, h) if lm is not None else None
        rec.add_center(m, lm, w, h)
        rec.add_focus(m)                    # 指标稳定性（判断角度可不可信）
        quality, qmsg = gm.view_quality(m, rec.jitter)
        rec.last = (m, quality, sel, sels.note)

        note = rec.note if time.time() - last_note < 2.5 else ""
        cv2.imshow("pen grip probe  ([/] lock hand  p compare  z zoom  q quit)",
                   draw(fr, lm, m, quality, rec.counts(), note, zoom=zoom,
                        hands=hands, sel=sel,
                        sel_lines=sels.label(hands, sel, w, h),
                        lock=sels.lock, focus_std=rec.focus_std))
        k = cv2.waitKey(1) & 0xFF
        key = chr(k) if 0 <= k < 128 else ""

        if k in (ord("q"), 27):
            break
        elif key == "z":
            zoom = not zoom
            print(f"  拇指-食指放大镜：{'开' if zoom else '关'}")
        elif key in ("[", "]"):
            # 按**画面位置**锁定，不用 MediaPipe 的 handedness 标签（那个会报反）
            sels.lock = "screen_left" if key == "[" else "screen_right"
            cfg["hand_lock"] = sels.lock
            save_cfg(cfg)
            print(f"  ✓ 已锁定：{'画面左半' if key == '[' else '画面右半'}那只"
                  f"（已记住，下次沿用）")
            rec.note = "hand locked"
            last_note = time.time()
        elif key == "\\":
            sels.lock = None
            cfg.pop("hand_lock", None)
            save_cfg(cfg)
            print("  ✓ 已取消锁定，回到自动（按运动量猜）")
            rec.note = "auto hand"
            last_note = time.time()
        elif key in CLASSES:
            if sel is None:
                # ⚠️ 分不出哪只手是写字的手时**拒绝记录**。
                #    录到错的手上的数据比没数据更糟 —— 会污染结论。
                rec.note = "NO HAND SELECTED"
                print(f"  ✗ 拒绝记录（{CLASSES[key][0]}）：{sels.note}")
                print("     按 [ 或 ] 指定写字的那只手（画面正中有分界线）")
            elif quality != "ok":
                # ⚠️ 同理：视角不合格就拒绝记录。
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
            cv2.imwrite(p, draw(fr, lm, m, quality, rec.counts(), note,
                                zoom=zoom, hands=hands, sel=sel,
                                sel_lines=sels.label(hands, sel, w, h),
                                lock=sels.lock, focus_std=rec.focus_std))
            print(f"  已存 {p}")

    cap.release()
    cv2.destroyAllWindows()
    hand.close()
    cnt = rec.counts()
    print("\n=== 退出小结 ===")
    print("  样本数：" + "  ".join(f"{CLASSES[k][0]}={cnt[k]}" for k in CLASSES))
    if sum(cnt.values()) >= 15:
        print("  跑一次完整对照：python hand_probe.py --replay 上面用 w 存的文件")
    return rec


def run_video(path, out_dir):
    """分析一段**录像**（不用推流，手机架好录一段就行）。

    为什么加这个模式（用户实测的硬件约束）：
      笔记本**内置摄像头做不到俯视** —— 它在屏幕顶部边框里，
      光轴垂直于屏幕朝着使用者；正常打开时摄像头是**略微朝上**的，
      看到的是人脸不是桌面。要朝下 45° 就得把屏幕压向键盘，屏幕就没法看了。
      所以俯视必须靠**手机或外接 USB 摄像头**架在桌面上方。

      但"把手机当网络摄像头推流"有额外配置成本。这个模式绕开它：
      手机架好、录一段、传到电脑、直接分析。

    按键
    ----
        空格        播放 / 暂停（**打开时默认暂停**，方便先看清画面）
        .  /  ,     前进 / 后退一帧
        [  ]        锁定画面左半 / 右半那只手为被测手
        \\           取消锁定
        1 2 3       把当前帧记成 正确 / 拇指包食指 / 其他错误
        p 对照表    w 存样本    z 放大镜    s 存图    q 退出
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"打不开这个视频：{path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    hand = make_hand(pg.find_model(HAND_MODEL))
    rec = Recorder()
    cfg = load_cfg()
    sels = HandSelector(lock=cfg.get("hand_lock"))

    print(f"视频：{os.path.basename(path)}  {total} 帧  {fps:.1f} fps")
    print("默认**暂停**。先按空格播放，或按 . 逐帧看。")
    print("确认左边 hand width / view / palm view 达标、且 MEASURING 在写字的手上，")
    print("再按 1 / 2 录样本。按 p 出对照表。\n")

    idx = 0
    paused = True
    zoom = True
    tick = 0
    last_note = 0.0
    frame = None

    def goto(i):
        i = max(0, min(total - 1, i)) if total else max(0, i)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        return i, (fr if ok else None)

    while True:
        if frame is None:
            idx, frame = goto(idx)
            if frame is None:
                print("  读到末尾了")
                break
        h, w = frame.shape[:2]
        rgb = frame[:, :, ::-1].copy()
        tick += 33
        r = hand.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), tick)

        hands = []
        for i, h_lm in enumerate(r.hand_landmarks or []):
            hd = "?"
            if r.handedness and i < len(r.handedness) and r.handedness[i]:
                cat = r.handedness[i][0]
                hd = getattr(cat, "category_name", None) or "?"
            hands.append((h_lm, hd))

        sel = sels.update(hands, w, h)
        lm = hands[sel][0] if sel is not None else None
        m = gm.compute_grip_metrics(lm, w, h) if lm is not None else None
        rec.add_center(m, lm, w, h)
        rec.add_focus(m)
        quality, qmsg = gm.view_quality(m, rec.jitter)

        note = rec.note if time.time() - last_note < 2.5 else ""
        vis = draw(frame, lm, m, quality, rec.counts(), note, zoom=zoom,
                   hands=hands, sel=sel,
                   sel_lines=sels.label(hands, sel, w, h),
                   lock=sels.lock, focus_std=rec.focus_std)
        cv2.putText(vis, f"frame {idx}/{total}  {'PAUSED' if paused else 'PLAY'}",
                    (12, h - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 255), 2)
        cv2.imshow("pen grip probe - VIDEO MODE  (space=play  ./,=step)", vis)

        k = cv2.waitKey(0 if paused else int(1000 / max(1.0, fps))) & 0xFF
        key = chr(k) if 0 <= k < 128 else ""

        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            paused = not paused
        elif key in (".", ">", "l"):
            paused = True
            idx, frame = goto(idx + 1)
            continue
        elif key in (",", "<", "j"):
            paused = True
            idx, frame = goto(idx - 1)
            continue
        elif key == "z":
            zoom = not zoom
        elif key in ("[", "]"):
            sels.lock = "screen_left" if key == "[" else "screen_right"
            cfg["hand_lock"] = sels.lock
            save_cfg(cfg)
            print(f"  ✓ 已锁定：{'画面左半' if key == '[' else '画面右半'}那只（已记住）")
        elif key == "\\":
            sels.lock = None
            cfg.pop("hand_lock", None)
            save_cfg(cfg)
            print("  ✓ 已取消锁定")
        elif key in CLASSES:
            if sel is None:
                print(f"  ✗ 拒绝记录：{sels.note}（按 [ 或 ] 指定）")
            elif quality != "ok":
                print(f"  ✗ 拒绝记录（{CLASSES[key][0]}）：{qmsg}")
            else:
                rec.record(key, m)
                print(f"  ✓ 记录 {CLASSES[key][0]}（共 {rec.counts()[key]} 条）"
                      f"  帧 {idx}  {_brief(m)}")
            rec.note = f"recorded {key}"
            last_note = time.time()
        elif key == "p":
            report, _ = rec.compare()
            print("\n" + "=" * 68 + "\n" + report + "\n" + "=" * 68 + "\n")
            rec.note = "report -> console"
            last_note = time.time()
        elif key == "w":
            print(f"  已保存 {save_samples(rec, out_dir)}")
        elif key == "s":
            os.makedirs(out_dir, exist_ok=True)
            p = os.path.join(out_dir, time.strftime("video_%Y%m%d-%H%M%S.jpg"))
            cv2.imwrite(p, vis)
            print(f"  已存 {p}")

        if not paused:
            idx += 1
            frame = None

    cap.release()
    cv2.destroyAllWindows()
    hand.close()
    cnt = rec.counts()
    print("\n=== 退出小结 ===")
    print("  样本数：" + "  ".join(f"{CLASSES[k][0]}={cnt[k]}" for k in CLASSES))
    if sum(cnt.values()) >= 10:
        print(f"  重新出表：python hand_probe.py --replay <用 w 存的文件>")
    return rec


def scan_video(path, every=3):
    """不弹窗口、不交互，直接报告整段录像的机位质量。

    用途：**录完先跑这个**，快速确认「手够不够大、角度对不对、跟得稳不稳」，
    不用把整段视频从头看到尾。机位不合格就别费劲逐帧录样本了。

    ⚠️ 为什么需要它：笔记本做俯视时盖子要压下去、**屏幕根本看不见**，
    所以只能盲录。盲录完最需要的就是一个"不用看图"的体检报告。
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"打不开这个视频：{path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hand = make_hand(pg.find_model(HAND_MODEL))
    sels = HandSelector(lock=load_cfg().get("hand_lock"))
    rec = Recorder()

    print("=" * 64)
    print(f"扫描：{os.path.basename(path)}   {total} 帧  {fps:.1f} fps"
          f"   （每 {every} 帧取一帧）")
    print("=" * 64)

    tick = 0
    n = 0
    got = 0
    multi = 0
    widths = []
    angles = []
    qcount = {"ok": 0, "marginal": 0, "bad": 0}
    stab = {"good": 0, "fair": 0, "poor": 0}

    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        n += 1
        if n % every:
            continue
        h, w = fr.shape[:2]
        tick += 33
        r = hand.detect_for_video(mp.Image(
            image_format=mp.ImageFormat.SRGB, data=fr[:, :, ::-1].copy()), tick)
        hands = [(h_lm, "?") for h_lm in (r.hand_landmarks or [])]
        if len(hands) > 1:
            multi += 1
        if not hands:
            continue
        sel = sels.update(hands, w, h)
        if sel is None:
            continue
        lm = hands[sel][0]
        m = gm.compute_grip_metrics(lm, w, h)
        got += 1
        widths.append(m["px_w"])
        if m["palm_view_angle"] is not None:
            angles.append(m["palm_view_angle"])
        qc = gm.view_quality(m)[0]
        qcount[qc] = qcount.get(qc, 0) + 1
        rec.add_focus(m)
        s0, s1 = rec.focus_std
        if s0 is not None:
            worst = max(s0 / 0.08, (s1 or 0) / 0.05)
            stab["good" if worst < 1.0 else
                 ("fair" if worst < 2.0 else "poor")] += 1

    cap.release()
    hand.close()
    processed = max(1, len(widths))

    print(f"\n取帧数        {n}")
    print(f"检到手        {got} 帧（{got / max(1, n) * 100:.0f}%）")
    print(f"同时两只手     {multi} 帧"
          f"{'  ← 左手也在画面里，属正常' if multi else ''}")
    if not widths:
        print("\n❌ 全程没检到可用的手。先解决：")
        print("   1. 手在不在画面里？（盖子压下去后角度变了，很容易拍到别处）")
        print("   2. 剪一段有手的重录，或把笔记本挪近/调盖子角度")
        print("   3. 用 tools\\record.py 录完先看它存的快照 jpg")
        return 1

    widths.sort()
    print(f"\n手部像素宽    平均 {sum(widths) / len(widths):.0f}px   "
          f"中位 {widths[len(widths) // 2]:.0f}px   "
          f"最小 {widths[0]:.0f}  最大 {widths[-1]:.0f}")
    ok_px = sum(1 for x in widths if x >= gm.HAND_PX_OK)
    print(f"  ≥{gm.HAND_PX_OK:.0f}px 的比例   {ok_px / processed * 100:.0f}%")
    if angles:
        print(f"\n手掌视角角    平均 {sum(angles) / len(angles):.1f}°"
              f"   （90° = 从正上方看手；越小越接近平视）")
        aok = sum(1 for a in angles if abs(90 - a) <= gm.PALM_ANGLE_OK)
        print(f"  合格比例       {aok / len(angles) * 100:.0f}%")
    print(f"\n视角判定       ok {qcount['ok']}   marginal {qcount['marginal']}"
          f"   bad {qcount['bad']}")
    print(f"指标稳定性     GOOD {stab['good']}   FAIR {stab['fair']}"
          f"   POOR {stab['poor']}")

    print("\n" + "-" * 64)
    bad = []
    if ok_px / processed < 0.8:
        bad.append(f"手不够大（只有 {ok_px / processed * 100:.0f}% 的帧 ≥{gm.HAND_PX_OK:.0f}px）"
                   f"—— 把笔记本挪近些")
    if angles and sum(1 for a in angles if abs(90 - a) <= gm.PALM_ANGLE_OK) / len(angles) < 0.8:
        bad.append("视角偏斜（手掌视角角离 90° 太远）—— 盖子再压一点，"
                   "或把笔记本挪到手的正前方")
    if qcount["ok"] / processed < 0.8:
        bad.append(f"综合视角合格率只有 {qcount['ok'] / processed * 100:.0f}%")
    if stab["poor"] > processed * 0.2:
        bad.append("指标稳定性差 —— 模型在猜被挡住的点，这个角度不可信")

    if not bad:
        print("✅ 机位合格。可以开始逐帧录样本了：")
        print(f"   envs\\pg\\Scripts\\python.exe hand_probe.py "
              f"--video \"{path}\"")
    else:
        print("⚠️ 机位还需要调：")
        for b in bad:
            print("   · " + b)
        print("\n   ⚠️ 如果所有项都差，可能是「从侧面平视」而不是俯视 ——"
              "那样横向的拇指-食指关系会被压扁，判不出来。")
        print("   （笔记本要俯视只能把盖子往键盘方向压，屏幕会看不见，属正常）")
    print("=" * 64)
    return 0 if not bad else 1


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
    n_hands = len(r.hand_landmarks or [])
    m = gm.compute_grip_metrics(lm, w, h) if lm is not None else None
    quality, qmsg = gm.view_quality(m)
    hand.close()

    print(f"图片 {w}x{h}")
    if lm is None:
        print("❌ 画面里没检测到手 —— 先解决「手进不了画面」。")
        return 1
    if n_hands > 1:
        print(f"⚠️ 画面里检测到 **{n_hands} 只手**。静态图片里没法按运动量区分，")
        print("   这里只取了第一只来算 —— 结果**可能不是写字的那只手**。")
        print("   要正确区分请用 --camera 模式，按 [ 或 ] 锁定被测手。")
    print(f"手部像素宽：{m['px_w']:.0f}px   掌宽：{m['palm_w']:.0f}px")
    print(f"视角判定：{quality} —— {qmsg}")
    vlv, vmsg = gm.palm_view_verdict(m)
    print(f"视角角　：{vlv} —— {vmsg}")
    print("          （90° = 从正上方看手；越小说明越接近在手掌平面内平视，")
    print("            横向的拇指-食指关系会被压扁）")
    print("\n候选指标：")
    for k in ("fist", "curl_index", "tip_close", "spread", "thumb_index_gap",
              "thumb_side", "thumb_index_dist", "thumb_index_pos",
              "thumb_index_angle", "palm_view_angle"):
        v = m.get(k)
        print(f"  {gm.METRIC_INFO[k][0]:<16} {'n/a' if v is None else f'{v:.3f}'}")
    out = os.path.splitext(path)[0] + "_probe.jpg"
    cv2.imwrite(out, draw(fr, lm, m, quality, {k: 0 for k in CLASSES}))
    print(f"\n标注图已存：{out}")
    print("\n⚠️ 单张图只能说「看得见吗」。要判断「分得开吗」，"
          "得给两种握法各录一组，跑 --camera 模式。")
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
    ap.add_argument("--video", default=None,
                    help="分析一段录像（手机架好录一段，不用推流）")
    ap.add_argument("--scan", default=None,
                    help="只体检一段录像的机位质量，不弹窗、不交互")
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
    if args.scan:
        return scan_video(args.scan)
    if args.video:
        run_video(args.video, args.out_dir)
        return 0
    if args.camera is None:
        ap.error("给 --camera N / --image 路径 / --video 路径 / --scan 路径 "
                 "/ --replay 文件 / --selftest 之一")
    run_live(args.camera, args.backend, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
