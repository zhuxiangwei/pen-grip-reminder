#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
握笔姿势 · 手部指标计算
=======================

从 MediaPipe Hand Landmarker 的 **21 个手部关键点**算出与握笔相关的候选指标。

⚠️ 先读这段，免得对它期待过高
----------------------------
**这个模块不"识别握笔姿势"。** 它只提供候选特征值。
"正确握笔"和"错误握笔"之间到底哪个指标能分开，是一个**必须用真实数据回答的
经验问题** —— 而且文献给出的预期相当悲观：

  · MediaPipe 手部关节角相对 Vicon 真值的 RMSE 是 **22.5°**
    （Sensors 2025，Leica... 见 docs/04）
  · 而"握笔对不对"要靠的关节角差异往往只有 10~30°

误差和信号同一个量级。所以本模块的设计目标是：
  **把能算的都算出来，然后用 tools 里的对照工具去实测哪个真的分得开。**
不要因为某个指标"看起来合理"就把它当成判据。

设计约束（每一条都有理由）
------------------------
1. **必须先把归一化坐标换成像素坐标再算。**
   MediaPipe 的 x、y 是**各自**归一化到 [0,1] 的，1280x720 上 dx=0.1 是 128px、
   dy=0.1 只有 72px。直接在归一化空间算距离/角度会被宽高比压扁
   —— 坐姿项目里这个坑导致"侧面角度系统性漏报 44%"，这里一次性做对。
   回归锁见 tests：把整只手在像素空间里刚性旋转，角度类指标必须不变。

2. **用「掌宽」做归一化基准，不用包围盒。**
   包围盒大小会随手指张开程度变化，掌宽（食指根↔小指根）不随手指动作变。
   这样得到的指标不受"手离摄像头远近"和"手在画面哪个位置"影响。

3. **只用 2D，不用 MediaPipe 给的 z。**
   实测 z 方向的标准差是 x/y 的 1.5 倍以上（见 docs/01），不可靠。

4. **所有指标都是手部内禀的** —— 只用关键点之间的距离和相对方向，
   不依赖手在画面中的朝向。这样孩子手转一下不会让指标乱跳。

指标方向约定
-----------
每个指标的"哪个方向算异常"都写在函数文档里，但**那只是假设**，
要用手部对照工具实测确认。
"""
import math

# ---------------------------------------------------------------- 关键点索引
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

# 指标名 -> (中文名, 单位, 假设的"异常方向")。界面和日志都从这里取，
# 避免同一个指标在三个地方写三遍不同的话。
METRIC_INFO = {
    "px_w": ("手部像素宽", "px", ""),
    "px_h": ("手部像素高", "px", ""),
    "palm_w": ("掌宽", "px", ""),
    "hand_ratio": ("手宽占画面比", "%", "越小说明离得越远、越不可判"),
    "fist": ("握拳程度", "", "越小越像攥拳头（正常握笔应偏大）"),
    "curl_index": ("食指弯曲度", "", "越小越蜷"),
    "curl_middle": ("中指弯曲度", "", ""),
    "curl_ring": ("无名指弯曲度", "", ""),
    "curl_pinky": ("小指弯曲度", "", ""),
    "tip_close": ("指尖贴掌程度", "", "越小=指尖都缩回掌心，像攥拳"),
    "spread": ("指尖张开度", "", "越小越挤在一起"),
    "thumb_index_gap": ("拇指食指间距", "", "偏大可能没捏住笔"),
    "thumb_side": ("拇指横向越过量", "", "**偏正 = 拇指越到食指外侧（包食指）**"),
    "thumb_index_dist": ("拇指尖到食指的距离", "",
                         "**越小 = 真的搭在食指上**"),
    "thumb_index_pos": ("搭在食指哪一段", "",
                        "0=食指根，1=食指尖。≈0.3~0.9 = 压在中段"),
    "thumb_index_angle": ("拇指食指夹角", "°", "偏小可能捏太紧/包住"),
    "palm_view_angle": ("手掌视角角", "°", "**越接近 90° 越好**：说明是从上方看手"),
}

# 视角角的目标区间。依据见 palm_view_verdict()。
PALM_ANGLE_IDEAL = 90.0
PALM_ANGLE_OK = 30.0        # 与 90° 相差 ≤30° 算够俯
PALM_ANGLE_MARGINAL = 55.0  # 相差 ≤55° 勉强；再偏就是在手掌平面内看了

# 「拇指包住食指」这个具体问题的判据由**两个**指标合成，缺一不可：
#
#   thumb_side       拇指尖相对「食指根→小指根」这条横轴的偏移
#                    （正 = 越到食指外侧 = 包住）
#   thumb_index_dist 拇指尖到食指折线的最短距离（小 = 真的搭上去了）
#
# ⚠️ 为什么必须两个一起看：
#    只看 thumb_side → 拇指**伸得很长越过食指尖**、或者**根本没碰到食指**
#    也可能偏正，但那是别的形态，纠正方式完全不同。
#    只看距离 → 分不清"搭在食指的哪一侧"（正常捏笔时拇指也贴着食指）。
#    两个一起才能说清「压过食指」这件事。
#
# ⚠️ 为什么用「到折线的距离」而不是「沿食指轴的投影」：
#    食指一弯，MCP→TIP 这条轴就转向侧面了，投影会变成负数、语义失效。
#    距离对弯曲免疫。这个坑是离线自检 C9/C10 抓出来的。
THUMB_OVER_INDEX_PAIR = ("thumb_side", "thumb_index_dist")


def _px(lm, w, h, i):
    """关键点 -> 像素坐标。必须先做这一步，理由见模块文档第 1 条。"""
    p = lm[i]
    return (p.x * w, p.y * h)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _norm(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-9 else (0.0, 0.0)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _angle_between(v1, v2):
    """两个向量的夹角（度）。"""
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    c = max(-1.0, min(1.0, _dot(v1, v2) / (n1 * n2)))
    return math.degrees(math.acos(c))


def _angle_at(a, vertex, b):
    """a-vertex-b 的夹角（度）。"""
    v1, v2 = _sub(a, vertex), _sub(b, vertex)
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    c = max(-1.0, min(1.0, _dot(v1, v2) / (n1 * n2)))
    return math.degrees(math.acos(c))


def _seg_dist(p, a, b):
    """点 p 到线段 ab 的最短距离，以及最近点在线段上的参数 t∈[0,1]。"""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(p[0] - ax, p[1] - ay), 0.0
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy)), t


def _curl(p_mcp, p_pip, p_dip, p_tip):
    """手指弯曲度：指尖到指根的距离 / 三节指骨长度之和。

    = 1   完全伸直（指尖到指根的距离等于骨长之和）
    → 越小 越蜷（蜷起来时指尖绕回指根附近）
    """
    reach = _dist(p_mcp, p_tip)
    chain = _dist(p_mcp, p_pip) + _dist(p_pip, p_dip) + _dist(p_dip, p_tip)
    return (reach / chain) if chain > 1e-9 else None


def compute_grip_metrics(lm, w, h):
    """21 个手部关键点 -> 指标字典。算不出来的项给 None（而不是 0）。

    lm : MediaPipe 的 hand_landmarks（21 个 NormalizedLandmark）
    w,h: 这一帧图像的像素宽高（**必需** —— 见模块文档第 1 条）

    返回 dict，键见 METRIC_INFO。数据不足时对应值为 None。
    """
    P = [_px(lm, w, h, i) for i in range(21)]
    out = {k: None for k in METRIC_INFO}

    xs = [p[0] for p in P]
    ys = [p[1] for p in P]
    out["px_w"] = max(xs) - min(xs)
    out["px_h"] = max(ys) - min(ys)

    # 归一化基准：掌宽（食指根 ↔ 小指根）。不随手指张合变化，所以稳定。
    palm = _dist(P[INDEX_MCP], P[PINKY_MCP])
    out["palm_w"] = palm
    if palm < 1e-6:
        return out                      # 手退化成一条线，所有比值都没意义
    out["hand_ratio"] = palm / float(w) if w else None

    # ---- 各手指弯曲度 ----
    curls = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        curls[name] = _curl(P[mcp], P[pip], P[dip], P[tip])
        out[f"curl_{name}"] = curls[name]
    got = [v for v in curls.values() if v is not None]
    out["fist"] = (sum(got) / len(got)) if got else None

    # ---- 指尖到掌心（拳握时全部缩回，值很小）----
    cx = sum(P[i][0] for i in (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)) / 5.0
    cy = sum(P[i][1] for i in (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)) / 5.0
    tips = [_dist(P[t], (cx, cy)) / palm for t in
            (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)]
    out["tip_close"] = sum(tips) / len(tips)

    # ---- 指尖张开度：相邻指尖两两距离的均值 ----
    order = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    gaps = [_dist(P[order[i]], P[order[i + 1]]) / palm for i in range(len(order) - 1)]
    out["spread"] = sum(gaps) / len(gaps) if gaps else None

    # ---- 拇指-食指 ----
    out["thumb_index_gap"] = _dist(P[THUMB_TIP], P[INDEX_TIP]) / palm

    # 拇指横向"越过"多少：把拇指尖相对食指根的位移，投影到掌内横向轴上
    # （食指根 → 小指根 的方向）。
    #   负 = 拇指在食指靠拇指的一侧（正常捏笔）
    #   正 = 拇指越到了食指另一侧（**假设**这就是「拇指包食指」）
    # ⚠️ 这个方向是假设，必须用对照工具实测确认。
    across = _norm(_sub(P[PINKY_MCP], P[INDEX_MCP]))
    out["thumb_side"] = _dot(_sub(P[THUMB_TIP], P[INDEX_MCP]), across) / palm

    # 拇指尖到**食指折线**（根→尖三段）的最短距离，以及最近点落在食指哪一段。
    #
    # ⚠️ 为什么不用「拇指尖沿食指轴的投影」衡量位置：
    #    食指一弯，MCP→食指尖 这条轴就转向侧面了，投影会变成负数、语义失效。
    #    距离对弯曲免疫。这个坑是离线自检 C9/C10 抓出来的。
    chain = (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    segs = [(P[chain[i]], P[chain[i + 1]]) for i in range(3)]
    lens = [_dist(a, b) for a, b in segs]
    total = sum(lens) if sum(lens) > 1e-9 else 1e-9
    best_d, best_pos, acc = None, None, 0.0
    for (a, b), L in zip(segs, lens):
        d, t = _seg_dist(P[THUMB_TIP], a, b)
        if best_d is None or d < best_d:
            best_d, best_pos = d, (acc + t * L) / total
        acc += L
    out["thumb_index_dist"] = best_d / palm
    out["thumb_index_pos"] = best_pos

    out["thumb_index_angle"] = _angle_at(P[THUMB_MCP], P[INDEX_MCP], P[INDEX_PIP])

    # ---- 视角角：判断"你是从哪个方向看这只手" ----
    #
    # 手掌平面上有两条基本正交的轴：
    #     across = 食指根 → 小指根      （掌的横向轴）
    #     along  = 手腕   → 中指根      （掌的纵向轴）
    #
    # 从**正上方**（沿手掌法线）看时，这两条轴在画面里接近垂直；
    # 如果是在**手掌平面内**平视，其中一条会被投影压短，夹角塌向 0° 或 180°。
    #
    # 所以这个夹角直接告诉我们：视角够不够"俯"。
    # 对判断「拇指有没有越过食指」很关键 —— 那是个**横向**关系，
    # 在手掌平面内平视时横向信息被压扁，等于看不见。
    out["palm_view_angle"] = _angle_between(
        _sub(P[PINKY_MCP], P[INDEX_MCP]), _sub(P[MIDDLE_MCP], P[WRIST]))
    return out


# ---------------------------------------------------------------- 视角质量门控

# 手部像素宽度门槛。依据见 docs/04：MediaPipe 的掌部检测器把 ROI 缩到 224x224，
# 手太小就抓不稳；而我们要分辨几十度的形态差异，需要更高余量。
# ⚠️ 经验值，不是论文阈值 —— 真正确认要看手部对照工具的实测。
# 2026-09-27 调整：用户实测固定距离的「正前方俯视」机位下，手宽最多只能到
# ~150px（200px 物理上达不到）。把门槛放宽到 130/80，让 150+ 稳定判为 ok，
# 避免所有样本都被标成 marginal 一直告警。指标本身大多按掌宽归一化，
# 对绝对像素大小不敏感，放宽门槛不影响判据可靠性。
HAND_PX_OK = 130.0
HAND_PX_MARGINAL = 80.0


def view_quality(m, stability_px=None):
    """判断这一帧的指标**值不值得拿去下结论**。

    ⚠️ 这是从坐姿项目学到的教训：一定要有门控。
       文献里记录过 MediaPipe 在遮挡+运动时**连续约 4 秒完全取不到有效坐标**；
       没有门控就会在这些帧上疯狂误报。
       "跟不住的时候不下结论" 比 "尽量多判几次" 重要得多。

    返回 (level, 说明)。level ∈ ok / marginal / bad
    """
    if not m or m.get("palm_w") is None:
        return "bad", "没检测到手"
    w = m["px_w"]
    if w < HAND_PX_MARGINAL:
        return "bad", f"手部仅 {w:.0f}px 宽 —— 太小，这个距离下判定不可靠"
    if w < HAND_PX_OK:
        return "marginal", f"手部 {w:.0f}px 宽 —— 偏小，先把机位拉近"
    if stability_px is not None and stability_px > HAND_PX_OK * 0.25:
        return "marginal", f"跟踪抖动 {stability_px:.0f}px —— 手在动，先稳一下"
    return "ok", f"手部 {w:.0f}px 宽 —— 可以判"


def palm_view_verdict(m):
    """判断「你是从哪个方向看这只手」—— 决定横向的拇指-食指关系看不看得见。

    返回 (level, 说明)。level ∈ ok / marginal / bad

    ⚠️ 为什么单独给这个判据（用户实测的直接教训）：
       用户原本的机位是从**拇指那一侧平着看**，要看的食指被拇指挡住。
       "把摄像头抬高、改成俯视"能解决，但"够不够俯"光靠感觉说不准。
       这个角度是可以算出来的 —— 算出来就不会调错方向。
    """
    if not m:
        return "bad", "没有数据"
    a = m.get("palm_view_angle")
    if a is None:
        return "bad", "算不出视角角（关键点退化）"
    d = abs(PALM_ANGLE_IDEAL - a)
    if d <= PALM_ANGLE_OK:
        return "ok", f"视角角 {a:.0f}°（接近正交）—— 够俯，横向关系看得见"
    if d <= PALM_ANGLE_MARGINAL:
        return "marginal", (f"视角角 {a:.0f}° —— 偏斜，能判但要打折扣；"
                            f"再把摄像头抬高些")
    return "bad", (f"视角角 {a:.0f}° —— 几乎在手掌平面内看，"
                   f"拇指和食指的左右关系被压扁了；**必须改成俯视**")


# ---------------------------------------------------------------- 合成手（测试/演示）

def synthetic_hand(curl=(0.55, 0.75, 0.80, 0.85), thumb_side=0.0,
                   thumb_spread=1.0, thumb_along_shift=0.0,
                   scale=1.0, rot_deg=0.0, dx=0.0, dy=0.0):
    """造一只可控的合成手，返回 21 个像素坐标。

    **只用于测试和演示，不参与运行时判定。** 存在的意义是：让"指标算得对不对"
    这件事可以离线验证 —— 不需要摄像头、不需要真的孩子配合。

    curl             : 四指弯曲度 0(伸直)~1(蜷到底)，顺序 index/middle/ring/pinky
    thumb_side       : 拇指横向偏移（正 = 往小指侧摆，模拟"包住食指"）
    thumb_spread     : 拇指远离食指的程度（越小越贴着食指）
    thumb_along_shift: 拇指沿食指方向平移（正 = 往食指尖方向伸）
    其余参数对整个手做刚性变换（用来验证指标的不变性）

    坐标系：+y 指向指尖方向，x 横跨手掌（负 = 拇指侧）。单位约等于掌宽。
    """
    def rot(v, deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)

    # 掌心骨架（单位：掌宽）
    base = {
        WRIST: (0.00, 0.00),
        THUMB_CMC: (-0.42, 0.18),
        THUMB_MCP: (-0.62, 0.42),
        THUMB_IP: (-0.74, 0.70),
        THUMB_TIP: (-0.80, 0.95),
        INDEX_MCP: (-0.40, 0.32),
        MIDDLE_MCP: (0.00, 0.35),
        RING_MCP: (0.34, 0.30),
        PINKY_MCP: (0.62, 0.18),
    }
    pts = dict(base)

    # 指骨长度（近→远）
    segs = {"index": (0.34, 0.22, 0.16), "middle": (0.38, 0.25, 0.17),
            "ring": (0.34, 0.23, 0.16), "pinky": (0.26, 0.17, 0.12)}
    # 每节关节的最大弯曲角。三节合计 255° ≈ 真实手指蜷到底的程度
    # （真实大致是 MCP 90° + PIP 100° + DIP 70°）。
    # ⚠️ 一开始取 62°（合计 186°），测出来"蜷到底"的 curl 只降到 0.67，
    #    根本不像拳头 —— 合成手的活动范围不够，测试就失去意义了。
    BEND = 85.0

    for idx, name in enumerate(("index", "middle", "ring", "pinky")):
        mcp, pip, dip, tip = FINGERS[name]
        d = _norm(_sub(base[mcp], base[WRIST]))     # 手指自然张开方向
        th = BEND * max(0.0, min(1.0, curl[idx]))
        cur = base[mcp]
        for j, (aidx, ln) in enumerate(zip((pip, dip, tip), segs[name])):
            d = rot(d, th)                          # 每过一节关节就多弯一点
            cur = (cur[0] + d[0] * ln, cur[1] + d[1] * ln)
            pts[aidx] = cur

    # 拇指：横向偏移 + 与食指的分开程度 + 沿手指方向平移
    # ⚠️ 平移必须用**纯 +y**，不能用「食指根→食指尖」那种带 x 分量的方向 ——
    #    否则调 thumb_along_shift 时会顺带改变 thumb_side，两个参数被耦合，
    #    "独立变化"的测试就没法做了。（离线自检 C10 专门盯这个耦合。）
    for aidx, k in ((THUMB_MCP, 0.5), (THUMB_IP, 0.75), (THUMB_TIP, 1.0)):
        x, y = pts[aidx]
        pts[aidx] = (x + thumb_side * k,
                     y - (1.0 - thumb_spread) * 0.30 * k + thumb_along_shift * k)

    out = []
    for i in range(21):
        x, y = pts[i]
        x = x * scale
        y = y * scale
        x, y = rot((x, y), rot_deg)
        out.append((x + dx, y + dy))
    return out


class _LM:
    """把像素坐标伪装成 MediaPipe 关键点（只需 .x/.y，且是归一化值）。"""
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x, self.y = x, y


def to_normalized(pixel_pts, w, h):
    """像素坐标 -> MediaPipe 那种「x、y 各自归一化」的形式。

    ⚠️ 这个转换是**有损的**（丢掉了宽高比），真正的价值在于它能把
       compute_grip_metrics 的"必须先转回像素"这一步测出来。
    """
    return [_LM(x / w, y / h) for (x, y) in pixel_pts]


def synthetic_landmarks(w=1280, h=720, **kw):
    """造一只合成手并返回 (归一化关键点, w, h)，直接喂给 compute_grip_metrics。"""
    pts = synthetic_hand(**kw)
    return to_normalized(pts, w, h), w, h
