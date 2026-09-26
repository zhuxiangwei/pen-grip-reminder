# 握笔姿势提醒（Pen Grip Reminder）

给家里的孩子做的**握笔姿势提醒**。用一颗对着桌面的摄像头拍写字的手，
从手部关键点判断握笔形态，不对就语音提醒一句。

**当前状态：验证阶段 —— 还没写提醒逻辑，也没写界面。**
这是故意的，原因见下。

---

## 先说清楚：这个项目可能做不成

这不是谦虚，是文献数据摆在那儿：

| 事实 | 数值 | 影响 |
|---|---|---|
| MediaPipe 手部关节角相对真值（Vicon）的误差 | **22.5°** | 而"握笔对不对"要靠的差异只有 **10~30°** —— 误差和信号同一个量级 |
| 误差最大的关节 | **PIP（近端指间关节）** | 恰恰是判握笔最需要的那个 |
| 运动中的手部检出率 | 从 99% 掉到 **83~85%** | 写字一直在动 |
| 记录到的最长连续丢失 | **约 4 秒** | 遮挡+运动场景 |

所以**只能做粗判**（拳握、拇指包食指这种大形态差异），别指望分清
"拇指搭在食指哪一节"。完整评估：[`docs/00-feasibility.md`](docs/00-feasibility.md)

**这也是为什么先设两个叫停点、先不写功能。**

---

## 两个叫停点

| | 问题 | 通过判据 | 不通过 |
|---|---|---|---|
| **第 0 关** | 看得见吗？ | 手宽 **≥200px**、检出率 **≥90%** | 改机位/换相机；仍不行 → **停** |
| **第 1 关** | 分得开吗？ | 至少一个指标 **margin ≥ 1.5**，且最优阈值下误报漏报都很少 | → 退回握笔器 |

进度：

| 项目 | 状态 |
|---|---|
| 指标算法 | ✅ 完成，离线自检全部通过 |
| 对照实验工具 | ✅ 完成 |
| **第 0 关实测（真实摄像头）** | ⏳ **待做 —— 需要你** |
| **第 1 关实测（三种握法对照）** | ⏳ **待做 —— 需要你和孩子** |
| 界面 / 语音 / 提醒 | 🚫 故意未开始 |

---

## 怎么跑

### 1. 环境

依赖很少（只有 opencv + mediapipe + numpy）。可以直接复用坐姿项目的 venv：

```bash
..\posture-reminder\envs\posture\Scripts\python.exe --version
```

或者自己建（步骤和理由见 `requirements.txt`）：

```bash
python -m venv envs\pg
envs\pg\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
envs\pg\Scripts\python.exe -m pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple mediapipe
envs\pg\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple absl-py certifi flatbuffers numpy
```

### 2. 下模型

```bash
python tools\fetch_models.py
```

### 3. 第 0 关：先确认看得见（0 成本）

拿手机拍一张孩子握笔的照片，或把手机装在桌上当摄像头：

```bash
python hand_probe.py --image D:\握笔照片.jpg
python hand_probe.py --camera 0
```

看画面左侧两个数：**hand width ≥200px**、**view: OK**。

### 4. 第 1 关：对照实验（重点）

```bash
python hand_probe.py --camera 0
```

按键：

| 键 | 作用 |
|---|---|
| `1` / `2` / `3` | 把当前帧记成「正确握笔」/「拇指包食指」/「拳握」 |
| `p` | **打印对照表** |
| `r` | 清空重来 |
| `w` | 存样本 json |
| `s` | 存一张标注画面 |
| `q` | 退出 |

**实验要点**：每种握法录 5~10 帧，而且**要换 3 个位置各录一遍**
（画面中央、偏左、偏右）。只在一个位置录出来的高可分性是假的 ——
那可能只是在认位置，不是在认握法。

⚠️ 只在 `view: OK` 时才接受记录。手太小/抖得厉害时录的样本没意义，会被拒绝。

### 5. 算法自检（不需要摄像头）

```bash
python tests\test_grip_offline.py
```

---

## 项目结构

```
pen-grip-reminder/
├── grip_metrics.py            核心：21 个手部点 -> 候选指标（纯函数，可离线测）
├── pg_utils.py                相机/模型/阈值工具（自包含）
├── hand_probe.py              对照实验工具（视角门控 + 记录分类 + 出对照表）
├── tests/test_grip_offline.py 指标算法自检
├── tools/fetch_models.py      下模型
├── docs/
│   ├── 00-feasibility.md      可行性评估（含文献数据、方案对比）
│   └── 01-plan.md             实施计划（里程碑、实验协议、退路）
└── bench/models/              模型（不入库）
```

---

## 为什么是独立项目

坐姿项目（`posture-reminder`）已经有一套成熟的东西，为什么不全放一起？

**因为两者机位在几何上互斥**：

| | 坐姿（侧面） | 握笔 |
|---|---|---|
| 取景 | 上半身 | 手 + 笔，约 15×15cm |
| 距离 | 1~1.5m | 30~60cm |
| 角度 | 侧方 80~90° | 俯视 |

要拍到躯干，**手在画面里只有 60~100px**；要把手拍大，身体就出画。
而且侧视角下笔杆几乎与视线平行、被手指完全遮住。

**必须两颗摄像头、两个独立进程。** 设计结论复用（相机封装、阈值建议、
迟滞规则的做法），但**代码不 import** —— 两个仓库之间不留硬依赖。

---

## 退路

如果两关没过，**不要硬做视觉方案**：

**握笔器，5~20 元，物理约束，成熟几十年，当天见效。**

两条路不冲突 —— 可以先用握笔器，视觉方案当"抽查"。
具体的退路判据见 [`docs/01-plan.md`](docs/01-plan.md) §6。

---

## 许可

暂未附许可协议（默认「保留所有权利」）。
