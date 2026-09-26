#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载手部关键点模型
==================

这个项目只需要一个模型：`hand_landmarker.task`（约 7.5MB）。
仓库里不放模型（体积 + GitHub 单文件限制），用这个脚本按需下载。

    python tools/fetch_models.py
    python tools/fetch_models.py --list     # 只看状态

⚠️ URL 里 `hand_landmarker` **出现两次**，写错会拿到错误页。
   本脚本用两条校验挡住：文件头不能是 XML/HTML、大小必须 ≥1MB。
"""
import argparse
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
MODEL_DIR = os.path.join(ROOT, "bench", "models")

MODEL = "hand_landmarker.task"
URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/1/hand_landmarker.task")

MIN_BYTES = 1 * 1024 * 1024      # 实际约 7.5MB，低于 1MB 一定是错误页


def main():
    ap = argparse.ArgumentParser(description="下载手部模型")
    ap.add_argument("--list", action="store_true", help="只看状态")
    ap.add_argument("--force", action="store_true", help="已存在也重下")
    args = ap.parse_args()

    dst = os.path.join(MODEL_DIR, MODEL)
    have = os.path.getsize(dst) if os.path.isfile(dst) else 0

    if args.list:
        print(f"模型目录：{MODEL_DIR}\n")
        mark = f"✅ {have / 1024 / 1024:.1f} MB" if have >= MIN_BYTES else "—　未下载"
        print(f"  {MODEL}   约 7.5MB   {mark}")
        return 0

    if have >= MIN_BYTES and not args.force:
        print(f"已存在（{have / 1024 / 1024:.1f} MB），跳过。要重下加 --force")
        return 0

    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = dst + ".part"
    print(f"下载 {MODEL} …（约 7.5MB）")
    try:
        with urllib.request.urlopen(URL, timeout=180) as r, open(tmp, "wb") as f:
            head = r.read(64)
            f.write(head)
            # ⚠️ 校验 1：错误页是 XML/HTML，正常模型是二进制 protobuf
            low = head.lstrip().lower()
            if low.startswith(b"<?xml") or low.startswith(b"<html") or low.startswith(b"<error"):
                os.remove(tmp)
                sys.exit(f"拿到的是错误页不是模型（URL 写错？）：\n  {head[:120]!r}")
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        sys.exit(f"下载失败：{e}")

    # ⚠️ 校验 2：大小下限
    size = os.path.getsize(tmp)
    if size < MIN_BYTES:
        os.remove(tmp)
        sys.exit(f"文件只有 {size} 字节，判定为无效")
    os.replace(tmp, dst)
    print(f"  ✅ {size / 1024 / 1024:.1f} MB -> {dst}")
    print("\n下一步：python hand_probe.py --camera 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
