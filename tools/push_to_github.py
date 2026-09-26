#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 pen-grip-reminder 推到 GitHub：检查远端、配 remote、试推。"""
import json
import os
import subprocess
import sys
import urllib.request

REPO = "zhuxiangwei/pen-grip-reminder"
API = f"https://api.github.com/repos/{REPO}"
PROJ = r"C:\Users\tmpee\Downloads\pen-grip-reminder"
GCM = "C:/Program Files/Git/mingw64/bin/git-credential-manager.exe"
PROMPT_TIMEOUT = 90


def git(*args, timeout=PROMPT_TIMEOUT, env_extra=None):
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"          # 需要终端输入时立刻失败，别挂住
    if env_extra:
        env.update(env_extra)
    try:
        r = subprocess.run(["git", *args], cwd=PROJ, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=env)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 999, f"!! 超时（{timeout}s）—— 大概率卡在认证交互上"


def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "setup"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)


def main():
    print("=" * 66)
    print(f"把本地仓库推到 {REPO}")
    print("=" * 66)

    # ---- 1. 远端仓库状态 ----
    print("\n【1】远端仓库状态")
    try:
        d = api(API)
        print(f"  full_name : {d['full_name']}")
        print(f"  可见性    : {'私有' if d['private'] else '★ 公开'}")
        print(f"  默认分支  : {d['default_branch']}")
        print(f"  size      : {d['size']} KB（0 = 空仓库）")
    except Exception as e:
        sys.exit(f"  ❌ 读不到仓库：{e}\n     确认地址对不对：{API}")

    # ---- 2. 本地状态 ----
    print("\n【2】本地仓库状态")
    rc, out = git("rev-parse", "--abbrev-ref", "HEAD")
    print(f"  分支      : {out.strip()}")
    rc, out = git("rev-list", "--count", "HEAD")
    print(f"  提交数    : {out.strip()}")
    rc, out = git("ls-files")
    print(f"  文件数    : {len(out.strip().splitlines())}")
    rc, out = git("status", "--porcelain")
    print(f"  工作区    : {'干净' if not out.strip() else '有改动: ' + out.strip()[:100]}")
    rc, out = git("tag", "-l")
    print(f"  标签      : {out.strip() or '（无）'}")

    # ---- 3. 配 remote ----
    print("\n【3】配置 remote")
    rc, out = git("remote", "get-url", "origin")
    if rc == 0 and out.strip():
        print(f"  已存在：{out.strip()}")
    else:
        url = f"https://github.com/{REPO}.git"
        rc, out = git("remote", "add", "origin", url)
        print(f"  已添加 origin -> {url}" if rc == 0 else f"  ❌ 失败：{out}")

    # ---- 4. 凭据助手（本仓库范围，不动全局）----
    print("\n【4】凭据助手")
    rc, out = git("config", "--local", "credential.helper")
    print(f"  当前（local）：{out.strip() or '（未设）'}")
    if os.path.isfile(GCM):
        git("config", "--local", "credential.helper", GCM)
        rc, out = git("config", "--local", "credential.helper")
        print(f"  已指向可用的 GCM：{out.strip()}")
        print("  （只改本仓库，不动你的全局配置、不影响别的仓库）")
    else:
        print(f"  ⚠️ 没找到 {GCM}，沿用全局配置")

    # ---- 5. 远端是否已有分支 ----
    print("\n【5】远端分支情况")
    rc, out = git("ls-remote", "--heads", "origin")
    if rc != 0:
        print(f"  ⚠️ 读不到远端：{out.strip()[:200]}")
        remote_empty = None
    else:
        heads = [l for l in out.strip().splitlines() if l.strip()]
        if heads:
            print("  远端已有分支：")
            for h in heads:
                print(f"    {h}")
            remote_empty = False
        else:
            print("  远端是空的 ✓（干净的首次推送）")
            remote_empty = True

    # ---- 6. 试推 ----
    print("\n【6】推送")
    print("  （最多等 90 秒，卡在认证交互就直接失败，不挂住）")
    rc, out = git("push", "-u", "origin", "main", timeout=PROMPT_TIMEOUT)
    print(f"  exit={rc}")
    for line in (out.strip().splitlines() or ["（无输出）"])[:15]:
        print(f"    {line}")

    if rc == 0:
        print("\n" + "=" * 66)
        print("✅ 推送成功")
        print(f"   https://github.com/{REPO}")
        print("=" * 66)
        return 0

    print("\n" + "=" * 66)
    print("❌ 没推上去。判断一下是哪种情况：")
    print()
    if "Authentication" in out or "credential" in out or "could not read Username" in out:
        print("  → 认证问题。**这一步只能你来**：")
        print("     双击本目录的「推送到GitHub.bat」，")
        print("     或手动跑： git push -u origin main")
        print("     首次会弹浏览器让你登录 GitHub，跟着走完即可。")
    elif "rejected" in out or "non-fast-forward" in out or "fetch first" in out:
        print("  → 远端已有提交、和本地不是同一条历史。")
        print("     如果远端只是建仓库时自动生成的 README，可以：")
        print("       git pull --rebase origin main --allow-unrelated-histories")
        print("       git push -u origin main")
    elif "CONNECT" in out or "502" in out or "443" in out or "timed out" in out:
        print("  → 网络到不了 github.com（代理只放行了 api.github.com）。")
        print("     注意：**沙箱环境**下会这样，你本机不一定。")
        print("     在你自己的终端里跑通常就通了。")
    else:
        print("  → 看上面 exit 后面的原始输出。")
    print("=" * 66)
    return 1


if __name__ == "__main__":
    sys.exit(main())
