"""从 CHANGELOG.md 抽出指定版本的小节, 作为 GitHub Release 正文.

发布流水线 (.github/workflows/release.yml) 在打 tag 后调用本脚本, 把
CHANGELOG 里对应版本的段落写成一个文件, 再交给 action-gh-release 的
``body_path``. 这样版本说明只维护 CHANGELOG 一份, 不必每次手动往 Release
页面粘贴.

为什么不直接用 ``generate_release_notes: true``:
    它按 PR 归纳变更。本仓库的提交都直接进 master、没有 PR, 所以它只能吐出
    一行 "Full Changelog" 链接, 正文实际是空的。

链接改写:
    Release 正文里的相对链接不会解析到仓库根 (它相对的是 release 页面 URL),
    所以这里统一改写成指向本次 tag 的绝对地址 —— 指 tag 而不是 master, 是为
    了让旧版本的说明始终指向它当时的文档。

用法:
    python tools/release_notes.py v2.0.0 -o notes.md

抽不到对应小节时以退出码 1 结束, 由调用方决定是退回自动生成还是直接失败。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = "https://github.com/Grefer/DeltaLab"
CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"

# 匹配 "## [2.0.0] - 2026-08-15" 起, 到下一个同级标题或文件末尾为止。
_SECTION = r"^## \[{version}\][^\n]*\n(?P<body>.*?)(?=^## \[|\Z)"

# 跳过 http(s):// 与页内锚点 (#...), 其余一律视为仓库内相对路径。
_RELATIVE_LINK = re.compile(r"\]\((?!https?://|#)([^)]+)\)")


def extract(version: str, text: str) -> str | None:
    """抽出 ``version`` 对应的小节正文; 找不到返回 None."""
    pattern = _SECTION.format(version=re.escape(version))
    match = re.search(pattern, text, re.S | re.M)
    if match is None:
        return None
    # 小节之间用 "---" 分隔, 它属于 CHANGELOG 的排版而不属于本节内容。
    return match.group("body").strip().rstrip("-").strip()


def render(version: str, body: str, *, date: str = "") -> str:
    """给正文补标题、把相对链接改写成绝对地址、附上版本对比链接。"""
    blob = f"{REPO}/blob/v{version}/"
    body = _RELATIVE_LINK.sub(lambda m: f"]({blob}{m.group(1)})", body)

    title = f"# DeltaLab v{version}"
    if date:
        title += f" — {date}"

    return (
        f"{title}\n\n{body}\n\n---\n\n"
        f"完整更新日志见 [CHANGELOG.md]({blob}CHANGELOG.md)。\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="tag 名, 形如 v2.0.0 (前缀 v 可省)")
    parser.add_argument("-o", "--output", type=Path,
                        help="输出文件; 省略则写到 stdout")
    parser.add_argument("--changelog", type=Path, default=CHANGELOG)
    args = parser.parse_args(argv)

    version = args.tag.lstrip("v")
    text = args.changelog.read_text(encoding="utf-8")

    body = extract(version, text)
    if body is None:
        print(f"CHANGELOG.md 里没有 [{version}] 小节", file=sys.stderr)
        return 1

    # 标题行里的日期 ("## [2.0.0] - 2026-08-15") 拿来放进 Release 标题。
    header = re.search(rf"^## \[{re.escape(version)}\]\s*-\s*(\S+)",
                       text, re.M)
    notes = render(version, body, date=header.group(1) if header else "")

    if args.output:
        args.output.write_text(notes, encoding="utf-8")
        print(f"已写入 {args.output} ({len(notes.encode())} 字节)",
              file=sys.stderr)
    else:
        sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
