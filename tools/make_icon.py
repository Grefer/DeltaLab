"""生成 DeltaLab GUI 图标.

输出:
  assets/deltalab.png  (512x512, 透明圆角)
  assets/deltalab.ico  (16/32/48/64/128/256 多尺寸)

设计:
  - 圆角方形底 + 品牌蓝纵向渐变 (#2563EB -> #1E40AF)
  - 中心白色希腊字母 Δ
  - 大尺寸 (>=64) 叠加白色对冲盈亏曲线, 穿过 Δ
  - 底部金色装饰线 (大尺寸)

用法: python tools/make_icon.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---- 品牌色 (与 gui_app.py 中 PALETTE 对齐) ----
BLUE_TOP    = (37, 99, 235)     # #2563EB  primary
BLUE_BOTTOM = (30, 64, 175)     # #1E40AF  primary_act
GOLD        = (184, 134, 11)    # #B8860B  gold
WHITE       = (255, 255, 255)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


# ------------------------------------------------------------------
#  字体查找 (尽量挑一个带良好 Δ 字形的 sans-serif)
# ------------------------------------------------------------------
def _find_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVu Sans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ------------------------------------------------------------------
#  单张图标渲染
# ------------------------------------------------------------------
def _rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=radius, fill=255
    )
    return mask


def _gradient_bg(size: int) -> Image.Image:
    """纵向线性渐变, 略带对角漂移, 避免纯色呆板."""
    bg = Image.new("RGB", (size, size), BLUE_TOP)
    px = bg.load()
    for y in range(size):
        t = y / max(1, size - 1)
        r = int(BLUE_TOP[0] * (1 - t) + BLUE_BOTTOM[0] * t)
        g = int(BLUE_TOP[1] * (1 - t) + BLUE_BOTTOM[1] * t)
        b = int(BLUE_TOP[2] * (1 - t) + BLUE_BOTTOM[2] * t)
        for x in range(size):
            px[x, y] = (r, g, b)
    return bg


def _draw_delta(canvas: Image.Image, size: int) -> None:
    """在 canvas 上绘制居中的白色大写 Δ."""
    draw = ImageDraw.Draw(canvas)
    # 多数 sans-serif 中 Δ 视觉高度 ~ 字号的 0.70, 目标占 canvas 的 ~58%
    target_h = size * 0.58
    font_size = int(target_h / 0.70)
    font = _find_font(font_size)

    text = "\u0394"  # Greek capital letter Delta
    l, t, r, b = font.getbbox(text)
    w, h = r - l, b - t
    x = (size - w) / 2 - l
    # 视觉下沉一点点, 让 Δ 看起来更稳
    y = (size - h) / 2 - t + size * 0.015

    draw.text((x, y), text, font=font, fill=WHITE)


def _draw_pnl_curve(canvas: Image.Image, size: int) -> None:
    """在 Δ 中央绘制一条半透明的对冲盈亏曲线 (仅大尺寸使用)."""
    # 画到独立 RGBA 层再合成, 方便控制不透明度
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cx, cy = size / 2, size / 2
    amp = size * 0.09      # 振幅
    span = size * 0.56     # 水平跨度
    x0 = cx - span / 2
    points = []
    n = 80
    for i in range(n + 1):
        t = i / n
        x = x0 + t * span
        # 合成两个正弦, 模拟盈亏曲线 (不是对称的波动率微笑)
        phase = t * math.pi * 2.2
        y = cy + math.sin(phase) * amp * 0.7 + math.sin(phase * 2.1 + 0.8) * amp * 0.3
        # 整体略向右下漂一点点, 增强"累计盈亏"的观感
        y += (t - 0.5) * size * 0.02
        points.append((x, y))

    line_w = max(2, int(size * 0.018))
    # 先画一条很浅的底光, 再叠一条实线, 让曲线在 Δ 上更醒目
    draw.line(points, fill=(255, 255, 255, 90), width=line_w + 2, joint="curve")
    draw.line(points, fill=(255, 255, 255, 220), width=line_w, joint="curve")

    # 曲线端点放两个小圆点
    r = max(2, int(size * 0.018))
    for (px, py), alpha in ((points[0], 200), (points[-1], 200)):
        draw.ellipse((px - r, py - r, px + r, py + r), fill=(255, 255, 255, alpha))

    canvas.alpha_composite(overlay)


def _draw_gold_accent(canvas: Image.Image, size: int) -> None:
    """底部金色装饰线 (仅大尺寸)."""
    draw = ImageDraw.Draw(canvas)
    y = int(size * 0.82)
    x1 = int(size * 0.30)
    x2 = int(size * 0.70)
    h = max(2, int(size * 0.014))
    draw.rounded_rectangle(
        (x1, y, x2, y + h), radius=h // 2, fill=GOLD + (255,)
    )


def render_icon(size: int) -> Image.Image:
    """渲染单尺寸图标 (RGBA, 圆角透明背景)."""
    # 超采样: 2x 绘制再缩回, 抗锯齿更干净
    scale = 2 if size <= 128 else 1
    s = size * scale
    radius = int(s * 0.22)

    bg = _gradient_bg(s).convert("RGBA")
    mask = _rounded_mask(s, radius)

    canvas = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    canvas.paste(bg, (0, 0), mask)

    # 细节: 只有 >= 64 的尺寸才加曲线 / 金线, 否则信息太挤
    effective = size  # 注意用原尺寸判定, 不是超采样后的
    if effective >= 64:
        _draw_pnl_curve(canvas, s)
        _draw_gold_accent(canvas, s)

    _draw_delta(canvas, s)

    # 轻微内阴影, 提升质感 (仅大尺寸)
    if effective >= 128:
        shadow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle(
            (0, 0, s - 1, s - 1), radius=radius,
            outline=(0, 0, 0, 55), width=max(1, int(s * 0.006)),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=s * 0.004))
        canvas.alpha_composite(shadow)

    if scale != 1:
        canvas = canvas.resize((size, size), Image.LANCZOS)
    return canvas


# ------------------------------------------------------------------
#  主入口
# ------------------------------------------------------------------
def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    # PNG: 主图 512, 附加 256 / 128 / 64 作为备用
    sizes_png = [512, 256, 128, 64]
    main_png = ASSETS / "deltalab.png"
    render_icon(512).save(main_png, format="PNG")
    for s in sizes_png[1:]:
        render_icon(s).save(ASSETS / f"deltalab_{s}.png", format="PNG")

    # ICO: Windows 多尺寸
    ico_sizes = [16, 32, 48, 64, 128, 256]
    imgs = [render_icon(s) for s in ico_sizes]
    ico_path = ASSETS / "deltalab.ico"
    imgs[0].save(
        ico_path,
        format="ICO",
        sizes=[(s, s) for s in ico_sizes],
        append_images=imgs[1:],
    )

    print(f"wrote {main_png}")
    print(f"wrote {ico_path}")
    for s in sizes_png[1:]:
        print(f"wrote {ASSETS / f'deltalab_{s}.png'}")


if __name__ == "__main__":
    main()
