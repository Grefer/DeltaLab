"""
Test: verify charts dynamically size with window and left panel scales
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from gui_app import BacktestApp

app = BacktestApp()
app.update_idletasks()

print("\n" + "=" * 65)
print("  Window Maximize / Dynamic Chart Size Test")
print("=" * 65)

test_cases = [
    ("Small   1200x720",  "1200x720"),
    ("Normal  1600x1000", "1600x1000"),
    ("Large   1920x1080", "1920x1080"),
    ("2K      2560x1440", "2560x1440"),
]

chart_widths = []
panel_widths = []

for label, geom in test_cases:
    app.geometry(geom)
    app.update_idletasks()
    app.update()

    fs = app._container_figsize(app._chart_container, fallback=(10, 9))
    nb_w = app._nb.winfo_width()
    nb_h = app._nb.winfo_height()
    lp_w = app._left_canvas.winfo_width()

    chart_widths.append(fs[0])
    panel_widths.append(lp_w)

    print(f"\n  {label}:")
    print(f"    Notebook:    {nb_w}x{nb_h} px")
    print(f"    Left panel:  {lp_w} px")
    print(f"    Chart fig:   {fs[0]:.1f} x {fs[1]:.1f} inch ({fs[0]*96:.0f}x{fs[1]*96:.0f} px)")

print("\n" + "-" * 65)

# Verify chart sizes differ
unique_chart = len(set([round(w, 1) for w in chart_widths]))
unique_panel = len(set(panel_widths))

if unique_chart >= 3:
    print("  PASS: Chart figsize dynamically scales with window!")
else:
    print(f"  FAIL: Chart figsize not scaling (unique widths: {unique_chart})")

if unique_panel >= 3:
    print("  PASS: Left panel dynamically scales with window!")
else:
    print(f"  FAIL: Left panel not scaling (unique widths: {unique_panel})")

print("=" * 65 + "\n")
app.destroy()
