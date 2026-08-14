"""ModernCheckbox 外观的手工验收脚本。

这不是自动化测试，是用来肉眼确认勾选态配色/字重的一次性预览：窗口开 3 秒
自动关闭。原先它叫仓库根的 test_checkbox.py，会被 pytest 当测试收集，而
建窗口的代码又写在模块顶层——收集阶段就 tk.Tk() + mainloop()，让 `pytest`
裸跑直接卡死。现在挪进 tools/ 并把建窗口的部分收进 __main__，两条路都堵上。

用法：python3 tools/checkbox_preview.py
"""
import tkinter as tk
from tkinter import ttk


class ModernCheckbox(ttk.Frame):
    def __init__(self, parent, text, variable, command=None, state="normal", bg_color="#F3F5F9"):
        super().__init__(parent)
        self.variable = variable
        self.command = command
        self.state = state
        self.bg_color = bg_color
        
        # Checkbox box label
        self.box = tk.Label(self, text="", width=2, height=1, font=("Arial", 12),
                            bg=self.bg_color, fg="#FFFFFF", relief="solid", bd=1)
        self.box.pack(side="left")
        
        # Text label
        self.label = tk.Label(self, text=" " + text, bg=self.bg_color, fg="#333333")
        self.label.pack(side="left")
        
        # Bind clicks
        self.box.bind("<Button-1>", self.toggle)
        self.label.bind("<Button-1>", self.toggle)
        self.variable.trace_add("write", self.update_view)
        self.update_view()

    def toggle(self, event=None):
        if self.state == "disabled": return
        self.variable.set(not self.variable.get())
        if self.command: self.command()
        
    def update_view(self, *args):
        if self.variable.get():
            self.box.config(text="✔", bg="#2563EB", bd=0) # primary blue
        else:
            self.box.config(text="", bg="#FFFFFF", bd=1) # white box, default border


def main():
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")

    var = tk.BooleanVar(value=True)
    c = ModernCheckbox(root, text="Test Checked Retina", variable=var)
    c.pack(padx=20, pady=20)

    root.after(3000, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
