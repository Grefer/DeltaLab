import tkinter as tk
from tkinter import ttk

root = tk.Tk()
style = ttk.Style()
style.theme_use("clam")

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

var = tk.BooleanVar(value=True)
c = ModernCheckbox(root, text="Test Checked Retina", variable=var)
c.pack(padx=20, pady=20)

root.after(3000, root.destroy)
root.mainloop()
