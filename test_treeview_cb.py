import tkinter as tk
from tkinter import ttk

root = tk.Tk()
style = ttk.Style()
style.theme_use("clam")

# Generate base64 images for checkboxes
img_checked = tk.PhotoImage(data="iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAZElEQVR4nGP8/PXnfwYKABMlmhkYGBhYYAyjvE8kaTw3iY86LqC/AbfmiJBvALpmrAZgU4QsrpbyhrAL0A3BpRmrATBFME34NON0AbohuDTjNABZEz7NeA0gRjNBA4gBjAOeGwEXWyXhhSE6UgAAAABJRU5ErkJggg==")
img_unchecked = tk.PhotoImage(data="iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAASElEQVR4nGP8/PXnfwYKABMlmhkYGBhYYIynL9+TpFFaXBDVAAYGBgZ1RXGiNN+8/xLOptgLowaMGsDAgJYSkVMYsYBxwHMjAKdYEIYsHHJgAAAAAElFTkSuQmCC")

tree = ttk.Treeview(root, columns=("name",), show="tree headings", height=5)
# Configure #0 column
tree.heading("#0", text="勾选")
tree.column("#0", width=60, anchor="center")
tree.heading("name", text="策略")
tree.column("name", width=150)

tree.insert("", "end", image=img_checked, values=("策略 A (Checked)",))
tree.insert("", "end", image=img_unchecked, values=("策略 B (Unchecked)",))

tree.pack(padx=20, pady=20)

root.after(3000, root.destroy)
root.mainloop()
