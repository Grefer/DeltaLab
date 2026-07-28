from PIL import Image
import base64
from io import BytesIO

def create_crisp_checkbox(checked, bg_color_hex="#F3F5F9"):
    # 16x16 size
    img = Image.new("RGBA", (16, 16), bg_color_hex)
    pixels = img.load()
    
    # helper colors
    border = (216, 222, 232, 255) # #D8DEE8
    primary = (37, 99, 235, 255)  # #2563EB
    white = (255, 255, 255, 255)  # #FFFFFF
    
    def hex_to_rgba(h):
        return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)
    
    bg = hex_to_rgba(bg_color_hex)

    if checked:
        # Draw solid primary square
        for y in range(2, 14):
            for x in range(2, 14):
                pixels[x, y] = primary
        
        # Draw crisp checkmark pixels
        check_coords = [
            (4, 7), (4, 8),
            (5, 8), (5, 9),
            (6, 9), (6, 10),
            (7, 10), (7, 11),
            (8, 9), (8, 10),
            (9, 8), (9, 9),
            (10, 7), (10, 8),
            (11, 6), (11, 7),
            (12, 5), (12, 6)
        ]
        for (x, y) in check_coords:
            pixels[x, y] = white
    else:
        # Draw border
        for y in range(2, 14):
            for x in range(2, 14):
                if x == 2 or x == 13 or y == 2 or y == 13:
                    pixels[x, y] = border
                else:
                    pixels[x, y] = white # inner bg
                    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

print("bg_checked =", create_crisp_checkbox(True, "#F3F5F9"))
print("bg_unchecked =", create_crisp_checkbox(False, "#F3F5F9"))

print("sf_checked =", create_crisp_checkbox(True, "#FFFFFF"))
print("sf_unchecked =", create_crisp_checkbox(False, "#FFFFFF"))
