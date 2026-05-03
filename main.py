import cv2
import numpy as np
import trimesh
import os
import xml.etree.ElementTree as ET
import customtkinter as ctk
from tkinter import filedialog
import threading
import uuid

# ================= SVG =================
def parse_svg_size(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()

    def to_mm(val):
        if val is None:
            return None
        val = val.strip()
        if val.endswith("mm"):
            return float(val.replace("mm", ""))
        if val.endswith("cm"):
            return float(val.replace("cm", "")) * 10
        if val.endswith("in"):
            return float(val.replace("in", "")) * 25.4
        return float(val)

    w = to_mm(root.get("width"))
    h = to_mm(root.get("height"))

    if w is None or h is None:
        raise ValueError("SVG thiếu width/height")

    return w, h


# ================= MESH =================
def create_mesh(heightmap, hole_mask, pitch_x, pitch_y, base_thickness):
    rows, cols = heightmap.shape

    x = np.arange(cols) * pitch_x
    y = np.arange(rows) * pitch_y
    xg, yg = np.meshgrid(x, y)

    top = heightmap + base_thickness

    v_top = np.stack([xg.flatten(), yg.flatten(), top.flatten()], axis=1)
    v_bot = np.stack([xg.flatten(), yg.flatten(), np.zeros_like(top).flatten()], axis=1)

    vertices = np.vstack([v_top, v_bot])
    num_v = rows * cols

    faces = []

    for i in range(rows - 1):
        for j in range(cols - 1):

            if hole_mask[i, j]:
                continue

            idx = i * cols + j
            b = idx + num_v

            faces += [
                [idx, idx+1, idx+cols],
                [idx+1, idx+cols+1, idx+cols],
                [b, b+cols, b+1],
                [b+1, b+cols, b+cols+1],
            ]

            # Thêm các mặt bên (tường) để biến lưới thành khối đặc (solid mesh)
            if i == 0 or hole_mask[i-1, j]:
                faces += [[idx, b, idx+1], [idx+1, b, b+1]]
                
            if i == rows - 2 or hole_mask[i+1, j]:
                faces += [[idx+cols, idx+cols+1, b+cols], [idx+cols+1, b+cols+1, b+cols]]
                
            if j == 0 or hole_mask[i, j-1]:
                faces += [[idx, idx+cols, b], [idx+cols, b+cols, b]]
                
            if j == cols - 2 or hole_mask[i, j+1]:
                faces += [[idx+1, b+1, idx+cols+1], [idx+cols+1, b+1, b+cols+1]]

    return trimesh.Trimesh(vertices=vertices, faces=faces)


# ================= PROCESS =================
def process(png_file, svg_file, base_thickness, trace_height, mode="Lom", skip_holes=False,
            custom_hole_mask=None, output_path=None, log_cb=None, prog_cb=None):

    def log(msg):
        if log_cb:
            log_cb(msg)

    def prog(v):
        if prog_cb:
            prog_cb(v)

    if not os.path.exists(png_file):
        raise FileNotFoundError("PNG khong ton tai")

    if not os.path.exists(svg_file):
        raise FileNotFoundError("SVG khong ton tai")

    # ===== OUTPUT =====
    if not output_path:
        folder = os.path.dirname(svg_file)
        name = f"pcb_{uuid.uuid4().hex[:8]}.stl"
        output_path = os.path.join(folder, name)

    log(f"Output: {output_path}")

    # ===== PROCESS =====
    log("Doc PNG...")
    img = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape
    prog(0.1)

    log("Doc SVG...")
    svg_w, svg_h = parse_svg_size(svg_file)
    pitch_x = svg_w / cols
    pitch_y = svg_h / rows
    prog(0.2)

    log("Phat hien lo khoan...")
    _, th = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if custom_hole_mask is not None:
        hole_mask = custom_hole_mask
    else:
        hole_mask = np.zeros_like(img, dtype=bool)

        if not skip_holes:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 2 < area < 1200:
                    peri = cv2.arcLength(cnt, True)
                    if peri > 0:
                        circ = 4 * np.pi * area / (peri * peri)
                        if circ > 0.6:
                            cv2.drawContours(hole_mask.view(np.uint8), [cnt], -1, 1, -1)

    prog(0.4)

    log(f"Tao ban do do cao ({mode})...")
    img_float = img.astype(float) / 255.0
    
    if mode == "Noi":
        # Trace (den=0) -> cao hon, Nen (trang=1) -> thap hon
        heightmap = (1.0 - img_float) * trace_height
    else:
        # Trace (den=0) -> thap hon (mac dinh)
        heightmap = (img_float - 1.0) * trace_height
        
    prog(0.6)

    log("Tao mo hinh luoi (Mesh)...")
    mesh = create_mesh(heightmap, hole_mask, pitch_x, pitch_y, base_thickness)
    prog(0.8)

    mesh.export(output_path)

    prog(1.0)
    log("HOAN TAT ✔")

    return output_path


# ================= GUI =================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Tao 3D PCB PRO")
        self.geometry("1100x750")

        self.png_path = None
        self.svg_path = None
        self.output_path = None
        self.custom_hole_mask = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ===== SIDEBAR =====
        self.sidebar = ctk.CTkFrame(self, width=220)
        self.sidebar.grid(row=0, column=0, sticky="nswe")

        ctk.CTkLabel(self.sidebar, text="CONG CU PCB", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)

        ctk.CTkButton(self.sidebar, text="Tai PNG", command=self.load_png).pack(pady=10, padx=10)
        ctk.CTkButton(self.sidebar, text="Tai SVG", command=self.load_svg).pack(pady=10, padx=10)
        ctk.CTkButton(self.sidebar, text="Chon Noi Luu (tuy chon)", command=self.choose_output).pack(pady=10, padx=10)

        self.run_btn = ctk.CTkButton(self.sidebar, text="Tao File STL", command=self.run_thread, fg_color="#2ecc71", hover_color="#27ae60")
        self.run_btn.pack(pady=20, padx=10)

        self.progress = ctk.CTkProgressBar(self.sidebar)
        self.progress.pack(pady=10, padx=10)
        self.progress.set(0)

        # ===== MAIN =====
        self.main = ctk.CTkFrame(self)
        self.main.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)

        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(3, weight=1)

        self.info = ctk.CTkLabel(self.main, text="Chua chon file", anchor="w", justify="left")
        self.info.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        # CONFIG
        self.cfg = ctk.CTkFrame(self.main)
        self.cfg.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Hang 1 cua Config
        row1 = ctk.CTkFrame(self.cfg, fg_color="transparent")
        row1.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(row1, text="Do day de (mm):").pack(side="left", padx=5)
        self.base_input = ctk.CTkEntry(row1, placeholder_text="1.2", width=100)
        self.base_input.pack(side="left", padx=5)

        ctk.CTkLabel(row1, text="Cao mach (mm):").pack(side="left", padx=15)
        self.trace_input = ctk.CTkEntry(row1, placeholder_text="0.6", width=100)
        self.trace_input.pack(side="left", padx=5)

        # Hang 2 cua Config: Kieu mach (Noi/Lom)
        row2 = ctk.CTkFrame(self.cfg, fg_color="transparent")
        row2.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(row2, text="Kieu duong mach:").pack(side="left", padx=5)
        self.mode_var = ctk.StringVar(value="Lom")
        self.mode_switch = ctk.CTkSegmentedButton(row2, values=["Lom", "Noi"], variable=self.mode_var)
        self.mode_switch.pack(side="left", padx=10)

        self.skip_holes_var = ctk.BooleanVar(value=False)
        self.skip_holes_cb = ctk.CTkCheckBox(row2, text="Bo duc lo", variable=self.skip_holes_var)
        self.skip_holes_cb.pack(side="left", padx=20)

        self.edit_holes_btn = ctk.CTkButton(row2, text="Chinh sua lo", command=self.edit_holes, width=120)
        self.edit_holes_btn.pack(side="left", padx=10)

        # LOG
        self.logbox = ctk.CTkTextbox(self.main)
        self.logbox.grid(row=3, column=0, sticky="nswe", padx=10, pady=10)

    # ===== ACTION =====
    def log(self, msg):
        self.logbox.insert("end", msg + "\n")
        self.logbox.see("end")

    def set_progress(self, v):
        self.progress.set(v)

    def load_png(self):
        self.png_path = filedialog.askopenfilename(filetypes=[("PNG", "*.png")])
        self.custom_hole_mask = None
        self.update_info()

    def load_svg(self):
        self.svg_path = filedialog.askopenfilename(filetypes=[("SVG", "*.svg")])
        self.update_info()

    def choose_output(self):
        self.output_path = filedialog.asksaveasfilename(defaultextension=".stl",
                                                        filetypes=[("STL", "*.stl")])
        self.update_info()

    def update_info(self):
        self.info.configure(text=
            f"PNG: {self.png_path}\n"
            f"SVG: {self.svg_path}\n"
            f"LUU TAI: {self.output_path if self.output_path else 'Tu dong (cung thu muc SVG)'}"
        )

    def edit_holes(self):
        if not self.png_path or not os.path.exists(self.png_path):
            self.log("LOI: Chua chon PNG")
            return

        img = cv2.imread(self.png_path, cv2.IMREAD_GRAYSCALE)
        
        if self.custom_hole_mask is None:
            _, th = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            self.custom_hole_mask = np.zeros_like(img, dtype=bool)
            if not self.skip_holes_var.get():
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 2 < area < 1200:
                        peri = cv2.arcLength(cnt, True)
                        if peri > 0:
                            circ = 4 * np.pi * area / (peri * peri)
                            if circ > 0.6:
                                cv2.drawContours(self.custom_hole_mask.view(np.uint8), [cnt], -1, 1, -1)

        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        drawing = False
        brush_size = 15
        win_name = "Chinh sua lo (Trai: Xoa, Phai: Them, Z/Ctrl+Z: Hoan tac, ESC: Luu)"

        history = []
        mouse_x, mouse_y = -1, -1

        def save_state():
            history.append(self.custom_hole_mask.copy())
            if len(history) > 20:
                history.pop(0)

        def undo():
            if history:
                self.custom_hole_mask = history.pop()
                update_display()

        def update_display():
            disp = display_img.copy()
            disp[self.custom_hole_mask] = [0, 0, 255] # Lo mau do
            if mouse_x >= 0 and mouse_y >= 0:
                cv2.circle(disp, (mouse_x, mouse_y), brush_size, (255, 255, 0), 2) # Vien mau vang
            cv2.imshow(win_name, disp)

        def mouse_event(event, x, y, flags, param):
            nonlocal drawing, mouse_x, mouse_y
            mouse_x, mouse_y = x, y
            if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                save_state()
                drawing = True
                if flags & cv2.EVENT_FLAG_LBUTTON:
                    cv2.circle(self.custom_hole_mask.view(np.uint8), (x, y), brush_size, 0, -1)
                elif flags & cv2.EVENT_FLAG_RBUTTON:
                    cv2.circle(self.custom_hole_mask.view(np.uint8), (x, y), brush_size, 1, -1)
                update_display()
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    if flags & cv2.EVENT_FLAG_LBUTTON:
                        cv2.circle(self.custom_hole_mask.view(np.uint8), (x, y), brush_size, 0, -1)
                    elif flags & cv2.EVENT_FLAG_RBUTTON:
                        cv2.circle(self.custom_hole_mask.view(np.uint8), (x, y), brush_size, 1, -1)
                update_display()
            elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
                drawing = False
                update_display()

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1200, 800)
        cv2.setMouseCallback(win_name, mouse_event)
        
        update_display()
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k == 27: # ESC
                break
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k == 26 or k == ord('z') or k == ord('Z'):
                undo()
            elif k == ord('+') or k == ord('='):
                brush_size += 2
                update_display()
            elif k == ord('-'):
                brush_size = max(1, brush_size - 2)
                update_display()

        cv2.destroyAllWindows()
        self.log("Da luu cac chinh sua lo khoan.")

    def run_thread(self):
        threading.Thread(target=self.run).start()

    def run(self):
        try:
            self.log("==== BAT DAU ====")
            self.set_progress(0)

            base = float(self.base_input.get() or 1.2)
            trace = float(self.trace_input.get() or 0.6)
            mode = self.mode_var.get()
            skip_holes = self.skip_holes_var.get()

            out = process(
                self.png_path,
                self.svg_path,
                base,
                trace,
                mode=mode,
                skip_holes=skip_holes,
                custom_hole_mask=self.custom_hole_mask,
                output_path=self.output_path,
                log_cb=self.log,
                prog_cb=self.set_progress
            )

            self.log(f"DA LUU TAI: {out}")

        except Exception as e:
            self.log(f"LOI: {e}")


# ================= RUN =================
if __name__ == "__main__":
    app = App()
    app.mainloop()