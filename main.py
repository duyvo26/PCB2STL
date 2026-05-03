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
    
    vertices = []
    faces = []
    
    # Ham ho tro de them vertex va tra ve index
    v_dict = {}
    def get_v(x, y, z):
        v = (float(x), float(y), float(z))
        if v not in v_dict:
            v_dict[v] = len(vertices)
            vertices.append(v)
        return v_dict[v]

    # Duyet tung pixel de tao mat Tren va vach dung
    for i in range(rows):
        for j in range(cols):
            if hole_mask[i, j]:
                continue
                
            h = heightmap[i, j] + base_thickness
            z_top = h
            z_bot = 0
            
            x0, x1 = j * pitch_x, (j + 1) * pitch_x
            y0, y1 = i * pitch_y, (i + 1) * pitch_y
            
            # --- MAT TREN (TOP) ---
            v0 = get_v(x0, y0, z_top)
            v1 = get_v(x1, y0, z_top)
            v2 = get_v(x1, y1, z_top)
            v3 = get_v(x0, y1, z_top)
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
            
            # --- MAT DUOI (BOTTOM) ---
            b0 = get_v(x0, y0, z_bot)
            b1 = get_v(x1, y0, z_bot)
            b2 = get_v(x1, y1, z_bot)
            b3 = get_v(x0, y1, z_bot)
            faces.append([b0, b2, b1])
            faces.append([b0, b3, b2])
            
            # --- VACH DUNG (VERTICAL WALLS) ---
            # Kiem tra cac lang gieng de tao vach neu co su chenh lech do cao
            # Ben trai
            if j == 0 or hole_mask[i, j-1] or heightmap[i, j-1] != heightmap[i, j]:
                h_neigh = (heightmap[i, j-1] + base_thickness) if (j > 0 and not hole_mask[i, j-1]) else 0
                if j == 0 or hole_mask[i, j-1]: h_neigh = 0 # Canh board hoac lo khoan thi xuong tan day
                
                # Chi tao vach tu h_neigh len z_top
                low = h_neigh
                if low < z_top:
                    v_low0 = get_v(x0, y0, low)
                    v_low1 = get_v(x0, y1, low)
                    faces.append([v_low0, v0, v3])
                    faces.append([v_low0, v3, v_low1])

            # Ben phai
            if j == cols - 1 or hole_mask[i, j+1] or heightmap[i, j+1] != heightmap[i, j]:
                h_neigh = (heightmap[i, j+1] + base_thickness) if (j < cols - 1 and not hole_mask[i, j+1]) else 0
                if j == cols - 1 or hole_mask[i, j+1]: h_neigh = 0
                
                if h_neigh < z_top:
                    v_low0 = get_v(x1, y0, h_neigh)
                    v_low1 = get_v(x1, y1, h_neigh)
                    faces.append([v1, v_low0, v_low1])
                    faces.append([v1, v_low1, v2])

            # Tren
            if i == 0 or hole_mask[i-1, j] or heightmap[i-1, j] != heightmap[i, j]:
                h_neigh = (heightmap[i-1, j] + base_thickness) if (i > 0 and not hole_mask[i-1, j]) else 0
                if i == 0 or hole_mask[i-1, j]: h_neigh = 0
                
                if h_neigh < z_top:
                    v_low0 = get_v(x0, y0, h_neigh)
                    v_low1 = get_v(x1, y0, h_neigh)
                    faces.append([v0, v_low1, v1])
                    faces.append([v0, v_low0, v_low1])

            # Duoi
            if i == rows - 1 or hole_mask[i+1, j] or heightmap[i+1, j] != heightmap[i, j]:
                h_neigh = (heightmap[i+1, j] + base_thickness) if (i < rows - 1 and not hole_mask[i+1, j]) else 0
                if i == rows - 1 or hole_mask[i+1, j]: h_neigh = 0
                
                if h_neigh < z_top:
                    v_low0 = get_v(x0, y1, h_neigh)
                    v_low1 = get_v(x1, y1, h_neigh)
                    faces.append([v3, v2, v_low1])
                    faces.append([v3, v_low1, v_low0])

    return trimesh.Trimesh(vertices=vertices, faces=faces)


# ================= PROCESS =================
def process(png_file, svg_file, base_thickness, trace_height, mode="Lom", smooth_val=0, skip_holes=False,
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

    # Xu ly lam min va khu nhieu
    if smooth_val > 0:
        log(f"Lam min duong mach (level {smooth_val})...")
        ksize = int(smooth_val * 2 + 1)
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Bat buoc dua ve nhi phan (Hard Threshold) de vach dung duoc thang
    log("Ap dung Hard Threshold de lam sac net vach dung...")
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    log("Phat hien lo khoan...")
    # ... (giu nguyen phan phat hien lo)
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
        heightmap = (1.0 - img_float) * trace_height
    else:
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
        self.geometry("1100x800")

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
        self.base_input = ctk.CTkEntry(row1, placeholder_text="1.2", width=80)
        self.base_input.insert(0, "1.2")
        self.base_input.pack(side="left", padx=5)

        ctk.CTkLabel(row1, text="Cao mach (mm):").pack(side="left", padx=10)
        self.trace_input = ctk.CTkEntry(row1, placeholder_text="0.6", width=80)
        self.trace_input.insert(0, "0.6")
        self.trace_input.pack(side="left", padx=5)

        # Hang 2: Kieu mach & Lam min
        row2 = ctk.CTkFrame(self.cfg, fg_color="transparent")
        row2.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(row2, text="Kieu:").pack(side="left", padx=5)
        self.mode_var = ctk.StringVar(value="Lom")
        self.mode_switch = ctk.CTkSegmentedButton(row2, values=["Lom", "Noi"], variable=self.mode_var)
        self.mode_switch.pack(side="left", padx=5)

        ctk.CTkLabel(row2, text="Lam min:").pack(side="left", padx=15)
        self.smooth_slider = ctk.CTkSlider(row2, from_=0, to=5, number_of_steps=5)
        self.smooth_slider.set(0)
        self.smooth_slider.pack(side="left", padx=5)

        # Hang 3: Tuy chon khac
        row3 = ctk.CTkFrame(self.cfg, fg_color="transparent")
        row3.pack(fill="x", padx=5, pady=5)

        self.skip_holes_var = ctk.BooleanVar(value=False)
        self.skip_holes_cb = ctk.CTkCheckBox(row3, text="Bo duc lo", variable=self.skip_holes_var)
        self.skip_holes_cb.pack(side="left", padx=10)

        self.edit_holes_btn = ctk.CTkButton(row3, text="Chinh sua lo", command=self.edit_holes, width=120)
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
            smooth_val = self.smooth_slider.get()
            skip_holes = self.skip_holes_var.get()

            out = process(
                self.png_path,
                self.svg_path,
                base,
                trace,
                mode=mode,
                smooth_val=smooth_val,
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