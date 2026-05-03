import cv2
import numpy as np
import trimesh
import os
import json
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
        raise ValueError("SVG thieu width/height")

    return w, h


# ================= MESH =================
def create_mesh(heightmap, hole_mask, pitch_x, pitch_y, base_thickness):
    rows, cols = heightmap.shape
    
    vertices = []
    faces = []
    
    v_dict = {}
    def get_v(x, y, z):
        v = (float(x), float(y), float(z))
        if v not in v_dict:
            v_dict[v] = len(vertices)
            vertices.append(v)
        return v_dict[v]

    for i in range(rows):
        for j in range(cols):
            if hole_mask[i, j]:
                continue
                
            h = heightmap[i, j] + base_thickness
            z_top = h
            z_bot = 0
            
            x0, x1 = j * pitch_x, (j + 1) * pitch_x
            y0, y1 = i * pitch_y, (i + 1) * pitch_y
            
            v0 = get_v(x0, y0, z_top)
            v1 = get_v(x1, y0, z_top)
            v2 = get_v(x1, y1, z_top)
            v3 = get_v(x0, y1, z_top)
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
            
            b0 = get_v(x0, y0, z_bot)
            b1 = get_v(x1, y0, z_bot)
            b2 = get_v(x1, y1, z_bot)
            b3 = get_v(x0, y1, z_bot)
            faces.append([b0, b2, b1])
            faces.append([b0, b3, b2])
            
            # Left
            if j == 0 or hole_mask[i, j-1] or heightmap[i, j-1] != heightmap[i, j]:
                h_neigh = (heightmap[i, j-1] + base_thickness) if (j > 0 and not hole_mask[i, j-1]) else 0
                if j == 0 or hole_mask[i, j-1]: h_neigh = 0
                if h_neigh < z_top:
                    v_low0 = get_v(x0, y0, h_neigh); v_low1 = get_v(x0, y1, h_neigh)
                    faces.append([v_low0, v0, v3]); faces.append([v_low0, v3, v_low1])
            # Right
            if j == cols - 1 or hole_mask[i, j+1] or heightmap[i, j+1] != heightmap[i, j]:
                h_neigh = (heightmap[i, j+1] + base_thickness) if (j < cols - 1 and not hole_mask[i, j+1]) else 0
                if j == cols - 1 or hole_mask[i, j+1]: h_neigh = 0
                if h_neigh < z_top:
                    v_low0 = get_v(x1, y0, h_neigh); v_low1 = get_v(x1, y1, h_neigh)
                    faces.append([v1, v_low0, v_low1]); faces.append([v1, v_low1, v2])
            # Top
            if i == 0 or hole_mask[i-1, j] or heightmap[i-1, j] != heightmap[i, j]:
                h_neigh = (heightmap[i-1, j] + base_thickness) if (i > 0 and not hole_mask[i-1, j]) else 0
                if i == 0 or hole_mask[i-1, j]: h_neigh = 0
                if h_neigh < z_top:
                    v_low0 = get_v(x0, y0, h_neigh); v_low1 = get_v(x1, y0, h_neigh)
                    faces.append([v0, v_low1, v1]); faces.append([v0, v_low0, v_low1])
            # Bottom
            if i == rows - 1 or hole_mask[i+1, j] or heightmap[i+1, j] != heightmap[i, j]:
                h_neigh = (heightmap[i+1, j] + base_thickness) if (i < rows - 1 and not hole_mask[i+1, j]) else 0
                if i == rows - 1 or hole_mask[i+1, j]: h_neigh = 0
                if h_neigh < z_top:
                    v_low0 = get_v(x0, y1, h_neigh); v_low1 = get_v(x1, y1, h_neigh)
                    faces.append([v3, v2, v_low1]); faces.append([v3, v_low1, v_low0])

    return trimesh.Trimesh(vertices=vertices, faces=faces)


# ================= PROCESS =================
def process(png_file, svg_file, base_thickness, trace_height, mode="Lom", smooth_val=0, sharpen_val=0, hole_size_ratio=0.6, noise_val=0, skip_holes=False,
            custom_hole_mask=None, output_path=None, log_cb=None, prog_cb=None, texts=None):

    def log(msg_key, **kwargs):
        if log_cb and texts:
            msg = texts.get(msg_key, msg_key).format(**kwargs)
            log_cb(msg)

    def prog(v):
        if prog_cb:
            prog_cb(v)

    if not os.path.exists(png_file): raise FileNotFoundError("PNG missing")
    if not os.path.exists(svg_file): raise FileNotFoundError("SVG missing")

    if not output_path:
        folder = os.path.dirname(svg_file)
        name = f"pcb_{uuid.uuid4().hex[:8]}.stl"
        output_path = os.path.join(folder, name)

    log("Output: {p}", p=output_path)
    log("log_reading_png")
    img = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape
    prog(0.1)

    log("log_reading_svg")
    svg_w, svg_h = parse_svg_size(svg_file)
    pitch_x, pitch_y = svg_w / cols, svg_h / rows
    prog(0.2)

    if smooth_val > 0:
        log("log_smoothing", v=smooth_val)
        ksize = int(smooth_val * 2 + 1)
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    if sharpen_val > 0:
        log("log_sharpening")
        # Contrast Enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        # Sharpening (Unsharp Mask)
        sigma = sharpen_val * 0.5
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        img = cv2.addWeighted(img, 1.5 + (sharpen_val * 0.2), blurred, -0.5 - (sharpen_val * 0.2), 0)

    log("log_threshold")
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    if noise_val > 0:
        log("log_removing_noise")
        # Invert to find black blobs (text/noise)
        img_inv = cv2.bitwise_not(img)
        cnts, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < noise_val:
                cv2.drawContours(img, [c], -1, 255, -1) # Fill with background (white)

    log("log_separating")
    # Morphological opening to separate close traces
    sep_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, sep_kernel)

    log("log_detecting_holes")
    # Detect holes from processed image
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if custom_hole_mask is not None:
        hole_mask = custom_hole_mask
    else:
        hole_mask = np.zeros_like(img, dtype=bool)
        if not skip_holes:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 5 < area < 3000:
                    peri = cv2.arcLength(cnt, True)
                    if peri > 0:
                        circ = 4 * np.pi * area / (peri * peri)
                        if circ > 0.4:
                            # Use minEnclosingCircle for perfectly round holes
                            (x, y), radius = cv2.minEnclosingCircle(cnt)
                            center = (int(x), int(y))
                            drill_r = int(radius * hole_size_ratio)
                            if drill_r > 0:
                                cv2.circle(hole_mask.view(np.uint8), center, drill_r, 1, -1)
                                # Clear the pad area from the trace mask as requested (cut the connection)
                                cv2.drawContours(img, [cnt], -1, 0, -1)

    prog(0.4)
    log("log_creating_heightmap", m=mode)
    img_float = img.astype(float) / 255.0
    heightmap = (1.0 - img_float) * trace_height if mode == "Noi" else (img_float - 1.0) * trace_height
    prog(0.6)

    log("log_creating_mesh")
    mesh = create_mesh(heightmap, hole_mask, pitch_x, pitch_y, base_thickness)
    prog(0.8)
    mesh.export(output_path)
    prog(1.0); log("log_finish")
    return output_path


# ================= GUI =================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.all_langs = {}
        self.load_lang_file()
        self.curr_lang = "vi"
        
        self.title("PCB to 3D STL PRO")
        self.geometry("1150x850")
        
        self.png_path = None; self.svg_path = None; self.output_path = None; self.custom_hole_mask = None
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ----- SIDEBAR -----
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nswe")
        
        self.side_title = ctk.CTkLabel(self.sidebar, text="🛠️ CONG CU PCB", font=ctk.CTkFont(size=22, weight="bold"))
        self.side_title.pack(pady=(30, 40), padx=20)

        self.btn_png = ctk.CTkButton(self.sidebar, text="📁 Tai PNG", command=self.load_png, height=40)
        self.btn_png.pack(pady=10, padx=20, fill="x")

        self.btn_svg = ctk.CTkButton(self.sidebar, text="📁 Tai SVG", command=self.load_svg, height=40)
        self.btn_svg.pack(pady=10, padx=20, fill="x")

        self.btn_save = ctk.CTkButton(self.sidebar, text="💾 Chon Noi Luu", command=self.choose_output, height=40, fg_color="gray30")
        self.btn_save.pack(pady=10, padx=20, fill="x")

        self.lang_var = ctk.StringVar(value="🇻🇳 Vietnamese")
        self.lang_menu = ctk.CTkOptionMenu(self.sidebar, values=["🇻🇳 Vietnamese", "🇺🇸 English"], 
                                         command=self.change_lang, variable=self.lang_var, height=35)
        self.lang_menu.pack(pady=30, padx=20, side="bottom")

        self.run_btn = ctk.CTkButton(self.sidebar, text="🚀 Tao File STL", command=self.run_thread, 
                                    fg_color="#2ecc71", hover_color="#27ae60", height=50, font=ctk.CTkFont(size=16, weight="bold"))
        self.run_btn.pack(pady=(0, 20), padx=20, side="bottom", fill="x")

        self.progress = ctk.CTkProgressBar(self.sidebar, height=10)
        self.progress.pack(pady=20, padx=20, side="bottom", fill="x")
        self.progress.set(0)

        # ----- MAIN AREA -----
        self.main = ctk.CTkFrame(self, fg_color="transparent")
        self.main.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(2, weight=1)

        # Info Card
        self.info_card = ctk.CTkFrame(self.main, corner_radius=15)
        self.info_card.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 20))
        self.info = ctk.CTkLabel(self.info_card, text="Chua chon file", anchor="w", justify="left", 
                                font=ctk.CTkFont(size=13), padx=20, pady=15)
        self.info.pack(fill="x")

        # Config Card
        self.cfg_card = ctk.CTkFrame(self.main, corner_radius=15, border_width=1, border_color="gray30")
        self.cfg_card.grid(row=1, column=0, sticky="ew", padx=0, pady=0)
        
        self.cfg_inner = ctk.CTkFrame(self.cfg_card, fg_color="transparent")
        self.cfg_inner.pack(padx=20, pady=20, fill="x")

        # Row 1
        row1 = ctk.CTkFrame(self.cfg_inner, fg_color="transparent")
        row1.pack(fill="x", pady=10)
        self.lbl_base = ctk.CTkLabel(row1, text="Do day de (mm):", width=120, anchor="w")
        self.lbl_base.pack(side="left")
        self.base_input = ctk.CTkEntry(row1, width=100); self.base_input.insert(0, "20.0"); self.base_input.pack(side="left", padx=10)
        self.lbl_trace = ctk.CTkLabel(row1, text="Cao mach (mm):", width=120, anchor="w")
        self.lbl_trace.pack(side="left", padx=(20, 0))
        self.trace_input = ctk.CTkEntry(row1, width=100); self.trace_input.insert(0, "0.6"); self.trace_input.pack(side="left", padx=10)

        # Row 2
        row2 = ctk.CTkFrame(self.cfg_inner, fg_color="transparent")
        row2.pack(fill="x", pady=10)
        self.lbl_mode = ctk.CTkLabel(row2, text="Kieu:", width=120, anchor="w")
        self.lbl_mode.pack(side="left")
        self.mode_var = ctk.StringVar(value="Lom")
        self.mode_switch = ctk.CTkSegmentedButton(row2, values=["Lom", "Noi"], variable=self.mode_var, height=35)
        self.mode_switch.pack(side="left", padx=10)
        self.lbl_smooth = ctk.CTkLabel(row2, text="Lam min:", width=80, anchor="w")
        self.lbl_smooth.pack(side="left", padx=(40, 0))
        self.smooth_slider = ctk.CTkSlider(row2, from_=0, to=5, number_of_steps=5, width=120)
        self.smooth_slider.set(0); self.smooth_slider.pack(side="left", padx=10)

        self.lbl_sharpen = ctk.CTkLabel(row2, text="Tang net:", width=80, anchor="w")
        self.lbl_sharpen.pack(side="left", padx=(20, 0))
        self.sharpen_slider = ctk.CTkSlider(row2, from_=0, to=5, number_of_steps=5, width=120)
        self.sharpen_slider.set(0); self.sharpen_slider.pack(side="left", padx=10)

        self.lbl_noise = ctk.CTkLabel(row2, text="Xoa chu:", width=80, anchor="w")
        self.lbl_noise.pack(side="left", padx=(20, 0))
        self.noise_slider = ctk.CTkSlider(row2, from_=0, to=1000, number_of_steps=20, width=120)
        self.noise_slider.set(0); self.noise_slider.pack(side="left", padx=10)

        # Row 3
        row3 = ctk.CTkFrame(self.cfg_inner, fg_color="transparent")
        row3.pack(fill="x", pady=10)
        self.skip_holes_var = ctk.BooleanVar(value=False)
        self.skip_holes_cb = ctk.CTkCheckBox(row3, text="Bo duc lo", variable=self.skip_holes_var)
        self.skip_holes_cb.pack(side="left")

        self.lbl_hole_size = ctk.CTkLabel(row3, text="Size lo:", width=80, anchor="w")
        self.lbl_hole_size.pack(side="left", padx=(40, 0))
        self.hole_size_slider = ctk.CTkSlider(row3, from_=0.1, to=1.0, width=150)
        self.hole_size_slider.set(0.6); self.hole_size_slider.pack(side="left", padx=10)

        self.edit_holes_btn = ctk.CTkButton(row3, text="✏️ Chinh sua lo", command=self.edit_holes, width=150, height=35, fg_color="gray25")
        self.edit_holes_btn.pack(side="right")

        # Log Box
        self.log_label = ctk.CTkLabel(self.main, text="📜 PROCESS LOGS", font=ctk.CTkFont(size=14, weight="bold"))
        self.log_label.grid(row=2, column=0, sticky="w", pady=(20, 5))
        self.logbox = ctk.CTkTextbox(self.main, corner_radius=15, border_width=1, border_color="gray30")
        self.logbox.grid(row=3, column=0, sticky="nswe", pady=(0, 0))

        self.update_ui_text()

    def load_lang_file(self):
        try:
            if os.path.exists("lang.json"):
                with open("lang.json", "r", encoding="utf-8") as f: self.all_langs = json.load(f)
        except Exception as e: print(f"Error: {e}")

    def t(self, key): return self.all_langs.get(self.curr_lang, {}).get(key, key)

    def change_lang(self, choice):
        self.curr_lang = "vi" if "Vietnamese" in choice else "en"
        self.update_ui_text()

    def update_ui_text(self):
        self.title(self.t("title"))
        self.side_title.configure(text="🛠️ " + self.t("sidebar_title"))
        self.btn_png.configure(text="📁 " + self.t("btn_load_png"))
        self.btn_svg.configure(text="📁 " + self.t("btn_load_svg"))
        self.btn_save.configure(text="💾 " + self.t("btn_save_as"))
        self.run_btn.configure(text="🚀 " + self.t("btn_run"))
        self.lbl_base.configure(text=self.t("label_base_thickness"))
        self.lbl_trace.configure(text=self.t("label_trace_height"))
        self.lbl_mode.configure(text=self.t("label_mode"))
        self.lbl_smooth.configure(text=self.t("label_smooth"))
        self.lbl_sharpen.configure(text=self.t("label_sharpen"))
        self.lbl_noise.configure(text=self.t("label_remove_noise"))
        self.lbl_hole_size.configure(text=self.t("label_hole_size"))
        self.skip_holes_cb.configure(text=self.t("cb_skip_holes"))
        self.edit_holes_btn.configure(text="✏️ " + self.t("btn_edit_holes"))
        self.update_info()

    def log(self, msg): self.logbox.insert("end", msg + "\n"); self.logbox.see("end")
    def set_progress(self, v): self.progress.set(v)
    def load_png(self): self.png_path = filedialog.askopenfilename(filetypes=[("PNG", "*.png")]); self.custom_hole_mask = None; self.update_info()
    def load_svg(self): self.svg_path = filedialog.askopenfilename(filetypes=[("SVG", "*.svg")]); self.update_info()
    def choose_output(self): self.output_path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL", "*.stl")]); self.update_info()

    def update_info(self):
        txt_none, txt_auto = self.t("info_none"), self.t("info_save_auto")
        self.info.configure(text=f"🖼️ PNG: {self.png_path if self.png_path else txt_none}\n"
                                f"📐 SVG: {self.svg_path if self.svg_path else txt_none}\n"
                                f"📦 SAVE: {self.output_path if self.output_path else txt_auto}")

    def edit_holes(self):
        if not self.png_path or not os.path.exists(self.png_path): self.log(self.t("log_error") + " PNG?"); return
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
                            if circ > 0.6: cv2.drawContours(self.custom_hole_mask.view(np.uint8), [cnt], -1, 1, -1)
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR); drawing = False; brush_size = 15; win_name = self.t("win_edit_holes")
        history = []; mouse_x, mouse_y = -1, -1
        def save_state(): history.append(self.custom_hole_mask.copy()); (history.pop(0) if len(history)>20 else None)
        def undo(): (setattr(self, 'custom_hole_mask', history.pop()) or update_display()) if history else None
        def update_display():
            disp = display_img.copy(); disp[self.custom_hole_mask] = [0,0,255]
            if mouse_x>=0 and mouse_y>=0: cv2.circle(disp, (mouse_x,mouse_y), brush_size, (255,255,0), 2)
            cv2.imshow(win_name, disp)
        def mouse_event(event, x, y, flags, param):
            nonlocal drawing, mouse_x, mouse_y, brush_size; mouse_x, mouse_y = x, y
            if event == cv2.EVENT_MOUSEWHEEL:
                if flags > 0: brush_size += 2
                else: brush_size = max(1, brush_size - 2)
            elif event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
                save_state(); drawing = True
                val = 0 if flags & cv2.EVENT_FLAG_LBUTTON else 1
                cv2.circle(self.custom_hole_mask.view(np.uint8), (x,y), brush_size, val, -1)
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                val = 0 if flags & cv2.EVENT_FLAG_LBUTTON else 1
                cv2.circle(self.custom_hole_mask.view(np.uint8), (x,y), brush_size, val, -1)
            elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]: drawing = False
            update_display()
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL); cv2.resizeWindow(win_name, 1200, 800); cv2.setMouseCallback(win_name, mouse_event); update_display()
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k == 27 or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1: break
            if k in [26, ord('z'), ord('Z')]: undo()
            elif k in [ord('+'), ord('=')]: brush_size += 2; update_display()
            elif k == ord('-'): brush_size = max(1, brush_size - 2); update_display()
        cv2.destroyAllWindows(); self.log(self.t("msg_save_holes"))

    def run_thread(self): threading.Thread(target=self.run).start()
    def run(self):
        try:
            self.log(self.t("log_start")); self.set_progress(0)
            base, trace = float(self.base_input.get() or 1.2), float(self.trace_input.get() or 0.6)
            mode, smooth_val, sharpen_val, hole_size_ratio, noise_val, skip_holes = self.mode_var.get(), self.smooth_slider.get(), self.sharpen_slider.get(), self.hole_size_slider.get(), self.noise_slider.get(), self.skip_holes_var.get()
            out = process(self.png_path, self.svg_path, base, trace, mode=mode, smooth_val=smooth_val, sharpen_val=sharpen_val, hole_size_ratio=hole_size_ratio, noise_val=noise_val, skip_holes=skip_holes,
                custom_hole_mask=self.custom_hole_mask, output_path=self.output_path, log_cb=self.log, prog_cb=self.set_progress,
                texts=self.all_langs.get(self.curr_lang))
            self.log(self.t("log_output") + f" {out}")
        except Exception as e: self.log(self.t("log_error") + f" {e}")

if __name__ == "__main__":
    app = App(); app.mainloop()