import cv2
import numpy as np
import trimesh
import os

def create_solid_pcb(heightmap, pitch=1.0, base_thickness=1.6):
    """
    Tạo mô hình PCB dạng khối (Solid) để tránh bị 'lủng lỗ'.
    """
    rows, cols = heightmap.shape
    x = np.arange(cols) * pitch
    y = np.arange(rows) * pitch
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Đỉnh mặt trên (dựa trên heightmap)
    # Lưu ý: heightmap ở đây là độ cao phần mạch so với bề mặt board
    top_z = heightmap + base_thickness
    top_vertices = np.stack([x_grid.flatten(), y_grid.flatten(), top_z.flatten()], axis=1)
    
    # Đỉnh mặt dưới (phẳng ở Z=0)
    bottom_vertices = np.stack([x_grid.flatten(), y_grid.flatten(), np.zeros_like(top_z).flatten()], axis=1)
    
    # Gộp tất cả đỉnh
    vertices = np.vstack([top_vertices, bottom_vertices])
    num_v = rows * cols # Số lượng đỉnh của một mặt
    
    # --- Tạo các mặt (Faces) ---
    i, j = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing='ij')
    idx = i * cols + j
    
    # 1. Mặt trên (Top) - CCW
    top_f1 = np.stack([idx, idx + 1, idx + cols], axis=-1).reshape(-1, 3)
    top_f2 = np.stack([idx + 1, idx + cols + 1, idx + cols], axis=-1).reshape(-1, 3)
    
    # 2. Mặt dưới (Bottom) - CW để hướng xuống dưới
    b_idx = idx + num_v
    bot_f1 = np.stack([b_idx, b_idx + cols, b_idx + 1], axis=-1).reshape(-1, 3)
    bot_f2 = np.stack([b_idx + 1, b_idx + cols, b_idx + cols + 1], axis=-1).reshape(-1, 3)
    
    faces_list = [top_f1, top_f2, bot_f1, bot_f2]
    
    # 3. Các mặt bên (Sides)
    # Cạnh trái (j=0)
    i_side = np.arange(rows - 1)
    l_idx = i_side * cols
    bl_idx = l_idx + num_v
    faces_list.append(np.stack([l_idx, l_idx + cols, bl_idx], axis=-1).reshape(-1, 3))
    faces_list.append(np.stack([l_idx + cols, bl_idx + cols, bl_idx], axis=-1).reshape(-1, 3))
    
    # Cạnh phải (j=cols-1)
    r_idx = i_side * cols + (cols - 1)
    br_idx = r_idx + num_v
    faces_list.append(np.stack([r_idx, br_idx, r_idx + cols], axis=-1).reshape(-1, 3))
    faces_list.append(np.stack([r_idx + cols, br_idx, br_idx + cols], axis=-1).reshape(-1, 3))
    
    # Cạnh trên (i=0)
    j_side = np.arange(cols - 1)
    t_idx = j_side
    bt_idx = t_idx + num_v
    faces_list.append(np.stack([t_idx, bt_idx, t_idx + 1], axis=-1).reshape(-1, 3))
    faces_list.append(np.stack([t_idx + 1, bt_idx, bt_idx + 1], axis=-1).reshape(-1, 3))
    
    # Cạnh dưới (i=rows-1)
    b_idx_edge = (rows - 1) * cols + j_side
    bb_idx = b_idx_edge + num_v
    faces_list.append(np.stack([b_idx_edge, b_idx_edge + 1, bb_idx], axis=-1).reshape(-1, 3))
    faces_list.append(np.stack([b_idx_edge + 1, bb_idx + 1, bb_idx], axis=-1).reshape(-1, 3))
    
    all_faces = np.vstack(faces_list)
    return trimesh.Trimesh(vertices=vertices, faces=all_faces)

# --- Cấu hình ---
# File ảnh đầu vào (PNG)
png_file = "PCB_PCB_loa-ai-copy_2026-05-01.png"

# Kích thước mong muốn (Chiều rộng PCB tính bằng mm)
TARGET_WIDTH_MM = 50.0  # Bạn có thể sửa số này (ví dụ 100.0)

# Độ dày board và độ nổi của mạch (mm)
BASE_THICKNESS = 1.6   # Độ dày tấm FR4
TRACE_HEIGHT = 0.2     # Độ nổi của đường đồng (copper)

# --- Thực thi ---
if not os.path.exists(png_file):
    print(f"Error: Không tìm thấy file {png_file}")
    exit(1)

# Đọc ảnh PNG
img = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: Không thể đọc file {png_file}")
    exit(1)

rows, cols = img.shape
print(f"Image resolution: {cols}x{rows}")

# Tính toán pitch (mm per pixel) để đúng kích thước thực tế
pitch = TARGET_WIDTH_MM / cols
print(f"Calculated pitch: {pitch:.4f} mm/pixel")
print(f"Physical Size: {TARGET_WIDTH_MM:.1f} x {rows * pitch:.1f} mm")

# Xử lý heightmap: 
# Màu đen (0) là đường mạch lỏm xuống.
# Trắng (255) là bề mặt phẳng.
# Công thức: (img/255 - 1) * TRACE_HEIGHT -> Trắng thành 0, Đen thành -TRACE_HEIGHT
heightmap = (img.astype(float) / 255.0 - 1.0) * TRACE_HEIGHT

print("Generating solid 3D mesh...")
mesh = create_solid_pcb(heightmap, pitch=pitch, base_thickness=BASE_THICKNESS)

# Export sang STL
output_file = "pcb_solid.stl"
mesh.export(output_file)

print(f"Success! Exported to {output_file}")
print(f"PCB fixed: Solid block, no holes, target width: {TARGET_WIDTH_MM}mm.")