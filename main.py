import cv2
import numpy as np
import trimesh
import os

def create_solid_pcb_with_holes(heightmap, hole_mask, pitch=1.0, base_thickness=1.6):
    """
    Tạo mô hình PCB dạng khối và tự động đục lỗ dựa trên hole_mask.
    """
    rows, cols = heightmap.shape
    x = np.arange(cols) * pitch
    y = np.arange(rows) * pitch
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Đỉnh mặt trên (z = base + heightmap)
    top_z = heightmap + base_thickness
    top_vertices = np.stack([x_grid.flatten(), y_grid.flatten(), top_z.flatten()], axis=1)
    
    # Đỉnh mặt dưới (z = 0)
    bottom_vertices = np.stack([x_grid.flatten(), y_grid.flatten(), np.zeros_like(top_z).flatten()], axis=1)
    
    vertices = np.vstack([top_vertices, bottom_vertices])
    num_v = rows * cols
    
    faces_list = []
    
    # Duyệt qua từng ô lưới
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Một ô được coi là lỗ nếu nó nằm trong hole_mask
            if hole_mask[i, j]:
                continue 
            
            idx = i * cols + j
            b_idx = idx + num_v
            
            # 1. Mặt trên (Top)
            faces_list.append([idx, idx + 1, idx + cols])
            faces_list.append([idx + 1, idx + cols + 1, idx + cols])
            
            # 2. Mặt dưới (Bottom)
            faces_list.append([b_idx, b_idx + cols, b_idx + 1])
            faces_list.append([b_idx + 1, b_idx + cols, b_idx + cols + 1])
            
            # 3. Mặt dựng (Side walls) cho các lỗ
            # Cạnh trái
            if j > 0 and hole_mask[i, j-1]:
                faces_list.append([idx, b_idx, idx + cols])
                faces_list.append([idx + cols, b_idx, b_idx + cols])
            # Cạnh phải
            if j < cols - 2 and hole_mask[i, j+1]:
                faces_list.append([idx + 1, idx + cols + 1, b_idx + 1])
                faces_list.append([idx + cols + 1, b_idx + cols + 1, b_idx + 1])
            # Cạnh trên
            if i > 0 and hole_mask[i-1, j]:
                faces_list.append([idx, idx + 1, b_idx])
                faces_list.append([idx + 1, b_idx + 1, b_idx])
            # Cạnh dưới
            if i < rows - 2 and hole_mask[i+1, j]:
                faces_list.append([idx + cols, b_idx + cols, idx + cols + 1])
                faces_list.append([idx + cols + 1, b_idx + cols, b_idx + cols + 1])

    # 4. Các cạnh ngoài cùng của Board (Outer boundary)
    for i in range(rows - 1):
        if not hole_mask[i, 0]: # Trái
            idx = i * cols
            b_idx = idx + num_v
            faces_list.append([idx, idx + cols, b_idx])
            faces_list.append([idx + cols, b_idx + cols, b_idx])
        if not hole_mask[i, cols-2]: # Phải
            idx = i * cols + (cols - 1)
            b_idx = idx + num_v
            faces_list.append([idx, b_idx, idx + cols])
            faces_list.append([idx + cols, b_idx, b_idx + cols])
            
    for j in range(cols - 1):
        if not hole_mask[0, j]: # Trên
            idx = j
            b_idx = idx + num_v
            faces_list.append([idx, b_idx, idx + 1])
            faces_list.append([idx + 1, b_idx, b_idx + 1])
        if not hole_mask[rows-2, j]: # Dưới
            idx = (rows - 1) * cols + j
            b_idx = idx + num_v
            faces_list.append([idx, idx + 1, b_idx])
            faces_list.append([idx + 1, b_idx + 1, b_idx])

    return trimesh.Trimesh(vertices=vertices, faces=faces_list)

# --- Cấu hình ---
png_file = "PCB_PCB_loa-ai-copy_2026-05-01.png"
TARGET_WIDTH_MM = 50.0
BASE_THICKNESS = 1.6
TRACE_HEIGHT = 0.2

# --- Thực thi ---
if not os.path.exists(png_file):
    print(f"Error: Không tìm thấy file {png_file}")
    exit(1)

img = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
pitch = TARGET_WIDTH_MM / cols

# 1. Nhận diện Lỗ (Holes là vùng TRẮNG nhỏ và TRÒN)
_, white_thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(white_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

hole_mask = np.zeros_like(img, dtype=bool)
CIRCULARITY_THRESHOLD = 0.6  # Càng gần 1 càng tròn. 0.6 giúp lọc bớt đường dây.

for cnt in contours:
    area = cv2.contourArea(cnt)
    # Lọc diện tích (không quá lớn, không quá nhỏ)
    if 2 < area < 1200:
        peri = cv2.arcLength(cnt, True)
        if peri > 0:
            circularity = 4 * np.pi * area / (peri * peri)
            # Chỉ đục lỗ nếu vùng trắng đó tương đối tròn
            if circularity > CIRCULARITY_THRESHOLD:
                cv2.drawContours(hole_mask.view(np.uint8), [cnt], -1, 1, -1)

# 2. Xử lý heightmap (Mạch ĐEN lỏm xuống)
# Trắng -> 0, Đen -> -TRACE_HEIGHT
heightmap = (img.astype(float) / 255.0 - 1.0) * TRACE_HEIGHT

print(f"Generating 3D mesh: White holes will be punched (Found {np.sum(hole_mask > 0)} hole pixels)...")
mesh = create_solid_pcb_with_holes(heightmap, hole_mask, pitch=pitch, base_thickness=BASE_THICKNESS)

# 3. Export
output_file = "pcb_with_white_holes.stl"
mesh.export(output_file)
print(f"Success! Exported to {output_file}")
print(f"PCB fixed: Black traces are recessed, small white areas are punched as holes.")