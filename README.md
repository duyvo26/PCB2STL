# 3D PCB Generator (Tạo 3D PCB PRO)

[English](#english) | [Tiếng Việt](#tiếng-việt)

---

## English

A Python-based tool to convert 2D PCB layouts into 3D printable STL models. The application provides an interactive Graphical User Interface (GUI) to configure parameters, edit hole masks, and generate watertight 3D meshes suitable for 3D printing.

### Features

- **2D to 3D Conversion**: Transforms PNG images (for trace height mapping) and SVG files (for physical dimensions) into accurate 3D meshes.
- **Interactive Hole Mask Editing**: Built-in OpenCV-based editor to manually add or remove drill holes from the mesh before generation.
- **Customizable Parameters**: Dynamically adjust the base thickness and trace height to match specific manufacturing requirements.
- **Solid Mesh Generation**: Automatically creates solid side-walls to ensure the resulting STL is a watertight block, preventing structural artifacts during 3D printing.
- **Modern Interface**: User-friendly dark-themed GUI built with `customtkinter`.
- **Skip Holes Option**: Toggle to quickly bypass hole punching during the STL generation process.

### Requirements

Ensure you have Python 3.8 or newer installed. All required dependencies are listed in `requirements.txt`.

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pcb_3d_main
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Start the application:
   ```bash
   python main.py
   ```

2. Operating the GUI:
   - Click **Tải PNG** to load the PCB image mask (grayscale).
   - Click **Tải SVG** to load the corresponding SVG file containing physical dimensions.
   - (Optional) Click **Chọn Nơi Lưu** to specify the output STL file path.
   - Set **Độ dày đế** (Base Thickness) and **Độ cao mạch** (Trace Height) as needed.
   - (Optional) Click **Chỉnh sửa lỗ** to open the interactive hole editor:
     - **Left Click**: Remove holes
     - **Right Click**: Add holes
     - **Z** or **Ctrl+Z**: Undo
     - **+** / **-**: Increase or decrease brush size
     - **ESC**: Save and close the editor
   - Click **Tạo File STL** to generate and save the 3D model.

---

## Tiếng Việt

Công cụ bằng Python dùng để chuyển đổi bản thiết kế PCB 2D thành mô hình 3D (định dạng STL) có thể in được. Ứng dụng cung cấp giao diện đồ họa (GUI) trực quan để tinh chỉnh thông số, chỉnh sửa các lỗ khoan, và tạo ra lưới 3D dạng khối đặc phù hợp cho in 3D.

### Tính năng

- **Chuyển đổi 2D sang 3D**: Biến đổi hình ảnh PNG (bản đồ độ cao mạch) và file SVG (kích thước vật lý) thành mô hình 3D chính xác.
- **Chỉnh sửa lỗ khoan tương tác**: Trình chỉnh sửa tích hợp dựa trên OpenCV cho phép thêm hoặc xóa lỗ khoan thủ công trước khi tạo mô hình 3D.
- **Thông số tùy chỉnh**: Linh hoạt điều chỉnh độ dày của đế (Base Thickness) và độ cao của mạch (Trace Height) để phù hợp với yêu cầu sản xuất.
- **Tạo khối đặc (Solid Mesh)**: Tự động tạo các bức tường bao quanh để đảm bảo file STL đầu ra là một khối kín (watertight), ngăn chặn lỗi cấu trúc khi in 3D.
- **Giao diện hiện đại**: Giao diện tối thân thiện với người dùng được xây dựng bằng `customtkinter`.
- **Tùy chọn bỏ qua lỗ (Skip Holes)**: Nhanh chóng vô hiệu hóa quá trình đục lỗ khi xuất file STL.

### Yêu cầu hệ thống

Đảm bảo bạn đã cài đặt Python 3.8 trở lên. Danh sách các thư viện cần thiết nằm trong file `requirements.txt`.

### Cài đặt

1. Clone dự án về máy:
   ```bash
   git clone <repository-url>
   cd pcb_3d_main
   ```

2. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```

### Hướng dẫn sử dụng

1. Chạy ứng dụng:
   ```bash
   python main.py
   ```

2. Thao tác trên giao diện:
   - Bấm **Tải PNG** để tải ảnh mặt nạ PCB (ảnh xám/grayscale).
   - Bấm **Tải SVG** để tải file SVG chứa thông tin về kích thước vật lý.
   - (Tùy chọn) Bấm **Chọn Nơi Lưu** để chỉ định đường dẫn lưu file STL đầu ra.
   - Nhập thông số **Độ dày đế** (Base Thickness) và **Độ cao mạch** (Trace Height) theo nhu cầu.
   - (Tùy chọn) Bấm **Chỉnh sửa lỗ** để mở trình chỉnh sửa lỗ khoan:
     - **Chuột trái**: Xóa lỗ
     - **Chuột phải**: Thêm lỗ
     - **Z** hoặc **Ctrl+Z**: Hoàn tác (Undo)
     - **+** / **-**: Tăng hoặc giảm kích thước cọ vẽ
     - **ESC**: Lưu và đóng trình chỉnh sửa
   - Bấm **Tạo File STL** để tiến hành tạo và lưu mô hình 3D.
