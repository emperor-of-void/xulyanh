import tkinter as tk
from tkinter import filedialog, messagebox, Scale, HORIZONTAL, Checkbutton, IntVar, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

# --- CẤU HÌNH MÀU SẮC (DARK LUXURY THEME) ---
BG_COLOR = "#1e1e1e"       # Nền chính
PANEL_COLOR = "#252526"    # Nền panel
TAB_BG = "#333333"         # Nền tab
TEXT_COLOR = "#f0f0f0"     # Chữ trắng
ACCENT_COLOR = "#d4af37"   # Vàng Gold
BUTTON_COLOR = "#007acc"   # Xanh dương kiểu VS Code

class LuxuryRestoreApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultimate Photo Restore - Đồ Án Xử Lý Ảnh")
        self.root.geometry("1400x850")
        self.root.configure(bg=BG_COLOR)

        # Biến dữ liệu ảnh
        self.img_original_cv = None  # Gốc
        self.img_processed_cv = None # Sau xử lý
        self.preview_mode = False    # Chế độ xem so sánh

        # Cấu hình Style cho Tab (Notebook)
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        self.style.configure("TNotebook.Tab", background=PANEL_COLOR, foreground="white", padding=[10, 5], font=("Arial", 10))
        self.style.map("TNotebook.Tab", background=[("selected", ACCENT_COLOR)], foreground=[("selected", "black")])
        self.style.configure("TFrame", background=PANEL_COLOR)

        self.setup_ui()

    def setup_ui(self):
        # --- HEADER ---
        header = tk.Frame(self.root, bg=BG_COLOR, height=50)
        header.pack(fill="x", pady=5)
        tk.Label(header, text="✨ PHẦN MỀM PHỤC CHẾ ẢNH ĐA NĂNG", font=("Helvetica", 18, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR).pack()

        # --- MAIN LAYOUT ---
        main_container = tk.Frame(self.root, bg=BG_COLOR)
        main_container.pack(fill="both", expand=True, padx=10, pady=5)

        # 1. CỘT TRÁI: CÔNG CỤ (Dùng Notebook chia Tab)
        left_panel = tk.Frame(main_container, bg=BG_COLOR, width=350)
        left_panel.pack(side="left", fill="y", padx=(0, 10))

        # Nút tác vụ chung
        btn_frame = tk.Frame(left_panel, bg=BG_COLOR)
        btn_frame.pack(fill="x", pady=5)
        tk.Button(btn_frame, text="📂 Tải Ảnh", command=self.open_image, bg="#444", fg="white", font=("Arial", 10, "bold")).pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(btn_frame, text="↺ Reset", command=self.reset_params, bg="#c0392b", fg="white", font=("Arial", 10, "bold")).pack(side="left", fill="x", expand=True, padx=2)

        # HỆ THỐNG TAB
        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill="both", expand=True, pady=5)

        # ---> TAB 1: PHỤC CHẾ (Yêu cầu đề bài)
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="1. Phục Chế")
        self.setup_tab_restore(tab1)

        # ---> TAB 2: ÁNH SÁNG & MÀU
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="2. Màu Sắc")
        self.setup_tab_color(tab2)

        # ---> TAB 3: HIỆU ỨNG & CÔNG CỤ
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="3. Hiệu Ứng")
        self.setup_tab_effects(tab3)

        # Nút Lưu & So sánh (Dưới cùng cột trái)
        action_frame = tk.Frame(left_panel, bg=BG_COLOR)
        action_frame.pack(fill="x", pady=10, side="bottom")
        
        self.btn_compare = tk.Button(action_frame, text="👁️ Giữ để xem ảnh gốc", bg="#555", fg="white")
        self.btn_compare.bind('<ButtonPress-1>', self.start_compare)
        self.btn_compare.bind('<ButtonRelease-1>', self.end_compare)
        self.btn_compare.pack(fill="x", pady=5)

        tk.Button(action_frame, text="💾 LƯU ẢNH CHẤT LƯỢNG CAO", command=self.save_image, 
                  bg=ACCENT_COLOR, fg="black", font=("Arial", 12, "bold"), pady=10).pack(fill="x")

        # 2. CỘT PHẢI: HIỂN THỊ ẢNH
        right_panel = tk.Frame(main_container, bg="#000")
        right_panel.pack(side="right", fill="both", expand=True)
        
        self.lbl_display = tk.Label(right_panel, bg="#000", text="Vui lòng mở ảnh để bắt đầu...", fg="#666")
        self.lbl_display.pack(fill="both", expand=True)

    # --- CÁC HÀM TẠO GIAO DIỆN CON ---
    def create_slider(self, parent, label, vmin, vmax, vdef, res=1):
        tk.Label(parent, text=label, bg=PANEL_COLOR, fg="#aaa", font=("Arial", 9)).pack(anchor="w", pady=(10, 0))
        slider = tk.Scale(parent, from_=vmin, to=vmax, orient=HORIZONTAL, resolution=res, 
                          bg=PANEL_COLOR, fg="white", highlightthickness=0, activebackground=ACCENT_COLOR, 
                          command=self.process_pipeline)
        slider.set(vdef)
        slider.pack(fill="x")
        return slider

    def setup_tab_restore(self, parent):
        # Median Filter
        self.s_median = self.create_slider(parent, "🔸 Khử nhiễu hạt (Median)", 0, 15, 0, 2) # Mặc định 0 để không bị mờ lúc đầu
        
        # White Balance
        tk.Label(parent, text="🔸 Cân bằng trắng", bg=PANEL_COLOR, fg="#aaa", font=("Arial", 9)).pack(anchor="w", pady=(15, 0))
        self.v_wb = IntVar(value=0)
        Checkbutton(parent, text="Bật tự động (Gray World)", variable=self.v_wb, 
                    bg=PANEL_COLOR, fg=ACCENT_COLOR, selectcolor="#444", command=self.process_pipeline).pack(anchor="w")

        # Unsharp Mask
        self.s_sharp_str = self.create_slider(parent, "🔸 Độ sắc nét (Strength)", 0.0, 5.0, 0.0, 0.1)
        self.s_sharp_sigma = self.create_slider(parent, "🔸 Bán kính làm nét (Sigma)", 0.5, 5.0, 1.0, 0.5)

    def setup_tab_color(self, parent):
        self.s_bright = self.create_slider(parent, "🔹 Độ sáng (Brightness)", -100, 100, 0)
        self.s_contrast = self.create_slider(parent, "🔹 Tương phản (Contrast)", -100, 100, 0)
        self.s_sat = self.create_slider(parent, "🔹 Độ bão hòa màu (Saturation)", -100, 100, 0)

    def setup_tab_effects(self, parent):
        self.s_vignette = self.create_slider(parent, "🌑 Làm tối 4 góc (Vignette)", 0, 100, 0)
        
        tk.Label(parent, text="🎨 Bộ lọc màu", bg=PANEL_COLOR, fg="#aaa", font=("Arial", 9)).pack(anchor="w", pady=(15, 0))
        self.v_filter = tk.StringVar(value="None")
        modes = [("Không", "None"), ("Phim Cũ (Sepia)", "Sepia"), ("Đen Trắng (B&W)", "BW")]
        for text, mode in modes:
            tk.Radiobutton(parent, text=text, variable=self.v_filter, value=mode, 
                           bg=PANEL_COLOR, fg="white", selectcolor="#444", activebackground=PANEL_COLOR,
                           command=self.process_pipeline).pack(anchor="w")

        tk.Label(parent, text="🔄 Xoay Ảnh", bg=PANEL_COLOR, fg="#aaa", font=("Arial", 9)).pack(anchor="w", pady=(15, 0))
        tk.Button(parent, text="Xoay 90°", command=self.rotate_image, bg="#555", fg="white").pack(fill="x", pady=2)

    # --- LOGIC XỬ LÝ (PIPELINE) ---
    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg;*.png;*.jpeg")])
        if path:
            self.img_original_cv = cv2.imread(path)
            if self.img_original_cv is None: return
            # Reset thông số khi mở ảnh mới
            self.reset_params()
            self.process_pipeline()

    def reset_params(self):
        # Đặt lại tất cả thanh trượt về mặc định
        if hasattr(self, 's_median'): self.s_median.set(0)
        if hasattr(self, 'v_wb'): self.v_wb.set(0)
        if hasattr(self, 's_sharp_str'): self.s_sharp_str.set(0.0)
        if hasattr(self, 's_bright'): self.s_bright.set(0)
        if hasattr(self, 's_contrast'): self.s_contrast.set(0)
        if hasattr(self, 's_sat'): self.s_sat.set(0)
        if hasattr(self, 's_vignette'): self.s_vignette.set(0)
        if hasattr(self, 'v_filter'): self.v_filter.set("None")
        self.process_pipeline()

    def rotate_image(self):
        if self.img_original_cv is not None:
            self.img_original_cv = cv2.rotate(self.img_original_cv, cv2.ROTATE_90_CLOCKWISE)
            self.process_pipeline()

    def process_pipeline(self, event=None):
        if self.img_original_cv is None: return
        
        # Bắt đầu từ ảnh gốc
        img = self.img_original_cv.copy()

        # --- GIAI ĐOẠN 1: PHỤC CHẾ (TAB 1) ---
        # 1. Median Blur
        k_median = int(self.s_median.get())
        if k_median > 0:
            if k_median % 2 == 0: k_median += 1
            img = cv2.medianBlur(img, k_median)
        
        # 2. White Balance
        if self.v_wb.get() == 1:
            b, g, r = cv2.split(img)
            r_avg, g_avg, b_avg = np.mean(r), np.mean(g), np.mean(b)
            if r_avg > 0 and g_avg > 0 and b_avg > 0:
                k = (r_avg + g_avg + b_avg) / 3
                r = cv2.addWeighted(r, k/r_avg, 0, 0, 0)
                g = cv2.addWeighted(g, k/g_avg, 0, 0, 0)
                b = cv2.addWeighted(b, k/b_avg, 0, 0, 0)
                img = cv2.merge([b, g, r])

        # 3. Unsharp Mask
        strength = self.s_sharp_str.get()
        if strength > 0:
            sigma = self.s_sharp_sigma.get()
            blurred = cv2.GaussianBlur(img, (0, 0), sigma)
            img = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)

        # --- GIAI ĐOẠN 2: MÀU SẮC (TAB 2) ---
        # 4. Brightness & Contrast
        bright = self.s_bright.get()
        contrast = self.s_contrast.get()
        if bright != 0 or contrast != 0:
            # Công thức: new_img = alpha * old_img + beta
            # alpha = contrast control (1.0-3.0), beta = brightness control (0-100)
            alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            gamma = 127 * (1 - alpha)
            img = cv2.addWeighted(img, alpha, img, 0, gamma + bright)

        # 5. Saturation (Chuyển sang HSV để tăng màu)
        sat = self.s_sat.get()
        if sat != 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            # Tăng/giảm kênh S
            scale = 1 + (sat / 100.0)
            s = cv2.multiply(s, scale)
            s = np.clip(s, 0, 255).astype(np.uint8) # Giới hạn 0-255
            hsv = cv2.merge([h, s, v])
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # --- GIAI ĐOẠN 3: HIỆU ỨNG (TAB 3) ---
        # 6. Filter
        f_mode = self.v_filter.get()
        if f_mode == "BW":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif f_mode == "Sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            img = cv2.transform(img, kernel)
            img = np.clip(img, 0, 255).astype(np.uint8)

        # 7. Vignette (Làm tối góc)
        vig = self.s_vignette.get()
        if vig > 0:
            rows, cols = img.shape[:2]
            # Tạo mask Gaussian đơn giản
            X_result_kernel = cv2.getGaussianKernel(cols, cols/(vig/10 + 1))
            Y_result_kernel = cv2.getGaussianKernel(rows, rows/(vig/10 + 1))
            kernel = Y_result_kernel * X_result_kernel.T
            mask = 255 * kernel / np.linalg.norm(kernel)
            # Chuẩn hóa mask để áp dụng
            mask_stack = np.dstack([mask]*3) # Tạo 3 kênh
            
            # Blend: ảnh gốc càng ra xa tâm càng bị đen đi
            # (Cách này làm đơn giản để chạy nhanh)
            img = img.astype(float)
            # Scale mask cho khớp độ sáng
            img = img * (mask_stack / mask_stack.max())
            img = np.clip(img, 0, 255).astype(np.uint8)

        self.img_processed_cv = img
        
        if not self.preview_mode:
            self.display_image(self.img_processed_cv)

    # --- CÁC HÀM HỖ TRỢ ---
    def start_compare(self, event):
        if self.img_original_cv is not None:
            self.preview_mode = True
            self.display_image(self.img_original_cv)
            self.lbl_display.config(text="[ĐANG XEM ẢNH GỐC]")

    def end_compare(self, event):
        if self.img_processed_cv is not None:
            self.preview_mode = False
            self.display_image(self.img_processed_cv)

    def display_image(self, cv_img):
        # Chuyển BGR -> RGB
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Tính toán resize giữ tỉ lệ
        w_canvas = self.lbl_display.winfo_width()
        h_canvas = self.lbl_display.winfo_height()
        if w_canvas < 10: w_canvas = 800
        if h_canvas < 10: h_canvas = 600

        img_pil.thumbnail((w_canvas, h_canvas), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.lbl_display.config(image=img_tk, text="")
        self.lbl_display.image = img_tk

    def save_image(self):
        if self.img_processed_cv is None: return
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPG", "*.jpg"), ("PNG", "*.png")])
        if path:
            cv2.imwrite(path, self.img_processed_cv)
            messagebox.showinfo("Đã lưu", "Lưu ảnh thành công!")

if __name__ == "__main__":
    root = tk.Tk()
    app = LuxuryRestoreApp(root)
    root.mainloop()