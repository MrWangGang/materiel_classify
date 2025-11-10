# soft_modern.py (GUI 界面文件，优化了视觉效果)

import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import zipfile
import threading
import io
from typing import Optional, List, Dict, Tuple, Any, Callable

# 假设核心处理函数和常量位于 pdf_processor1.py 中
try:
    from pdf_processor2 import extract_part_contours_from_pdf
except ImportError:
    def extract_part_contours_from_pdf(*args, **kwargs):
        messagebox.showerror("错误", "找不到 pdf_processor1.py 文件。请确保该文件在同一目录下！")
        return {}
    print("警告：缺少 pdf_processor1.py，程序将无法正常工作。")

# ======================================================================
# 全局配置和常量
# ======================================================================
FINAL_ZIP_DEFAULT_NAME = "extracted_contours_data.zip"

class PDFExtractorApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("PDF 零件轮廓提取工具")
        self.geometry("800x650") # 稍微增加高度以容纳新布局
        self.minsize(700, 600)  # 设置最小窗口尺寸

        # --- 全局布局配置 ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1) # 让第三个框架（控制区）可以扩展

        self.pdf_file_path: Optional[str] = None
        self.zip_data_buffer: Optional[io.BytesIO] = None

        # --- UI 状态变量 ---
        self.pdf_path_var = ctk.StringVar(value="尚未选择任何 PDF 文件")
        self.progress_page_count_var = ctk.StringVar(value="0/0")
        self.status_var = ctk.StringVar(value="欢迎使用！请先选择一个 PDF 文件。")

        self._setup_ui()

    def _setup_ui(self):
        """搭建现代化的用户界面"""
        # --- 定义字体 ---
        self.title_font = ctk.CTkFont(size=16, weight="bold")
        self.label_font = ctk.CTkFont(size=13)
        self.status_font = ctk.CTkFont(size=12)

        # --- 1. 文件选择区 ---
        file_frame = ctk.CTkFrame(self, corner_radius=10)
        file_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        file_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(file_frame, text="1. 选择文件", font=self.title_font).grid(
            row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w")

        ctk.CTkLabel(file_frame, text="PDF 路径:", font=self.label_font).grid(
            row=1, column=0, padx=(15, 5), pady=10, sticky="w")

        path_label = ctk.CTkLabel(file_frame, textvariable=self.pdf_path_var, anchor="w", fg_color=("gray90", "gray25"), corner_radius=5)
        path_label.grid(row=1, column=1, padx=(0, 10), pady=10, sticky="ew")

        ctk.CTkButton(file_frame, text="浏览...", width=100, command=self._select_pdf).grid(
            row=1, column=2, padx=(0, 15), pady=10)

        # --- 2. 参数配置区 ---
        param_frame = ctk.CTkFrame(self, corner_radius=10)
        param_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        param_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(param_frame, text="2. 参数配置", font=self.title_font).grid(
            row=0, column=0, columnspan=2, padx=15, pady=(15, 10), sticky="w")

        self.param_vars: Dict[str, ctk.StringVar] = {}
        self._add_param_entry(param_frame, "渲染 DPI:", "600", 1, "TARGET_DPI")
        self._add_param_entry(param_frame, "起始页 (从1开始):", "7", 2, "PAGE_START")
        self._add_param_entry(param_frame, "结束页 (包含):", "200", 3, "PAGE_END")

        self.output_debug_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(param_frame, text="生成并包含调试过程图片", variable=self.output_debug_var,
                        font=self.label_font).grid(row=4, column=0, columnspan=2, padx=15, pady=(10, 15), sticky="w")


        # --- 3. 运行与下载区 ---
        control_frame = ctk.CTkFrame(self, corner_radius=10)
        control_frame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="nsew")
        control_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(control_frame, text="3. 运行与下载", font=self.title_font).grid(
            row=0, column=0, columnspan=2, padx=15, pady=(15, 10), sticky="w")

        # 运行按钮
        self.run_button = ctk.CTkButton(control_frame, text="开始处理", height=40,
                                        command=self._start_processing, state="disabled")
        self.run_button.grid(row=1, column=0, columnspan=2, padx=15, pady=10, sticky="ew")

        # 下载按钮
        self.download_button = ctk.CTkButton(control_frame, text="下载结果 (.zip)", height=40,
                                             command=self._download_zip, state="disabled")
        self.download_button.grid(row=2, column=0, columnspan=2, padx=15, pady=(5, 10), sticky="ew")

        # --- 状态与进度条 ---
        status_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        status_frame.grid(row=3, column=0, columnspan=2, padx=15, pady=(10, 15), sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(status_frame)
        self.progress_bar.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="ew")
        self.progress_bar.set(0)

        ctk.CTkLabel(status_frame, textvariable=self.progress_page_count_var, width=60, font=self.label_font).grid(
            row=0, column=1, pady=5, sticky="e")

        ctk.CTkLabel(status_frame, textvariable=self.status_var, font=self.status_font, anchor="w").grid(
            row=1, column=0, columnspan=2, pady=(5, 0), sticky="ew")


    def _add_param_entry(self, parent: ctk.CTkFrame, label_text: str, default_value: str, row: int, name: str):
        ctk.CTkLabel(parent, text=label_text, font=self.label_font).grid(
            row=row, column=0, padx=(15, 5), pady=5, sticky="w")
        var = ctk.StringVar(value=default_value)
        entry = ctk.CTkEntry(parent, textvariable=var, width=180) # 适当增加宽度
        entry.grid(row=row, column=1, padx=(0, 15), pady=5, sticky="w") # 改为靠左对齐，更整洁
        self.param_vars[name] = var

    def _update_progress(self, current: int, total: int):
        """主线程中安全更新进度条和标签的回调函数。"""
        if total > 0:
            percentage = current / total
            self.progress_bar.set(percentage)
            self.progress_page_count_var.set(f"{current}/{total}")
        else:
            self.progress_bar.set(0)
            self.progress_page_count_var.set("0/0")

        if current < total and current > 0:
            self.status_var.set(f"处理中... 第 {current} 页 / 共 {total} 页")

        self.update_idletasks() # 使用 idletasks 避免界面卡顿

    def _select_pdf(self):
        fpath = filedialog.askopenfilename(
            title="选择 PDF 文件",
            filetypes=[("PDF files", "*.pdf")]
        )
        if fpath:
            self.pdf_file_path = fpath
            # 显示文件名，如果路径太长则截断
            display_path = os.path.basename(fpath)
            if len(display_path) > 50:
                display_path = "..." + display_path[-47:]
            self.pdf_path_var.set(display_path)

            self.run_button.configure(state="normal")
            self.download_button.configure(state="disabled")
            self.zip_data_buffer = None
            self.progress_bar.set(0)
            self.progress_page_count_var.set("0/0")
            self.status_var.set(f"文件已就绪: {os.path.basename(fpath)}")
        else:
            self.status_var.set("用户取消了文件选择。")

    def _start_processing(self):
        if not self.pdf_file_path:
            messagebox.showerror("错误", "请先选择一个 PDF 文件。")
            return

        self.run_button.configure(state="disabled", text="正在处理中...")
        self.download_button.configure(state="disabled")
        self.status_var.set("正在初始化处理任务...")
        self.progress_bar.set(0)
        self.progress_page_count_var.set("0/0")
        self.update_idletasks()

        processing_thread = threading.Thread(target=self._process_pdf_thread, daemon=True)
        processing_thread.start()

    def _process_pdf_thread(self):
        """后台线程方法：执行耗时的 PDF 处理逻辑和内存打包"""
        success = False
        status_message = "处理失败。请检查参数或文件。"
        error_message = None
        extracted_data: Dict[str, bytes] = {}

        def thread_safe_callback(current, total):
            # +1 是为了让页码从 1 开始显示
            self.after(0, self._update_progress, current + 1, total)

        try:
            params = self._get_and_validate_params()
            if params is None:
                status_message = "参数校验失败，请检查输入。"
                self.after(0, lambda: self._finalize_processing_ui(False, status_message))
                return

            extracted_data = extract_part_contours_from_pdf(
                pdf_file=self.pdf_file_path,
                page_range=params['PAGE_RANGE'],
                target_dpi=params['TARGET_DPI'],
                ocr_languages=['en'],
                morph_kernel_size=1,
                min_contour_area=500,
                padding=10,
                output_debug_images=params['OUTPUT_DEBUG_IMAGES'],
                progress_callback=thread_safe_callback
            )

            if extracted_data:
                self.zip_data_buffer = self._create_zip_archive_in_memory(extracted_data)
                contour_count = sum(1 for k in extracted_data.keys() if k.startswith("轮廓图/"))
                status_message = f"处理完成！成功提取 {contour_count} 个零件轮廓。"
                success = True
            else:
                status_message = "处理完成，但未能从指定页面中提取到任何有效轮廓。"
                success = False

        except Exception as e:
            status_message = "处理过程中发生了一个意外错误。"
            error_message = f"错误详情: {e}"
            success = False

        self.after(0, lambda: self._finalize_processing_ui(success, status_message, error_message))

    def _create_zip_archive_in_memory(self, file_data: Dict[str, bytes]) -> io.BytesIO:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename, data_bytes in file_data.items():
                zipf.writestr(filename, data_bytes)
        zip_buffer.seek(0)
        return zip_buffer

    def _finalize_processing_ui(self, success: bool, status_message: str, error_message: Optional[str] = None):
        """在主线程中执行，用于更新UI状态"""
        self.progress_bar.set(1.0 if success else 0.0)
        self.status_var.set(status_message)
        self.run_button.configure(state="normal", text="重新开始")

        if success and self.zip_data_buffer is not None:
            self.download_button.configure(state="normal")
        else:
            self.download_button.configure(state="disabled")
            if error_message:
                messagebox.showerror("处理失败", f"{status_message}\n\n{error_message}")

    def _get_and_validate_params(self) -> Optional[Dict[str, Any]]:
        try:
            dpi = int(self.param_vars['TARGET_DPI'].get())
            p_start = int(self.param_vars['PAGE_START'].get())
            p_end = int(self.param_vars['PAGE_END'].get())
            output_debug = self.output_debug_var.get()
        except ValueError:
            messagebox.showerror("参数错误", "DPI 和页码必须是有效的整数。")
            return None

        if not (1 <= dpi <= 2400):
            messagebox.showerror("参数错误", "DPI 值建议在 1 到 2400 之间。")
            return None
        if p_start <= 0 or p_end < p_start:
            messagebox.showerror("参数错误", "页码范围无效。起始页必须大于0，且不能大于结束页。")
            return None

        return {
            'TARGET_DPI': dpi,
            'PAGE_RANGE': [p_start, p_end],
            'OUTPUT_DEBUG_IMAGES': output_debug
        }

    def _download_zip(self):
        if self.zip_data_buffer is None:
            messagebox.showerror("错误", "没有可供下载的结果。请先运行处理。")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            initialfile=FINAL_ZIP_DEFAULT_NAME,
            filetypes=[("ZIP 压缩包", "*.zip")],
            title="保存提取结果"
        )

        if save_path:
            try:
                with open(save_path, 'wb') as f:
                    f.write(self.zip_data_buffer.getvalue())
                self.status_var.set(f"下载成功！文件已保存。")
                messagebox.showinfo("下载完成", f"结果已成功保存到:\n{save_path}")
            except Exception as e:
                messagebox.showerror("下载失败", f"无法将文件写入目标位置: {e}")
        else:
            self.status_var.set("用户取消了下载。")


if __name__ == "__main__":
    # 推荐使用 'System' 模式，它可以自动适应操作系统的亮/暗模式
    ctk.set_appearance_mode("System")
    # 'dark-blue' 主题在暗色模式下观感更佳
    ctk.set_default_color_theme("dark-blue")

    app = PDFExtractorApp()
    app.mainloop()