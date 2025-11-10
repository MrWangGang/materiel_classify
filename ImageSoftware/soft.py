import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import zipfile
import os
import shutil
import glob
from pathlib import Path
from threading import Thread
import tempfile
import cv2
import easyocr
import numpy as np # 新增导入 numpy 以使用 cv2.imencode/tobytes

# 导入两个处理模块
# 这里的函数现在返回一个元组：(处理后的图像, 是否处理成功)
from model_fade_component import process_image_for_streamlit
from model_delete_component import process_image_by_x_cutoff

try:
    # 实例化 Reader，这个操作只在程序启动时执行一次。
    GLOBAL_EASYOCR_READER = easyocr.Reader(['en'])
    print("EasyOCR Reader 初始化成功 (只执行一次)。")
except Exception as e:
    print(f"错误：EasyOCR Reader 模块初始化失败: {e}")
    GLOBAL_EASYOCR_READER = None

class ImageProcessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 配置窗口
        self.title("图像处理与下载工具")
        self.geometry("600x500")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # UI 组件
        self.title_label = ctk.CTkLabel(self, text="图像处理与下载工具", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, pady=(20, 10))

        # 选项卡视图
        self.tabview = ctk.CTkTabview(self, command=self.on_tab_change)
        self.tabview.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.tabview.grid_columnconfigure(0, weight=1)

        # 创建两个选项卡，并保存对它们的直接引用
        self.tab_fade = self.tabview.add("淡化处理")
        self.tab_cutoff = self.tabview.add("消除处理")

        # 配置淡化处理选项卡
        self.create_tab_ui(self.tab_fade, "请上传一个包含 .jpg 或 .png 图像的压缩包 必须zip格式，并进行淡化处理。")

        # 配置消除处理选项卡
        self.create_tab_ui(self.tab_cutoff, "请上传一个包含 .jpg 或 .png 图像的压缩包 必须zip格式，并进行消除处理。")

        # 初始状态
        self.tabview.set("淡化处理")
        self.update_tab_buttons()

    def create_tab_ui(self, tab_frame, info_text):
        """创建单个选项卡的 UI 元素，并为每个选项卡创建独立的私有状态。"""
        # 为每个选项卡创建独立的变量，存储其私有状态
        tab_frame.uploaded_zip_path = None
        tab_frame.processed_zip_data = None
        tab_frame.download_zip_name = "processed_images.zip"

        tab_frame.grid_columnconfigure(0, weight=1)

        info_label = ctk.CTkLabel(tab_frame, text=info_text, font=ctk.CTkFont(size=14))
        info_label.grid(row=0, column=0, pady=(10, 10))

        browse_frame = ctk.CTkFrame(tab_frame)
        browse_frame.grid(row=1, column=0, pady=10, sticky="ew", padx=20)
        browse_frame.grid_columnconfigure(1, weight=1)

        file_path_label = ctk.CTkLabel(browse_frame, text="未选择文件", font=ctk.CTkFont(size=12))
        file_path_label.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        browse_button = ctk.CTkButton(browse_frame, text="选择压缩包", command=lambda: self.browse_zip_file(file_path_label))
        browse_button.grid(row=0, column=0, padx=10, pady=10)

        process_button = ctk.CTkButton(tab_frame, text="开始处理", command=self.start_processing_thread, state="disabled")
        process_button.grid(row=2, column=0, pady=10)

        progress_bar = ctk.CTkProgressBar(tab_frame, mode="determinate")
        progress_bar.grid(row=3, column=0, pady=10, sticky="ew", padx=20)
        progress_bar.set(0)

        status_label = ctk.CTkLabel(tab_frame, text="等待上传文件...", font=ctk.CTkFont(size=12))
        status_label.grid(row=4, column=0, pady=(0, 10))

        download_button = ctk.CTkButton(tab_frame, text="下载处理后的压缩包", command=self.download_processed_zip, state="disabled")
        download_button.grid(row=5, column=0, pady=(10, 20))

        # 将组件存储在字典中，以便按选项卡名称访问
        tab_frame.widgets = {
            "file_path_label": file_path_label,
            "browse_button": browse_button,
            "process_button": process_button,
            "progress_bar": progress_bar,
            "status_label": status_label,
            "download_button": download_button
        }

    def on_tab_change(self, tab_name=None):
        """选项卡切换时更新 UI 状态。"""
        # tab_name 参数由 CTkTabview 的 command 自动传入
        self.update_tab_buttons()

    def get_current_tab_frame(self):
        """获取当前选中的选项卡框架的引用。"""
        current_tab_name = self.tabview.get()
        if current_tab_name == "淡化处理":
            return self.tab_fade
        elif current_tab_name == "消除处理":
            return self.tab_cutoff
        return None

    def update_tab_buttons(self):
        """根据当前选项卡的文件选择状态更新其按钮。"""
        current_tab_frame = self.get_current_tab_frame()
        if not current_tab_frame:
            return

        process_button = current_tab_frame.widgets["process_button"]
        download_button = current_tab_frame.widgets["download_button"]

        # 现在每个选项卡有独立的 uploaded_zip_path 变量
        if current_tab_frame.uploaded_zip_path:
            process_button.configure(state="normal")
        else:
            process_button.configure(state="disabled")

        # 现在每个选项卡有独立的 processed_zip_data 变量
        if current_tab_frame.processed_zip_data:
            download_button.configure(state="normal")
        else:
            download_button.configure(state="disabled")

    def browse_zip_file(self, file_path_label):
        """打开文件对话框，让用户选择一个 ZIP 文件。"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Zip files", "*.zip"), ("All files", "*.*")]
        )
        if file_path:
            current_tab_frame = self.get_current_tab_frame()
            # 将文件路径存储在当前选项卡的私有变量中
            current_tab_frame.uploaded_zip_path = file_path

            file_path_label.configure(text=os.path.basename(file_path))
            self.update_tab_buttons()
            current_tab_frame.widgets["status_label"].configure(text="文件已选择，可以开始处理。")

    def start_processing_thread(self):
        """在单独的线程中开始图像处理。"""
        current_tab_frame = self.get_current_tab_frame()
        if not current_tab_frame or not current_tab_frame.uploaded_zip_path:
            return

        # 禁用所有按钮
        for widget in current_tab_frame.widgets.values():
            if isinstance(widget, ctk.CTkButton):
                widget.configure(state="disabled")

        current_tab_frame.widgets["status_label"].configure(text="正在处理中，请稍候...")
        current_tab_frame.widgets["progress_bar"].set(0)

        # 在后台线程中运行处理函数
        self.processing_thread = Thread(target=self.process_images)
        self.processing_thread.start()

    def process_images(self):
        """核心处理函数，在单独的线程中运行。"""
        try:
            current_tab_name = self.tabview.get()
            current_tab_frame = self.get_current_tab_frame()
            if not current_tab_frame:
                return

            # 创建临时目录
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = Path(temp_dir_obj.name)
            input_dir = temp_dir / "input"
            output_dir = temp_dir / "output"
            input_dir.mkdir(exist_ok=True)
            output_dir.mkdir(exist_ok=True)

            # 创建处理后的分类文件夹
            processed_dir = output_dir / "已处理"
            unprocessed_dir = output_dir / "未处理"
            processed_dir.mkdir(exist_ok=True)
            unprocessed_dir.mkdir(exist_ok=True)

            # 解压文件
            with zipfile.ZipFile(current_tab_frame.uploaded_zip_path, 'r') as zip_ref:
                zip_ref.extractall(input_dir)

            image_files = [f for f in glob.glob(str(input_dir / "**" / "*.*"), recursive=True)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                self.after(0, lambda: messagebox.showerror("错误", "上传的压缩包中没有找到任何图像文件。"))
                return

            total_images = len(image_files)

            # 遍历并处理图像
            for i, image_path in enumerate(image_files):
                processed_image = None
                is_successful = False # 初始状态为处理失败

                # 根据当前选项卡选择不同的处理函数
                if current_tab_name == "淡化处理":
                    processed_image, is_successful = process_image_for_streamlit(image_path)
                elif current_tab_name == "消除处理":
                    processed_image, is_successful = process_image_by_x_cutoff(image_path,GLOBAL_EASYOCR_READER)

                # 获取相对于输入目录的相对路径，用于在输出中保持原始文件夹结构
                relative_path = Path(image_path).relative_to(input_dir)

                if is_successful:
                    output_path = processed_dir / relative_path
                    # 确保输出目录存在
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # 【！！！解决中文路径问题的核心修改！！！】
                    try:
                        # 1. 获取文件扩展名
                        extension = output_path.suffix

                        # 2. 使用 cv2.imencode 将图像编码为内存缓冲区
                        # 注意：对于 PNG，需要确保图像是 BGRA 或 BGR，我们的处理函数返回的是 BGRA
                        is_success, buffer = cv2.imencode(extension, processed_image)

                        if is_success:
                            # 3. 使用 Python 内置的 open 函数以 wb 模式写入文件
                            # Path 对象可以直接传入 open
                            with open(output_path, 'wb') as f:
                                # buffer 是 numpy 数组，需要转换为 bytes
                                f.write(buffer.tobytes())
                        else:
                            print(f"cv2.imencode 编码失败: {output_path}")
                            is_successful = False # 编码失败，视为处理失败
                    except Exception as file_error:
                        print(f"写入文件 {output_path} 失败: {file_error}")
                        is_successful = False # 写入失败，视为处理失败
                    # 【修改结束】

                    # 如果处理、编码或写入失败，则将原始图像复制到 '未处理' 文件夹
                    if not is_successful:
                        output_path = unprocessed_dir / relative_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        # 如果处理失败，将原始图像复制到 '未处理' 文件夹
                        shutil.copy(image_path, output_path)

                else:
                    # 如果处理函数返回 False（未找到截止线等），则走原逻辑：将原始图像复制到 '未处理' 文件夹
                    output_path = unprocessed_dir / relative_path
                    # 确保输出目录存在
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(image_path, output_path)


                # 更新进度条
                progress_percentage = (i + 1) / total_images
                self.after(0, lambda p=progress_percentage: current_tab_frame.widgets["progress_bar"].set(p))
                self.after(0, lambda: current_tab_frame.widgets["status_label"].configure(text=f"处理进度: {int(progress_percentage*100)}%"))

            # 打包处理后的文件，并根据选项卡名称命名
            current_tab_frame.download_zip_name = f"{current_tab_name}_processed_images.zip"
            final_zip_path = temp_dir / current_tab_frame.download_zip_name

            # 打包整个 output 目录，包含 '已处理' 和 '未处理' 两个子文件夹
            self.create_zip_from_folder(str(output_dir), str(final_zip_path))

            # 读取文件内容，以便稍后下载
            with open(final_zip_path, "rb") as f:
                # 将处理后的数据存储在当前选项卡的私有变量中
                current_tab_frame.processed_zip_data = f.read()

            # 清理临时目录
            temp_dir_obj.cleanup()

            # 更新UI状态
            self.after(0, self.on_processing_complete)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("错误", f"处理过程中发生错误: {e}"))
            self.after(0, self.on_processing_error)

    def create_zip_from_folder(self, folder_path, output_path):
        """将指定文件夹下的所有文件打包成ZIP文件。"""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)

    def on_processing_complete(self):
        """处理完成后的 UI 状态更新。"""
        current_tab_frame = self.get_current_tab_frame()
        if not current_tab_frame:
            return

        current_tab_frame.widgets["status_label"].configure(text="所有图像处理完成！")
        current_tab_frame.widgets["download_button"].configure(state="normal")
        current_tab_frame.widgets["process_button"].configure(state="disabled")
        current_tab_frame.widgets["browse_button"].configure(state="normal")
        current_tab_frame.widgets["progress_bar"].set(1)

    def on_processing_error(self):
        """处理出错后的 UI 状态恢复。"""
        current_tab_frame = self.get_current_tab_frame()
        if not current_tab_frame:
            return

        current_tab_frame.widgets["status_label"].configure(text="处理失败，请重试。")
        current_tab_frame.widgets["process_button"].configure(state="normal")
        current_tab_frame.widgets["browse_button"].configure(state="normal")
        current_tab_frame.widgets["download_button"].configure(state="disabled")
        current_tab_frame.widgets["progress_bar"].set(0)

    def download_processed_zip(self):
        """打开保存文件对话框，让用户保存处理后的 ZIP 文件。"""
        current_tab_frame = self.get_current_tab_frame()
        if not current_tab_frame or not current_tab_frame.processed_zip_data:
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("Zip files", "*.zip")],
            # 使用当前选项卡的私有变量作为默认文件名
            initialfile=current_tab_frame.download_zip_name
        )
        if save_path:
            with open(save_path, "wb") as f:
                f.write(current_tab_frame.processed_zip_data)
            messagebox.showinfo("成功", "文件已成功保存！")

if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()