@echo off
rem 切换到 UTF-8 编码以正确显示中文
chcp 65001 > nul

rem 设置当前目录为脚本所在的目录，确保能找到 pip
cd /d "%~dp0"

echo 正在使用清华镜像安装项目依赖，请稍候...
echo.

rem 使用 pip 安装所需的库
rem 界面依赖：customtkinter
rem 核心依赖：opencv-python, numpy, easyocr, PyMuPDF
rem EasyOCR运行时依赖：torch (建议保留以确保EasyOCR正常工作)
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple customtkinter opencv-python numpy easyocr torch PyMuPDF

echo.
echo 所有依赖都已安装完成！
echo.

pause