@echo off
rem 切换到脚本所在目录
cd /d "%~dp0"

rem 执行 Python 脚本
python soft.py

rem 运行结束后暂停，方便查看输出或错误信息
pause