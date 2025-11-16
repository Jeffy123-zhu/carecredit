@echo off
chcp 65001 >nul
echo ========================================
echo 阿尔茨海默病检测系统 - Web应用启动器
echo ========================================
echo.

REM 检查是否安装了streamlit
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [错误] 未检测到streamlit，正在安装...
    pip install streamlit
    if errorlevel 1 (
        echo [错误] streamlit安装失败，请手动运行: pip install streamlit
        pause
        exit /b 1
    )
)

echo [信息] 正在启动Web应用...
echo [信息] 浏览器将自动打开，如果没有自动打开，请访问: http://localhost:8501
echo.
echo 按 Ctrl+C 停止服务器
echo.

streamlit run alzheimers_detection_web_app.py

pause

