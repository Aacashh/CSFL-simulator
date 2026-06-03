@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run_windows.ps1" %*
exit /b %ERRORLEVEL%
