@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0show_missing_criticalfl_windows.ps1" %*
exit /b %ERRORLEVEL%
