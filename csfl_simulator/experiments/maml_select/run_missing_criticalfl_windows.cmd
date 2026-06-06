@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_missing_criticalfl_windows.ps1" %*
exit /b %ERRORLEVEL%
