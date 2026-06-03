@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0show_windows_campaign.ps1" %*
exit /b %ERRORLEVEL%
