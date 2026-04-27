@echo off
REM Wrapper for scripts/check.py — cmd.exe.
REM All args are forwarded: `scripts\check.bat --fix` works.
pushd "%~dp0\.."
python scripts\check.py %*
set EXITCODE=%ERRORLEVEL%
popd
exit /b %EXITCODE%
