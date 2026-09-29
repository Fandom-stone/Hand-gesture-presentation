@echo off
REM ===================================================================
REM  Launcher for the gesture presentation system.
REM
REM      .\go              present on DESKTOP PowerPoint (the default)
REM      .\go web          present on PowerPoint for the web instead
REM      .\go collect      record static pose samples
REM      .\go swipes       record swipe clips
REM      .\go tune         camera window + diagnostics, for debugging
REM      .\go dry          recognise only, send no keypresses
REM      .\go train        retrain both models
REM      .\go test         run the test suite
REM      .\go profile      measure where startup time goes
REM
REM  Extra flags pass straight through, e.g.  .\go tune --camera 1
REM
REM  NOTE: this deliberately calls venv\Scripts\python.exe by full path
REM  instead of running venv\Scripts\activate.bat first.
REM
REM  activate.bat expands %PATH% inside parenthesised if-blocks, and a
REM  ")" anywhere in the path -- as in "gesture-presentation-control
REM  (CLD)" -- closes that block early. The result is a mangled PATH and
REM  a silent fall back to system Python, which shows up as a confusing
REM  "No module named 'cv2'" even when the venv is fine. Naming the
REM  interpreter directly sidesteps activation completely, so the folder
REM  can be called anything and it still works.
REM ===================================================================

setlocal
cd /d "%~dp0"

set "PYEXE=%~dp0venv\Scripts\python.exe"
if not exist "%PYEXE%" goto no_venv

set "MODE=%~1"
if "%MODE%"=="" set "MODE=present"
if not "%~1"=="" shift

if /i "%MODE%"=="present" goto run_present
if /i "%MODE%"=="web"     goto run_web
if /i "%MODE%"=="desktop" goto run_present
if /i "%MODE%"=="collect" goto run_collect
if /i "%MODE%"=="swipes"  goto run_swipes
if /i "%MODE%"=="tune"    goto run_tune
if /i "%MODE%"=="dry"     goto run_dry
if /i "%MODE%"=="train"   goto run_train
if /i "%MODE%"=="test"    goto run_test
if /i "%MODE%"=="profile" goto run_profile
if /i "%MODE%"=="which"   goto run_which
goto usage

:run_present
echo   Presenting on DESKTOP PowerPoint. Click your slides to focus them. Ctrl+C to stop.
echo   Hold the pointing pose for the laser.
"%PYEXE%" -m src.infer_realtime --present --target desktop %1 %2 %3 %4
goto end

:run_web
echo   Presenting on PowerPoint for the WEB. Click your slides to focus them. Ctrl+C to stop.
echo   The web player has no laser mode - the pointing pose moves the ordinary cursor.
"%PYEXE%" -m src.infer_realtime --present --target web %1 %2 %3 %4
goto end

:run_collect
echo   Static pose collection. Keys 1-5 pick a pose, HOLD c to capture, q/Esc quits.
echo   Targets: 250+ NEUTRAL, 150+ each of the rest. Vary distance, angle and lighting.
"%PYEXE%" -m src.collect_data %1 %2 %3 %4
goto end

:run_swipes
echo   Swipe clip collection. Keys 1-3 pick a class, press c ONCE then do the motion.
echo   Targets: 40+ each swipe, 60+ NO_SWIPE. Hold still until the bar fills.
"%PYEXE%" -m src.collect_dynamic_data %1 %2 %3 %4
goto end

:run_tune
echo   Tuning mode: camera window on, diagnostics on. q or Esc to quit.
"%PYEXE%" -m src.infer_realtime %1 %2 %3 %4
goto end

:run_dry
echo   Dry run: gestures are recognised and printed, no keys are sent.
"%PYEXE%" -m src.infer_realtime --dry-run %1 %2 %3 %4
goto end

:run_train
echo   Retraining both models...
"%PYEXE%" -m src.train_model %1 %2 %3 %4
if errorlevel 1 goto end
echo.
"%PYEXE%" -m src.train_dynamic_model %1 %2 %3 %4
goto end

:run_test
"%PYEXE%" -m unittest discover -s tests -t . %1 %2 %3 %4
goto end

:run_profile
"%PYEXE%" scripts\profile_startup.py %1 %2 %3 %4
goto end

:run_which
REM Diagnostic: confirms which interpreter the launcher will actually use.
echo   Interpreter : %PYEXE%
"%PYEXE%" -c "import sys; print('   Version     :', sys.version.split()[0]); print('   Running from:', sys.executable)"
"%PYEXE%" -c "import cv2, mediapipe, sklearn; print('   cv2', cv2.__version__, '| mediapipe', mediapipe.__version__, '| sklearn', sklearn.__version__)"
goto end

:no_venv
echo.
echo   Could not find the project's Python at:
echo     %PYEXE%
echo.
echo   Create the virtual environment first:
echo     python -m venv venv
echo     venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
exit /b 1

:usage
echo.
echo   Unknown mode: %MODE%
echo.
echo   Usage:  .\go [present^|web^|collect^|swipes^|tune^|dry^|train^|test^|profile^|which]
echo.
echo     present   present on desktop PowerPoint - no window, quiet, laser
echo     web       same, but for PowerPoint in a browser
echo     collect   record static pose samples
echo     swipes    record swipe clips
echo     tune      camera window + diagnostics, for debugging
echo     dry       recognise only, send no keypresses
echo     train     retrain both models
echo     test      run the test suite
echo     profile   measure where startup time goes
echo     which     show which Python and libraries the launcher will use
echo.
echo   No mode given runs 'present'.
echo.
exit /b 1

:end
endlocal
