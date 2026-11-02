@echo off
setlocal
echo ===================================================
echo BookWise - Push to GitHub for Streamlit Deployment
echo ===================================================
echo.
set "PATH=C:\Users\Asus\AppData\Local\Programs\MinGit\cmd;%PATH%"

set /p REPO_URL="Enter your new GitHub repository URL (e.g. https://github.com/username/BookWise.git): "
if "%REPO_URL%"=="" (
    echo No repository URL entered. Exiting.
    pause
    exit /b 1
)

echo.
echo Linking remote repository...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo Setting primary branch to 'main'...
git branch -M main

echo.
echo Pushing codebase and pre-computed analytical models to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================================
    echo [SUCCESS] Codebase pushed successfully!
    echo.
    echo Next Steps for Streamlit Cloud Deployment:
    echo 1. Open: https://share.streamlit.io/
    echo 2. Click 'Create app'
    echo 3. Select your repository
    echo 4. Main file path: app.py
    echo 5. Click 'Deploy'!
    echo ===================================================
) else (
    echo.
    echo [NOTE] If prompted for credentials, use your GitHub username and Personal Access Token (PAT).
)

pause

