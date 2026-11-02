@echo off
setlocal
echo ===================================================
echo BookWise - Personalized Book Recommendation System
echo College Big Data Analytics Microproject
echo ===================================================
echo.

set "SCRIPT_DIR=%~dp0"
set "JAVA_HOME=%SCRIPT_DIR%jdk17"
set "PATH=%SCRIPT_DIR%jdk17\bin;C:\Users\Asus\AppData\Local\Programs\Python\Python314;C:\Users\Asus\AppData\Local\Programs\Python\Python314\Scripts;%PATH%"
set "PYSPARK_PYTHON=C:\Users\Asus\AppData\Local\Programs\Python\Python314\python.exe"
set "PYSPARK_DRIVER_PYTHON=C:\Users\Asus\AppData\Local\Programs\Python\Python314\python.exe"

echo Checking PySpark and dataset cache...
if not exist "data\processed\books_processed.csv" (
    echo Processed data not found. Running PySpark pipeline...
    python data_processing.py
)

echo Starting BookWise Streamlit Web Application...
python -m streamlit run app.py --server.port 8501

pause

