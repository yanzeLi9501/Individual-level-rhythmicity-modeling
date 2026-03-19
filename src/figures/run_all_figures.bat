@echo off
echo =========================================
echo  Running all figure generation modules
echo =========================================
echo.

cd /d D:\LDH_cancer\files\healthline

echo [1/7] Figure 1: Model Performance
python gen_fig1.py
if %ERRORLEVEL% neq 0 (echo ERROR in gen_fig1.py & pause & exit /b 1)
echo.

echo [2/7] Figure 2: Comorbidity Performance
python gen_fig2.py
if %ERRORLEVEL% neq 0 (echo ERROR in gen_fig2.py & pause & exit /b 1)
echo.

echo [3/7] Figure 3: Feature Importance
python gen_fig3.py
if %ERRORLEVEL% neq 0 (echo ERROR in gen_fig3.py & pause & exit /b 1)
echo.

echo [4/7] Figure 4: COVID Impact
python gen_fig4.py
if %ERRORLEVEL% neq 0 (echo ERROR in gen_fig4.py & pause & exit /b 1)
echo.

echo [5/7] Figure 5: COVID Test Stratification
python gen_fig5.py
if %ERRORLEVEL% neq 0 (echo ERROR in gen_fig5.py & pause & exit /b 1)
echo.

echo [6/7] Supplementary Figures S1-S6
python gen_supp_figs.py
if %ERRORLEVEL% neq 0 (echo ERROR in gen_supp_figs.py & pause & exit /b 1)
echo.

echo [7/7] DOCX Paper
python gen_paper.py
if %ERRORLEVEL% neq 0 (echo ERROR in gen_paper.py & pause & exit /b 1)
echo.

echo =========================================
echo  ALL MODULES COMPLETE!
echo  Output: readmission_output\figures_v2\
echo  Paper:  readmission_output\manuscript_with_figures.docx
echo =========================================
pause
