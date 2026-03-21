@echo off
echo =========================================
echo  Running all figure generation modules
echo =========================================
echo.

set BASE=D:\LDH_cancer\files\healthline\readmission_output\figures_v2\merge\Individual-level-rhythmicity-modeling
set FIG=%BASE%\src\figures
set SUPP=%BASE%\src\supplementary

echo [1/13] Figure 1: Epidemiology and Behavioral Profiles
python "%FIG%\gen_fig_new1.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_fig_new1.py & pause & exit /b 1)
echo.

echo [2/13] Figure 2: Sentinel Discovery
python "%FIG%\gen_fig_new2.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_fig_new2.py & pause & exit /b 1)
echo.

echo [3/13] Figure 3: Early Warning / RDI
python "%FIG%\gen_fig_new3.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_fig_new3.py & pause & exit /b 1)
echo.

echo [4/13] Figure 4 + S6: Post-Pandemic Validation
python "%FIG%\gen_fig_new4.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_fig_new4.py & pause & exit /b 1)
echo.

echo [5/13] Figure 5: Model Generalizability
python "%FIG%\gen_fig_new5.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_fig_new5.py & pause & exit /b 1)
echo.

echo [6/16] Supplementary S1,S9,S11-S12,S14
python "%SUPP%\gen_supp_figs.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_supp_figs.py & pause & exit /b 1)
echo.

echo [7/16] Supplementary S10: Best Model Analysis
python "%SUPP%\gen_supp_best_model.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_supp_best_model.py & pause & exit /b 1)
echo.

echo [8/16] Supplementary S2: Pandemic Impact
python "%SUPP%\gen_supp_pandemic_impact.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_supp_pandemic_impact.py & pause & exit /b 1)
echo.

echo [9/16] Supplementary S3: Permutation Test
python "%SUPP%\gen_supp_permutation.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_supp_permutation.py & pause & exit /b 1)
echo.

echo [10/16] Supplementary S5: Early Warning Extended
python "%SUPP%\gen_supp_early_warning.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_supp_early_warning.py & pause & exit /b 1)
echo.

echo [11/16] Supplementary S9B: Lead-Time Threshold
python "%SUPP%\gen_supp_leadtime.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_supp_leadtime.py & pause & exit /b 1)
echo.

echo [12/16] Supplementary S13: Model Extended
python "%SUPP%\gen_supp_model_extended.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_supp_model_extended.py & pause & exit /b 1)
echo.

echo [13/16] Supplementary S4: Dimension Ablation
python "%SUPP%\gen_supp_ablation.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_supp_ablation.py & pause & exit /b 1)
echo.

echo [14/16] Supplementary S7: SCH RDI Workflow
python "%SUPP%\gen_supp_sch_rdi.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_supp_sch_rdi.py & pause & exit /b 1)
echo.

echo [15/16] Supplementary S8: Prospective Validation (20K)
python "%SUPP%\gen_supp_prospective_validation.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_supp_prospective_validation.py & pause & exit /b 1)
echo.

echo [16/16] DOCX Manuscript + Supplementary Materials
python "%FIG%\gen_paper_v2.py"
if %ERRORLEVEL% neq 0 (echo ERROR in gen_paper_v2.py & pause & exit /b 1)
echo.

echo =========================================
echo  ALL MODULES COMPLETE!
echo  Figures: figures_v2\merge\
echo  Paper:   figures_v2\merge\manuscript.docx
echo  Supp:    figures_v2\merge\supplementary_materials.docx
echo =========================================
pause
