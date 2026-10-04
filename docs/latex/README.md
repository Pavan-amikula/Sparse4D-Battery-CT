# Sparse4D IEEE LaTeX report

The report uses IEEEtran in conference mode, A4, and pdfLaTeX. The date is removed from the title block. Results and source references are preserved from the updated project report.

## Overleaf
1. Choose New Project, then Upload Project.
2. Upload the complete ZIP file.
3. Select main.tex as the main document and pdfLaTeX as compiler.
4. Click Recompile.

## Local compilation
Run `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice from this directory.

Edit main.tex for content and references.tex for the numbered bibliography. Figures are in figures/. The included unmodified IEEEtran.cls (version 1.8b) was downloaded from CTAN and retains its original license header. No BibTeX or Biber step is needed, because the bibliography uses explicit bibitems.

This is a project report, not a published IEEE paper. Check venue-specific requirements before submitting. The simulation results and measured-data fit metrics describe different experiments. The prototype does not demonstrate a live scanner connection or calibrated defect detection.
