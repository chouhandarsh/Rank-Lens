# Data folder

- `raw/` holds the clean 2017–2025 rankings master, the supplementary parameter workbook and the Tableau score file. Do not rename them unless you also update `scripts/build_data.py`.
- `processed/` holds standardized ranking and integrated parameter tables used by Streamlit.
- `final/` holds `nirf_final_dataset.csv`: the app dataset plus its calculated fields, for use outside the app.
- `reports/` holds quality and merge-audit outputs.

The main app reads `processed/nirf_analysis_dataset.csv`. Rebuild it and the final file after replacing a raw source file by running `python scripts/build_data.py` from the project root.
Master rows link to the Tableau file exactly by year, category and original Institute ID, using its `Original Category` column; the build stops if a row is missing or its score or rank disagrees. All TLR, RPC, GO, OI, PR, SFC, QNR and QLR values come from the Tableau file. The workbook still supplies `institution_type` and the legacy audit, joined to Overall rows exactly by `(Year, Institute ID)` and `(Year, Institute_ID_Original)`; its parameter values are no longer used.
Always join, group, and filter institutions by `Institution_ID`, never by name. Rows from Tableau-only categories get their ID through `Institution UID`; see `reports/tableau_merge_audit.csv`.
