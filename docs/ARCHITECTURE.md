# Architecture

```text
User
  │
  ▼
Streamlit UI (app.py)
  ├── Overview
  ├── Level 1: question dropdown (40 supplied questions)
  ├── Level 2: custom analysis controls
  └── Chatbot: natural-language NIRF queries, results, and charts
          │
          ▼
     Shared global filters (nirf/filters.py)
          │
          ├───────────────┐
          ▼               ▼
     Level 1 engine    Level 2 engine
     level1.py         custom_analysis.py
          │               │
          └───────┬───────┘
                  ▼
             Plotly charts
                  │
                  ▼
           Charts + metrics + tables

Chatbot data path:

Streamlit chat UI ──► validated SQL (DATABASE_URL) ──► public."NIRF" database table
        │
        ├── result table and interactive chart
        └── generated plotting code (displayed, never executed)

Data preparation path:

Clean 2017–2025 ranking master ─┐
Overall parameter workbook ─────┼─ scripts/build_data.py
Tableau score file ─────────────┘          │
                                          ▼
                            data/processed/nirf_analysis_dataset.csv
                            data/final/nirf_final_dataset.csv
                                          │
                                          ▼
                                  nirf/data_loader.py
                                          │
                                          ▼
                                    Streamlit UI
```

## Design choices

1. **One analytical fact table for the app.** `nirf_analysis_dataset.csv` contains every category-level record from `nirf_rankings_2017_2025_clean.csv`, plus rows from six categories found only in `nirf_tableau_ready_cleaned_v2.csv`. Its compatibility columns (`year`, `category`, `institution`, `state`, `score`, and `rank_numeric`) are mapped from the master where a master row exists. Parameter scores come from the Tableau file, and the build stops if that file disagrees with a master score or rank. Parameter analyses use one category at a time because an institution has one row per category and weights can differ. Status fields preserve withdrawn and unscored records; null scores are excluded from score analyses.
2. **Data preparation is separate from the UI.** The build script creates standardized datasets, coverage tables and audit reports. The Streamlit app reads the prepared CSV and does not run the workbook merge on every page refresh.
3. **Level 1 and Level 2 are separate engines.** `level1.py` contains handlers for the 40 supplied questions. `custom_analysis.py` exposes generic analysis operations for combinations not covered by those questions.
4. **Institution identity is canonical.** Always join, filter, or group institutions by `Institution_ID` (exposed as `institution_id` and the compatibility alias `institution_key`). Never use institution names as keys; distinct IDs may share a name. Rows from Tableau-only categories get their ID through the Tableau `Institution UID`, and each assignment is recorded in `data/reports/tableau_merge_audit.csv`.
5. **Global filters are shared.** Year, category, state, institution ID and rank selections feed both analysis engines. Historical movement pages bypass the rank-range restriction so a selected rank window does not hide an institution's prior or later rank.
6. **The UI does not accept file uploads.** Data sources are fixed in `data/raw/`; processed tables are fixed in `data/processed/`.
7. **Chatbot integration.** The copied chatbot is available from the dashboard navigation and its standalone Streamlit entry point. Both launch paths share one chat/results UI. Generated plot code is shown for review and is never executed by the app.
