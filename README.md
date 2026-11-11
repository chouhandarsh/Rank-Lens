<div align="center">

<img src="chatbot/assets/nirf_logo.png" alt="RankLens logo" width="110"/>

# RankLens

### See India's higher-education rankings clearly

**An interactive analytics dashboard and AI chatbot for NIRF rankings from 2017 to 2025**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75?logo=plotly&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase-4169E1?logo=postgresql&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Sarvam%20AI-1C3C3C)

</div>

---

**RankLens** turns nine years of India's **National Institutional Ranking Framework (NIRF)** data into something you can explore, question, and map. It has three parts:

1. **A guided analytics dashboard.** It answers 40 research questions with charts, metrics, and the tables behind them.
2. **A custom analysis explorer.** You pick your own combination of filters, variables, and chart types.
3. **An AI chatbot.** Ask a question in plain English. It writes the SQL, runs it on PostgreSQL, and returns an answer, a table, and an interactive chart.

The dataset covers **6,980 ranking records**, **1,082 institutions**, **18 NIRF categories**, and **33 states and union territories**.

---

## Table of contents

- [Screenshots](#screenshots)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Configuration (`.env`)](#configuration-env)
- [Running the app](#running-the-app)
- [Rebuilding the data](#rebuilding-the-data)
- [Running the tests](#running-the-tests)
- [How it works](#how-it-works)
- [Data sources and integration](#data-sources-and-integration)
- [Data quality notes](#data-quality-notes)
- [Security notes](#security-notes)
- [Further documentation](#further-documentation)
- [Acknowledgements and licences](#acknowledgements-and-licences)

---

## Screenshots

| Level 1: guided questions | Geographic view |
|---|---|
| ![Level 1](docs/screenshots/level1-questions.png) | ![Geographic view](docs/screenshots/geographic-view.png) |
| **Multivariate analysis** | **State gainers and losers** |
| ![Multivariate](docs/screenshots/multivariate-analysis.png) | ![Gainers & losers](docs/screenshots/state-gainers-losers.png) |
| **AI chatbot: results** | **AI chatbot: map chart** |
| ![Chatbot results](docs/screenshots/chatbot-results.png) | ![Chatbot map](docs/screenshots/chatbot-map.png) |

---

## Features

### Overview
- KPI cards: records, institutions, categories, years, and states.
- Score trend and category breakdown charts.
- **Geographic view:** an India choropleth with year, category, and top-N controls.

### Level 1: 40 sample questions
One dropdown holds 40 analytical questions in 8 themes, 5 questions each:

| # | Theme |
|---|---|
| 1 | Time-series and trend analysis |
| 2 | Geospatial and regional analysis |
| 3 | Univariate analysis |
| 4 | Bivariate analysis and correlation |
| 5 | Multivariate analysis |
| 6 | Distribution and statistical comparison |
| 7 | Ranking transitions and institutional mobility |
| 8 | Outlier detection and performance gaps |

Each question runs its own calculation. You get a written answer, summary metrics, a Plotly chart, and the data table behind it.

### Level 2: custom analysis explorer
Eleven reusable analysis modes cover combinations the 40 questions don't:

`Compare groups` · `Trend over time` · `Forecast / prediction` · `India map` · `Distribution` · `Relationship / scatter` · `Correlation matrix` · `Top / bottom institutions` · `Institution parameter profile` · `Rank-band transitions` · `Summary statistics`

Maps can show a single year, two years side by side, the change between two years, or an animation across years.

### Chatbot: AI analysis
- Ask a question in plain English, for example *"Which states have improved their representation in the top 100 over time?"*
- The **Sarvam AI `sarvam-105b`** model, called through LangChain, writes the SQL. The SQL is validated with **SQLGlot** and runs against PostgreSQL.
- Each answer has four tabs: **Results**, **Chart**, **Generated SQL**, and **Plot code**.
- You can change the chart type, axes, and grouping. State-level results can be drawn on the India map.
- Failed queries get up to **two automatic repair attempts**.
- Generated plotting code is shown and can be downloaded, but it is **never executed**.

### Global filters
Year range, NIRF category, state, institution, and rank range. All pages share these filters.

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit (custom dark theme) |
| Analysis | pandas, NumPy |
| Charts and maps | Plotly, bundled OpenStreetMap GeoJSON (works offline) |
| Data pipeline | pandas, openpyxl |
| LLM | LangChain + `langchain-sarvam` (`sarvam-105b`) |
| Database | PostgreSQL (Supabase) via SQLAlchemy + psycopg 3 |
| SQL safety | SQLGlot parsing and normalisation |
| Configuration | python-dotenv, pydantic-settings |
| Optional API | FastAPI + Uvicorn |

---

## Project structure

```text
RankLens/
├── app.py                       # Streamlit entry point: navigation, filters, all pages
├── requirements.txt             # Python dependencies
├── .env.example                 # Template for the chatbot's secrets (copy to .env)
├── .streamlit/
│   └── config.toml              # Dark theme and server settings
│
├── nirf/                        # Dashboard analytics package
│   ├── data_loader.py           # Loads and validates the processed dataset
│   ├── filters.py               # Global filters and parameter-row selection
│   ├── schema.py                # Shared field and parameter definitions
│   ├── questions.py             # The 40 Level 1 question definitions
│   ├── level1.py                # Handlers for the 40 questions
│   ├── custom_analysis.py       # Level 2 analysis engine (11 modes)
│   ├── custom_ui.py             # Level 2 controls
│   ├── maps.py                  # India choropleth, state-name mapping, attribution
│   ├── score_chart.py           # Institution score chart data (never sums scores)
│   └── theme.py                 # Colour palette and CSS
│
├── chatbot/                     # AI chatbot
│   ├── streamlit_app.py         # Standalone chatbot app
│   ├── assets/nirf_logo.png
│   └── src/nirf_chatbot/
│       ├── chat_service.py      # Orchestrates: shortcuts, LLM SQL, repair, answer
│       ├── sql_engine.py        # Schema-aware SQL prompt and few-shot examples
│       ├── sql_validator.py     # One statement only, adds LIMIT, normalisation
│       ├── sql_executor.py      # Runs SQL (60 s timeout, 1,000-row cap)
│       ├── answer_engine.py     # Turns result rows into a natural-language answer
│       ├── visualization.py     # Chart spec inference/validation and Plotly figures
│       ├── streamlit_ui.py      # Chat UI: Results / Chart / SQL / Plot code tabs
│       ├── llm.py, Settings.py  # Sarvam model and API-key loading
│       ├── database.py          # SQLAlchemy engine from DATABASE_URL
│       ├── prompts.py           # Answer-generation prompt
│       ├── main.py, routes.py   # Optional FastAPI server (POST /api/chat)
│       └── test_sql.py          # CLI: run one SQL query against the database
│
├── scripts/
│   ├── build_data.py            # Rebuilds data/processed, final, reports from raw
│   └── create_readonly_role.sql # Creates a read-only Postgres role for the chatbot
│
├── data/
│   ├── raw/                     # The three source files (inputs to the build)
│   ├── processed/               # App-ready tables, incl. nirf_analysis_dataset.csv
│   ├── final/                   # nirf_final_dataset.csv (+ calculated fields)
│   ├── reports/                 # Data-quality and merge-audit reports
│   └── geo/                     # india_states_osm.geojson (ODbL)
│
├── tests/                       # Data, analysis, map, chatbot and UI tests
│
└── docs/
    ├── ARCHITECTURE.md          # System diagram and design choices
    ├── TECHNICAL_OVERVIEW.md    # Detailed technical reference
    ├── DATASET_INCONSISTENCY_REPORT.pdf
    └── screenshots/
```

---

## Getting started

### Prerequisites
- **Python 3.11 or newer**
- Git
- *(Chatbot only)* a PostgreSQL database with a `public."NIRF"` table and a [Sarvam AI](https://www.sarvam.ai/) API key

### 1. Clone the repository

```bash
git clone https://github.com/chouhandarsh/NIRF-Analytics.git
cd NIRF-Analytics
```

### 2. Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure secrets (chatbot only)

```bash
cp .env.example .env              # Windows: Copy-Item .env.example .env
```

Then open `.env` and fill in your values. See the next section for details.

> The **Overview, Level 1, and Level 2** pages run fully offline on the bundled CSVs and **do not need a `.env`**.

---

## Configuration (`.env`)

The chatbot reads these settings from `.env` in the project root, or from the process environment.

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Chatbot only | SQLAlchemy URL of the database that holds `public."NIRF"`, for example `postgresql+psycopg://user:pass@host:5432/postgres` |
| `SARVAM` | Chatbot only | Sarvam AI API key for the `sarvam-105b` model |
| `CHATBOT_CORS_ORIGINS` | Optional | Extra comma-separated CORS origins for the FastAPI server |
| `PORT` | Optional | FastAPI server port (default `8000`) |
| `NIRF_RUN_DB_TESTS` | Optional | Set to `1` to run the live-database integration tests |

`.env` is listed in `.gitignore`. **Never commit real credentials.**

---

## Running the app

### Full dashboard (all four pages)

```bash
python -m streamlit run app.py
```

Open <http://localhost:8501>.

### Standalone chatbot

```bash
python -m streamlit run chatbot/streamlit_app.py
```

### Optional REST API

```bash
cd chatbot/src
python -m uvicorn nirf_chatbot.main:app --reload --port 8000
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/db_health` | Database connectivity check |
| `POST` | `/api/chat` | Body: `{"question": "..."}`. Returns the answer, the generated SQL, and the result rows |

### Test one SQL query against the database

```bash
cd chatbot/src
python -m nirf_chatbot.test_sql "SELECT \"Year\", \"Institution_Name\" FROM public.\"NIRF\" LIMIT 5"
```

---

## Rebuilding the data

The processed datasets are already in the repository, so you don't need to rebuild anything to run the app. If you replace a file in `data/raw/`, regenerate everything with:

```bash
python scripts/build_data.py
```

This rewrites `data/processed/`, `data/final/`, and `data/reports/`. Restart Streamlit afterwards so the cached data reloads.

The build stops with an error if:
- a master row has no matching row in the Tableau score file,
- a score or rank disagrees between sources,
- a merge creates a duplicate `(year, category, institution_id)` key, or
- Overall parameter scores, weighted 30/30/20/10/10, miss the published score by more than 0.05.

---

## Running the tests

Each test file is a standalone `unittest` script. Run them from the project root:

```bash
python tests/smoke_test.py            # All 40 Level 1 questions + Level 2 modes
python tests/master_data_test.py      # Dataset invariants and build checks
python tests/analysis_test.py         # Analysis calculations
python tests/maps_test.py             # GeoJSON matching and choropleths
python tests/app_pages_test.py        # Streamlit page rendering (AppTest)
python tests/chatbot_sql_test.py      # SQL validation, normalisation, chat service
python tests/chat_chart_test.py       # Chatbot chart specs and figures
python tests/chat_ui_test.py          # Chatbot UI parsing and rendering
```

The live-database integration tests are opt-in and only run `SELECT` queries:

```bash
# PowerShell:  $env:NIRF_RUN_DB_TESTS="1"; python tests/chatbot_db_integration_test.py
NIRF_RUN_DB_TESTS=1 python tests/chatbot_db_integration_test.py
```

---

## How it works

```text
                         ┌───────────────────────────── Streamlit UI (app.py) ─────────────────────────────┐
                         │   Overview      Level 1 (40 Qs)      Level 2 (custom)      Chatbot (AI)         │
                         └──────┬──────────────┬───────────────────┬───────────────────────┬───────────────┘
                                │              │                   │                       │
                                ▼              ▼                   ▼                       ▼
                        shared global filters (nirf/filters.py)              chat_service.py
                                │              │                   │                       │
                                ▼              ▼                   ▼            Sarvam LLM → SQL
                         nirf/level1.py   nirf/custom_analysis.py                          │
                                │              │                                SQLGlot validation
                                └──────┬───────┘                                           │
                                       ▼                                    PostgreSQL public."NIRF"
                        data/processed/nirf_analysis_dataset.csv                           │
                                       ▲                                 Results · Chart · SQL · Code
                                       │
             scripts/build_data.py  ◄──┴──  data/raw/ (3 source files)
```

- **Dashboard path:** works fully offline. It reads the processed CSV, applies the shared filters, and builds Plotly charts locally.
- **Chatbot path:** uses the live PostgreSQL table, so it is separate from the CSVs. Rebuilding the CSVs does **not** update the database.
- **Identity rule:** institutions are always joined and grouped by `Institution_ID`, never by name, because different institutions can share a name.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/TECHNICAL_OVERVIEW.md`](docs/TECHNICAL_OVERVIEW.md) for the full details.

---

## Data sources and integration

| Source file (`data/raw/`) | Rows | Role |
|---|---|---|
| `nirf_rankings_2017_2025_clean.csv` | 6,734 | **Ranking master.** Supplies canonical `Institution_ID`, names, states, scores, ranks, and status fields for 12 categories |
| `nirf_tableau_ready_cleaned_v2.csv` | 6,980 | **Parameter scores** (TLR, RPC, GO, OI, PR, plus SFC, QNR, QLR for Research). Adds 246 rows from six categories missing from the master |
| `NIRF_master_2017_2025.csv.xlsx` | 900 | Legacy Overall workbook. Now used only for `institution_type` and the audit trail |

**Derived outputs:**

| File | Purpose |
|---|---|
| `data/processed/nirf_analysis_dataset.csv` | **The main table the app reads:** rankings, parameters, and source fields |
| `data/processed/fact_rankings.csv` | Standardised ranking records |
| `data/processed/fact_parameters.csv` | Rows with parameter scores |
| `data/processed/parameter_match_audit.csv` | Match and quality flags for all 900 workbook rows |
| `data/processed/category_year_coverage.csv` | Record coverage by year and category |
| `data/processed/data_dictionary.csv` | Field-level data dictionary |
| `data/final/nirf_final_dataset.csv` | Main table plus `score_change`, `rank_improvement`, and `parameter_gap` |
| `data/reports/data_quality_report.csv` | Data-quality summary |
| `data/reports/parameter_merge_issues.csv` | Unmatched rows and score discrepancies |
| `data/reports/tableau_merge_audit.csv` | How each Tableau-only row got its `Institution_ID` |

---

## Data quality notes

- There are **no duplicate** `(year, category, institution_id)` keys, and each ID maps to one canonical name.
- **57 null scores** are kept: rank-only listings, 2018 Special Mentions, and one withdrawn record. Score analyses drop them.
- **Parameter analyses use one category at a time:** Overall when it is selected, otherwise the first selected category in alphabetical order. This is because parameter weights differ between categories.
- **Rank bands:** `Top 10`, `11–50`, `51–100`, `101+`, and `Unranked / not numeric`.
- Overall has 100 records per year, except 2020, which has 101 because of a tie.
- Compare ranks **only within the same category**. Rank 5 in Engineering is not comparable with rank 5 in Overall.
- *"First observed year"* means the first year an institution appears in this 2017–2025 dataset. It is not the year the institution was founded.
- State boundaries come from community-mapped **OpenStreetMap** data. They are **not** an official Survey of India source.

---

## Security notes

> **Read this before you deploy the chatbot publicly.**

- The chatbot runs SQL written by the model, using the privileges of `DATABASE_URL`. The validator allows **only one statement** per request and adds a row limit, but it is **not** a permission boundary.
- **Always connect with a read-only role.** Run [`scripts/create_readonly_role.sql`](scripts/create_readonly_role.sql) once as an admin, replace the password placeholder first, and use that role in `DATABASE_URL`.
- Plot code from the model is **displayed only, never executed**.
- Keep `.env` out of version control. It is already in `.gitignore`.

---

## Further documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): system diagram and design decisions
- [`docs/TECHNICAL_OVERVIEW.md`](docs/TECHNICAL_OVERVIEW.md): module-by-module reference, request flows, and caveats
- [`docs/DATASET_INCONSISTENCY_REPORT.pdf`](docs/DATASET_INCONSISTENCY_REPORT.pdf): inconsistencies found in the source data
- [`data/README.md`](data/README.md) and [`data/geo/README.md`](data/geo/README.md): data folder and map data notes

---

## Acknowledgements and licences

- Ranking data comes from the **National Institutional Ranking Framework (NIRF)**, Ministry of Education, Government of India ([nirfindia.org](https://www.nirfindia.org/)).
- Map boundaries: © OpenStreetMap contributors, available under the [Open Database License (ODbL) 1.0](https://www.openstreetmap.org/copyright).
- The LLM is provided by [Sarvam AI](https://www.sarvam.ai/).

<div align="center">

Made by <a href="https://github.com/chouhandarsh">chouhandarsh</a>

</div>
