# NIRF Analytics: Technical Overview

This document describes the current dashboard, data preparation pipeline,
geographic visualizations, and chatbot. It distinguishes local dashboard data
from the chatbot's live database so their different sources of truth are clear.

## 1. System at a glance

| Area | Main technologies | Data source |
|---|---|---|
| Analytics dashboard | Python, Streamlit, pandas, NumPy, Plotly | Local processed CSVs in `data/processed/` |
| Data preparation | Python, pandas, openpyxl | Clean NIRF master CSV and supplementary XLSX |
| State maps | Plotly choropleth, bundled GeoJSON, Streamlit resource cache | Local data plus `data/geo/india_states_osm.geojson` |
| Chatbot interface | Streamlit | User question, chatbot service |
| Chatbot SQL generation | LangChain and `ChatSarvam` (`sarvam-105b`) | PostgreSQL schema prompt and user question |
| Chatbot database access | SQLAlchemy, psycopg, PostgreSQL/Supabase | `public."NIRF"` in the configured database |
| SQL safety | SQLGlot plus application validation | Generated SQL before execution |

The dashboard and chatbot are separate data paths. The dashboard reads its
packaged CSV and workbook-derived outputs; the chatbot runs SQL
against the configured PostgreSQL database. Rebuilding the local CSVs does not
update the Supabase table, and changes to Supabase do not update the local
dashboard files.

## 2. Repository map

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit entry point, page navigation, global sidebar filters, Overview, Level 1, Level 2, and chatbot page integration |
| `nirf/data_loader.py` | Loads and validates the processed analysis dataframe; creates its dataset summary |
| `nirf/filters.py` | Builds state/institution options and applies year, category, state, institution, and rank filters; selects parameter rows one category at a time |
| `nirf/schema.py` | Shared field and analysis configuration |
| `nirf/questions.py` | Level 1 question definitions, labels, and question-specific control metadata |
| `nirf/level1.py` | Calculations and result data for the 40 Level 1 sample questions |
| `nirf/custom_analysis.py` | Reusable Level 2 group, trend, distribution, relationship, correlation, profile, transition, and summary calculations |
| `nirf/score_chart.py` | Prepares institution score-chart data without summing scores across years |
| `nirf/maps.py` | Shared state aliases, map metrics, offline GeoJSON matching, choropleth rendering, and attribution |
| `scripts/build_data.py` | Rebuilds processed ranking, parameter, audit, and coverage files |
| `chatbot/src/nirf_chatbot/chat_service.py` | Deterministic question handlers, LLM fallback, query execution coordination, and grounded answers |
| `chatbot/src/nirf_chatbot/sql_engine.py` | Database schema discovery, SQL-generation model, system/schema instructions, and few-shot guidance |
| `chatbot/src/nirf_chatbot/sql_validator.py` | Validates and normalizes generated SQL before it can reach the database |
| `chatbot/src/nirf_chatbot/sql_executor.py` | Opens the configured database connection and executes validated queries |
| `chatbot/src/nirf_chatbot/visualization.py` | Chart-spec inference/validation and Plotly chart construction for chatbot results |
| `chatbot/src/nirf_chatbot/streamlit_ui.py` | Chat history, result tabs, charts, generated SQL, plot-code display, and interaction |
| `chatbot/streamlit_app.py` | Standalone chatbot Streamlit entry point |
| `tests/` | Data invariants, dashboard smoke tests, maps, chatbot SQL behavior, charts, and UI tests |

## 3. Data and local preparation

### Inputs

The local build uses:

- `data/raw/nirf_rankings_2017_2025_clean.csv` — the clean rankings master.
- `data/raw/nirf_tableau_ready_cleaned_v2.csv` — all parameter scores, plus six
  categories absent from the master.
- `data/raw/NIRF_master_2017_2025.csv.xlsx` — legacy Overall workbook, now used
  only for institution type and audit fields.

The build keeps the master's records and adds 246 Tableau-only rows, giving
6,980 records, 18 categories and 1,082 institution IDs.

The master contains 6,734 records over 2017–2025, 12 categories, 1,045
institution IDs, and 33 canonical state/UT names. Its key is
`(Year, Category, Institution_ID)`. One institution ID maps to one canonical
name, but different IDs can share a name; therefore identity joins and
grouping use `Institution_ID`, never name alone.

The dataset retains null-score and withdrawn records with status fields.
There are 57 null scores (52 rank-only listings, four 2018 Special Mentions and
a withdrawn record), and two rows with a withdrawn rank status. Score analyses exclude
null scores. Overall has 100 records per year, except 2020 with 101 because of
a tie.

### Build sequence

Run `python scripts/build_data.py` from the repository root to regenerate:

1. `fact_rankings.csv` from the clean master plus the Tableau-only rows, mapped
   to the established analysis schema.
2. `fact_parameters.csv`, holding every row with parameter scores. Master rows
   link to the Tableau file by exact year, category and original Institute ID.
   The join is validated as one-to-one; fuzzy name matching is not used.
3. `parameter_match_audit.csv`, which preserves match status and score/rank
   discrepancy flags for the legacy workbook rows.
4. `nirf_analysis_dataset.csv`, the app table with parameter scores and source
   fields, and `data/final/nirf_final_dataset.csv`, which adds calculated fields.
5. Category/year coverage and data-quality reports under `data/processed/`
   and `data/reports/`, including `tableau_merge_audit.csv`.

The build stops if a master row has no Tableau row, if a score or rank
disagrees, if a merge creates a duplicate key, or if Overall parameters weighted
30/30/20/10/10 miss the published score by more than 0.05 points.

Of the 900 legacy workbook rows, 894 match the master and six are retained as
unmatched in the audit. Ninety-four matched rows, all from 2020, conflict with
the master, and their parameter values fail the weighted-score check. This is why
the Tableau file replaced the workbook as the parameter source. No rows currently
need review, so the sidebar control was removed.

Parameter analyses use one category at a time: Overall when selected, otherwise
the first selected category alphabetically. Research rows have SFC, QNR, QLR, OI
and PR, so `parameters_available` is False for them.

The produced analysis schema adapts master fields to names used by existing
dashboard analysis code (for example `year`, `category`, `institution`,
`institution_key`, `state`, `city`, `score`, and `rank_num`). Rank bands are
derived from numeric ranks. Refer to `data_dictionary.csv` and
`scripts/build_data.py` for the maintained field-level contract.

### Dataset facts currently verified for the local data

- Years: 2017–2025 (nine years).
- Categories: Overall, Engineering, University, College, Management,
  Pharmacy, Medical, Law, Architecture, Dental, Research, Innovation,
  Agriculture & Allied Sectors, State Public University, Open University,
  Skill University, Sustainable Development Goals (SDG), and Special Mention (2018).
- States/UTs: 33 canonical display names, including Puducherry.
- Parameter coverage: 6,673 rows with the five standard parameters and 250
  Research rows with SFC, QNR, QLR, OI and PR.
- An institution's first observed year means its first presence in this
  supplied 2017–2025 dataset, not its founding year or necessarily its first
  participation in NIRF.

## 4. Dashboard request and rendering flow

1. Streamlit starts at `app.py`; navigation selects Overview, Level 1 Sample
   Questions, Level 2 Custom Analysis, or Chatbot AI Analysis.
2. The cached data loader reads `data/processed/nirf_analysis_dataset.csv`.
3. The sidebar creates shared year, category, state, institution, and rank
   selections. The filter layer applies them to the analysis
   dataframe. Institution options retain IDs as identity and names as labels.
4. Overview renders its existing KPI cards, score trend, and category chart,
   followed by the Geographic view. The Geographic view derives map values
   from the filtered data and includes year, category, and top-N controls
   where applicable.
5. Level 1 selects one of the defined questions, calls `run_level1`, and
   renders the returned metrics, tables, and charts. Its geospatial questions
   add map views below the existing answer; other questions only expose a map
   selector where a state-level answer is meaningful.
6. Level 2 collects an analysis mode and options, then calls
   `run_custom_analysis` to generate its result table, metrics, and Plotly
   figure.

The dashboard's local analyses use pandas/NumPy dataframes and Plotly; they do
not send these calculations to the chatbot LLM.

### Score chart semantics

`nirf/score_chart.py` drops null scores and rejects values outside 0–100. With
one selected year it selects that year's score. With multiple years it groups
by `(institution_key, category)` and uses the mean score, recording the number
of years represented. It never sums scores. Institution IDs are the grouping
identity; when IDs share a display name, the city is appended to the label.

### Shared map flow

`nirf/maps.py` loads the bundled GeoJSON through Streamlit's resource cache,
maps canonical data state names to the GeoJSON `properties.name`, and reports
unmatched data names or boundary features. State metrics include top-N
institution counts/shares, average score, best rank, newly observed
institutions, and differences between two years' counts. Scores are averaged,
not summed. Empty or fully filtered map input produces an informational
message instead of an empty plot.

The shared map file is `data/geo/india_states_osm.geojson`: it has 37 named
features and the `properties.name` field. The canonical data states are
matched through the explicit mapping in `nirf/maps.py`; four geographic
features do not correspond to data states and appear as no-data areas.
`data/geo/README.md` documents the source, license, and caveat. This is
community-mapped OpenStreetMap boundary data under ODbL 1.0, **not an
official Survey of India boundary source**. No boundary or geocoding data is
downloaded at runtime. There is no city-coordinate file, so city bubble maps
are not available.

## 5. Chatbot request and execution flow

1. The dashboard chatbot page calls the UI in
   `chatbot/src/nirf_chatbot/streamlit_ui.py`. The standalone entry point is
   `chatbot/streamlit_app.py`.
2. The UI sends the current question and relevant conversation context to
   `chat_service.process_question`.
3. Recognized common question patterns can use deterministic handlers. The
   dataset-overview handler directly queries aggregate facts (such as row and
   year counts) and builds its answer from those results, bypassing SQL
   generation. Other known handlers cover certain score, institution, and
   state questions.
4. Questions not handled deterministically use the SQL generation path in
   `sql_engine.py`, which uses LangChain's `ChatSarvam` model
   (`sarvam-105b`) with the configured database schema, rules, and examples.
5. `sql_validator.py` strips comments and one trailing semicolon, then rejects
   empty output or multiple statements. It does not decide whether a query can
   write. `normalize_sql()` adds `LIMIT 1000` when no limit is present, while
   preserving valid CTE, join, aggregate, and window query forms.
6. `sql_executor.py` uses SQLAlchemy and psycopg with `DATABASE_URL`, and sets
   a 60-second statement timeout. It fetches at most 1,000 rows. Failed queries
   get up to two model repair attempts. No read-only role or transaction is
   applied, so generated SQL runs with that connection's full privileges; the
   chatbot should not be exposed to untrusted users with an admin connection.
7. The service grounds answer text in returned data where a deterministic
   answer path is available. The UI exposes Results, Chart, Generated SQL,
   and Plot code tabs. Visualization infers and validates a chart spec from
   the result columns/question, then generates Plotly code from that validated
   spec; it does not execute free-form model-authored plotting code.
8. For state metrics, the chatbot visualization uses the shared map renderer
   in `nirf/maps.py`, so it uses the same local GeoJSON, state-name mapping,
   and attribution as dashboard maps.

The chatbot queries `public."NIRF"` in the configured database. It does not
read `data/raw/nirf_rankings_2017_2025_clean.csv` or
`data/processed/nirf_analysis_dataset.csv`. Local data facts in this report
describe the checked-in master/build, not a live verification of the deployed
Supabase table.

### Configuration and secrets

The chatbot reads `DATABASE_URL` and `SARVAM` from the process environment
or the root `.env` file. `DATABASE_URL` configures the PostgreSQL connection;
`SARVAM` supplies the Sarvam API credential. Never commit `.env` or paste
credential values into reports, logs, or issue descriptions. The
dashboard's local-data pages do not require these chatbot credentials;
chatbot requests require network access to both the configured database and
the model service.

## 6. Dependencies and runtime

The root `requirements.txt` lists the direct dependencies of the combined
dashboard and chatbot. The project requires Python 3.11 or newer. Principal libraries are:

- Streamlit for the web application and interactive controls.
- pandas and NumPy for tabular data and calculations.
- Plotly for charts, animation, and choropleths.
- SQLAlchemy and psycopg for PostgreSQL access.
- LangChain and `langchain-sarvam` for model integration.
- SQLGlot for SQL parsing/validation support.
- python-dotenv and pydantic-settings for configuration.
- openpyxl for the supplementary Excel workbook.

The dashboard can run offline against the supplied processed data. Chatbot
model and database requests require connectivity. Maps are offline with
respect to geographic boundaries: the GeoJSON is bundled locally.

## 7. Running and rebuilding

From the project root on Windows PowerShell:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For chatbot features, configure `DATABASE_URL` and `SARVAM` locally before
launching. The chatbot can also be started independently:

```powershell
python -m streamlit run chatbot/streamlit_app.py
```

To rebuild dashboard data after updating the source files:

```powershell
python scripts/build_data.py
```

The build reads the named master CSV, Tableau file and legacy workbook, then
regenerates the processed tables, final dataset and audit reports. It does not migrate or synchronize
Supabase. Restart Streamlit after rebuilding so the cached dataframe reloads.

## 8. Tests and verification

Tests under `tests/` cover master-data invariants, dashboard calculations,
maps, chatbot SQL, chart specification, and chatbot UI. Useful focused
commands from the project root include:

```powershell
python tests/master_data_test.py
python tests/maps_test.py
python tests/chatbot_sql_test.py
python tests/chat_chart_test.py
python tests/chat_ui_test.py
python tests/smoke_test.py
```

The test command names above reflect the test files present in this
repository; some chatbot tests exercise a real database path and therefore
need valid network/database configuration. In the most recent combined
chatbot test run, the chart tests (5) and UI tests (5) passed. One SQL test
(`test_least_answer_includes_states_with_zero_top_ranked_institutions`) could
not complete because the configured Supabase hostname failed DNS resolution
(`getaddrinfo failed`). This is an environment/connectivity failure, not
evidence that the live query passed. Earlier map tests passed (7), the Level 1
smoke set passed (40), the Level 2 smoke set passed (10), and the dashboard
returned HTTP 200. The full set of Level 1 map dropdown combinations has not
been exhaustively UI-tested.

## 9. Operational boundaries and caveats

- The local dashboard and chatbot database can temporarily contain different
  data. Confirm the relevant source before comparing an answer to a dashboard
  value.
- Research rows have no TLR, RPC or GO, and rank-only categories have no
  parameters. Do not substitute a category score for a parameter, and do not
  compare parameter values across categories, because weights can differ.
- The chatbot's database does not contain the merged scores or the six added
  categories until `public."NIRF"` is updated separately.
- Rank values are category-specific and are not directly comparable between
  NIRF categories.
- “Top N” and state representation depend on the selected category and year.
- The supplied master covers the categories listed above, not necessarily
  every category NIRF has published in every year.
- State boundaries are a visualization aid sourced from community-mapped OSM
  data; they are not an authoritative or legal administrative reference.
- The chatbot's active path is direct SQL against PostgreSQL; there is no
  RAG or document-ingestion step.
- The app displays generated SQL and chart code for transparency. SQL is
  validated before execution; chart rendering is built from a validated
  specification rather than executing arbitrary model code.
