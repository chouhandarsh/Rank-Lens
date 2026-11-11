from __future__ import annotations
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
REPORTS = ROOT / 'data' / 'reports'
FINAL = ROOT / 'data' / 'final'
RANKINGS_FILE = RAW / 'nirf_rankings_2017_2025_clean.csv'
PARAMETERS_FILE = RAW / 'NIRF_master_2017_2025.csv.xlsx'
TABLEAU_FILE = RAW / 'nirf_tableau_ready_cleaned_v2.csv'

STANDARD_PARAMETERS = {
    'TLR Score': 'tlr', 'RPC Score': 'rpc', 'GO Score': 'go',
    'OI Score': 'oi', 'Perception Score': 'pr',
}
RESEARCH_PARAMETERS = {
    'Research SFC Score': 'research_sfc',
    'Research QNR Score': 'research_qnr',
    'Research QLR Score': 'research_qlr',
}
TABLEAU_TEXT_FIELDS = {
    'Parameter Structure': 'parameter_structure',
    'Data Quality Flags': 'data_quality_flags',
    'PDF URL': 'pdf_url',
    'Source URL': 'source_url',
}
# NIRF's published Overall weights; used only to check the parameter source.
OVERALL_WEIGHTS = {'tlr': .30, 'rpc': .30, 'go': .20, 'oi': .10, 'pr': .10}
MAX_OVERALL_RESIDUAL = 0.05
# Categories present only in the Tableau file, with their master-style group.
EXTRA_CATEGORY_GROUPS = {
    'Agriculture & Allied Sectors': 'Discipline',
    'State Public University': 'Institution type',
    'Open University': 'Institution type',
    'Skill University': 'Institution type',
    'Sustainable Development Goals (SDG)': 'Thematic',
    'Special Mention (2018)': 'Special recognition',
}

CANONICAL_STATES = {
    'Andaman and Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh',
    'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh', 'Delhi', 'Goa',
    'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir',
    'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
    'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry',
    'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
    'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
}


def rank_band(rank: float) -> str:
    if pd.isna(rank): return 'Unranked / not numeric'
    if rank <= 10: return 'Top 10'
    if rank <= 50: return '11–50'
    if rank <= 100: return '51–100'
    return '101+'


def load_rankings() -> pd.DataFrame:
    if not RANKINGS_FILE.exists():
        raise FileNotFoundError(f'Missing primary rankings file: {RANKINGS_FILE}')
    raw = pd.read_csv(RANKINGS_FILE, dtype=str, keep_default_na=False)
    required = [
        'Year', 'Category', 'Institution_ID', 'Institution_Name', 'City',
        'State', 'Score', 'Score_Status', 'Rank_Numeric', 'Rank_Status',
        'Rank_Original', 'Band',
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f'Clean master CSV is missing columns: {missing}')
    df = raw.rename(columns={
        'Year': 'year',
        'Category': 'category',
        'Institution_ID': 'institution_id',
        'Institution_Name': 'institution',
        'City': 'city',
        'State': 'state',
        'Score': 'score',
        'Score_Status': 'score_status',
        'Rank_Numeric': 'rank_numeric',
        'Rank_Status': 'rank_status',
        'Rank_Original': 'rank_label',
        'Rank_Suffix': 'rank_suffix',
        'Band': 'source_band',
        'Name_Standardized': 'name_standardized',
        'City_Imputed': 'city_imputed',
    }).copy()
    df['year'] = pd.to_numeric(df['year'], errors='raise').astype(int)
    df['score'] = pd.to_numeric(df['score'].replace('', np.nan), errors='coerce')
    df['rank_numeric'] = pd.to_numeric(df['rank_numeric'].replace('', np.nan), errors='coerce')
    df['rank_parse_method'] = np.select(
        [
            df['rank_status'].eq('Rank Withdrawn'),
            df['rank_numeric'].notna(),
        ],
        ['non_numeric', 'exact_numeric'],
        default='missing',
    )
    df['rank_band'] = df['rank_numeric'].apply(rank_band)
    df['institution_key'] = df['institution_id']
    df['state'] = df['state'].str.strip()
    invalid_states = sorted(set(df['state'].dropna()) - CANONICAL_STATES)
    if invalid_states:
        raise ValueError(f'Clean master contains non-canonical states: {invalid_states}')
    canonical_state_keys = [state.casefold() for state in CANONICAL_STATES]
    if len(canonical_state_keys) != len(set(canonical_state_keys)):
        raise ValueError('Canonical state list contains case-insensitive duplicates')
    df['score_raw'] = df['Score_Original']
    df['rank_source'] = df['rank_label']
    df['source_rankings'] = RANKINGS_FILE.name
    keep = [
        'year', 'category', 'institution_id', 'institution', 'institution_key',
        'city', 'city_imputed', 'state', 'rank_numeric', 'rank_label',
        'rank_suffix', 'rank_status', 'rank_parse_method', 'rank_band', 'score',
        'score_status', 'score_raw', 'rank_source', 'source_band',
        'name_standardized', 'source_rankings', 'Institute_ID_Original',
        'Canonical_ID_Category', 'Category_Group', 'Is_Core_Category',
    ]
    return df[keep]


def load_parameter_source(rankings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PARAMETERS_FILE.exists():
        raise FileNotFoundError(f'Missing supplementary parameter workbook: {PARAMETERS_FILE}')
    src = pd.read_excel(PARAMETERS_FILE, dtype={'Institute ID': str}, engine='openpyxl')
    required = ['Year','Institution','Institute ID','City','State','Rank','Score','TLR','RPC','GO','OI','PR']
    missing = [c for c in required if c not in src.columns]
    if missing:
        raise ValueError(f'Parameter workbook is missing columns: {missing}')
    p = src.rename(columns={
        'Year': 'year',
        'Institution': 'institution_legacy',
        'Institute ID': 'institute_id_original',
        'City': 'city_legacy',
        'State': 'state_legacy',
        'Institution Type': 'institution_type',
        'Rank': 'rank_legacy',
        'Score': 'score_legacy',
        'TLR': 'tlr',
        'RPC': 'rpc',
        'GO': 'go',
        'OI': 'oi',
        'PR': 'pr',
    }).copy()
    p['year'] = pd.to_numeric(p['year'], errors='raise').astype(int)
    p['institute_id_original'] = p['institute_id_original'].fillna('').astype(str)
    p['rank_legacy_numeric'] = pd.to_numeric(p['rank_legacy'], errors='coerce')
    for col in ['score_legacy','tlr','rpc','go','oi','pr']:
        p[col] = pd.to_numeric(p[col], errors='coerce')
    for col in ['institution_type','institution_legacy','city_legacy','state_legacy']:
        if col not in p.columns: p[col] = ''

    overall = rankings.loc[rankings['category'].eq('Overall'), [
        'year', 'Institute_ID_Original', 'institution_id', 'institution',
        'city', 'state', 'rank_numeric', 'score',
    ]].rename(columns={
        'Institute_ID_Original': 'institute_id_original',
        'institution_id': 'institution_id_primary',
        'institution': 'institution_primary',
        'city': 'city_primary',
        'state': 'state_primary',
        'rank_numeric': 'rank_primary',
        'score': 'score_primary',
    })
    audit = p.merge(
        overall,
        on=['year', 'institute_id_original'],
        how='left',
        validate='one_to_one',
    )
    audit['matched_to_primary'] = audit['institution_primary'].notna()
    audit['score_discrepancy'] = (
        audit['matched_to_primary'] & audit['score_primary'].notna() & audit['score_legacy'].notna()
        & ((audit['score_primary'] - audit['score_legacy']).abs() > 0.01)
    )
    audit['rank_discrepancy'] = (
        audit['matched_to_primary'] & audit['rank_primary'].notna() & audit['rank_legacy_numeric'].notna()
        & ((audit['rank_primary'] - audit['rank_legacy_numeric']).abs() > 0.01)
    )
    audit['parameter_review_required'] = audit['score_discrepancy'] | audit['rank_discrepancy']
    audit['parameter_quality_note'] = np.select(
        [~audit['matched_to_primary'], audit['score_discrepancy'] & audit['rank_discrepancy'],
         audit['score_discrepancy'], audit['rank_discrepancy']],
        ['No exact year + Institute ID match in primary Overall data',
         'Legacy overall rank and score differ from primary ranking CSV; parameter row should be reviewed',
         'Legacy overall score differs from primary ranking CSV; parameter row should be reviewed',
         'Legacy overall rank differs from primary ranking CSV; parameter row should be reviewed'],
        default='Matched by exact year + Institute ID'
    )
    audit['category'] = 'Overall'
    audit['institution_id'] = audit['institution_id_primary'].fillna(
        audit['institute_id_original']
    )
    audit['institution'] = audit['institution_primary'].fillna(audit['institution_legacy'])
    audit['city'] = audit['city_primary'].fillna(audit['city_legacy'])
    audit['state'] = audit['state_primary'].fillna(audit['state_legacy'])
    audit['institution_key'] = audit['institution_id']
    audit['source_parameters'] = 'NIRF_master_2017_2025.csv.xlsx'
    cols = [
        'year', 'category', 'institution_id', 'institute_id_original',
        'institution', 'institution_key', 'city', 'state', 'institution_type',
        'tlr', 'rpc', 'go', 'oi', 'pr', 'rank_legacy', 'rank_legacy_numeric',
        'score_legacy', 'score_primary', 'rank_primary', 'matched_to_primary',
        'score_discrepancy', 'rank_discrepancy', 'parameter_review_required',
        'parameter_quality_note', 'source_parameters',
    ]
    parameter_source = audit[cols].copy()
    matched = parameter_source.loc[parameter_source['matched_to_primary']].copy()
    return parameter_source, matched


def load_tableau() -> pd.DataFrame:
    if not TABLEAU_FILE.exists():
        raise FileNotFoundError(f'Missing Tableau score file: {TABLEAU_FILE}')
    tableau = pd.read_csv(TABLEAU_FILE, dtype=str, keep_default_na=False)
    required = [
        'Year', 'Category', 'Original Category', 'Institution UID', 'Institute ID',
        'Standardized Institution Name', 'Standardized Location', 'Is Location Imputed',
        'Standardized State', 'Composite Score', 'Analytical Rank', 'Rank Label',
        *STANDARD_PARAMETERS, *RESEARCH_PARAMETERS, *TABLEAU_TEXT_FIELDS,
    ]
    missing = [c for c in required if c not in tableau.columns]
    if missing:
        raise ValueError(f'Tableau score file is missing columns: {missing}')
    tableau = tableau.reset_index(names='tableau_row')
    tableau['year'] = tableau['Year'].astype(int)
    return tableau


def match_master_to_tableau(rankings: pd.DataFrame, tableau: pd.DataFrame) -> pd.DataFrame:
    """Link every master row to its Tableau row, and refuse to build if they disagree."""
    shared = tableau[tableau['Original Category'].isin(rankings['category'].unique())]
    matched = rankings.merge(
        shared,
        left_on=['year', 'category', 'Institute_ID_Original'],
        right_on=['year', 'Original Category', 'Institute ID'],
        how='left',
        validate='one_to_one',
    )
    if matched['tableau_row'].isna().any():
        raise ValueError(f"{int(matched['tableau_row'].isna().sum())} master rows have no Tableau row")
    composite = pd.to_numeric(matched['Composite Score'], errors='coerce')
    if (matched['score'].isna() != composite.isna()).any() or ((matched['score'] - composite).abs() > 0.01).any():
        raise ValueError('Tableau composite scores disagree with the clean master')
    analytical = pd.to_numeric(matched['Analytical Rank'], errors='coerce')
    if ((matched['rank_numeric'] - analytical).abs() > 0.01).any():
        raise ValueError('Tableau analytical ranks disagree with the clean master')
    return matched


def _normalized_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', str(name).casefold().replace('&', 'and')).strip()


def _uid_sort_key(uid: str) -> tuple[str, int]:
    prefix, _, number = uid.partition('-')
    return prefix, int(number) if number.isdigit() else 0


def build_extra_rows(
    rankings: pd.DataFrame, tableau: pd.DataFrame, matched: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build rows for the Tableau-only categories and audit how each identity was assigned.

    Institution UID links a new row to master rows. A UID with several master IDs
    uses the ID whose normalized name matches; failing that, the most frequent ID.
    A UID absent from the master becomes a new institution with a new stable ID.
    The master merges a few distinct UIDs under one ID (e.g. the Agricultural
    Sciences universities in Bengaluru and Dharwad), so each master ID belongs to
    only the UID with the most rows, then the latest year; other UIDs get new IDs.
    """
    extras = tableau[~tableau['Original Category'].isin(rankings['category'].unique())].copy()
    unknown = sorted(set(extras['Category']) - set(EXTRA_CATEGORY_GROUPS))
    if unknown:
        raise ValueError(f'Tableau file has unmapped new categories: {unknown}')
    counts = matched.groupby(['Institution UID', 'institution_id'], as_index=False).agg(
        rows=('year', 'size'), latest_year=('year', 'max')
    )
    owners = counts.sort_values(
        ['rows', 'latest_year', 'Institution UID'], ascending=[False, False, True]
    ).drop_duplicates('institution_id').set_index('institution_id')['Institution UID']
    names = rankings.drop_duplicates('institution_id').set_index('institution_id')['institution']
    latest = rankings.sort_values('year').groupby('institution_id').agg(
        latest_city=('city', 'last'), latest_state=('state', 'last')
    )
    extras = extras.sort_values('year')
    next_id = int(rankings['institution_id'].str.extract(r'(\d+)$')[0].astype(int).max()) + 1
    identity = {}
    for uid in sorted(extras['Institution UID'].unique(), key=_uid_sort_key):
        rows = extras[extras['Institution UID'].eq(uid)]
        source_name = rows['Standardized Institution Name'].iloc[-1]
        linked = counts[counts['Institution UID'].eq(uid)]
        candidates = linked[linked['institution_id'].map(owners).eq(uid)]
        if len(candidates) == 1:
            identity[uid] = (candidates['institution_id'].iloc[0], 'single master ID for UID')
        elif len(candidates) > 1:
            same_name = [
                value for value in candidates['institution_id']
                if _normalized_name(names[value]) == _normalized_name(source_name)
            ]
            if len(same_name) == 1:
                identity[uid] = (same_name[0], 'several master IDs for UID; name match')
            else:
                best = candidates.sort_values(['rows', 'institution_id'], ascending=[False, True]).iloc[0]
                identity[uid] = (best['institution_id'], 'several master IDs for UID; most frequent ID')
        else:
            method = 'new institution; master ID belongs to another UID' if len(linked) else 'new institution'
            identity[uid] = (f'INST-{next_id:04d}', method)
            next_id += 1

    extras['institution_id'] = extras['Institution UID'].map(lambda uid: identity[uid][0])
    extras['identity_method'] = extras['Institution UID'].map(lambda uid: identity[uid][1])
    new_latest = extras.groupby('institution_id').agg(
        source_name=('Standardized Institution Name', 'last'),
        source_city=('Standardized Location', 'last'),
        source_state=('Standardized State', 'last'),
    )
    ids = extras['institution_id']
    rank = pd.to_numeric(extras['Analytical Rank'], errors='coerce')
    score = pd.to_numeric(extras['Composite Score'], errors='coerce')
    is_special = extras['Category'].eq('Special Mention (2018)')
    rows = pd.DataFrame({
        'tableau_row': extras['tableau_row'],
        'year': extras['year'],
        'category': extras['Category'],
        'institution_id': ids,
        'institution': ids.map(names).fillna(ids.map(new_latest['source_name'])),
        'institution_key': ids,
        'city': ids.map(latest['latest_city']).fillna(ids.map(new_latest['source_city'])),
        'city_imputed': extras['Is Location Imputed'].str.title(),
        'state': ids.map(latest['latest_state']).fillna(ids.map(new_latest['source_state'])).str.strip(),
        'rank_numeric': rank,
        'rank_label': extras['Rank Label'],
        'rank_suffix': '',
        'rank_status': np.where(rank.notna(), 'Ranked', 'Not ranked (Special Mention)'),
        'rank_parse_method': np.where(rank.notna(), 'exact_numeric', 'missing'),
        'rank_band': rank.apply(rank_band),
        'score': score,
        'score_status': np.select(
            [score.notna(), is_special],
            ['Scored', 'Not scored (Special Mention)'],
            default='Not scored (rank-only listing)',
        ),
        'score_raw': extras['Composite Score'],
        'rank_source': extras['Rank Label'],
        'source_band': '',
        'name_standardized': extras['Standardized Institution Name'],
        'source_rankings': TABLEAU_FILE.name,
        'Institute_ID_Original': extras['Institute ID'],
        'Canonical_ID_Category': '',
        'Category_Group': extras['Category'].map(EXTRA_CATEGORY_GROUPS),
        'Is_Core_Category': 'False',
    })
    invalid_states = sorted(set(rows['state']) - CANONICAL_STATES)
    if invalid_states:
        raise ValueError(f'Tableau rows contain non-canonical states: {invalid_states}')
    audit = rows[['year', 'category', 'institution_id', 'institution', 'state', 'rank_numeric', 'score']].copy()
    audit.insert(4, 'source_name', extras['Standardized Institution Name'])
    audit.insert(5, 'source_state', extras['Standardized State'])
    audit.insert(6, 'city', rows['city'])
    audit.insert(7, 'source_city', extras['Standardized Location'])
    audit['institution_uid'] = extras['Institution UID']
    audit['institute_id_original'] = extras['Institute ID']
    audit['identity_method'] = extras['identity_method']
    return rows, audit


def add_tableau_scores(enriched: pd.DataFrame, tableau: pd.DataFrame) -> pd.DataFrame:
    """Copy parameter scores and source fields from each row's Tableau record."""
    source = tableau.set_index('tableau_row').loc[enriched['tableau_row'].astype(int)]
    out = enriched.copy()
    for field, column in {**STANDARD_PARAMETERS, **RESEARCH_PARAMETERS}.items():
        out[column] = pd.to_numeric(source[field].replace('', np.nan), errors='coerce').to_numpy()
    for field, column in TABLEAU_TEXT_FIELDS.items():
        out[column] = source[field].to_numpy()
    standard = out[list(STANDARD_PARAMETERS.values())].notna().all(axis=1)
    research = out[list(RESEARCH_PARAMETERS.values())].notna().all(axis=1) & out[['oi', 'pr']].notna().all(axis=1)
    # Means the five standard parameters; use parameter_structure to find Research rows.
    out['parameters_available'] = standard
    has_scores = standard | research
    out['parameter_source'] = np.where(has_scores, TABLEAU_FILE.name, '')
    out['parameter_quality_note'] = np.where(
        has_scores,
        'Linked to the Tableau file; its composite score agrees with the ranking record',
        'No parameter scores published in source',
    )
    out['parameter_review_required'] = False
    return out


def overall_score_residual(enriched: pd.DataFrame) -> float:
    """Largest gap between the published Overall score and its weighted parameters."""
    overall = enriched[enriched['category'].eq('Overall') & enriched['parameters_available']]
    weighted = sum(overall[column] * weight for column, weight in OVERALL_WEIGHTS.items())
    return float((weighted - overall['score']).abs().max())


def build() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    master = load_rankings()
    parameter_source, parameters = load_parameter_source(master)
    tableau = load_tableau()
    matched = match_master_to_tableau(master, tableau)
    extra_rows, merge_audit = build_extra_rows(master, tableau, matched)
    linked = master.assign(tableau_row=matched['tableau_row'].astype(int).to_numpy())
    all_rows = pd.concat([linked, extra_rows], ignore_index=True)
    if all_rows.duplicated(['year', 'category', 'institution_id']).any():
        raise ValueError('Merging Tableau rows created duplicate year + category + institution rows')
    rankings = all_rows.drop(columns='tableau_row')
    # The legacy workbook still supplies institution type and its audit fields;
    # all parameter scores now come from the Tableau file.
    pcols = [
        'year', 'institution_id', 'institution_type', 'score_legacy',
        'score_discrepancy', 'rank_discrepancy',
    ]
    enriched = all_rows.merge(
        parameters[pcols],
        on=['year', 'institution_id'],
        how='left',
        validate='many_to_one',
    )
    if len(enriched) != len(all_rows):
        raise ValueError('Parameter enrichment changed the ranking row count')
    non_overall = enriched['category'].ne('Overall')
    for col in pcols[2:]:
        enriched.loc[non_overall, col] = np.nan
    enriched = add_tableau_scores(enriched, tableau).drop(columns='tableau_row')
    enriched['score_discrepancy'] = enriched['score_discrepancy'].map(lambda v: False if pd.isna(v) else bool(v))
    enriched['rank_discrepancy'] = enriched['rank_discrepancy'].map(lambda v: False if pd.isna(v) else bool(v))
    residual = overall_score_residual(enriched)
    if residual > MAX_OVERALL_RESIDUAL:
        raise ValueError(f'Overall parameters miss the published score by up to {residual:.3f} points')

    coverage = rankings.groupby(['year','category'], as_index=False).agg(
        ranking_records=('institution_id','size'),
        records_with_numeric_rank=('rank_numeric','count'),
        records_with_score=('score','count')
    )
    param_coverage = enriched.groupby(['year','category'], as_index=False).agg(
        parameter_records=('parameters_available','sum'),
        parameter_rows_flagged_for_review=('parameter_review_required','sum')
    )
    coverage = coverage.merge(param_coverage, on=['year','category'], how='left')
    coverage[['parameter_records','parameter_rows_flagged_for_review']] = coverage[
        ['parameter_records','parameter_rows_flagged_for_review']].fillna(0).astype(int)
    fact_parameters = enriched.loc[enriched['parameter_source'].ne(''), [
        'year', 'category', 'institution_id', 'Institute_ID_Original', 'institution',
        'institution_key', 'city', 'state', 'institution_type', 'score',
        *STANDARD_PARAMETERS.values(), *RESEARCH_PARAMETERS.values(),
        'parameter_structure', 'parameter_source', 'pdf_url', 'source_url',
    ]]

    quality_rows = []
    for col in rankings.columns:
        missing_mask = rankings[col].isna() | rankings[col].astype(str).str.strip().eq('')
        quality_rows.append({'dataset':'primary_rankings','check':f'missing_{col}',
                             'value':int(missing_mask.sum()),'details':'Missing, blank or whitespace-only values in standardized field'})
    quality_rows.extend([
        {'dataset':'primary_rankings','check':'total_rows','value':len(rankings),'details':'Clean master records plus rows from Tableau-only categories'},
        {'dataset':'primary_rankings','check':'clean_master_rows','value':len(master),'details':'Records from the clean master CSV'},
        {'dataset':'tableau_merge','check':'rows_added','value':len(extra_rows),'details':'Records in categories absent from the clean master'},
        {'dataset':'tableau_merge','check':'new_institutions','value':int(merge_audit.loc[merge_audit['identity_method'].str.startswith('new institution'),'institution_id'].nunique()),'details':'Institutions given a new Institution_ID; see tableau_merge_audit.csv'},
        {'dataset':'tableau_merge','check':'rows_with_standard_parameters','value':int(enriched['parameters_available'].sum()),'details':'All five of TLR, RPC, GO, OI and PR present'},
        {'dataset':'tableau_merge','check':'rows_with_research_parameters','value':int(enriched['research_sfc'].notna().sum()),'details':'Research-category SFC, QNR and QLR present'},
        {'dataset':'tableau_merge','check':'overall_weighted_score_max_residual','value':round(residual,3),'details':'Largest gap between Overall score and 30/30/20/10/10 weighted parameters'},
        {'dataset':'primary_rankings','check':'unique_categories','value':rankings['category'].nunique(),'details':', '.join(sorted(rankings['category'].unique()))},
        {'dataset':'primary_rankings','check':'duplicate_year_category_institute_id','value':int(rankings.duplicated(['year','category','institution_id']).sum()),'details':'Expected 0'},
        {'dataset':'parameter_workbook','check':'source_rows','value':len(parameter_source),'details':'Supplementary Overall-category parameter records'},
        {'dataset':'parameter_workbook','check':'matched_to_primary','value':int(parameter_source['matched_to_primary'].sum()),'details':'Exact year + Institute ID to master Institute_ID_Original match within Overall category'},
        {'dataset':'parameter_workbook','check':'unmatched_to_primary','value':int((~parameter_source['matched_to_primary']).sum()),'details':'Retained in parameter_match_audit.csv; not joined to rankings'},
        {'dataset':'parameter_workbook','check':'score_discrepancies','value':int(parameter_source['score_discrepancy'].sum()),'details':'Legacy overall score differs from primary CSV by more than 0.01'},
        {'dataset':'parameter_workbook','check':'rank_discrepancies','value':int(parameter_source['rank_discrepancy'].sum()),'details':'Legacy overall rank differs from primary CSV'},
        {'dataset':'parameter_workbook','check':'matched_parameter_rows_default_usable','value':int((parameter_source['matched_to_primary'] & ~parameter_source['parameter_review_required']).sum()),'details':'Matched workbook records not flagged by overall rank/score discrepancy; workbook parameters are no longer used'},
        {'dataset':'enriched_fact_table','check':'rows_with_parameters','value':int(enriched['parameters_available'].sum()),'details':'Rows with all five standard parameters, across categories'}
    ])
    quality = pd.DataFrame(quality_rows)
    rankings.to_csv(PROCESSED/'fact_rankings.csv', index=False)
    parameter_source.to_csv(PROCESSED/'parameter_match_audit.csv', index=False)
    fact_parameters.to_csv(PROCESSED/'fact_parameters.csv', index=False)
    enriched.to_csv(PROCESSED/'nirf_analysis_dataset.csv', index=False)
    coverage.to_csv(PROCESSED/'category_year_coverage.csv', index=False)
    quality.to_csv(REPORTS/'data_quality_report.csv', index=False)
    merge_audit.to_csv(REPORTS/'tableau_merge_audit.csv', index=False)
    issues = parameter_source.loc[(~parameter_source['matched_to_primary']) | parameter_source['score_discrepancy'] | parameter_source['rank_discrepancy'],
        ['year','institution_id','institution','rank_legacy','rank_primary','score_legacy','score_primary',
         'matched_to_primary','rank_discrepancy','score_discrepancy','parameter_quality_note']]
    issues.to_csv(REPORTS/'parameter_merge_issues.csv', index=False)

    # The final file is what the app loads, including its calculated fields.
    sys.path.insert(0, str(ROOT))
    from nirf.data_loader import load_data
    final = load_data(PROCESSED/'nirf_analysis_dataset.csv')
    derived = ['score_change', 'rank_improvement', 'parameter_gap']
    final[derived] = final[derived].round(4)
    final.to_csv(FINAL/'nirf_final_dataset.csv', index=False, encoding='utf-8-sig')

    print('Data build complete')
    print(f'Ranking rows: {len(rankings):,} ({len(master):,} master + {len(extra_rows):,} Tableau-only)')
    print(f'Years: {rankings.year.min()}-{rankings.year.max()} | categories: {rankings.category.nunique()} | institutions: {rankings.institution_id.nunique():,}')
    print(f'Rows with standard parameters: {int(enriched.parameters_available.sum()):,}')
    print(f'Rows with Research parameters: {int(enriched.research_sfc.notna().sum()):,}')
    print(f'Largest Overall weighted-score gap: {residual:.3f}')
    print(f'Identity methods: {merge_audit.identity_method.value_counts().to_dict()}')
    print(f'Legacy workbook rows matched (audit only): {int(parameter_source.matched_to_primary.sum()):,} of {len(parameter_source):,}')
    print(f'Final dataset rows: {len(final):,}')

if __name__ == '__main__':
    build()
