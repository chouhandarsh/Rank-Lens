from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'data' / 'processed' / 'nirf_analysis_dataset.csv'


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    file_path = Path(path) if path else DATA_FILE
    if not file_path.exists():
        raise FileNotFoundError(
            f'Processed dataset not found at {file_path}. Run: python scripts/build_data.py'
        )
    df = pd.read_csv(file_path)
    numeric = ['year','rank_numeric','score','tlr','rpc','go','oi','pr','research_sfc','research_qnr','research_qlr','score_legacy','score_primary','rank_primary']
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['parameters_available','parameter_review_required','score_discrepancy','rank_discrepancy','matched_to_primary']:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    df['category'] = df['category'].fillna('Unknown').astype(str)
    for col in ['institution','institution_id','institution_key','state','city','rank_band','rank_label']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    return add_derived_measures(df)


def add_derived_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Add year-on-year changes (consecutive years only) and the parameter gap."""
    out = df.copy()
    ordered = out.sort_values(['category', 'institution_id', 'year'])
    grouped = ordered.groupby(['category', 'institution_id'])
    consecutive = ordered['year'].sub(grouped['year'].shift()).eq(1)
    out['score_change'] = (ordered['score'] - grouped['score'].shift()).where(consecutive)
    # A lower rank number is better, so a positive value means places gained.
    out['rank_improvement'] = (grouped['rank_numeric'].shift() - ordered['rank_numeric']).where(consecutive)
    parameters = [col for col in ['tlr', 'rpc', 'go', 'oi', 'pr'] if col in out.columns]
    if parameters:
        # Blank unless all five exist: Research rows only have OI and PR in these columns.
        out['parameter_gap'] = (out[parameters].max(axis=1) - out[parameters].min(axis=1)).where(
            out[parameters].notna().all(axis=1)
        )
    return out


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        'records': len(df),
        'years': int(df['year'].nunique()),
        'year_min': int(df['year'].min()),
        'year_max': int(df['year'].max()),
        'categories': int(df['category'].nunique()),
        'institutions': int(df['institution_key'].nunique()),
        'parameter_records': int(df['parameters_available'].sum()),
        'parameter_review_records': int(df['parameter_review_required'].sum()),
    }
