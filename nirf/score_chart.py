from __future__ import annotations

import pandas as pd


def top_score_chart_data(
    df: pd.DataFrame,
    identities: pd.DataFrame,
    single_year_selected: bool,
) -> pd.DataFrame:
    scores = df.dropna(subset=['score']).copy()
    if not scores['score'].between(0, 100).all():
        raise ValueError('NIRF scores used in charts must be between 0 and 100.')

    grouping = ['institution_key', 'category']
    if single_year_selected:
        chart_data = scores.groupby(grouping, as_index=False).agg(
            institution=('institution', 'first'),
            city=('city', 'first'),
            score=('score', 'first'),
            year=('year', 'first'),
            years_averaged=('year', 'nunique'),
        )
    else:
        chart_data = scores.groupby(grouping, as_index=False).agg(
            institution=('institution', 'first'),
            city=('city', 'first'),
            score=('score', 'mean'),
            years_averaged=('year', 'nunique'),
        )

    duplicate_names = identities.groupby('institution')['institution_id'].nunique()
    duplicate_names = set(duplicate_names[duplicate_names.gt(1)].index)
    chart_data['display_institution'] = chart_data['institution']
    duplicate_mask = chart_data['institution'].isin(duplicate_names)
    chart_data.loc[duplicate_mask, 'display_institution'] = (
        chart_data.loc[duplicate_mask, 'institution']
        + ' ('
        + chart_data.loc[duplicate_mask, 'city'].fillna('').astype(str)
        + ')'
    )
    chart_data['institution_id'] = chart_data['institution_key']
    return chart_data
