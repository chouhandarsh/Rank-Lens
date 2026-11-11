from __future__ import annotations
import pandas as pd


def build_institution_options(df: pd.DataFrame) -> pd.DataFrame:
    options = df[['institution_id', 'institution', 'city']].dropna(
        subset=['institution_id']
    )
    options['institution_id'] = options['institution_id'].astype(str)
    options = options.drop_duplicates('institution_id')
    duplicate_names = options.groupby('institution')['institution_id'].nunique()
    duplicate_names = set(duplicate_names[duplicate_names.gt(1)].index)
    options['label'] = options['institution']
    duplicate_mask = options['institution'].isin(duplicate_names)
    options.loc[duplicate_mask, 'label'] = (
        options.loc[duplicate_mask, 'institution']
        + ' ('
        + options.loc[duplicate_mask, 'city'].fillna('').astype(str)
        + ')'
    )
    return options.sort_values(
        'label', key=lambda values: values.str.casefold()
    ).reset_index(drop=True)


def build_state_options(df: pd.DataFrame) -> list[str]:
    states = df['state'].dropna().astype(str).str.strip()
    counts = states.value_counts()
    by_normalized_name = {}
    for state, count in sorted(
        counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0])
    ):
        by_normalized_name.setdefault(state.casefold(), state)
    return sorted(by_normalized_name.values(), key=str.casefold)


def apply_filters(
    df: pd.DataFrame,
    year_range: tuple[int, int] | None = None,
    categories: list[str] | None = None,
    states: list[str] | None = None,
    institution_ids: list[str] | None = None,
    rank_range: tuple[int, int] | None = None,
    ignore_categories: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    if year_range:
        out = out[out['year'].between(int(year_range[0]), int(year_range[1]))]
    if categories is not None and not ignore_categories:
        out = out[out['category'].isin(categories)]
    if states:
        selected_states = {str(state).strip().casefold() for state in states}
        out = out[
            out['state'].astype(str).str.strip().str.casefold().isin(selected_states)
        ]
    if institution_ids:
        out = out[out['institution_id'].astype(str).isin(institution_ids)]
    if rank_range:
        lo, hi = int(rank_range[0]), int(rank_range[1])
        # Missing and non-numeric ranks stay in the dataset; rank-based analyses drop them explicitly.
        out = out[out['rank_numeric'].isna() | out['rank_numeric'].between(lo, hi)]
    return out.reset_index(drop=True)


def parameter_rows(df: pd.DataFrame, include_review: bool = False) -> pd.DataFrame:
    required = ['tlr', 'rpc', 'go', 'oi', 'pr']
    out = df[df['parameters_available']].copy()
    if not include_review:
        out = out[~out['parameter_review_required']]
    for col in required:
        out = out[out[col].notna()]
    # An institution has one row per category, and categories are weighted
    # differently, so parameter analyses must never pool categories.
    categories = sorted(out['category'].unique())
    if len(categories) > 1:
        chosen = 'Overall' if 'Overall' in categories else categories[0]
        out = out[out['category'].eq(chosen)]
    return out.reset_index(drop=True)
