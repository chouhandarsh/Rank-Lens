"""Streamlit controls for the Level 2 custom analysis explorer."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .custom_analysis import (
    AGGREGATIONS,
    COMPARE_CHARTS,
    CORRELATION_CHARTS,
    COUNT_MEASURES,
    DISTRIBUTION_CHARTS,
    GROUP_FIELDS,
    MAP_COUNT_METRICS,
    MAP_VIEWS,
    MEASURES,
    MODES,
    NUMERIC_MEASURES,
    PROFILE_CHARTS,
    TRANSITION_CHARTS,
    TREND_CHARTS,
    group_value_options,
    institution_labels,
)
from .filters import parameter_rows

OPTIONAL_GROUPS = ['No grouping', *GROUP_FIELDS]


def _measure_box(label, options, key, index=0):
    return options[st.selectbox(label, list(options), index=index, key=key)]


def _aggregation_box(measure, key):
    if measure in COUNT_MEASURES:
        return 'Count'
    return st.selectbox('Calculation', AGGREGATIONS, key=key)


def _group_picker(data, field, key, default_top_n=15, include_top_n=True):
    """Let a viewer name specific groups, or keep the leading ones."""
    config = {}
    if field is None or field == 'year':
        return config
    values, labels = group_value_options(data, field)
    if not values:
        return config
    config['selected_groups'] = st.multiselect(
        f'Only these ({len(values)} available; leave empty for the leading groups)',
        values,
        format_func=lambda value: labels.get(value, str(value)),
        key=f'{key}_{field}_values',
    )
    if include_top_n and len(values) > 2:
        top = min(default_top_n, len(values))
        config['top_n'] = st.slider('Groups to show', 2, min(50, len(values)), top, key=f'{key}_{field}_top')
    return config


def _year_pair(years, key):
    first, second = st.columns(2)
    with first:
        start = st.selectbox('From year', years[:-1], index=0, key=f'{key}_from')
    later = [year for year in years if year > start]
    with second:
        end = st.selectbox('To year', later, index=len(later) - 1, key=f'{key}_to')
    return start, end


def build_config(filtered_df, history_df, all_categories_df, include_review):
    """Render the controls. Returns the config and the data the analysis should use."""
    mode = st.selectbox('Analysis type', MODES, key='custom_mode')
    config = {'mode': mode, 'include_review': include_review}
    data = filtered_df
    years = sorted(filtered_df['year'].dropna().astype(int).unique())

    def pick_scope(*fields):
        # Comparing categories needs every category, not only the sidebar's selection.
        return all_categories_df if 'category' in fields or 'Category_Group' in fields else filtered_df

    if mode == 'Compare groups':
        c1, c2, c3 = st.columns(3)
        with c1:
            config['measure'] = _measure_box('Compare this attribute', MEASURES, 'custom_compare_measure')
        with c2:
            config['group_by'] = st.selectbox('Across', list(GROUP_FIELDS), index=3, key='custom_compare_group')
        with c3:
            config['split_by'] = st.selectbox('Split each group by (optional)', OPTIONAL_GROUPS, key='custom_compare_split')
        c1, c2 = st.columns(2)
        with c1:
            config['aggregation'] = _aggregation_box(config['measure'], 'custom_compare_agg')
        with c2:
            config['chart'] = st.selectbox('Chart type', COMPARE_CHARTS, key='custom_compare_chart')
        group, split = GROUP_FIELDS[config['group_by']], GROUP_FIELDS.get(config['split_by'])
        data = pick_scope(group, split)
        config.update(_group_picker(data, group, 'custom_compare'))
    elif mode == 'Trend over time':
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            config['measure'] = _measure_box('Attribute', MEASURES, 'custom_trend_measure')
        with c2:
            config['group_by'] = st.selectbox('Compare trend across', OPTIONAL_GROUPS, key='custom_trend_group')
        with c3:
            config['aggregation'] = _aggregation_box(config['measure'], 'custom_trend_agg')
        with c4:
            config['chart'] = st.selectbox('Chart type', TREND_CHARTS, key='custom_trend_chart')
        config['show_trendline'] = st.checkbox('Add a straight trend line', key='custom_trend_line')
        group = GROUP_FIELDS.get(config['group_by'])
        data = pick_scope(group)
        config.update(_group_picker(data, group, 'custom_trend', default_top_n=8))
    elif mode == 'Forecast / prediction':
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            config['measure'] = _measure_box('Predict this attribute', MEASURES, 'custom_forecast_measure')
        with c2:
            config['group_by'] = st.selectbox('For each', OPTIONAL_GROUPS, index=OPTIONAL_GROUPS.index('Institution'), key='custom_forecast_group')
        with c3:
            config['aggregation'] = _aggregation_box(config['measure'], 'custom_forecast_agg')
        with c4:
            config['horizon'] = st.slider('Years ahead', 1, 5, 3, key='custom_forecast_horizon')
        group = GROUP_FIELDS.get(config['group_by'])
        data = pick_scope(group)
        config.update(_group_picker(data, group, 'custom_forecast', default_top_n=5))
    elif mode == 'India map':
        labels = {**MAP_COUNT_METRICS, **MEASURES}
        c1, c2, c3 = st.columns(3)
        with c1:
            config['metric'] = _measure_box('Show on the map', labels, 'custom_map_metric')
        with c2:
            config['view'] = st.selectbox('Map view', MAP_VIEWS, key='custom_map_view')
        with c3:
            categories = sorted(filtered_df['category'].dropna().unique())
            if categories:
                config['category'] = st.selectbox('Category', categories, key='custom_map_category')
        c1, c2 = st.columns(2)
        with c1:
            if config['metric'] in {'top_n_count', 'top_n_share'}:
                config['top_n'] = st.selectbox('Top N', [10, 25, 50, 100, 150, 200], index=3, key='custom_map_top_n')
            elif config['metric'] in NUMERIC_MEASURES.values():
                config['aggregation'] = st.selectbox('Calculation', AGGREGATIONS, key='custom_map_agg')
        with c2:
            if config['view'] == 'Single year' and years:
                config['year'] = st.selectbox('Year', years, index=len(years) - 1, key='custom_map_year')
        if config['view'] in {'Two years side by side', 'Change between two years'} and len(years) >= 2:
            config['from_year'], config['to_year'] = _year_pair(years, 'custom_map')
    elif mode == 'Distribution':
        c1, c2, c3 = st.columns(3)
        with c1:
            config['measure'] = _measure_box('Attribute', NUMERIC_MEASURES, 'custom_dist_measure')
        with c2:
            config['color_by'] = st.selectbox('Compare across', OPTIONAL_GROUPS, key='custom_dist_group')
        with c3:
            config['chart'] = st.selectbox('Chart type', DISTRIBUTION_CHARTS, key='custom_dist_chart')
        group = GROUP_FIELDS.get(config['color_by'])
        data = pick_scope(group)
        config.update(_group_picker(data, group, 'custom_dist', default_top_n=8))
    elif mode == 'Relationship / scatter':
        c1, c2, c3 = st.columns(3)
        names = list(NUMERIC_MEASURES)
        with c1:
            config['x'] = NUMERIC_MEASURES[st.selectbox('X attribute', names, index=names.index('TLR (teaching)'), key='custom_x')]
        with c2:
            config['y'] = NUMERIC_MEASURES[st.selectbox('Y attribute', names, index=names.index('Overall score'), key='custom_y')]
        with c3:
            config['color_by'] = st.selectbox('Color by', OPTIONAL_GROUPS, key='custom_scatter_color')
        c1, c2 = st.columns(2)
        with c1:
            size = st.selectbox('Bubble size (optional)', ['None', *names], key='custom_scatter_size')
            config['size'] = NUMERIC_MEASURES.get(size)
        with c2:
            st.write('')
            config['animate'] = st.checkbox('Animate by year', key='custom_scatter_animate')
        group = GROUP_FIELDS.get(config['color_by'])
        data = pick_scope(group)
        config.update(_group_picker(data, group, 'custom_scatter', default_top_n=10))
    elif mode == 'Correlation matrix':
        names = list(NUMERIC_MEASURES)
        defaults = ['Overall score', 'Rank', 'TLR (teaching)', 'RPC (research)', 'GO (graduation outcome)', 'OI (outreach and inclusivity)', 'PR (perception)']
        selected = st.multiselect('Numeric attributes', names, default=defaults, key='custom_corr_vars')
        config['variables'] = [NUMERIC_MEASURES[name] for name in selected]
        config['chart'] = st.selectbox('Chart type', CORRELATION_CHARTS, key='custom_corr_chart')
    elif mode == 'Top / bottom institutions':
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            config['measure'] = _measure_box('Attribute', NUMERIC_MEASURES, 'custom_top_measure')
        with c2:
            config['direction'] = st.selectbox('Show', ['Best', 'Worst'], key='custom_top_direction')
        with c3:
            config['year'] = st.selectbox('Year', ['All selected years', *years[::-1]], index=1 if years else 0, key='custom_top_year')
        with c4:
            config['top_n'] = st.slider('How many', 3, 50, 10, key='custom_top_n')
    elif mode == 'Institution parameter profile':
        rows = parameter_rows(filtered_df, include_review)
        if rows.empty:
            st.info('No parameter records match the current sidebar filters. Research, Innovation and rank-only categories have no standard parameter set.')
        else:
            labels = institution_labels(rows)
            best = rows.sort_values(['year', 'score'], ascending=[False, False])['institution_id'].drop_duplicates().head(3).tolist()
            config['institution_ids'] = st.multiselect('Institutions to compare', list(labels), default=best, format_func=lambda value: labels[value], key='custom_profile_institutions')
            c1, c2 = st.columns(2)
            with c1:
                config['chart'] = st.selectbox('Chart type', PROFILE_CHARTS, key='custom_profile_chart')
            with c2:
                profile_years = sorted(rows['year'].astype(int).unique())
                if config['chart'] != 'Trend over time':
                    config['year'] = st.selectbox('Year', profile_years, index=len(profile_years) - 1, key='custom_profile_year')
    elif mode == 'Rank-band transitions':
        data = history_df
        history_years = sorted(history_df['year'].dropna().astype(int).unique())
        if len(history_years) >= 2:
            config['from_year'], config['to_year'] = _year_pair(history_years, 'custom_transition')
        config['chart'] = st.selectbox('Chart type', TRANSITION_CHARTS, key='custom_transition_chart')
    elif mode == 'Summary statistics':
        c1, c2, c3 = st.columns(3)
        with c1:
            config['measure'] = _measure_box('Attribute', MEASURES, 'custom_summary_measure')
        with c2:
            config['group_by'] = st.selectbox('Group by', list(GROUP_FIELDS), key='custom_summary_group')
        with c3:
            config['aggregation'] = _aggregation_box(config['measure'], 'custom_summary_agg')
        group = GROUP_FIELDS[config['group_by']]
        data = pick_scope(group)
        config.update(_group_picker(data, group, 'custom_summary'))
    return config, data


def scope_note(data: pd.DataFrame, filtered_df: pd.DataFrame) -> str | None:
    if data is not filtered_df and data['category'].nunique() > filtered_df['category'].nunique():
        return 'This view compares all categories, so it ignores the sidebar category filter.'
    return None
