from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .filters import build_institution_options, parameter_rows
from .level1 import BAND_ORDER, PARAMETER_WARNING
from .maps import make_choropleth
from .theme import DIVERGING_SCALE, SEQUENTIAL_SCALE, SEQUENTIAL_SCALE_REVERSED, WHITE
from .schema import PARAMETERS

GROUP_FIELDS = {
    'Year': 'year',
    'Category': 'category',
    'Category group': 'Category_Group',
    'State': 'state',
    'City': 'city',
    'Rank band': 'rank_band',
    'Institution type (Overall only)': 'institution_type',
    'Institution': 'institution_id',
}
MEASURES = {
    'Overall score': 'score',
    'Rank': 'rank_numeric',
    'TLR (teaching)': 'tlr',
    'RPC (research)': 'rpc',
    'GO (graduation outcome)': 'go',
    'OI (outreach and inclusivity)': 'oi',
    'PR (perception)': 'pr',
    'Score change from previous year': 'score_change',
    'Rank improvement from previous year': 'rank_improvement',
    'Parameter gap (strongest − weakest)': 'parameter_gap',
    'Number of ranking entries': 'institution_id',
    'Number of unique institutions': 'unique_institutions',
}
COUNT_MEASURES = {'institution_id', 'unique_institutions'}
NUMERIC_MEASURES = {label: value for label, value in MEASURES.items() if value not in COUNT_MEASURES}
PARAMETER_MEASURES = {'tlr', 'rpc', 'go', 'oi', 'pr', 'parameter_gap'}
PARAMETER_COLUMNS = [parameter.lower() for parameter in PARAMETERS]
MEASURE_LABELS = {value: label for label, value in MEASURES.items()}
GROUP_LABELS = {value: label for label, value in GROUP_FIELDS.items()}
AGGREGATIONS = ['Mean', 'Median', 'Count', 'Minimum', 'Maximum', 'Standard deviation']
MODES = [
    'Compare groups',
    'Trend over time',
    'Forecast / prediction',
    'India map',
    'Distribution',
    'Relationship / scatter',
    'Correlation matrix',
    'Top / bottom institutions',
    'Institution parameter profile',
    'Rank-band transitions',
    'Summary statistics',
]
COMPARE_CHARTS = ['Bar', 'Horizontal bar', 'Grouped bar', 'Stacked bar', 'Line', 'Heatmap', 'Box plot', 'Violin', 'Strip plot', 'Pie', 'Treemap']
TREND_CHARTS = ['Line', 'Area', 'Bar']
DISTRIBUTION_CHARTS = ['Histogram', 'Box plot', 'Violin', 'Strip plot', 'Cumulative (ECDF)']
CORRELATION_CHARTS = ['Heatmap', 'Scatter matrix', 'Parallel coordinates']
PROFILE_CHARTS = ['Radar', 'Bar', 'Trend over time']
TRANSITION_CHARTS = ['Heatmap', 'Sankey flow']
MAP_COUNT_METRICS = {
    'Institutions in top N': 'top_n_count',
    'Share of top N (%)': 'top_n_share',
    'Newly appearing institutions': 'new_institutions',
}
MAP_VIEWS = ['Single year', 'Two years side by side', 'Change between two years', 'Animated by year']
LOWER_IS_BETTER = {'rank_numeric'}
SCORE_BOUNDED = {'score', 'tlr', 'rpc', 'go', 'oi', 'pr'}
SPLIT_LIMIT = 8
_T_95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


def _empty(message):
    return {'figure': None, 'table': None, 'metrics': {}, 'note': message, 'warning': None, 'answer': None, 'extra_figures': []}


def _result(fig=None, table=None, metrics=None, note=None, warnings=None, answer=None, extra=None, is_map=False):
    return {
        'figure': fig,
        'table': table,
        'metrics': metrics or {},
        'note': note,
        'warning': ' '.join(dict.fromkeys(warnings)) if warnings else None,
        'answer': answer,
        'extra_figures': [item for item in (extra or []) if item is not None],
        'is_map': is_map,
    }


def _field(value, default='category'):
    """Accept either a display label or a raw column name."""
    if value in GROUP_FIELDS:
        return GROUP_FIELDS[value]
    return value if value in GROUP_FIELDS.values() else default


def _optional_field(value):
    return _field(value, None) if value not in (None, 'No grouping') else None


def _label(measure):
    return MEASURE_LABELS.get(measure, str(measure))


def _agg_name(aggregation: str) -> str:
    return {'Mean': 'mean', 'Median': 'median', 'Count': 'count', 'Minimum': 'min', 'Maximum': 'max', 'Standard deviation': 'std'}[aggregation]


def _prepare(df: pd.DataFrame, measure: str, include_review: bool = False) -> pd.DataFrame:
    out = parameter_rows(df, include_review=include_review) if measure in PARAMETER_MEASURES else df.copy()
    if measure not in COUNT_MEASURES:
        out = out[out[measure].notna()]
    return out


def _aggregate(x: pd.DataFrame, keys: list[str], measure: str, aggregation: str) -> pd.DataFrame:
    grouped = x.groupby(keys, as_index=False, observed=True)
    if measure == 'institution_id':
        return grouped.size().rename(columns={'size': 'value'})
    if measure == 'unique_institutions':
        # Built without as_index=False because institution_id can also be a group key.
        return x.groupby(keys, observed=True)['institution_id'].nunique().rename('value').reset_index()
    return grouped[measure].agg(_agg_name(aggregation)).rename(columns={measure: 'value'})


def institution_labels(df: pd.DataFrame) -> dict:
    options = build_institution_options(df)
    return dict(zip(options['institution_id'], options['label']))


def _display(x: pd.DataFrame, field: str | None) -> tuple[pd.DataFrame, str | None]:
    """Drop blank group values, and give institutions a readable column."""
    if field is None:
        return x, None
    if field not in x.columns:
        raise KeyError(field)
    out = x[x[field].notna() & x[field].astype(str).str.strip().ne('')].copy()
    if field == 'institution_id':
        out['institution_label'] = out['institution_id'].astype(str).map(institution_labels(out))
        return out, 'institution_label'
    if field == 'rank_band':
        out = out[out['rank_band'].isin(BAND_ORDER)]
    return out, field


def group_value_options(df: pd.DataFrame, field: str) -> tuple[list, dict]:
    """Values a viewer can pick for a group field, with display labels."""
    if field not in df.columns:
        return [], {}
    if field == 'institution_id':
        labels = institution_labels(df)
        return list(labels), labels
    values = df[field].dropna()
    values = values[values.astype(str).str.strip().ne('')]
    ordered = sorted(values.unique().tolist(), key=lambda value: (str(type(value)), value))
    if field == 'rank_band':
        ordered = [band for band in BAND_ORDER if band in ordered]
    return ordered, {value: str(value) for value in ordered}


def _limit(x, field, measure, aggregation, config, notes):
    """Keep the chosen groups, or the leading ones so that charts stay readable."""
    if field is None or field == 'year':
        return x
    selected = config.get('selected_groups') or []
    if selected:
        return x[x[field].isin(selected)]
    limit = int(config.get('top_n', 15))
    if x[field].nunique() <= limit:
        return x
    order = _aggregate(x, [field], measure, aggregation).dropna(subset=['value'])
    ascending = measure in LOWER_IS_BETTER and aggregation != 'Count'
    keep = order.sort_values('value', ascending=ascending)[field].head(limit)
    notes.append(f'Showing the leading {limit} of {x[field].nunique()} groups. Change this with "Groups to show", or pick specific groups.')
    return x[x[field].isin(keep)]


def _limit_split(x, split, notes):
    if split is None or split == 'year' or x[split].nunique() <= SPLIT_LIMIT:
        return x
    keep = x[split].value_counts().head(SPLIT_LIMIT).index
    notes.append(f'The split shows the {SPLIT_LIMIT} most frequent of {x[split].nunique()} values.')
    return x[x[split].isin(keep)]


def _orders(*fields):
    return {'rank_band': BAND_ORDER} if 'rank_band' in fields else {}


def _finish(fig, measure=None):
    if measure in LOWER_IS_BETTER:
        fig.update_yaxes(autorange='reversed')
    return fig


def _no_parameter_data(measure):
    if measure in PARAMETER_MEASURES:
        return _empty('No parameter scores for this selection. Research, Innovation and rank-only categories have no standard parameter set; select another category in the sidebar.')
    return _empty('No rows remain for this selection. Rank-only categories have no scores, so try another category or wider sidebar filters.')


def _warnings_for(*measures):
    return [PARAMETER_WARNING] if any(measure in PARAMETER_MEASURES for measure in measures) else []


def _compare(df, config, include_review):
    measure = config.get('measure', 'score')
    group = _field(config.get('group_by', 'Category'))
    split = _optional_field(config.get('split_by'))
    aggregation = config.get('aggregation', 'Mean')
    chart = config.get('chart', 'Bar')
    warnings, notes = _warnings_for(measure), []
    x = _prepare(df, measure, include_review)
    if x.empty:
        return _no_parameter_data(measure)
    if split == group:
        split = None
    x, display = _display(x, group)
    x = _limit(x, group, measure, aggregation, config, notes)
    if split:
        x, split_display = _display(x, split)
        x = _limit_split(x, split, notes)
    else:
        split_display = None
    if x.empty:
        return _no_parameter_data(measure)
    if 'institution_type' in (group, split):
        warnings.append('Institution type comes from the supplementary Overall-category workbook, so other categories have no value for it.')

    is_count = measure in COUNT_MEASURES
    y_title = _label(measure) if is_count else f'{aggregation} of {_label(measure)}'
    title = f'{y_title} by {GROUP_LABELS.get(group, group)}' + (f' and {GROUP_LABELS.get(split, split)}' if split else '')

    if chart in {'Box plot', 'Violin', 'Strip plot'} and not is_count:
        maker = {'Box plot': px.box, 'Violin': px.violin, 'Strip plot': px.strip}[chart]
        kwargs = {'points': 'outliers'} if chart == 'Box plot' else {'box': True} if chart == 'Violin' else {}
        fig = maker(x, x=display, y=measure, color=split_display, title=f'Spread of {_label(measure)} by {GROUP_LABELS.get(group, group)}', labels={measure: _label(measure), display: GROUP_LABELS.get(group, group)}, category_orders=_orders(group, split), **kwargs)
        _finish(fig, measure)
        table = x.groupby(display, observed=True)[measure].describe().round(2).reset_index()
        return _result(fig, table, {'Groups': x[group].nunique(), 'Records': len(x)}, ' '.join(notes) or None, warnings)

    if chart == 'Heatmap':
        other = split_display or 'year'
        if other == display:
            other = 'category'
        keys = [display, other]
    else:
        other = None
        keys = [display, split_display] if split_display else [display]
    tab = _aggregate(x, keys, measure, aggregation).dropna(subset=['value'])
    if tab.empty:
        return _no_parameter_data(measure)
    tab['value'] = tab['value'].round(2)
    ascending = measure in LOWER_IS_BETTER and aggregation != 'Count'
    tab = tab.sort_values(display if group == 'year' else 'value', ascending=True if group == 'year' else ascending)
    labels = {'value': y_title, display: GROUP_LABELS.get(group, group)}
    if split_display:
        labels[split_display] = GROUP_LABELS.get(split, split)
    orders = _orders(group, split)

    if chart in {'Pie', 'Treemap'} and tab['value'].le(0).any():
        notes.append(f'{chart} charts need positive values, so this is shown as a bar chart.')
        chart = 'Bar'
    if chart == 'Heatmap':
        matrix = tab.pivot_table(index=display, columns=other, values='value', aggfunc='first')
        fig = px.imshow(matrix, text_auto=True, aspect='auto', color_continuous_scale=SEQUENTIAL_SCALE, title=f'{y_title}: {GROUP_LABELS.get(group, group)} by {GROUP_LABELS.get(other, other)}', labels={'color': y_title, 'x': GROUP_LABELS.get(other, other), 'y': ''})
    elif chart == 'Horizontal bar':
        fig = px.bar(tab, x='value', y=display, color=split_display, orientation='h', title=title, labels=labels, category_orders=orders)
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    elif chart == 'Line':
        line = tab.sort_values(display)
        fig = px.line(line, x=display, y='value', color=split_display, markers=True, title=title, labels=labels, category_orders=orders)
        _finish(fig, measure)
    elif chart == 'Pie':
        fig = px.pie(tab.groupby(display, as_index=False, observed=True)['value'].sum() if split_display else tab, names=display, values='value', title=title, hole=.35, labels=labels)
    elif chart == 'Treemap':
        tree = tab.copy()
        tree[display] = tree[display].astype(str)
        path = [display]
        if split_display:
            tree[split_display] = tree[split_display].astype(str)
            path.append(split_display)
        fig = px.treemap(tree, path=path, values='value', color='value', color_continuous_scale=SEQUENTIAL_SCALE, title=title, labels=labels)
    else:
        barmode = 'stack' if chart == 'Stacked bar' else 'group'
        fig = px.bar(tab, x=display, y='value', color=split_display, barmode=barmode, title=title, labels=labels, category_orders=orders)
        _finish(fig, measure)
    if group == 'year' and chart not in {'Pie', 'Treemap', 'Heatmap', 'Horizontal bar'}:
        fig.update_xaxes(dtick=1)
    elif chart in {'Bar', 'Grouped bar', 'Stacked bar', 'Line'}:
        fig.update_layout(xaxis_tickangle=-35)

    answer = None
    if not split_display and chart != 'Heatmap' and len(tab) > 1:
        ordered = tab.sort_values('value', ascending=ascending)
        answer = f'{ordered[display].iloc[0]} is best on {y_title.lower()} ({ordered["value"].iloc[0]:,.2f}), and {ordered[display].iloc[-1]} is lowest ({ordered["value"].iloc[-1]:,.2f}).' if measure in LOWER_IS_BETTER else f'{ordered[display].iloc[0]} is highest on {y_title.lower()} ({ordered["value"].iloc[0]:,.2f}), and {ordered[display].iloc[-1]} is lowest ({ordered["value"].iloc[-1]:,.2f}).'
    table = tab.rename(columns={'value': y_title})
    return _result(fig, table, {'Groups': x[group].nunique(), 'Records': len(x)}, ' '.join(notes) or None, warnings, answer)


def _trend_rows(df, config, include_review, notes):
    measure = config.get('measure', 'score')
    group = _optional_field(config.get('group_by'))
    aggregation = config.get('aggregation', 'Mean')
    x = _prepare(df, measure, include_review)
    if x.empty:
        return x, None, None, None
    display = None
    if group:
        x, display = _display(x, group)
        x = _limit(x, group, measure, aggregation, config, notes)
    keys = ['year', display] if display else ['year']
    tab = _aggregate(x, keys, measure, aggregation).dropna(subset=['value']).sort_values(keys)
    return x, tab, display, group


def _linear_fit(years, values):
    slope, intercept = np.polyfit(years, values, 1)
    return float(slope), float(intercept)


def _trend(df, config, include_review):
    measure = config.get('measure', 'score')
    aggregation = config.get('aggregation', 'Mean')
    chart = config.get('chart', 'Line')
    warnings, notes = _warnings_for(measure), []
    x, tab, display, group = _trend_rows(df, config, include_review, notes)
    if tab is None or tab.empty:
        return _no_parameter_data(measure)
    y_title = _label(measure) if measure in COUNT_MEASURES else f'{aggregation} of {_label(measure)}'
    labels = {'value': y_title, 'year': 'Year'}
    if display:
        labels[display] = GROUP_LABELS.get(group, group)
    title = f'{y_title} over time'
    orders = _orders(group)
    if chart == 'Bar':
        fig = px.bar(tab, x='year', y='value', color=display, barmode='group', title=title, labels=labels, category_orders=orders)
    elif chart == 'Area':
        fig = px.area(tab, x='year', y='value', color=display, title=title, labels=labels, category_orders=orders)
    else:
        fig = px.line(tab, x='year', y='value', color=display, markers=True, title=title, labels=labels, category_orders=orders)
    fig.update_xaxes(dtick=1)
    _finish(fig, measure)

    series = [(None, tab)] if not display else list(tab.groupby(display, observed=True))
    summary = []
    for name, rows in series:
        if rows['year'].nunique() < 2:
            continue
        slope, intercept = _linear_fit(rows['year'].astype(float), rows['value'].astype(float))
        summary.append({'group': name or 'All selected', 'first_year': int(rows.year.iloc[0]), 'first_value': round(float(rows.value.iloc[0]), 2),
                        'last_year': int(rows.year.iloc[-1]), 'last_value': round(float(rows.value.iloc[-1]), 2),
                        'total_change': round(float(rows.value.iloc[-1] - rows.value.iloc[0]), 2), 'trend_per_year': round(slope, 2)})
        if config.get('show_trendline') and chart != 'Bar':
            years = rows['year'].astype(float)
            fig.add_trace(go.Scatter(x=years, y=slope * years + intercept, mode='lines', line={'dash': 'dash', 'width': 1.5}, name=f'{name or "Overall"} trend'))
    table = pd.DataFrame(summary) if summary else tab
    answer = None
    if len(summary) == 1:
        row = summary[0]
        direction = 'rose' if row['total_change'] > 0 else 'fell' if row['total_change'] < 0 else 'stayed flat'
        answer = f"{y_title} {direction} from {row['first_value']:,.2f} in {row['first_year']} to {row['last_value']:,.2f} in {row['last_year']} (linear trend {row['trend_per_year']:+.2f} per year)."
    elif summary:
        leader = max(summary, key=lambda row: row['trend_per_year'])
        answer = f"{leader['group']} has the steepest upward trend ({leader['trend_per_year']:+.2f} per year)."
    return _result(fig, table, {'Records': len(x), 'Years': int(tab['year'].nunique())}, ' '.join(notes) or None, warnings, answer)


def _forecast_series(years, values, horizon, measure):
    """Linear fit with a 95% prediction interval and a last-year holdout check."""
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(years)
    slope, intercept = _linear_fit(years, values)
    fitted = slope * years + intercept
    residuals = values - fitted
    total = ((values - values.mean()) ** 2).sum()
    r_squared = 1 - (residuals ** 2).sum() / total if total > 0 else 1.0
    spread = np.sqrt((residuals ** 2).sum() / (n - 2))
    t_value = _T_95.get(n - 2, 2.0)
    future = np.arange(years.max() + 1, years.max() + 1 + horizon)
    sxx = ((years - years.mean()) ** 2).sum()
    margin = t_value * spread * np.sqrt(1 + 1 / n + (future - years.mean()) ** 2 / sxx)
    predicted = slope * future + intercept
    low, high = predicted - margin, predicted + margin
    if measure in SCORE_BOUNDED:
        predicted, low, high = (np.clip(item, 0, 100) for item in (predicted, low, high))
    elif measure in LOWER_IS_BETTER:
        predicted, low, high = (np.clip(item, 1, None) for item in (predicted, low, high))
    elif measure in COUNT_MEASURES:
        predicted, low, high = (np.clip(item, 0, None) for item in (predicted, low, high))
    holdout = None
    if n >= 4:
        s, i = _linear_fit(years[:-1], values[:-1])
        holdout = abs(s * years[-1] + i - values[-1])
    return {'future': future, 'predicted': predicted, 'low': low, 'high': high, 'slope': slope, 'r_squared': r_squared, 'holdout_error': holdout, 'points': n}


def _forecast(df, config, include_review):
    measure = config.get('measure', 'score')
    aggregation = config.get('aggregation', 'Mean')
    horizon = int(config.get('horizon', 3))
    warnings, notes = _warnings_for(measure), []
    config = {**config, 'top_n': min(int(config.get('top_n', 5)), 10)}
    # Choose the top groups from those that can be forecast, or a few short
    # histories could fill the top N and leave nothing to draw.
    grouping = _optional_field(config.get('group_by'))
    if grouping:
        prepared = _prepare(df, measure, include_review)
        years = prepared.groupby(grouping, observed=True)['year'].nunique()
        eligible = years[years.ge(3)].index
        if len(eligible):
            df = prepared[prepared[grouping].isin(eligible)]
    x, tab, display, group = _trend_rows(df, config, include_review, notes)
    if tab is None or tab.empty:
        return _no_parameter_data(measure)
    y_title = _label(measure) if measure in COUNT_MEASURES else f'{aggregation} of {_label(measure)}'
    series = [('All selected', tab)] if not display else list(tab.groupby(display, observed=True))
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    rows, skipped = [], []
    for index, (name, part) in enumerate(series):
        if part['year'].nunique() < 3:
            skipped.append(str(name))
            continue
        part = part.sort_values('year')
        fit = _forecast_series(part['year'], part['value'], horizon, measure)
        color = colors[index % len(colors)]
        fig.add_trace(go.Scatter(x=part['year'], y=part['value'], mode='lines+markers', name=str(name), line={'color': color}, legendgroup=str(name)))
        link_x = [part['year'].iloc[-1], *fit['future']]
        link_y = [part['value'].iloc[-1], *fit['predicted']]
        fig.add_trace(go.Scatter(x=link_x, y=link_y, mode='lines+markers', name=f'{name} forecast', line={'color': color, 'dash': 'dash'}, marker={'symbol': 'diamond'}, legendgroup=str(name)))
        if len(series) <= 4:
            fig.add_trace(go.Scatter(x=[*fit['future'], *fit['future'][::-1]], y=[*fit['high'], *fit['low'][::-1]], fill='toself', fillcolor=color, opacity=.15, line={'width': 0}, hoverinfo='skip', name=f'{name} 95% range', legendgroup=str(name), showlegend=False))
        for year, value, low, high in zip(fit['future'], fit['predicted'], fit['low'], fit['high']):
            rows.append({'group': name, 'year': int(year), 'forecast': round(float(value), 2), 'low_95': round(float(low), 2), 'high_95': round(float(high), 2),
                         'trend_per_year': round(fit['slope'], 2), 'r_squared': round(fit['r_squared'], 3),
                         'last_year_test_error': None if fit['holdout_error'] is None else round(float(fit['holdout_error']), 2), 'years_of_data': fit['points']})
    if not rows:
        return _empty('Forecasting needs at least three years of data for a group. Widen the year range or choose another group.')
    if skipped:
        notes.append('Not forecast because they have fewer than three years of data: ' + ', '.join(skipped) + '.')
    fig.update_layout(title=f'{y_title}: history and {horizon}-year forecast', xaxis_title='Year', yaxis_title=y_title)
    fig.update_xaxes(dtick=1)
    _finish(fig, measure)
    table = pd.DataFrame(rows)
    final = table[table['year'].eq(table['year'].max())].iloc[0]
    answer = f"{final['group']}: forecast {final['forecast']:,.2f} in {final['year']}, with a 95% range of {final['low_95']:,.2f} to {final['high_95']:,.2f}."
    metrics = {'Trend per year': f"{final['trend_per_year']:+.2f}", 'Fit (R²)': final['r_squared']}
    if final['last_year_test_error'] is not None:
        metrics['Last-year test error'] = final['last_year_test_error']
    weak = table[table['r_squared'].lt(.5)]['group'].unique().tolist()
    if weak:
        warnings.append('The straight-line fit is weak (R² below 0.5) for: ' + ', '.join(map(str, weak)) + '. Treat these forecasts as rough.')
    return _result(fig, table, metrics, ' '.join(notes + ['This extends the past straight-line trend. It cannot foresee methodology changes or new entrants, so read it as an indication, not a certainty. Last-year test error is how far a fit without the latest year missed that year.']), warnings, answer)


def _state_year_table(df, metric, aggregation, top_n, source_df, include_review):
    if metric in {'top_n_count', 'top_n_share'}:
        rows = df[df['rank_numeric'].le(top_n)]
        tab = rows.groupby(['state', 'year'], as_index=False)['institution_id'].nunique().rename(columns={'institution_id': 'value'})
        tab['institution_count'] = tab['value']
    elif metric == 'new_institutions':
        source = source_df if source_df is not None else df
        source = source[source['category'].isin(df['category'].unique())]
        first_seen = source.groupby('institution_id')['year'].min()
        baseline = source['year'].min()
        rows = df[df['institution_id'].map(first_seen).eq(df['year']) & df['year'].gt(baseline)]
        tab = rows.groupby(['state', 'year'], as_index=False)['institution_id'].nunique().rename(columns={'institution_id': 'value'})
        tab['institution_count'] = tab['value']
    else:
        x = _prepare(df, metric, include_review)
        tab = _aggregate(x, ['state', 'year'], metric, aggregation)
        counts = x.groupby(['state', 'year'], as_index=False)['institution_id'].nunique().rename(columns={'institution_id': 'institution_count'})
        tab = tab.merge(counts, on=['state', 'year'], how='left')
    tab = tab[tab['state'].astype(str).str.strip().ne('')]
    if metric in {'top_n_count', 'top_n_share', 'new_institutions', 'institution_id', 'unique_institutions'}:
        # Absence from a count is a real zero for a ranked state.
        states = df.loc[df['state'].astype(str).str.strip().ne(''), 'state'].unique()
        grid = pd.MultiIndex.from_product([states, sorted(df['year'].unique())], names=['state', 'year']).to_frame(index=False)
        tab = grid.merge(tab, on=['state', 'year'], how='left').fillna({'value': 0, 'institution_count': 0})
    if metric == 'top_n_share':
        totals = tab.groupby('year')['value'].transform('sum')
        tab['value'] = (100 * tab['value'] / totals.where(totals > 0)).fillna(0)
    tab['value'] = tab['value'].astype(float).round(2)
    return tab


def _map(df, config, include_review, source_df):
    metric = config.get('metric', 'top_n_count')
    aggregation = config.get('aggregation', 'Mean')
    top_n = int(config.get('top_n', 100))
    view = config.get('view', 'Single year')
    warnings, notes = _warnings_for(metric), []
    categories = sorted(df['category'].dropna().unique())
    if not categories:
        return _no_parameter_data(metric)
    category = config.get('category') or ('Overall' if metric in PARAMETER_MEASURES else None)
    if category not in categories:
        category = categories[0]
    data = df[df['category'].eq(category)]
    tab = _state_year_table(data, metric, aggregation, top_n, source_df, include_review)
    if tab.empty:
        return _no_parameter_data(metric)
    label = next((name for name, value in MAP_COUNT_METRICS.items() if value == metric), None)
    if label:
        label = label.replace('top N', f'top {top_n}')
    else:
        label = _label(metric) if metric in COUNT_MEASURES else f'{aggregation} of {_label(metric)}'
    years = sorted(tab['year'].astype(int).unique())
    colorscale = SEQUENTIAL_SCALE_REVERSED if metric in LOWER_IS_BETTER else SEQUENTIAL_SCALE
    if metric == 'new_institutions':
        notes.append("A category's first year is its baseline, so new institutions cannot be identified for that year.")
    rankings_table = None
    extra = []
    answer = None

    def plot(frame, title, **kwargs):
        return make_choropleth(frame, 'value', title, kwargs.pop('colorscale', colorscale), metric_label=label, **kwargs)

    if view in {'Two years side by side', 'Change between two years'}:
        if len(years) < 2:
            return _empty('Select a year range with at least two years for this map.')
        start = config.get('from_year') if config.get('from_year') in years else years[0]
        end = config.get('to_year') if config.get('to_year') in years and config.get('to_year') > start else years[-1]
        if start >= end:
            start, end = years[0], years[-1]
        first, last = tab[tab['year'].eq(start)], tab[tab['year'].eq(end)]
        if view == 'Two years side by side':
            joint = pd.concat([first, last])['value']
            limits = (float(joint.min()), float(joint.max())) if joint.max() > joint.min() else None
            fig = plot(first, f'{label} — {start} ({category})', color_range=limits)
            extra.append(plot(last, f'{label} — {end} ({category})', color_range=limits))
            rankings_table = first[['state', 'value']].rename(columns={'value': str(start)}).merge(last[['state', 'value']].rename(columns={'value': str(end)}), on='state', how='outer').sort_values(str(end), ascending=False)
        else:
            merged = first[['state', 'value']].merge(last[['state', 'value', 'institution_count']], on='state', suffixes=('_first', ''))
            merged['value'] = (merged['value'] - merged['value_first']).round(2)
            changed = merged.drop(columns='value_first')
            if metric in LOWER_IS_BETTER:
                notes.append('For rank, a negative change is an improvement.')
            fig = make_choropleth(changed, 'value', f'Change in {label}: {start} to {end} ({category})', DIVERGING_SCALE, diverging=True, metric_label=f'Change in {label}')
            rankings_table = changed.rename(columns={'value': 'change'}).sort_values('change', ascending=False)
            if not rankings_table.empty:
                top = rankings_table.iloc[0]
                answer = f'{top.state} changed the most upward: {top.change:+,.2f} from {start} to {end}.'
    elif view == 'Animated by year':
        high, low = float(tab['value'].max()), float(tab['value'].min())
        fig = plot(tab.sort_values('year'), f'{label} by year ({category}) — press play', color_range=(low, high) if high > low else None, animation_frame='year')
        rankings_table = tab.sort_values(['year', 'value'], ascending=[True, False])
    else:
        year = config.get('year') if config.get('year') in years else years[-1]
        selected = tab[tab['year'].eq(year)]
        fig = plot(selected, f'{label} by state — {year} ({category})')
        ascending = metric in LOWER_IS_BETTER
        rankings_table = selected[['state', 'value', 'institution_count']].sort_values('value', ascending=ascending).rename(columns={'value': label})
        lead = rankings_table.iloc[0]
        answer = f'{lead.state} leads on {label.lower()} in {year}: {lead[label]:,.2f}.'
        bars = rankings_table.head(15)
        extra.append(px.bar(bars, x=label, y='state', orientation='h', title='Leading states').update_layout(yaxis={'categoryorder': 'total descending' if ascending else 'total ascending'}, yaxis_title=''))
    return _result(fig, rankings_table, {'Category': category, 'States': int(tab['state'].nunique()), 'Years available': len(years)}, ' '.join(notes) or None, warnings, answer, extra, is_map=True)


def _distribution(df, config, include_review):
    measure = config.get('measure', 'score')
    if measure in COUNT_MEASURES:
        return _empty('Distribution charts need a numeric measure. Select a score, rank or parameter.')
    chart = config.get('chart', 'Histogram')
    color = _optional_field(config.get('color_by'))
    warnings, notes = _warnings_for(measure), []
    x = _prepare(df, measure, include_review)
    if x.empty:
        return _no_parameter_data(measure)
    display = None
    if color:
        x, display = _display(x, color)
        x = _limit(x, color, measure, 'Count', config, notes)
    labels = {measure: _label(measure)}
    if display:
        labels[display] = GROUP_LABELS.get(color, color)
    title = f'Distribution of {_label(measure)}'
    orders = _orders(color)
    if chart == 'Box plot':
        fig = px.box(x, x=display, y=measure, color=display, points='outliers', title=title, labels=labels, category_orders=orders)
    elif chart == 'Violin':
        fig = px.violin(x, x=display, y=measure, color=display, box=True, title=title, labels=labels, category_orders=orders)
    elif chart == 'Strip plot':
        fig = px.strip(x, x=display, y=measure, color=display, hover_name='institution', title=title, labels=labels, category_orders=orders)
    elif chart == 'Cumulative (ECDF)':
        fig = px.ecdf(x, x=measure, color=display, title=f'Cumulative distribution of {_label(measure)}', labels=labels, category_orders=orders)
    else:
        fig = px.histogram(x, x=measure, color=display, nbins=25, barmode='overlay' if display else 'relative', opacity=.7 if display else 1, title=title, labels=labels, category_orders=orders)
        fig.add_vline(x=float(x[measure].mean()), line_dash='dash', annotation_text=f'Mean {x[measure].mean():.1f}')
        fig.update_layout(yaxis_title='Records')
    trend = None
    if x['year'].nunique() > 1:
        trend = px.box(x, x='year', y=measure, title=f'{_label(measure)} year by year', labels=labels)
        trend.update_xaxes(type='category', categoryorder='category ascending')
    table = (x.groupby(display, observed=True)[measure].describe().reset_index() if display else x[[measure]].describe().T.reset_index(drop=True)).round(2)
    series = x[measure]
    answer = f'{_label(measure)} ranges from {series.min():,.2f} to {series.max():,.2f}, with a median of {series.median():,.2f} and a mean of {series.mean():,.2f}.'
    return _result(fig, table, {'Observations': len(x), 'Mean': round(float(series.mean()), 2), 'Median': round(float(series.median()), 2), 'Std dev': round(float(series.std()), 2)}, ' '.join(notes) or None, warnings, answer, [trend])


def _relationship(df, config, include_review):
    vx, vy = config.get('x', 'tlr'), config.get('y', 'rpc')
    if vx in COUNT_MEASURES or vy in COUNT_MEASURES:
        return _empty('Scatter plots need two numeric measures.')
    if vx == vy:
        return _empty('Choose two different variables to compare.')
    size = config.get('size')
    size = size if size in NUMERIC_MEASURES.values() and size not in {vx, vy} else None
    measures = [item for item in (vx, vy, size) if item]
    warnings, notes = _warnings_for(*measures), []
    x = parameter_rows(df, include_review) if any(item in PARAMETER_MEASURES for item in measures) else df.copy()
    data = x.dropna(subset=measures).copy()
    if data.empty:
        return _empty('The selected variables have no overlapping records in this selection.')
    data['institution_label'] = data['institution_id'].astype(str).map(institution_labels(data))
    color = _optional_field(config.get('color_by'))
    display = None
    if color:
        data, display = _display(data, color)
        data = _limit(data, color, vy, 'Count', config, notes)
    if color == 'year':
        data = data.assign(year_label=data['year'].astype(int).astype(str))
        display = 'year_label'
    labels = {vx: _label(vx), vy: _label(vy), 'year_label': 'Year'}
    if display and display != 'year_label':
        labels[display] = GROUP_LABELS.get(color, color)
    kwargs = {}
    if size:
        data['_size'] = data[size] - data[size].min() + 1
        kwargs['size'] = '_size'
        labels['_size'] = _label(size)
    animate = bool(config.get('animate')) and data['year'].nunique() > 1
    if animate:
        data = data.sort_values('year')
        kwargs['animation_frame'] = 'year'
        kwargs['range_x'] = [float(data[vx].min()), float(data[vx].max())]
        kwargs['range_y'] = [float(data[vy].min()), float(data[vy].max())]
    fig = px.scatter(data, x=vx, y=vy, color=display, hover_name='institution_label', hover_data=['year', 'category', 'state'], title=f'{_label(vx)} vs {_label(vy)}', labels=labels, category_orders=_orders(color), **kwargs)
    correlation = data[vx].corr(data[vy])
    spearman = data[vx].rank().corr(data[vy].rank())
    slope = None
    if data[vx].nunique() > 1:
        slope, intercept = np.polyfit(data[vx], data[vy], 1)
        if not animate:
            xs = np.linspace(float(data[vx].min()), float(data[vx].max()), 100)
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode='lines', name='Linear trend', line={'color': WHITE, 'dash': 'dash'}))
    _finish(fig, vy)
    if vx in LOWER_IS_BETTER:
        fig.update_xaxes(autorange='reversed')
    size_of = abs(correlation) if pd.notna(correlation) else 0
    level = 'very strong' if size_of >= .8 else 'strong' if size_of >= .6 else 'moderate' if size_of >= .4 else 'weak' if size_of >= .2 else 'very weak'
    answer = None
    if pd.notna(correlation):
        answer = f'{_label(vx)} and {_label(vy)} have a {level} {"positive" if correlation >= 0 else "negative"} relationship (r = {correlation:.3f}, n = {len(data)}).'
    columns = [column for column in ['year', 'category', 'institution_label', 'state', vx, vy, size] if column]
    metrics = {'Pearson correlation': None if pd.isna(correlation) else round(float(correlation), 3), 'Spearman correlation': None if pd.isna(spearman) else round(float(spearman), 3), 'Observations': len(data)}
    if slope is not None:
        metrics['Trend slope'] = round(float(slope), 3)
    return _result(fig, data[columns].head(200), metrics, ' '.join(notes + ['Correlation measures association, not cause. Pearson measures straight-line association, and Spearman measures rank order.']), warnings, answer)


def _correlation(df, config, include_review):
    variables = [v for v in config.get('variables', ['score', 'tlr', 'rpc', 'go', 'oi', 'pr']) if v not in COUNT_MEASURES]
    chart = config.get('chart', 'Heatmap')
    x = parameter_rows(df, include_review) if any(v in PARAMETER_MEASURES for v in variables) else df.copy()
    variables = [v for v in variables if v in x.columns and x[v].notna().any()]
    if len(variables) < 2:
        return _empty('Choose at least two numeric variables that are available in the selection. Research, Innovation and rank-only categories have no standard parameter set.')
    warnings = _warnings_for(*variables)
    named = {v: _label(v) for v in variables}
    corr = x[variables].corr().rename(index=named, columns=named)
    if chart == 'Scatter matrix':
        fig = px.scatter_matrix(x, dimensions=variables, hover_name='institution', title='Scatter matrix', labels=named)
        fig.update_traces(diagonal_visible=False, marker={'size': 4, 'opacity': .6})
        fig.update_layout(height=max(500, 150 * len(variables)))
    elif chart == 'Parallel coordinates':
        clean = x[variables].dropna()
        fig = px.parallel_coordinates(clean, dimensions=variables, color=variables[0], labels=named, color_continuous_scale=SEQUENTIAL_SCALE, title='Parallel coordinates: each line is one record')
    else:
        fig = px.imshow(corr.round(2), text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale=DIVERGING_SCALE, title='Pearson correlation matrix')
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    answer = None
    if not pairs.empty:
        strongest = pairs.abs().idxmax()
        answer = f'The strongest correlation is between {strongest[0]} and {strongest[1]} (r = {pairs.loc[strongest]:.3f}).'
    return _result(fig, corr.round(3).reset_index().rename(columns={'index': 'variable'}), {'Variables': len(variables), 'Records': len(x)}, 'Correlation measures association, not cause.', warnings, answer)


def _top_bottom(df, config, include_review):
    measure = config.get('measure', 'score')
    if measure in COUNT_MEASURES:
        return _empty('Choose a numeric measure, such as score, rank or rank improvement.')
    size = int(config.get('top_n', 10))
    direction = config.get('direction', 'Best')
    warnings = _warnings_for(measure)
    x = _prepare(df, measure, include_review)
    year = config.get('year')
    if year not in (None, 'All selected years'):
        x = x[x['year'].eq(year)]
    if x.empty:
        return _no_parameter_data(measure)
    x = x.copy()
    x['institution'] = x['institution_id'].astype(str).map(institution_labels(x))
    lower_first = measure in LOWER_IS_BETTER
    ascending = lower_first if direction == 'Best' else not lower_first
    rows = x.sort_values(measure, ascending=ascending).head(size).copy()
    rows['entry'] = rows['institution'] + ' (' + rows['year'].astype(int).astype(str) + ')'
    label = _label(measure)
    fig = px.bar(rows, x=measure, y='entry', color='category', orientation='h', title=f'{direction} {size} by {label}', labels={measure: label, 'entry': ''}, hover_data=['state', 'score', 'rank_numeric'])
    fig.update_yaxes(categoryorder='array', categoryarray=rows['entry'].tolist()[::-1])
    lead = rows.iloc[0]
    table = rows[['institution', 'year', 'category', 'state', measure]].rename(columns={measure: label}).round(2)
    return _result(fig, table, {'Records searched': len(x)}, None, warnings, f'{direction} on {label.lower()}: {lead.institution} in {int(lead.year)} ({lead[measure]:,.2f}).')


def _profile(df, config, include_review):
    x = parameter_rows(df, include_review)
    if x.empty:
        return _empty('No eligible parameter records. Research, Innovation and rank-only categories have no standard parameter set.')
    ids = config.get('institution_ids') or ([config['institution_id']] if config.get('institution_id') else [])
    ids = [value for value in ids if value in set(x['institution_id'])] or [x.sort_values('score', ascending=False)['institution_id'].iloc[0]]
    labels = institution_labels(x)
    chart = config.get('chart', 'Radar')
    selected = x[x['institution_id'].isin(ids)].copy()
    selected['institution'] = selected['institution_id'].map(labels)
    if chart == 'Trend over time':
        long = selected.melt(id_vars=['institution', 'year'], value_vars=PARAMETER_COLUMNS, var_name='parameter', value_name='parameter_score')
        long['parameter'] = long['parameter'].str.upper()
        fig = px.line(long.sort_values('year'), x='year', y='parameter_score', color='parameter', line_dash='institution' if len(ids) > 1 else None, markers=True, title='Parameter scores over time')
        fig.update_xaxes(dtick=1)
        return _result(fig, long, {'Institutions': len(ids)}, None, [PARAMETER_WARNING])
    year = config.get('year')
    if year not in set(selected['year']):
        year = int(selected['year'].max())
    rows = selected[selected['year'].eq(year)]
    if rows.empty:
        return _empty('The selected institutions have no parameter scores in the chosen year.')
    long = rows.melt(id_vars=['institution'], value_vars=PARAMETER_COLUMNS, var_name='parameter', value_name='parameter_score')
    long['parameter'] = long['parameter'].str.upper()
    if chart == 'Radar':
        fig = go.Figure()
        for row in rows.itertuples():
            values = [getattr(row, column) for column in PARAMETER_COLUMNS]
            fig.add_trace(go.Scatterpolar(r=values + values[:1], theta=PARAMETERS + PARAMETERS[:1], fill='toself', opacity=.65, name=row.institution))
        fig.update_layout(title=f'Parameter profile — {year}', polar={'radialaxis': {'visible': True, 'range': [0, 100]}})
    else:
        fig = px.bar(long, x='parameter', y='parameter_score', color='institution', barmode='group', title=f'Parameter profile — {year}')
    return _result(fig, long, {'Institutions': rows['institution_id'].nunique(), 'Year': year}, None, [PARAMETER_WARNING])


def _transitions(df, config):
    years = sorted(df['year'].dropna().astype(int).unique())
    if len(years) < 2:
        return _empty('At least two years are required.')
    start, end = config.get('from_year'), config.get('to_year')
    if start not in years or end not in years or start >= end:
        start, end = years[0], years[-1]
    x = df[df['year'].isin([start, end]) & df['rank_numeric'].notna()]
    keys = ['category', 'institution_id']
    old = x[x['year'].eq(start)].drop_duplicates(keys)[[*keys, 'rank_band']]
    new = x[x['year'].eq(end)].drop_duplicates(keys)[[*keys, 'rank_band']]
    joined = old.merge(new, on=keys, suffixes=('_from', '_to'))
    if joined.empty:
        return _empty('No matching institutions across those years.')
    tab = joined.groupby(['rank_band_from', 'rank_band_to']).size().reset_index(name='institutions')
    stay = joined['rank_band_from'].eq(joined['rank_band_to']).mean() * 100
    if config.get('chart') == 'Sankey flow':
        bands = [band for band in BAND_ORDER if band in set(tab['rank_band_from']) | set(tab['rank_band_to'])]
        left = {band: index for index, band in enumerate(bands)}
        right = {band: index + len(bands) for index, band in enumerate(bands)}
        fig = go.Figure(go.Sankey(
            node={'label': [f'{band} ({start})' for band in bands] + [f'{band} ({end})' for band in bands], 'pad': 18},
            link={'source': tab['rank_band_from'].map(left), 'target': tab['rank_band_to'].map(right), 'value': tab['institutions']}))
        fig.update_layout(title=f'Rank-band flow: {start} to {end}')
    else:
        matrix = tab.pivot(index='rank_band_from', columns='rank_band_to', values='institutions').fillna(0)
        matrix = matrix.reindex(index=[b for b in BAND_ORDER if b in matrix.index], columns=[b for b in BAND_ORDER if b in matrix.columns], fill_value=0)
        fig = px.imshow(matrix, text_auto=True, aspect='auto', color_continuous_scale=SEQUENTIAL_SCALE, title=f'Rank-band transitions: {start} to {end}', labels={'x': 'To band', 'y': 'From band', 'color': 'Institutions'})
    return _result(fig, tab, {'Matched institutions': len(joined), 'Stayed in band': f'{stay:.1f}%', 'From year': start, 'To year': end}, 'Institutions are matched by institution ID, and only those ranked in both years are included.',
                   answer=f'{stay:.1f}% of the {len(joined)} matched institutions stayed in the same rank band from {start} to {end}.')


def _summary(df, config, include_review):
    measure = config.get('measure', 'score')
    group = _field(config.get('group_by', 'Category'))
    aggregation = config.get('aggregation', 'Mean')
    warnings, notes = _warnings_for(measure), []
    x = _prepare(df, measure, include_review)
    if x.empty:
        return _no_parameter_data(measure)
    x, display = _display(x, group)
    x = _limit(x, group, measure, aggregation, config, notes)
    if x.empty:
        return _no_parameter_data(measure)
    tab = _aggregate(x, [display], measure, aggregation).dropna(subset=['value'])
    is_count = measure in COUNT_MEASURES
    y_title = _label(measure) if is_count else f'{aggregation} of {_label(measure)}'
    fig = px.bar(tab.sort_values(display if group == 'year' else 'value', ascending=group == 'year' or measure in LOWER_IS_BETTER), x=display, y='value', title=f'{y_title} by {GROUP_LABELS.get(group, group)}', labels={'value': y_title, display: GROUP_LABELS.get(group, group)}, category_orders=_orders(group))
    if group == 'year':
        fig.update_xaxes(dtick=1)
    else:
        fig.update_layout(xaxis_tickangle=-35)
    if is_count:
        table = tab.rename(columns={'value': y_title}).sort_values(y_title, ascending=False)
    else:
        stats = x.groupby(display, observed=True)[measure]
        table = stats.agg(count='count', mean='mean', median='median', std='std', minimum='min', q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75), maximum='max').reset_index()
        table['iqr'] = table['q3'] - table['q1']
        table['cv_percent'] = 100 * table['std'] / table['mean'].where(table['mean'].abs() > 1e-9)
        table = table.round(2).sort_values('mean', ascending=measure in LOWER_IS_BETTER)
    return _result(fig, table, {'Groups': len(tab), 'Records': len(x)}, ' '.join(notes) or None, warnings)


def run_custom_analysis(df: pd.DataFrame, config: dict, source_df: pd.DataFrame | None = None) -> dict:
    """Run a user-configured Level 2 analysis and return figures, a table and a written answer."""
    mode = config.get('mode', 'Compare groups')
    include_review = bool(config.get('include_review', False))
    if df.empty:
        return _empty('No rows remain for this selection. Try widening the sidebar filters.')
    if mode == 'Compare groups':
        return _compare(df, config, include_review)
    if mode == 'Trend over time':
        return _trend(df, config, include_review)
    if mode == 'Forecast / prediction':
        return _forecast(df, config, include_review)
    if mode == 'India map':
        return _map(df, config, include_review, source_df)
    if mode == 'Distribution':
        return _distribution(df, config, include_review)
    if mode == 'Relationship / scatter':
        return _relationship(df, config, include_review)
    if mode == 'Correlation matrix':
        return _correlation(df, config, include_review)
    if mode == 'Top / bottom institutions':
        return _top_bottom(df, config, include_review)
    if mode == 'Institution parameter profile':
        return _profile(df, config, include_review)
    if mode == 'Rank-band transitions':
        return _transitions(df, config)
    if mode == 'Summary statistics':
        return _summary(df, config, include_review)
    return _empty('Unsupported custom analysis mode.')
