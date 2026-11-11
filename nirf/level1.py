from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .filters import build_institution_options, parameter_rows
from .schema import PARAMETERS
from .theme import (
    DIVERGING_SCALE,
    GREEN,
    GREY,
    QUARTILE_COLORS,
    RED,
    SEQUENTIAL_SCALE,
    SEQUENTIAL_SCALE_REVERSED,
    WARM_SCALE,
    WHITE,
)

PARAMETER_COLUMNS = [parameter.lower() for parameter in PARAMETERS]
PARAMETER_WARNING = (
    'Parameter analyses use one category at a time: Overall when it is selected, '
    'otherwise the first selected category alphabetically. Research, Innovation and '
    'rank-only categories have no standard parameter set.'
)
QUARTILES = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']
BAND_ORDER = ['Top 10', '11–50', '51–100', '101+']
MOVEMENT_COLORS = {'Improved': GREEN, 'Declined': RED}


def _out(fig=None, table=None, metrics=None, note=None, warning=None, answer=None, extra=None):
    return {
        'figure': fig,
        'table': table,
        'metrics': metrics or {},
        'note': note,
        'warning': warning,
        'answer': answer    ,
        'extra_figures': [item for item in (extra or []) if item is not None],
    }


def _empty(message='There are not enough records after applying the current filters.'):
    return _out(note=message)


def _ranked(df):
    return df[df['rank_numeric'].notna()].copy()


def _selected_year(df, options):
    years = sorted(df['year'].dropna().astype(int).unique())
    if not years:
        return None
    requested = options.get('year')
    return int(requested) if requested in years else years[-1]


def _year_pair(df, options):
    years = sorted(df['year'].dropna().astype(int).unique())
    if len(years) < 2:
        return None, None
    start = options.get('from_year')
    end = options.get('to_year')
    if start not in years or end not in years or start >= end:
        return years[0], years[-1]
    return int(start), int(end)


def _institution_labels(df):
    options = build_institution_options(df)
    return dict(zip(options['institution_id'], options['label']))


def _names(values, limit=3):
    items = [str(value) for value in values][:limit]
    if not items:
        return 'none'
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + ' and ' + items[-1]


def _number(value, digits=2):
    if value is None or pd.isna(value):
        return 'n/a'
    value = float(value)
    return f'{value:.0f}' if value == int(value) else f'{value:.{digits}f}'


def _strength(correlation):
    size = abs(correlation)
    level = 'very strong' if size >= .8 else 'strong' if size >= .6 else 'moderate' if size >= .4 else 'weak' if size >= .2 else 'very weak'
    return f"{level} {'positive' if correlation >= 0 else 'negative'}"


def _horizontal(fig):
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig


def _trend_line(fig, x, column_x, column_y):
    if x[column_x].nunique() > 1:
        slope, intercept = np.polyfit(x[column_x], x[column_y], 1)
        xs = np.linspace(float(x[column_x].min()), float(x[column_x].max()), 100)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode='lines', name='Linear trend', line={'color': WHITE, 'dash': 'dash'}))
    return fig


def _consecutive_change(df, column):
    """Change in a column between consecutive years for each institution in a category."""
    work = df.dropna(subset=[column]).sort_values(['category', 'institution_id', 'year']).copy()
    grouped = work.groupby(['category', 'institution_id'])
    work['previous_year'] = grouped['year'].shift()
    work['previous_value'] = grouped[column].shift()
    work = work[work['year'].sub(work['previous_year']).eq(1)]
    work['change'] = work[column] - work['previous_value']
    return work


def _transitions(df, start, end):
    """Institutions ranked in either year, including those ranked in only one of them."""
    ranked = _ranked(df[df['year'].isin([start, end])])
    columns = ['category', 'institution_id', 'institution', 'state', 'rank_numeric', 'rank_band']
    old = ranked[ranked['year'].eq(start)][columns].drop_duplicates(['category', 'institution_id'])
    new = ranked[ranked['year'].eq(end)][columns].drop_duplicates(['category', 'institution_id'])
    joined = old.merge(new, on=['category', 'institution_id'], how='outer', suffixes=('_old', '_new'))
    joined['institution'] = joined['institution_new'].fillna(joined['institution_old'])
    joined['state'] = joined['state_new'].fillna(joined['state_old'])
    joined['rank_improvement'] = joined['rank_numeric_old'] - joined['rank_numeric_new']
    return joined


def _rank_paths(d, ids, title):
    """Yearly rank lines, which show the route behind a start-to-end comparison."""
    rows = _ranked(d[d['institution_id'].isin(ids)]).sort_values('year')
    if rows.empty:
        return None
    label = 'institution' if rows['category'].nunique() == 1 else rows['institution'] + ' · ' + rows['category']
    fig = px.line(rows, x='year', y='rank_numeric', color=label, markers=True, title=title, labels={'color': 'Institution', 'institution': 'Institution'})
    fig.update_yaxes(autorange='reversed', title='Rank (lower is better)')
    fig.update_xaxes(dtick=1)
    return fig


def _spread_by_year(x, title):
    fig = px.box(x, x='year', y='score', points='outliers', title=title)
    fig.update_xaxes(dtick=1, type='category', categoryorder='category ascending')
    return fig


def _trend_answer(frame, value):
    first, last = frame.iloc[0], frame.iloc[-1]
    change = float(last[value]) - float(first[value])
    direction = 'rose' if change > 0 else 'fell' if change < 0 else 'stayed flat'
    return direction, first, last, change


def run_level1(question_id: str, df: pd.DataFrame, all_categories_df: pd.DataFrame | None = None,
               source_df: pd.DataFrame | None = None, options: dict | None = None):
    options = options or {}
    d = df.copy()
    if d.empty:
        return _empty()
    # Distinct IDs can share a name, so charts use a label that keeps them apart.
    labels = _institution_labels(source_df if source_df is not None else d)
    d['institution'] = d['institution_id'].map(labels).fillna(d['institution'])
    start_year, end_year = _year_pair(d, options)
    year = _selected_year(d, options)
    p = parameter_rows(d, include_review=bool(options.get('include_review', False)))

    # 1. Time-series and trend analysis
    if question_id == '1.1':
        institution_id = options.get('institution_id') or d['institution_id'].iloc[0]
        x = _ranked(d[d['institution_id'].eq(institution_id)]).sort_values('year')
        if x.empty:
            return _empty('No exact numeric ranks are available for the selected institution.')
        name = x['institution'].iloc[0]
        color = 'category' if x['category'].nunique() > 1 else None
        fig = px.line(x, x='year', y='rank_numeric', color=color, markers=True, title=f'Rank trend: {name}', hover_data=['score', 'category'])
        fig.update_yaxes(autorange='reversed', title='Rank (lower is better)')
        fig.update_xaxes(dtick=1)
        scores = x.dropna(subset=['score'])
        extra = None
        if not scores.empty:
            extra = px.line(scores, x='year', y='score', color=color, markers=True, title=f'Score trend: {name}')
            extra.update_xaxes(dtick=1)
        first, last = x.iloc[0], x.iloc[-1]
        movement = first.rank_numeric - last.rank_numeric
        wording = f'improved by {_number(movement)} places' if movement > 0 else f'fell by {_number(-movement)} places' if movement < 0 else 'did not change'
        answer = (f'{name} ranked {_number(first.rank_numeric)} in {int(first.year)} and {_number(last.rank_numeric)} in {int(last.year)}, '
                  f'so its rank {wording}. Its best rank was {_number(x.rank_numeric.min())}.')
        return _out(fig, x[['year', 'category', 'institution', 'rank_label', 'score']], {'Best rank': _number(x.rank_numeric.min()), 'Years ranked': x.year.nunique()}, answer=answer, extra=[extra])

    if question_id in ('1.2', '1.3'):
        if start_year is None:
            return _empty('Select at least two years with numeric ranks.')
        joined = _transitions(d, start_year, end_year).dropna(subset=['rank_improvement'])
        if joined.empty:
            return _empty('No institutions have comparable ranks in both selected years.')
        improving = question_id == '1.2'
        view = joined.sort_values('rank_improvement', ascending=not improving).head(15)
        table = view[['category', 'institution', 'state', 'rank_numeric_old', 'rank_numeric_new', 'rank_improvement']].rename(
            columns={'rank_numeric_old': f'rank_{start_year}', 'rank_numeric_new': f'rank_{end_year}'})
        fig = px.bar(view, x='rank_improvement', y='institution', color='category', orientation='h',
                     title=f"{'Largest rank improvements' if improving else 'Largest rank declines'}: {start_year} to {end_year}")
        fig.update_layout(xaxis_title='Places gained (negative = lost)')
        fig.update_yaxes(categoryorder='total ascending' if improving else 'total descending')
        best = view.iloc[0]
        verb = 'gained' if improving else 'lost'
        answer = (f'{best.institution} {verb} the most: {_number(abs(best.rank_improvement))} places, from rank {_number(best.rank_numeric_old)} '
                  f'in {start_year} to {_number(best.rank_numeric_new)} in {end_year} ({best.category}).')
        paths = _rank_paths(d, view['institution_id'].head(5), 'Year-by-year route of the top five')
        return _out(fig, table, {'Institutions compared': len(joined), 'From year': start_year, 'To year': end_year}, answer=answer, extra=[paths],
                    note='Only institutions ranked in both selected years can be compared. Institutions are matched by institution ID.')

    if question_id == '1.4':
        scores = d.dropna(subset=['score'])
        x = scores.groupby('year', as_index=False)['score'].agg(average_score='mean', median_score='median', records='count').round(2)
        if x.empty:
            return _empty('No scores are available.')
        fig = px.line(x, x='year', y='average_score', markers=True, title='Average NIRF score over time')
        fig.update_xaxes(dtick=1)
        direction, first, last, change = _trend_answer(x, 'average_score')
        answer = f'The average score {direction} from {first.average_score:.2f} in {int(first.year)} to {last.average_score:.2f} in {int(last.year)}, a change of {change:+.2f}.'
        extra = None
        if scores['category'].nunique() > 1:
            by_category = scores.groupby(['year', 'category'], as_index=False)['score'].mean()
            extra = px.line(by_category, x='year', y='score', color='category', markers=True, title='Average score by category')
        return _out(fig, x, {'Change': f'{change:+.2f}'}, answer=answer, extra=[extra])

    if question_id == '1.5':
        x = _ranked(d[d['rank_numeric'].le(10)])
        counts = x.groupby(['category', 'institution_id', 'institution'], as_index=False)['year'].nunique().rename(columns={'year': 'top10_years'})
        counts = counts[counts['top10_years'] >= 2].sort_values('top10_years', ascending=False).head(20)
        if counts.empty:
            return _empty('No institution appears in the top 10 in at least two selected years.')
        fig = _horizontal(px.bar(counts, x='top10_years', y='institution', color='category', orientation='h', title='Institutions repeatedly appearing in the top 10'))
        span = d['year'].nunique()
        always = counts[counts['top10_years'].eq(span)]
        answer = f'{len(always)} institutions were in the top 10 in all {span} selected years: {_names(always.institution, 10)}.'
        timeline = x[x['institution_id'].isin(counts['institution_id']) & x['category'].eq(counts['category'].iloc[0])]
        matrix = timeline.pivot_table(index='institution', columns='year', values='rank_numeric', aggfunc='min')
        matrix = matrix.loc[matrix.notna().sum(axis=1).sort_values(ascending=False).index]
        extra = px.imshow(matrix, text_auto=True, aspect='auto', color_continuous_scale=SEQUENTIAL_SCALE_REVERSED, title=f'Top-10 rank by year ({counts["category"].iloc[0]}; blank = outside the top 10)', labels={'x': 'Year', 'y': '', 'color': 'Rank'})
        extra.update_xaxes(dtick=1)
        return _out(fig, counts, {'Institutions': len(counts), 'In every selected year': len(always)}, answer=answer, extra=[extra],
                    note='Counts the distinct selected years in which an institution ranked in the top 10.')

    # 2. Geospatial and regional analysis
    if question_id in ('2.1', '2.2', '2.3', '2.4', '2.5'):
        ranked = _ranked(d)
        top = ranked[ranked['rank_numeric'].le(100)]
        if question_id == '2.1':
            yearly = top[top['year'].eq(year)]
            x = yearly.groupby('state', as_index=False)['institution_id'].nunique().rename(columns={'institution_id': 'institutions_in_top_100'}).sort_values('institutions_in_top_100', ascending=False)
            if x.empty:
                return _empty('No top-100 records remain for the selected year.')
            fig = px.bar(x, x='state', y='institutions_in_top_100', title=f'Institutions in the top 100 by state — {year}')
            fig.update_layout(xaxis_tickangle=-35)
            leaders = x[x['institutions_in_top_100'].eq(x['institutions_in_top_100'].max())]
            answer = f'{_names(leaders.state, 5)} led in {year} with {int(leaders.institutions_in_top_100.iloc[0])} institutions in the top 100.'
            return _out(fig, x, {'States represented': len(x), 'Year': year}, answer=answer,
                        note='Counts unique institutions for the selected year. With several categories selected, an institution counts once per state.')
        if question_id == '2.2':
            x = top.groupby(['year', 'state'], as_index=False)['institution_id'].nunique().rename(columns={'institution_id': 'institutions'})
            if x.empty:
                return _empty('No top-100 records remain.')
            leaders = x.groupby('state')['institutions'].sum().nlargest(10).index
            fig = px.line(x[x['state'].isin(leaders)], x='year', y='institutions', color='state', markers=True, title='Top-100 institutions by state over time (10 largest states)')
            fig.update_xaxes(dtick=1)
            matrix = x.pivot(index='state', columns='year', values='institutions').fillna(0)
            matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
            extra = px.imshow(matrix, text_auto=True, aspect='auto', color_continuous_scale=SEQUENTIAL_SCALE, title='Every state by year', labels={'x': 'Year', 'y': '', 'color': 'Institutions'})
            extra.update_xaxes(dtick=1)
            spread = x.groupby('year')['state'].nunique()
            first_year, last_year = int(spread.index.min()), int(spread.index.max())
            lead = x[x['year'].eq(last_year)].nlargest(3, 'institutions')
            answer = (f'{int(spread.loc[first_year])} states were represented in {first_year} and {int(spread.loc[last_year])} in {last_year}. '
                      f'In {last_year} the largest shares were in {_names(lead.state)}.')
            return _out(fig, x, {'States shown': len(leaders)}, answer=answer, extra=[extra])
        if question_id == '2.3':
            if start_year is None:
                return _empty('At least two years are required.')
            pivot = top[top['year'].isin([start_year, end_year])].groupby(['year', 'state'])['institution_id'].nunique().unstack(fill_value=0)
            if start_year not in pivot.index or end_year not in pivot.index:
                return _empty('One of the selected years has no ranked top-100 data.')
            change = (pivot.loc[end_year] - pivot.loc[start_year]).rename('change').reset_index().sort_values('change', ascending=False)
            change['direction'] = np.select([change['change'].gt(0), change['change'].lt(0)], ['Gained', 'Lost'], 'No change')
            fig = px.bar(change, x='state', y='change', color='direction', title=f'Change in top-100 institutions: {start_year} to {end_year}',
                         color_discrete_map={'Gained': GREEN, 'Lost': RED, 'No change': GREY})
            fig.update_layout(xaxis_tickangle=-35, xaxis={'categoryorder': 'total descending'})
            gainers = change[change['change'].gt(0)]
            answer = ('No state gained top-100 institutions over this period.' if gainers.empty else
                      f'{len(gainers)} states improved. The largest gains: ' + ', '.join(f'{row.state} (+{row.change})' for row in gainers.head(3).itertuples()) + '.')
            yearly = top[top['state'].isin(gainers['state'].head(5))].groupby(['year', 'state'], as_index=False)['institution_id'].nunique().rename(columns={'institution_id': 'institutions'})
            extra = None
            if not yearly.empty:
                extra = px.line(yearly, x='year', y='institutions', color='state', markers=True, title='Year-by-year path of the largest gainers')
                extra.update_xaxes(dtick=1)
            return _out(fig, change.drop(columns='direction'), {'From year': start_year, 'To year': end_year, 'States that gained': len(gainers)}, answer=answer, extra=[extra],
                        note='Positive values mean more institutions in the top 100.')
        if question_id == '2.4':
            states = options.get('states_pair') or ['Karnataka', 'Tamil Nadu']
            selected = top[top['state'].isin(states)]
            if selected.empty:
                return _empty('The selected states have no top-100 institutions in this selection.')
            x = selected.groupby(['year', 'state'], as_index=False).agg(average_rank=('rank_numeric', 'mean'), institutions=('institution_id', 'nunique'), average_score=('score', 'mean'), best_rank=('rank_numeric', 'min')).round(2)
            fig = px.line(x, x='year', y='average_rank', color='state', markers=True, title=f'Average top-100 rank: {" vs ".join(states)}')
            fig.update_yaxes(autorange='reversed', title='Average rank (lower is better)')
            fig.update_xaxes(dtick=1)
            counts = px.bar(x, x='year', y='institutions', color='state', barmode='group', title='Institutions in the top 100')
            counts.update_xaxes(dtick=1)
            spread = px.box(selected, x='state', y='rank_numeric', color='state', points='all', hover_name='institution', title='Spread of individual ranks, all selected years')
            spread.update_yaxes(autorange='reversed', title='Rank (lower is better)')
            latest = x[x['year'].eq(x['year'].max())].set_index('state')
            summary = '; '.join(f'{state}: {int(latest.loc[state].institutions)} institutions, average rank {latest.loc[state].average_rank:.1f}, best rank {_number(latest.loc[state].best_rank)}' for state in latest.index)
            return _out(fig, x, {'States': ', '.join(states)}, answer=f'In {int(x.year.max())} — {summary}.', extra=[counts, spread])
        # A category's first year is its baseline, so its institutions are not "new".
        src = source_df if source_df is not None else d
        first_seen = src.groupby(['category', 'institution_id'], as_index=False)['year'].min().rename(columns={'year': 'first_observed_year'})
        baseline = src.groupby('category', as_index=False)['year'].min().rename(columns={'year': 'category_first_year'})
        x = d.merge(first_seen, on=['category', 'institution_id'], how='left').merge(baseline, on='category', how='left')
        x = x[x['year'].eq(x['first_observed_year']) & x['year'].gt(x['category_first_year'])]
        if start_year is not None:
            x = x[x['year'].between(start_year, end_year)]
        counts = x.groupby('state', as_index=False)['institution_id'].nunique().rename(columns={'institution_id': 'new_institutions'}).sort_values('new_institutions', ascending=False)
        if counts.empty:
            return _empty('No newly appearing institutions in this selection.')
        fig = px.bar(counts.head(15), x='state', y='new_institutions', title='States with the most newly appearing institutions')
        fig.update_layout(xaxis_tickangle=-35)
        by_year = x.groupby('year', as_index=False)['institution_id'].nunique().rename(columns={'institution_id': 'new_institutions'})
        extra = px.bar(by_year, x='year', y='new_institutions', title='New institutions by year')
        extra.update_xaxes(dtick=1)
        lead = counts.iloc[0]
        return _out(fig, counts, {'New institutions': int(counts.new_institutions.sum())}, extra=[extra],
                    answer=f'{lead.state} produced the most, with {int(lead.new_institutions)} newly appearing institutions.',
                    note="New means first seen after a category's first year in this dataset, so that first year is treated as a baseline. It is not an institution's founding year or first-ever NIRF entry.")

    # 3. Univariate analysis
    if question_id == '3.1':
        all_scores = d[d['score'].notna()]
        x = all_scores[all_scores['year'].eq(year)]
        if x.empty:
            return _empty('No scores for the selected year.')
        s = x['score']
        fig = px.histogram(x, x='score', nbins=20, title=f'NIRF score distribution — {year}', labels={'score': 'Score'})
        fig.update_layout(yaxis_title='Institutions', bargap=.05)
        fig.add_vline(x=float(s.mean()), line_dash='dash', line_color=RED, annotation_text=f'Mean {s.mean():.1f}', annotation_position='top right')
        fig.add_vline(x=float(s.median()), line_dash='dot', line_color=WHITE, annotation_text=f'Median {s.median():.1f}', annotation_position='top left')
        shape = 'right-skewed: most institutions score low and a few score very high' if s.mean() > s.median() + .25 else 'left-skewed' if s.mean() < s.median() - .25 else 'roughly symmetric'
        answer = f'In {year}, scores ran from {s.min():.2f} to {s.max():.2f} with a median of {s.median():.2f}. The distribution is {shape}.'
        trend = _spread_by_year(all_scores, 'How the distribution changes year by year')
        table = x.groupby('category')['score'].describe().round(2).reset_index()
        return _out(fig, table, {'Records': len(x), 'Mean': round(s.mean(), 2), 'Median': round(s.median(), 2), 'Std dev': round(s.std(), 2)}, answer=answer, extra=[trend])
    if question_id == '3.2':
        x = _ranked(d)
        if x.empty:
            return _empty('No numeric ranks are available.')
        high = int(x['rank_numeric'].max())
        fig = px.histogram(x, x='rank_numeric', title='Rank positions, all selected years', labels={'rank_numeric': 'Rank'})
        fig.update_traces(xbins={'start': .5, 'end': high + .5, 'size': 10})
        fig.update_layout(yaxis_title='Ranking records', bargap=.05)
        bands = x[x['rank_band'].isin(BAND_ORDER)].groupby(['year', 'rank_band'], as_index=False).size().rename(columns={'size': 'records'})
        stacked = px.bar(bands, x='year', y='records', color='rank_band', title='Records in each rank band by year', category_orders={'rank_band': BAND_ORDER})
        stacked.update_xaxes(dtick=1)
        average = x.groupby(['category', 'institution_id'], as_index=False)['rank_numeric'].mean().rename(columns={'rank_numeric': 'average_rank'})
        typical = px.histogram(average, x='average_rank', nbins=20, title='Average rank of each institution', labels={'average_rank': 'Average rank across selected years'})
        typical.update_layout(yaxis_title='Institutions', bargap=.05)
        answer = (f'Across {len(x):,} ranking records the median rank is {_number(x.rank_numeric.median(), 1)}. Rank positions are close to evenly spread because each year fills '
                  f'each position about once. Average rank per institution is more revealing: {int(average.average_rank.le(10).sum())} of {len(average):,} institutions average inside the top 10.')
        table = x.groupby('rank_band')['rank_numeric'].agg(records='count', best_rank='min', worst_rank='max').reindex([band for band in BAND_ORDER if band in set(x['rank_band'])]).reset_index()
        return _out(fig, table, {'Ranked records': len(x), 'Median rank': round(x.rank_numeric.median(), 1), 'Institutions': x.institution_id.nunique()}, answer=answer, extra=[typical, stacked])
    if question_id == '3.3':
        scored = d[d['score'].notna()]
        x = scored[scored['year'].eq(year)]
        if x.empty:
            return _empty('No scores for the selected year.')
        s = x['score']
        stats = scored.groupby('year')['score'].agg(Average='mean', Median='median', Minimum='min', Maximum='max', Records='count').round(2).reset_index()
        long = stats.melt(id_vars='year', value_vars=['Maximum', 'Average', 'Median', 'Minimum'], var_name='Statistic', value_name='Score')
        fig = px.line(long, x='year', y='Score', color='Statistic', markers=True, title=f'Score summary over time ({year} marked)')
        fig.update_xaxes(dtick=1)
        if stats['year'].nunique() > 1:
            fig.add_vline(x=year, line_dash='dot', line_color=GREY)
        answer = f'In {year}: average {s.mean():.2f}, median {s.median():.2f}, minimum {s.min():.2f}, maximum {s.max():.2f}.'
        return _out(fig, stats, {'Average': round(s.mean(), 2), 'Median': round(s.median(), 2), 'Minimum': round(s.min(), 2), 'Maximum': round(s.max(), 2)}, answer=answer)
    if question_id == '3.4':
        scored = d[d['score'].notna() & d['rank_numeric'].notna()].copy()
        if scored[scored['year'].eq(year)].empty:
            return _empty('No scores for the selected year.')
        # Cut-offs come from all selected years. Quartiles of one year alone are 25% each by definition.
        cutoffs = []
        for category, group in scored.groupby('category'):
            q1, q2, q3 = group['score'].quantile([.25, .5, .75])
            bins = [-np.inf, q1, q2, q3, np.inf]
            scored.loc[group.index, 'quartile'] = pd.cut(group['score'], bins=bins, labels=QUARTILES).astype(str)
            cutoffs.append({'category': category, 'Q1 upper limit': round(q1, 2), 'Q2 upper limit': round(q2, 2), 'Q3 upper limit': round(q3, 2)})
        counts = scored.groupby(['category', 'year', 'quartile'], as_index=False).size().rename(columns={'size': 'records'})
        counts['percentage'] = (100 * counts['records'] / counts.groupby(['category', 'year'])['records'].transform('sum')).round(2)
        selected = counts[counts['year'].eq(year)]
        fig = px.bar(selected, x='quartile', y='percentage', color='category', barmode='group', text='percentage', title=f'Share of ranked institutions in each score quartile — {year}', category_orders={'quartile': QUARTILES})
        fig.update_layout(yaxis_title='Institutions (%)', xaxis_title='Score quartile')
        trend = px.bar(counts, x='year', y='percentage', color='quartile', title='Quartile shares over time', category_orders={'quartile': QUARTILES},
                       facet_col='category' if counts['category'].nunique() > 1 else None, color_discrete_sequence=QUARTILE_COLORS)
        trend.update_layout(yaxis_title='Institutions (%)')
        trend.update_xaxes(dtick=1)
        first = selected.sort_values('category').iloc[0]['category']
        shares = selected[selected['category'].eq(first)].set_index('quartile')['percentage']
        answer = f'{first}, {year}: ' + ', '.join(f'{name} {shares.get(name, 0):.1f}%' for name in QUARTILES) + '.'
        return _out(fig, pd.DataFrame(cutoffs), {'Records': int(selected.records.sum()), 'Top-quartile share': f'{shares.get(QUARTILES[-1], 0):.1f}%'}, answer=answer, extra=[trend],
                    note='Quartile cut-offs (table below) are fixed across all selected years within each category, so shares can move over time. Within one year alone, quartiles would always be 25% each.')
    if question_id == '3.5':
        x = all_categories_df.copy() if all_categories_df is not None else d.copy()
        x = x[x['score'].notna()]
        if x.empty:
            return _empty('No category score data is available.')
        table = x.groupby('category')['score'].agg(records='count', median='median', mean='mean', std='std', q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75)).reset_index()
        table['iqr'] = table['q3'] - table['q1']
        table = table.round(2).sort_values('iqr', ascending=False)
        fig = px.box(x, x='category', y='score', points='outliers', title='Score spread across categories (widest first)', category_orders={'category': table['category'].tolist()})
        fig.update_layout(xaxis_tickangle=-35)
        by_year = x.groupby(['category', 'year'])['score'].agg(lambda s: s.quantile(.75) - s.quantile(.25)).unstack()
        by_year = by_year.reindex(table['category'])
        extra = px.imshow(by_year.round(1), text_auto=True, aspect='auto', color_continuous_scale=WARM_SCALE, title='Interquartile range by category and year (blank = no data)', labels={'x': 'Year', 'y': '', 'color': 'IQR'})
        extra.update_xaxes(dtick=1)
        widest, narrowest = table.iloc[0], table.iloc[-1]
        return _out(fig, table, {'Categories': x.category.nunique()}, extra=[extra],
                    answer=f'{widest.category} has the widest middle-50% spread (IQR {widest.iqr:.2f}) and {narrowest.category} the narrowest (IQR {narrowest.iqr:.2f}).',
                    note='This question includes every category, regardless of the sidebar category selection.')

    # 4. Bivariate analysis and correlation (parameter_rows keeps one category.)
    if question_id in ('4.1', '4.2', '4.3', '4.4', '4.5'):
        x = p
        if x.empty:
            return _empty('No rows with parameter scores remain after filters. Research, Innovation and rank-only categories have no standard parameter set.')
        if question_id in ('4.1', '4.2', '4.4'):
            vx, vy = {'4.1': ('rpc', 'score'), '4.2': ('pr', 'score'), '4.4': ('tlr', 'rpc')}[question_id]
            x = x.dropna(subset=[vx, vy])
            names = {'rpc': 'RPC', 'pr': 'PR', 'tlr': 'TLR', 'score': 'overall score'}
            scatter = x.assign(year=x['year'].astype(str))
            fig = px.scatter(scatter, x=vx, y=vy, color='year', hover_name='institution', hover_data=['state'], title=f'{names[vx]} vs {names[vy]}', labels={vx: names[vx], vy: names[vy]}, category_orders={'year': sorted(scatter['year'].unique())})
            _trend_line(fig, x, vx, vy)
            corr = x[vx].corr(x[vy])
            yearly = pd.DataFrame([{'year': int(value), 'correlation': group[vx].corr(group[vy])} for value, group in x.groupby('year')]).round(3)
            extra = px.line(yearly, x='year', y='correlation', markers=True, title='Correlation in each year')
            extra.update_xaxes(dtick=1)
            extra.update_yaxes(range=[-1, 1])
            answer = f'{names[vx]} and {names[vy]} have a {_strength(corr)} relationship (Pearson r = {corr:.3f}, n = {len(x)}).'
            return _out(fig, yearly, {'Pearson correlation': round(float(corr), 3), 'Records': len(x)}, answer=answer, extra=[extra], warning=PARAMETER_WARNING,
                        note='Correlation shows association, not cause.')
        if question_id == '4.3':
            tab = pd.DataFrame({'parameter': PARAMETERS, 'correlation': [x[col].corr(x['score']) for col in PARAMETER_COLUMNS]}).round(3)
            tab = tab.reindex(tab['correlation'].abs().sort_values(ascending=False).index)
            fig = px.bar(tab, x='parameter', y='correlation', text='correlation', title='Correlation of each parameter with overall score')
            fig.update_yaxes(range=[min(0, tab.correlation.min()) - .05, 1.05])
            matrix = x[['score', *PARAMETER_COLUMNS]].corr().round(2)
            matrix.index = matrix.columns = ['Overall', *PARAMETERS]
            extra = px.imshow(matrix, text_auto=True, zmin=-1, zmax=1, color_continuous_scale=DIVERGING_SCALE, title='All correlations')
            lead = tab.iloc[0]
            return _out(fig, tab, {'Strongest parameter': lead.parameter, 'Correlation': lead.correlation}, extra=[extra], warning=PARAMETER_WARNING,
                        answer=f'{lead.parameter} has the strongest correlation with the overall score (r = {lead.correlation:.3f}).')
        yearly = pd.DataFrame([{'year': int(value), 'correlation': group['rpc'].corr(group['score']), 'institutions': len(group)} for value, group in x.groupby('year')]).round(3)
        fig = px.line(yearly, x='year', y='correlation', markers=True, title='RPC–overall score correlation by year')
        fig.update_xaxes(dtick=1)
        direction, first, last, change = _trend_answer(yearly, 'correlation')
        return _out(fig, yearly, {'Change': f'{change:+.3f}'}, warning=PARAMETER_WARNING,
                    answer=f'The correlation {direction} from {first.correlation:.3f} in {int(first.year)} to {last.correlation:.3f} in {int(last.year)}.')

    # 5. Multivariate analysis
    if question_id in ('5.1', '5.2', '5.3', '5.4', '5.5'):
        x = p
        if x.empty:
            return _empty('No rows with parameter scores remain after filters. Research, Innovation and rank-only categories have no standard parameter set.')
        if question_id == '5.1':
            x = x[x['year'].eq(year)].sort_values('score', ascending=False).head(5)
            if x.empty:
                return _empty(f'No parameter rows for {year}. Try another year or a category with parameter scores.')
            long = x.melt(id_vars=['institution_id', 'institution', 'year'], value_vars=PARAMETER_COLUMNS, var_name='parameter', value_name='parameter_score')
            long['parameter'] = long['parameter'].str.upper()
            fig = px.bar(long, x='parameter', y='parameter_score', color='institution', barmode='group', title=f'Top five institutions across five parameters — {year}')
            radar = go.Figure()
            for row in x.itertuples():
                values = [getattr(row, column) for column in PARAMETER_COLUMNS]
                radar.add_trace(go.Scatterpolar(r=values + values[:1], theta=PARAMETERS + PARAMETERS[:1], name=row.institution))
            radar.update_layout(title='Parameter profile', polar={'radialaxis': {'visible': True, 'range': [0, 100]}})
            leaders = {parameter: x.loc[x[parameter.lower()].idxmax(), 'institution'] for parameter in PARAMETERS}
            answer = 'Highest within this group — ' + '; '.join(f'{parameter}: {name}' for parameter, name in leaders.items()) + '.'
            return _out(fig, long, {'Institutions': x.institution_id.nunique()}, answer=answer, extra=[radar], warning=PARAMETER_WARNING)
        if question_id in ('5.2', '5.3'):
            a, b = ('rpc', 'tlr') if question_id == '5.2' else ('tlr', 'pr')
            x = x[x['year'].eq(year)].dropna(subset=[a, b]).copy()
            if x.empty:
                return _empty(f'No parameter rows for {year}. Try another year or a category with parameter scores.')
            q75, q25 = x[a].quantile(.75), x[b].quantile(.25)
            x['group'] = np.where((x[a] >= q75) & (x[b] <= q25), f'High {a.upper()}, low {b.upper()}', 'Other institutions')
            flagged = x[x['group'].ne('Other institutions')].sort_values(a, ascending=False)
            fig = px.scatter(x, x=a, y=b, color='group', hover_name='institution', title=f'High {a.upper()} but low {b.upper()} — {year}', labels={a: a.upper(), b: b.upper()},
                             color_discrete_map={f'High {a.upper()}, low {b.upper()}': RED, 'Other institutions': GREY})
            fig.add_vline(x=q75, line_dash='dash')
            fig.add_hline(y=q25, line_dash='dash')
            answer = (f'{len(flagged)} institutions qualify in {year}' + (f', led by {_names(flagged.institution)}.' if len(flagged) else '.'))
            return _out(fig, flagged[['institution', 'year', 'state', a, b]], {'Institutions': len(flagged), 'High threshold': round(float(q75), 2), 'Low threshold': round(float(q25), 2)}, answer=answer,
                        note='High means the top quarter and low means the bottom quarter for the selected year.', warning=PARAMETER_WARNING)
        if question_id == '5.4':
            work = x.sort_values(['institution_id', 'year']).copy()
            grouped = work.groupby('institution_id')
            work['from_year'] = grouped['year'].shift()
            consecutive = work['year'].sub(work['from_year']).eq(1)
            work['score_change'] = work['score'] - grouped['score'].shift()
            declines = pd.DataFrame({parameter: (work[parameter.lower()] - grouped[parameter.lower()].shift()).lt(0) for parameter in PARAMETERS})
            work['declining_parameters'] = declines.apply(lambda row: ', '.join(row.index[row]), axis=1)
            tab = work[consecutive & work['score_change'].gt(0) & work['declining_parameters'].ne('')].copy()
            tab['from_year'] = tab['from_year'].astype(int)
            tab = tab.rename(columns={'year': 'to_year'})[['institution', 'from_year', 'to_year', 'score_change', 'declining_parameters']].sort_values('score_change', ascending=False).round(2)
            if tab.empty:
                return _out(answer='No institution meets this condition in the current selection.', warning=PARAMETER_WARNING)
            fig = _horizontal(px.bar(tab.head(20).assign(period=lambda f: f['institution'] + ' (' + f['to_year'].astype(str) + ')'), x='score_change', y='period', color='declining_parameters', orientation='h',
                                     title='Largest score rises despite a parameter decline'))
            fig.update_layout(yaxis_title='')
            lead = tab.iloc[0]
            return _out(fig, tab, {'Year-on-year cases': len(tab), 'Institutions': tab.institution.nunique()}, warning=PARAMETER_WARNING,
                        answer=f'{len(tab)} cases across {tab.institution.nunique()} institutions. The largest was {lead.institution}: +{lead.score_change:.2f} from {lead.from_year} to {lead.to_year} while {lead.declining_parameters} fell.',
                        note='Only consecutive year pairs are compared.')
        x = x[x['year'].eq(year)].copy()
        if x.empty:
            return _empty(f'No parameter rows for {year}. Try another year or a category with parameter scores.')
        x['parameter_gap'] = x[PARAMETER_COLUMNS].max(axis=1) - x[PARAMETER_COLUMNS].min(axis=1)
        tab = x.sort_values('parameter_gap', ascending=False).head(20)
        fig = _horizontal(px.bar(tab, x='parameter_gap', y='institution', orientation='h', title=f'Largest strongest-to-weakest parameter gaps — {year}'))
        lead = tab.iloc[0]
        strongest = PARAMETERS[int(np.argmax([lead[c] for c in PARAMETER_COLUMNS]))]
        weakest = PARAMETERS[int(np.argmin([lead[c] for c in PARAMETER_COLUMNS]))]
        return _out(fig, tab[['institution', 'state', 'parameter_gap', *PARAMETER_COLUMNS]].round(2), warning=PARAMETER_WARNING,
                    answer=f'{lead.institution} has the largest gap in {year}: {lead.parameter_gap:.2f} points between {strongest} and {weakest}.')

    # 6. Distribution and statistical comparison
    if question_id == '6.1':
        x = d[d['score'].notna() & d['rank_numeric'].notna()].copy()
        if x.empty:
            return _empty()
        x['rank_group'] = np.where(x['rank_numeric'].le(10), 'Top 10', 'Remaining institutions')
        fig = px.box(x, x='rank_group', y='score', color='rank_group', points='outliers', title='Score distribution: top 10 vs remaining institutions')
        tab = x.groupby('rank_group')['score'].agg(['count', 'median', 'mean', 'std', 'min', 'max']).round(2).reset_index()
        by_year = x.groupby(['year', 'rank_group'], as_index=False)['score'].median()
        extra = px.line(by_year, x='year', y='score', color='rank_group', markers=True, title='Median score by year')
        extra.update_xaxes(dtick=1)
        top, rest = tab.set_index('rank_group').loc['Top 10'], tab.set_index('rank_group').loc['Remaining institutions']
        return _out(fig, tab, extra=[extra],
                    answer=f'The top 10 have a median score of {top["median"]:.2f}, against {rest["median"]:.2f} for the rest, a gap of {top["median"] - rest["median"]:.2f} points. Their scores also vary {"less" if top["std"] < rest["std"] else "more"} (std dev {top["std"]:.2f} vs {rest["std"]:.2f}).')
    if question_id == '6.2':
        scored = d[d['score'].notna()]
        x = scored.groupby('year')['score'].agg(score_std='std', q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75)).reset_index()
        x['iqr'] = x['q3'] - x['q1']
        x = x.drop(columns=['q1', 'q3']).round(2)
        if x.empty:
            return _empty()
        fig = px.line(x, x='year', y='score_std', markers=True, title='Score dispersion (standard deviation) by year')
        fig.update_xaxes(dtick=1)
        high, low = x.loc[x['score_std'].idxmax()], x.loc[x['score_std'].idxmin()]
        return _out(fig, x, extra=[_spread_by_year(scored, 'Score spread in each year')],
                    answer=f'Yes. Scores were most dispersed in {int(high.year)} (std dev {high.score_std:.2f}) and least in {int(low.year)} ({low.score_std:.2f}).')
    if question_id == '6.3':
        x = p[p['year'].eq(year)]
        if x.empty:
            return _empty(f'No eligible parameter scores for {year}. Try another year or a category with parameter scores.')
        tab = x[PARAMETER_COLUMNS].std().rename_axis('parameter').reset_index(name='standard_deviation').sort_values('standard_deviation', ascending=False).round(2)
        tab['parameter'] = tab['parameter'].str.upper()
        fig = px.bar(tab, x='parameter', y='standard_deviation', text='standard_deviation', title=f'Parameter variability across institutions — {year}')
        long = x.melt(value_vars=PARAMETER_COLUMNS, var_name='parameter', value_name='parameter_score')
        long['parameter'] = long['parameter'].str.upper()
        extra = px.box(long, x='parameter', y='parameter_score', title='Spread of each parameter', category_orders={'parameter': tab['parameter'].tolist()})
        lead = tab.iloc[0]
        return _out(fig, tab, extra=[extra], warning=PARAMETER_WARNING, answer=f'{lead.parameter} varies the most in {year} (std dev {lead.standard_deviation:.2f}).')
    if question_id == '6.4':
        x = d[d['score'].notna()].groupby('year')['score'].agg(median='median', q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75)).reset_index()
        if x.empty:
            return _empty()
        x['iqr'] = x['q3'] - x['q1']
        x = x.round(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x.year, y=x.q3, mode='lines', line={'width': 0}, showlegend=False))
        fig.add_trace(go.Scatter(x=x.year, y=x.q1, mode='lines', fill='tonexty', name='Interquartile range'))
        fig.add_trace(go.Scatter(x=x.year, y=x['median'], mode='lines+markers', name='Median score'))
        fig.update_layout(title='Median and interquartile range over time', xaxis_title='Year', yaxis_title='Score')
        fig.update_xaxes(dtick=1)
        first, last = x.iloc[0], x.iloc[-1]
        return _out(fig, x, answer=f'From {int(first.year)} to {int(last.year)}, the median moved from {first["median"]:.2f} to {last["median"]:.2f} and the IQR from {first.iqr:.2f} to {last.iqr:.2f}.')
    if question_id == '6.5':
        x = d[d['score'].notna()].copy()
        if x.empty:
            return _empty()
        largest = x['state'].value_counts().head(12).index
        tab = x.groupby('state')['score'].agg(['count', 'median', 'mean', 'std']).round(2).reset_index().sort_values('median', ascending=False)
        order = [state for state in tab['state'] if state in set(largest)]
        fig = px.box(x[x['state'].isin(largest)], x='state', y='score', points='outliers', title='Score distribution across the 12 states with the most records', category_orders={'state': order})
        fig.update_layout(xaxis_tickangle=-35)
        steady = tab[tab['count'].ge(10)]
        lead = (steady if not steady.empty else tab).iloc[0]
        return _out(fig, tab, answer=f'Among states with at least 10 records, {lead.state} has the highest median score ({lead["median"]:.2f}).')

    # 7. Ranking transitions and institutional mobility
    if question_id in ('7.1', '7.2', '7.3', '7.4'):
        a, b = _year_pair(d, options)
        if a is None:
            return _empty('At least two years are required.')
        if question_id in {'7.2', '7.3'} and b != a + 1:
            return _empty('This question compares consecutive years. Choose a To year exactly one year after the From year.')
        joined = _transitions(d, a, b)
        if joined.empty:
            return _empty('No ranked institutions are available in the selected years.')
        table_columns = ['category', 'institution', 'state', 'rank_numeric_old', 'rank_numeric_new']
        if question_id == '7.1':
            band = joined[joined['rank_numeric_old'].between(51, 100)]
            moved = band[band['rank_numeric_new'].le(50)]
            stayed = band[band['rank_numeric_new'].between(51, 100)]
            left = len(band) - len(moved) - len(stayed)
            counts = pd.DataFrame({'outcome': ['Moved into top 50', 'Stayed in 51–100', 'Fell below 100 or not ranked'], 'institutions': [len(moved), len(stayed), left]})
            fig = px.bar(counts, x='outcome', y='institutions', color='outcome', text='institutions', title=f'Where the {a} band 51–100 went by {b}')
            fig.update_layout(showlegend=False)
            answer = f'{len(moved)} institutions moved from the 51–100 band into the top 50 between {a} and {b}' + (f', including {_names(moved.sort_values("rank_numeric_new").institution)}.' if len(moved) else '.')
            return _out(fig, moved.sort_values('rank_numeric_new')[table_columns], {'Moved into top 50': len(moved), 'From year': a, 'To year': b}, answer=answer)
        if question_id == '7.2':
            entered = joined[joined['rank_numeric_new'].le(10) & ~joined['rank_numeric_old'].le(10)].assign(transition='Entered top 10')
            exited = joined[joined['rank_numeric_old'].le(10) & ~joined['rank_numeric_new'].le(10)].assign(transition='Exited top 10')
            tab = pd.concat([entered, exited]).sort_values('transition')
            counts = pd.DataFrame({'transition': ['Entered top 10', 'Exited top 10'], 'institutions': [len(entered), len(exited)]})
            fig = px.bar(counts, x='transition', y='institutions', color='transition', text='institutions', title=f'Top-10 entries and exits: {a} to {b}')
            answer = f'Entered: {_names(entered.institution, 5)}. Exited: {_names(exited.institution, 5)}.'
            return _out(fig, tab[['transition', *table_columns]], {'Entered': len(entered), 'Exited': len(exited), 'From year': a, 'To year': b}, answer=answer)
        matched = joined.dropna(subset=['rank_numeric_old', 'rank_numeric_new'])
        if matched.empty:
            return _empty('No institutions could be matched across the selected years.')
        if question_id == '7.3':
            pct = 100 * (matched['rank_band_old'] == matched['rank_band_new']).mean()
            tab = matched.groupby(['rank_band_old', 'rank_band_new']).size().reset_index(name='institutions')
            matrix = tab.pivot(index='rank_band_old', columns='rank_band_new', values='institutions').fillna(0)
            matrix = matrix.reindex(index=[v for v in BAND_ORDER if v in matrix.index], columns=[v for v in BAND_ORDER if v in matrix.columns], fill_value=0)
            fig = px.imshow(matrix, text_auto=True, aspect='auto', color_continuous_scale=SEQUENTIAL_SCALE, title=f'Rank-band transitions: {a} to {b}', labels={'x': 'To band', 'y': 'From band', 'color': 'Institutions'})
            return _out(fig, tab, {'Same band percentage': round(float(pct), 2), 'Matched institutions': len(matched)},
                        answer=f'{pct:.1f}% of the {len(matched)} institutions ranked in both years stayed in the same rank band from {a} to {b}.')
        up = matched.sort_values('rank_improvement', ascending=False).head(10).assign(movement='Improved')
        down = matched.sort_values('rank_improvement').head(10).assign(movement='Declined')
        tab = pd.concat([up, down])
        fig = _horizontal(px.bar(tab, x='rank_improvement', y='institution', color='movement', orientation='h', title=f'Largest rank movements: {a} to {b}', color_discrete_map=MOVEMENT_COLORS))
        fig.update_layout(xaxis_title='Places gained (negative = lost)', yaxis_title='')
        best, worst = up.iloc[0], down.iloc[0]
        return _out(fig, tab[['movement', *table_columns, 'rank_improvement']], {'Matched institutions': len(matched)},
                    answer=f'Largest rise: {best.institution} (+{_number(best.rank_improvement)}). Largest fall: {worst.institution} ({_number(worst.rank_improvement)}).')

    if question_id == '7.5':
        changes = _consecutive_change(_ranked(d), 'rank_numeric')
        if changes.empty:
            return _empty('No consecutive year pairs have matched institutions.')
        # A lower rank number is an improvement.
        changes['movement'] = np.select([changes['change'].lt(0), changes['change'].gt(0)], ['Moved up', 'Moved down'], 'Retained rank')
        changes['period'] = (changes['year'] - 1).astype(int).astype(str) + '–' + changes['year'].astype(int).astype(str)
        tab = changes.groupby(['period', 'movement'], as_index=False).size().rename(columns={'size': 'institutions'})
        fig = px.bar(tab, x='period', y='institutions', color='movement', barmode='group', title='Institution rank movements by consecutive year',
                     color_discrete_map={'Moved up': GREEN, 'Moved down': RED, 'Retained rank': GREY})
        last = tab[tab['period'].eq(tab['period'].max())].set_index('movement')['institutions']
        return _out(fig, tab, answer=f'In {tab["period"].max()}: {int(last.get("Moved up", 0))} moved up, {int(last.get("Moved down", 0))} moved down and {int(last.get("Retained rank", 0))} kept their rank.')

    # 8. Outlier detection and performance gaps
    if question_id in ('8.1', '8.2', '8.4'):
        x = p[p['year'].eq(year)].copy()
        if x.empty:
            return _empty(f'No parameter rows for {year}. Try another year or a category with parameter scores.')
        if question_id == '8.1':
            x = x.dropna(subset=['rpc', 'score'])
            if len(x) < 3:
                return _empty('At least three parameter observations are required.')
            slope, intercept = np.polyfit(x['score'], x['rpc'], 1)
            x['rpc_residual'] = x['rpc'] - (slope * x['score'] + intercept)
            q1, q3 = x['rpc_residual'].quantile([.25, .75])
            cutoff = q3 + 1.5 * (q3 - q1)
            x['group'] = np.where(x['rpc_residual'] > cutoff, 'Unusually high research', 'Other institutions')
            fig = px.scatter(x, x='score', y='rpc', color='group', hover_name='institution', title=f'Research score relative to overall score — {year}', labels={'score': 'Overall score', 'rpc': 'RPC'},
                             color_discrete_map={'Unusually high research': RED, 'Other institutions': GREY})
            _trend_line(fig, x, 'score', 'rpc')
            tab = x[x['rpc_residual'] > cutoff].sort_values('rpc_residual', ascending=False)
            answer = (f'{len(tab)} institutions are outliers in {year}: {_names(tab.institution, 5)}.' if len(tab) else
                      f'No institution passes the outlier threshold in {year}. The closest is {x.loc[x.rpc_residual.idxmax(), "institution"]}, {x.rpc_residual.max():.2f} points above the trend.')
            return _out(fig, tab[['institution', 'year', 'score', 'rpc', 'rpc_residual']].round(2), {'Outliers': len(tab), 'Residual threshold': round(float(cutoff), 2)}, answer=answer, warning=PARAMETER_WARNING,
                        note='An outlier sits more than 1.5 × IQR above the line predicting RPC from overall score.')
        if question_id == '8.2':
            x['research_minus_teaching'] = x['rpc'] - x['tlr']
            tab = x.sort_values('research_minus_teaching', ascending=False).head(20)
            fig = _horizontal(px.bar(tab, x='research_minus_teaching', y='institution', orientation='h', title=f'Research score minus teaching score — {year}', labels={'research_minus_teaching': 'RPC − TLR'}))
            lead = tab.iloc[0]
            above = int(x['research_minus_teaching'].gt(0).sum())
            answer = (f'{above} of {len(x)} institutions score higher in research than teaching in {year}. ' +
                      (f'{lead.institution} has the largest lead, at {lead.research_minus_teaching:.2f} points.' if above else ''))
            return _out(fig, tab[['institution', 'year', 'rpc', 'tlr', 'research_minus_teaching']].round(2), {'Research above teaching': above}, answer=answer, warning=PARAMETER_WARNING)
        param = options.get('parameter', 'RPC').lower()
        x = x.dropna(subset=[param, 'rank_numeric'])
        if x.empty:
            return _empty()
        high, poor = x[param].quantile(.75), x['rank_numeric'].median()
        x['group'] = np.where((x[param] >= high) & (x['rank_numeric'] > poor), f'Strong {param.upper()}, weaker rank', 'Other institutions')
        tab = x[x['group'].ne('Other institutions')].sort_values(param, ascending=False)
        fig = px.scatter(x, x=param, y='rank_numeric', color='group', hover_name='institution', title=f'High {param.upper()} score but relatively poor rank — {year}', labels={param: param.upper(), 'rank_numeric': 'Rank'},
                         color_discrete_map={f'Strong {param.upper()}, weaker rank': RED, 'Other institutions': GREY})
        fig.update_yaxes(autorange='reversed')
        answer = f'{len(tab)} institutions qualify in {year}' + (f', led by {_names(tab.institution)}.' if len(tab) else '.')
        return _out(fig, tab[['institution', 'year', 'rank_numeric', param]].round(2), {'Institutions': len(tab), 'High threshold': round(float(high), 2), 'Poor-rank threshold': round(float(poor), 1)}, answer=answer, warning=PARAMETER_WARNING,
                    note='Strong means the top quarter for this parameter. Poor rank means worse than the median rank for the selected year.')

    if question_id == '8.3':
        tab = _consecutive_change(d, 'score')
        if tab.empty:
            return _empty('No consecutive year pairs have matched scores.')
        tab = tab.drop(columns='score_change', errors='ignore').rename(columns={'change': 'score_change', 'year': 'to_year'})
        tab['from_year'] = tab['to_year'] - 1
        q1, q3 = tab['score_change'].quantile([.25, .75])
        spread = q3 - q1
        tab['group'] = np.where((tab['score_change'] < q1 - 1.5 * spread) | (tab['score_change'] > q3 + 1.5 * spread), 'Unusual change', 'Typical change')
        unusual = tab[tab['group'].eq('Unusual change')].copy().sort_values('score_change', key=lambda s: s.abs(), ascending=False)
        fig = px.scatter(tab, x='to_year', y='score_change', color='group', hover_name='institution', hover_data=['category'], title='Year-to-year score changes and outliers', labels={'to_year': 'Year', 'score_change': 'Score change'},
                         color_discrete_map={'Unusual change': RED, 'Typical change': GREY})
        fig.update_xaxes(dtick=1)
        by_year = unusual.groupby('to_year', as_index=False).size().rename(columns={'size': 'unusual_changes', 'to_year': 'year'})
        extra = None
        if not by_year.empty:
            extra = px.bar(by_year, x='year', y='unusual_changes', title='Unusual changes by year')
            extra.update_xaxes(dtick=1)
        answer = 'No unusual changes were found.'
        if not unusual.empty:
            lead = unusual.iloc[0]
            answer = f'{len(unusual)} changes are unusual. The largest: {lead.institution} ({lead.category}), {lead.score_change:+.2f} points from {int(lead.from_year)} to {int(lead.to_year)}.'
        return _out(fig, unusual[['institution', 'category', 'state', 'from_year', 'to_year', 'score_change']].round(2), {'Unusual changes': len(unusual)}, answer=answer, extra=[extra],
                    note='Outliers use the 1.5 × IQR rule on year-to-year score changes. A cluster in one year can reflect a change in scoring across that category.')

    if question_id == '8.5':
        x = _ranked(d)
        volatility = x.groupby(['category', 'institution_id', 'institution'])['rank_numeric'].agg(years='count', rank_std='std', average_rank='mean').reset_index()
        volatility = volatility[volatility['years'] >= 3].dropna(subset=['rank_std'])
        if volatility.empty:
            return _empty('At least three ranked years per institution are required.')
        # Peers share a category and a similar average rank, because mid-table ranks move more than top ranks.
        volatility['peer_group'] = pd.cut(volatility['average_rank'], bins=[0, 25, 50, 75, np.inf], labels=['Average rank 1–25', '26–50', '51–75', '76+']).astype(str)
        peers = volatility.groupby(['category', 'peer_group'])['rank_std']
        q1, q3 = peers.transform(lambda s: s.quantile(.25)), peers.transform(lambda s: s.quantile(.75))
        volatility['peer_cutoff'] = q3 + 1.5 * (q3 - q1)
        volatility['group'] = np.where(volatility['rank_std'] > volatility['peer_cutoff'], 'Unusual for its peers', 'Within peer range')
        flagged = volatility[volatility['group'].eq('Unusual for its peers')].copy()
        flagged['excess_volatility'] = flagged['rank_std'] - flagged['peer_cutoff']
        flagged = flagged.sort_values('excess_volatility', ascending=False)
        fig = px.scatter(volatility, x='average_rank', y='rank_std', color='group', hover_name='institution', hover_data=['category', 'years', 'peer_group'], title='Rank volatility compared with similar institutions', labels={'average_rank': 'Average rank', 'rank_std': 'Rank volatility (std dev)'},
                         color_discrete_map={'Unusual for its peers': RED, 'Within peer range': GREY})
        view = volatility.sort_values('rank_std', ascending=False).head(20)
        extra = _horizontal(px.bar(view, x='rank_std', y='institution', color='group', orientation='h', title='Highest rank volatility overall', labels={'rank_std': 'Rank volatility (std dev)'},
                                   color_discrete_map={'Unusual for its peers': RED, 'Within peer range': GREY}))
        answer = (f'{len(flagged)} institutions are unusually volatile for their peer group, led by {_names(flagged.institution)}.' if len(flagged) else 'No institution is unusually volatile for its peer group.')
        table = (flagged if not flagged.empty else view)[['category', 'institution', 'peer_group', 'years', 'average_rank', 'rank_std', 'peer_cutoff']].round(2)
        return _out(fig, table, {'Unusual': len(flagged), 'Institutions with 3+ years': len(volatility)}, answer=answer, extra=[extra],
                    note='Volatility is the standard deviation of an institution\'s ranks. Similar institutions share a category and a similar average rank. Unusual means more than 1.5 × IQR above that peer group.')

    return _empty(f'Question handler {question_id} has not been implemented.')
