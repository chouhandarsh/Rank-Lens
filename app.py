from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SIDEBAR_LOGO_PATH = ROOT / 'chatbot' / 'assets' / 'nirf_logo.png'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CHATBOT_SOURCE = ROOT / 'chatbot' / 'src'
if str(CHATBOT_SOURCE) not in sys.path:
    sys.path.insert(0, str(CHATBOT_SOURCE))

from nirf.data_loader import load_data, dataset_summary
from nirf.filters import (
    apply_filters,
    build_institution_options,
    build_state_options,
)
from nirf.questions import QUESTIONS
from nirf.level1 import run_level1
from nirf.custom_analysis import run_custom_analysis
from nirf.custom_ui import build_config, scope_note
from nirf.score_chart import top_score_chart_data
from nirf.maps import (
    build_state_metrics,
    report_state_coverage,
    render_choropleth,
    show_map_attribution,
)
from nirf.theme import DIVERGING_SCALE, apply_theme
from nirf_chatbot.streamlit_ui import render_chat_page

st.set_page_config(page_title='NIRF Analytics', page_icon='📊', layout='wide', initial_sidebar_state='expanded')

apply_theme()


@st.cache_data(show_spinner='Loading NIRF datasets...')
def get_data():
    return load_data()


def reset_global_filters():
    for key in ['global_year_range','global_categories','global_states','global_institutions','global_rank_range']:
        st.session_state.pop(key, None)

try:
    full_df = get_data()
except Exception as exc:
    st.error(f'Could not load the processed NIRF dataset: {exc}')
    st.info('From the project root, run: `python scripts/build_data.py`')
    st.stop()

summary = dataset_summary(full_df)
min_year, max_year = int(full_df.year.min()), int(full_df.year.max())
HIDDEN_FILTER_CATEGORIES = {'Special Mention (2018)'}
all_categories = sorted(set(full_df.category.dropna()) - HIDDEN_FILTER_CATEGORIES)
# A session may still hold a hidden category selected before it was removed.
if 'global_categories' in st.session_state:
    st.session_state['global_categories'] = [
        category for category in st.session_state['global_categories']
        if category in all_categories
    ]
all_states = build_state_options(full_df)
institution_options = build_institution_options(full_df)
institution_labels = dict(zip(institution_options.institution_id, institution_options.label))
all_institution_ids = institution_options.institution_id.tolist()
rank_max_data = int(full_df.rank_numeric.max()) if full_df.rank_numeric.notna().any() else 200
rank_max_data = max(rank_max_data, 100)

# Sidebar: shared filters applied to both Level 1 and Level 2.
with st.sidebar:
    if SIDEBAR_LOGO_PATH:
        st.sidebar.image(str(SIDEBAR_LOGO_PATH), width=96)
    st.title('NIRF Analytics')
    st.caption('Ranking data exploration · 2017–2025')
    page = st.radio('Navigate', ['Overview', 'Level 1 · Sample Questions', 'Level 2 · Custom Analysis', 'Chatbot · AI Analysis'], index=0)
    st.divider()
    st.subheader('Global filters')
    default_years = (max(min_year, 2017), max_year)
    year_range = st.slider('Year range', min_value=min_year, max_value=max_year,
                           value=default_years, step=1, key='global_year_range')
    selected_categories = st.multiselect('NIRF categories', options=all_categories,
                                         default=['Overall'] if 'Overall' in all_categories else all_categories[:1],
                                         key='global_categories', help='Choose one or more ranking categories. Default is Overall; choose multiple categories for cross-category analysis.')
    selected_states = st.multiselect('States', options=all_states, default=[], key='global_states',
                                     help='Leave empty to include all states.')
    selected_institution_ids = st.multiselect(
        'Institutions (optional)',
        options=all_institution_ids,
        format_func=lambda institution_id: institution_labels[institution_id],
        default=[],
        key='global_institutions',
        help='Leave empty to include all institutions.',
    )
    rank_range = st.slider('Rank range', min_value=1, max_value=rank_max_data,
                           value=(1, rank_max_data), step=1, key='global_rank_range')
    include_review = False  # No rows are review-flagged since parameters moved to the Tableau file.
    st.button('Reset all filters', use_container_width=True, on_click=reset_global_filters)
    st.divider()
    st.caption('Dataset status')
    st.metric('Filtered ranking records', f'{len(apply_filters(full_df, year_range, selected_categories, selected_states, selected_institution_ids, rank_range)):,}')
    st.caption(f"Source: {summary['records']:,} category-year records · {summary['categories']} categories · {min_year}–{max_year}")

filtered_df = apply_filters(full_df, year_range, selected_categories, selected_states,
                            selected_institution_ids, rank_range)
all_categories_filtered = apply_filters(full_df, year_range, None, selected_states,
                                        selected_institution_ids, rank_range, ignore_categories=True)
history_filtered_df = apply_filters(full_df, year_range, selected_categories, selected_states,
                                    selected_institution_ids, None)


def show_metrics(metrics: dict):
    if not metrics: return
    items = list(metrics.items())
    cols = st.columns(min(4, len(items)))
    for i, (label, value) in enumerate(items):
        with cols[i % len(cols)]:
            st.metric(str(label), '—' if value is None else str(value))


def show_result(result: dict):
    if result.get('answer'):
        st.success(f"**Answer:** {result['answer']}")
    if result.get('warning'):
        st.warning(result['warning'])
    if result.get('note'):
        st.caption(result['note'])
    show_metrics(result.get('metrics', {}))
    figures = [result.get('figure'), *result.get('extra_figures', [])]
    for position, figure in enumerate(item for item in figures if item is not None):
        st.plotly_chart(figure, use_container_width=True, theme='streamlit', key=f'result_chart_{position}')
    if result.get('is_map'):
        show_map_attribution()
    if result.get('table') is not None:
        table = result['table']
        if isinstance(table, pd.DataFrame) and not table.empty:
            with st.expander('View underlying data', expanded=True):
                st.dataframe(table, use_container_width=True, hide_index=True)
        elif isinstance(table, pd.DataFrame):
            st.info('No matching rows were found for this analysis.')


def render_level1_geographic_maps(qid: str, question_data: pd.DataFrame,
                                  full_data: pd.DataFrame, options: dict):
    map_categories = sorted(
        question_data.category.dropna().unique().tolist()
    )
    if not map_categories:
        st.info('No categories remain for the geographic view.')
        return
    if len(map_categories) > 1:
        map_category = st.selectbox(
            'Category for map',
            map_categories,
            key=f'level1_{qid.replace(".", "")}_map_category',
        )
    else:
        map_category = map_categories[0]

    map_data = question_data[question_data.category.eq(map_category)]
    map_years = sorted(map_data.year.dropna().astype(int).unique().tolist())
    if not map_years:
        st.info('No years remain for the selected category.')
        return

    report_state_coverage(full_data)
    if qid == '2.1':
        map_view = st.selectbox(
            'Map view',
            ['Count in top 100', 'Share of top 100', 'Average score'],
            key='level1_21_map_view',
        )
        map_year = st.selectbox(
            'Map year',
            map_years,
            index=len(map_years) - 1,
            key='level1_21_map_year',
        )
        top_rows = map_data[
            pd.to_numeric(map_data.rank_numeric, errors='coerce').le(100)
        ]
        value_col = {
            'Count in top 100': 'institutions_in_top_n',
            'Share of top 100': 'share_of_top_n',
            'Average score': 'average_score',
        }[map_view]
        metric_source = (
            top_rows if map_view == 'Average score' else map_data
        )
        metric = (
            'average_score'
            if map_view == 'Average score'
            else value_col
        )
        metrics = build_state_metrics(
            metric_source,
            metric,
            year=map_year,
            category=map_category,
            top_n=100,
        )
        figure = render_choropleth(
            metrics,
            metric,
            f'{map_view} by state in {map_year} ({map_category})',
            key='level1_21_state_map',
        )
        if figure is not None:
            show_map_attribution()
        return

    if qid == '2.2':
        map_view = st.selectbox(
            'Map view',
            [
                'Single year',
                'Compare two years side by side',
                'Animated by year',
            ],
            key='level1_22_map_view',
        )
        if map_view == 'Single year':
            map_year = st.selectbox(
                'Map year',
                map_years,
                index=len(map_years) - 1,
                key='level1_22_map_year',
            )
            metrics = build_state_metrics(
                map_data,
                'institutions_in_top_n',
                year=map_year,
                category=map_category,
                top_n=100,
            )
            figure = render_choropleth(
                metrics,
                'institutions_in_top_n',
                f'Top-100 institutions by state in {map_year} ({map_category})',
                key='level1_22_single_map',
            )
            if figure is not None:
                show_map_attribution()
        elif len(map_years) < 2:
            st.info('Select at least two years in the global year range.')
        elif map_view == 'Compare two years side by side':
            map_controls = st.columns(2)
            with map_controls[0]:
                from_year = st.selectbox(
                    'From year',
                    map_years[:-1],
                    index=0,
                    key='level1_22_from_year',
                )
            to_year_options = [year for year in map_years if year > from_year]
            with map_controls[1]:
                to_year = st.selectbox(
                    'To year',
                    to_year_options,
                    index=len(to_year_options) - 1,
                    key='level1_22_to_year',
                )
            first = build_state_metrics(
                map_data,
                'institutions_in_top_n',
                year=from_year,
                category=map_category,
                top_n=100,
            )
            second = build_state_metrics(
                map_data,
                'institutions_in_top_n',
                year=to_year,
                category=map_category,
                top_n=100,
            )
            max_value = max(
                float(first.institutions_in_top_n.max()),
                float(second.institutions_in_top_n.max()),
            )
            color_range = (0, max_value) if max_value > 0 else None
            map_columns = st.columns(2)
            with map_columns[0]:
                first_figure = render_choropleth(
                    first,
                    'institutions_in_top_n',
                    f'Top-100 institutions by state in {from_year}',
                    key='level1_22_compare_from_map',
                    color_range=color_range,
                )
            with map_columns[1]:
                second_figure = render_choropleth(
                    second,
                    'institutions_in_top_n',
                    f'Top-100 institutions by state in {to_year}',
                    key='level1_22_compare_to_map',
                    color_range=color_range,
                )
            if first_figure is not None or second_figure is not None:
                show_map_attribution()
        else:
            year_frames = []
            for map_year in map_years:
                metrics = build_state_metrics(
                    map_data,
                    'institutions_in_top_n',
                    year=map_year,
                    category=map_category,
                    top_n=100,
                )
                metrics['year'] = map_year
                year_frames.append(metrics)
            animated = pd.concat(year_frames, ignore_index=True)
            max_value = float(animated.institutions_in_top_n.max())
            figure = render_choropleth(
                animated,
                'institutions_in_top_n',
                f'Top-100 institutions by state over time ({map_category})',
                key='level1_22_animated_map',
                color_range=(0, max_value) if max_value > 0 else None,
                animation_frame='year',
            )
            if figure is not None:
                show_map_attribution()
        return

    if qid == '2.3':
        if len(map_years) < 2:
            st.info('Select at least two years in the global year range.')
            return
        from_year = options.get('from_year')
        to_year = options.get('to_year')
        if (
            from_year not in map_years
            or to_year not in map_years
            or from_year >= to_year
        ):
            from_year, to_year = map_years[0], map_years[-1]
        metrics = build_state_metrics(
            map_data,
            'change_in_count',
            year=(from_year, to_year),
            category=map_category,
            top_n=100,
        )
        figure = render_choropleth(
            metrics,
            'change_in_count',
            f'Top-100 representation change by state: {from_year} to {to_year}',
            key='level1_23_change_map',
            diverging=True,
        )
        if figure is not None:
            show_map_attribution()
            import plotly.express as px

            gainers = metrics.nlargest(10, 'change_in_count')
            losers = metrics.nsmallest(10, 'change_in_count')
            gain_loss = pd.concat([gainers, losers]).drop_duplicates('state')
            gain_loss = gain_loss.sort_values('change_in_count')
            bar = px.bar(
                gain_loss,
                x='change_in_count',
                y='state',
                color='change_in_count',
                color_continuous_scale=DIVERGING_SCALE,
                color_continuous_midpoint=0,
                orientation='h',
                title=f'Top state gainers and losers: {from_year} to {to_year}',
                labels={
                    'change_in_count': 'Change in top-100 institutions',
                    'state': 'State',
                },
            )
            st.plotly_chart(
                bar,
                use_container_width=True,
                theme='streamlit',
                key='level1_23_gainers_losers',
            )
        return

    if qid == '2.4':
        states_pair = options.get('states_pair') or [
            'Karnataka',
            'Tamil Nadu',
        ]
        counts = build_state_metrics(
            map_data,
            'institutions_in_top_n',
            year=None,
            category=map_category,
            top_n=100,
        )
        highlights = counts[counts.state.isin(states_pair)].copy()
        highlights['selected_state'] = 1
        highlights['institution_count'] = highlights[
            'institutions_in_top_n'
        ]
        figure = render_choropleth(
            highlights,
            'selected_state',
            'Selected states: ' + ' and '.join(states_pair),
            key='level1_24_selected_states_map',
            color_range=(0, 1),
        )
        if figure is not None:
            show_map_attribution()
        return

    if qid == '2.5':
        if len(map_years) < 2:
            st.info('Select a year range with at least two years.')
            return
        year_controls = st.columns(2)
        with year_controls[0]:
            from_year = st.selectbox(
                'First year',
                map_years[:-1],
                index=0,
                key='level1_25_from_year',
            )
        to_year_options = [year for year in map_years if year > from_year]
        with year_controls[1]:
            to_year = st.selectbox(
                'Last year',
                to_year_options,
                index=len(to_year_options) - 1,
                key='level1_25_to_year',
            )
        year_frames = []
        for map_year in range(from_year, to_year + 1):
            if map_year not in map_years:
                continue
            metrics = build_state_metrics(
                map_data,
                'new_institutions',
                year=map_year,
                category=map_category,
                history_df=full_data,
            )
            year_frames.append(metrics)
        combined = (
            pd.concat(year_frames, ignore_index=True)
            .groupby('state', as_index=False)
            .agg(
                new_institutions=('new_institutions', 'sum'),
                institution_count=('institution_count', 'sum'),
            )
        )
        figure = render_choropleth(
            combined,
            'new_institutions',
            f'New institutions first observed: {from_year} to {to_year} ({map_category})',
            key='level1_25_new_institutions_map',
        )
        if figure is not None:
            show_map_attribution()


def render_state_score_map(question_data: pd.DataFrame):
    scores = question_data.dropna(subset=['score'])
    if scores.empty:
        st.info('No scored records remain after applying the global filters.')
        return
    state_metrics = scores.groupby('state', as_index=False).agg(
        median_score=('score', 'median'),
        institution_count=('institution_id', 'nunique'),
    )
    figure = render_choropleth(
        state_metrics,
        'median_score',
        'Median score by state',
        key='level1_65_score_map',
    )
    if figure is not None:
        st.caption('The map shows the median score; use Chart view for the full score distributions.')
        show_map_attribution()


def render_question(qid: str):
    """Controls and results for one Annexure A question."""
    qmeta = next(item for item in QUESTIONS if item[0] == qid)
    st.markdown(f'**Question {qid}**  {qmeta[2]}')

    options = {'include_review': include_review}
    years = sorted(filtered_df.year.dropna().astype(int).unique())
    if qid == '1.1':
        choices = build_institution_options(filtered_df)
        if not choices.empty:
            ids = choices.institution_id.tolist()
            labels = dict(zip(ids, choices.label))
            matching = choices.loc[choices.institution.eq('Indian Institute of Technology Madras'), 'institution_id']
            default_id = matching.iloc[0] if not matching.empty else ids[0]
            options['institution_id'] = st.selectbox(
                'Institution', ids, index=ids.index(default_id),
                format_func=lambda institution_id: labels[institution_id],
            )
    if qid == '2.4':
        states = sorted(state for state in filtered_df.state.dropna().unique() if state)
        c1, c2 = st.columns(2)
        if states:
            s1 = 'Karnataka' if 'Karnataka' in states else states[0]
            s2 = 'Tamil Nadu' if 'Tamil Nadu' in states else states[min(1, len(states) - 1)]
            with c1:
                a = st.selectbox('First state', states, index=states.index(s1))
            with c2:
                b = st.selectbox('Second state', states, index=states.index(s2))
            options['states_pair'] = [a, b]
    if qid in {'2.1', '3.1', '3.3', '3.4', '5.1', '5.2', '5.3', '5.5', '6.3', '8.1', '8.2', '8.4'} and years:
        options['year'] = st.selectbox('Year for this analysis', years, index=len(years) - 1)
    if qid in {'1.2', '1.3', '2.3', '2.5', '7.1', '7.2', '7.3', '7.4'} and len(years) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            default_from_index = len(years) - 2 if qid in {'7.2', '7.3'} else 0
            options['from_year'] = st.selectbox('From year', years, index=default_from_index)
        with c2:
            if qid in {'7.2', '7.3'}:
                next_years = [year for year in years if year == options['from_year'] + 1]
                if next_years:
                    options['to_year'] = st.selectbox('To year (consecutive)', next_years, index=0)
                else:
                    st.info('No consecutive next year is available in the current selection.')
            else:
                options['to_year'] = st.selectbox('To year', years, index=len(years) - 1)
    if qid == '8.4':
        options['parameter'] = st.selectbox('Parameter', ['TLR', 'RPC', 'GO', 'OI', 'PR'], index=1)

    st.divider()
    if qmeta[3] and not filtered_df['parameters_available'].any():
        st.info('This question requires parameter scores. Research, Innovation and rank-only categories have no standard parameter set; select another category in the sidebar.')
        return
    if qmeta[3]:
        st.caption('Parameter analyses use one category at a time: Overall when selected, otherwise the first selected category alphabetically.')
    question_data = history_filtered_df if qid in {'1.2', '1.3', '7.1', '7.2', '7.3', '7.4', '7.5'} else filtered_df
    result = run_level1(qid, question_data, all_categories_filtered, full_df, options)
    if qid == '6.5':
        view_as = st.selectbox('View as', ['Chart', 'Map'], key='level1_65_view_as')
        if view_as == 'Map':
            map_result = dict(result)
            map_result['figure'] = None
            show_result(map_result)
            render_state_score_map(question_data)
        else:
            show_result(result)
        return
    show_result(result)
    if qid in {'2.1', '2.2', '2.3', '2.4', '2.5'}:
        st.subheader('Geographic view')
        render_level1_geographic_maps(qid, question_data, full_df, options)


if page == 'Overview':
    st.title('NIRF Rankings Analytics Dashboard')
    st.markdown('<p class="page-subtitle">Explore institution rankings, categories, geography, score patterns and ranking mobility.</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Ranking records', f'{len(filtered_df):,}')
    c2.metric('Categories selected', f'{filtered_df.category.nunique():,}')
    c3.metric('Institutions selected', f'{filtered_df.institution_key.nunique():,}')
    valid_scores=filtered_df.score.dropna()
    c4.metric('Average score', f'{valid_scores.mean():.2f}' if not valid_scores.empty else '—')

    left,right=st.columns([1.4,1])
    with left:
        st.subheader('Average score by year')
        score_trend=filtered_df.dropna(subset=['score']).groupby('year',as_index=False).score.mean()
        import plotly.express as px
        fig=px.line(score_trend,x='year',y='score',markers=True,title='Average score trend')
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader('Records by category')
        cat_counts=filtered_df.groupby('category',as_index=False).size().sort_values('size',ascending=False)
        fig=px.bar(cat_counts,x='category',y='size',title='Ranking records by category')
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig,use_container_width=True)

    left,right=st.columns(2)
    with left:
        st.subheader('Top institutions by score')
        single_year_selected = year_range[0] == year_range[1]
        top = top_score_chart_data(
            filtered_df,
            full_df,
            single_year_selected=single_year_selected,
        )
        top=top.sort_values('score',ascending=False).groupby('category').head(10).head(20)
        hover_fields = ['institution_id', 'year'] if single_year_selected else [
            'institution_id', 'years_averaged'
        ]
        fig=px.bar(top,x='score',y='display_institution',color='category',orientation='h',
                   hover_data=hover_fields,title='Top scored records in current selection')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, barmode='group')
        fig.update_xaxes(
            range=[0,100],
            title_text='Score' if single_year_selected else 'Average score (selected years)',
        )
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader('State-wise top-100 representation')
        top100=filtered_df[filtered_df.rank_numeric.le(100)].groupby('state',as_index=False).size().sort_values('size',ascending=False).head(15)
        fig=px.bar(top100,x='state',y='size',title='Top-100 ranking entries by state')
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig,use_container_width=True)

    st.divider()
    st.subheader('Geographic view')
    map_view = st.selectbox(
        'Map view',
        [
            'Institutions in top N by state',
            'Average score by state',
            'Best rank by state',
            'Newly appearing institutions by state',
            'Change in representation between two years',
        ],
        key='overview_map_view',
    )
    if filtered_df.empty:
        st.info('No geographic data remains after applying the global filters.')
    else:
        map_categories = sorted(filtered_df.category.dropna().unique().tolist())
        if len(map_categories) > 1:
            map_category = st.selectbox(
                'Category for map',
                map_categories,
                key='overview_map_category',
            )
        elif map_categories:
            map_category = map_categories[0]
        else:
            map_category = None

        if map_category is None:
            st.info('Select a category in the global filters to build a state map.')
        else:
            map_data = filtered_df[filtered_df.category.eq(map_category)]
            map_years = sorted(
                map_data.year.dropna().astype(int).unique().tolist()
            )
            if not map_years:
                st.info('No years remain for the selected map category.')
            else:
                report_state_coverage(full_df)
                top_n = 100
                if map_view in {
                    'Institutions in top N by state',
                    'Change in representation between two years',
                }:
                    top_n = st.selectbox(
                        'Top N',
                        [10, 50, 100],
                        index=2,
                        key='overview_map_top_n',
                    )

                if map_view == 'Change in representation between two years':
                    if len(map_years) < 2:
                        st.info('Select at least two years in the global year range.')
                    else:
                        map_controls = st.columns(2)
                        with map_controls[0]:
                            from_year = st.selectbox(
                                'From year',
                                map_years[:-1],
                                index=0,
                                key='overview_map_from_year',
                            )
                        available_to_years = [
                            value for value in map_years if value > from_year
                        ]
                        with map_controls[1]:
                            to_year = st.selectbox(
                                'To year',
                                available_to_years,
                                index=len(available_to_years) - 1,
                                key='overview_map_to_year',
                            )
                        map_metrics = build_state_metrics(
                            map_data,
                            'change_in_count',
                            year=(from_year, to_year),
                            category=map_category,
                            top_n=top_n,
                        )
                        value_col = 'change_in_count'
                        map_title = (
                            f'Top-{top_n} representation change by state: '
                            f'{from_year} to {to_year} ({map_category})'
                        )
                        map_color_range = None
                        diverging = True
                else:
                    map_year = st.selectbox(
                        'Map year',
                        map_years,
                        index=len(map_years) - 1,
                        key='overview_map_year',
                    )
                    metric_by_view = {
                        'Institutions in top N by state': 'institutions_in_top_n',
                        'Average score by state': 'average_score',
                        'Best rank by state': 'best_rank',
                        'Newly appearing institutions by state': 'new_institutions',
                    }
                    value_col = metric_by_view[map_view]
                    map_metrics = build_state_metrics(
                        map_data,
                        value_col,
                        year=map_year,
                        category=map_category,
                        top_n=top_n,
                        history_df=full_df,
                    )
                    map_title = f'{value_col.replace("_", " ").title()} by state in {map_year} ({map_category})'
                    map_color_range = None
                    diverging = False
                    if value_col == 'new_institutions' and map_year == int(full_df.year.min()):
                        st.info(
                            'The first year in the dataset is a baseline; '
                            'new entrants cannot be identified for that year.'
                        )

                if map_view == 'Change in representation between two years' and len(map_years) < 2:
                    pass
                else:
                    figure = render_choropleth(
                        map_metrics,
                        value_col,
                        map_title,
                        key='overview_geographic_map',
                        diverging=diverging,
                        color_range=map_color_range,
                    )
                    if figure is not None:
                        show_map_attribution()
                        ascending = value_col == 'best_rank'
                        state_table = (
                            map_metrics[['state', value_col]]
                            .sort_values(value_col, ascending=ascending)
                            .head(10)
                            .rename(
                                columns={
                                    'state': 'State',
                                    value_col: value_col.replace('_', ' ').title(),
                                }
                            )
                        )
                        st.dataframe(
                            state_table,
                            use_container_width=True,
                            hide_index=True,
                        )

    with st.expander('Data coverage and quality notes'):
        st.markdown('''
    - The dataset has 6,980 ranking records across 18 category labels and 2017–2025: 6,734 from the clean master, plus 246 from six categories found only in the Tableau file.
    - Parameter scores come from the Tableau file and cover every category that publishes them. 6,673 rows have TLR, RPC, GO, OI and PR. The 250 Research rows have SFC, QNR, QLR, OI and PR instead.
    - Parameter weights can differ by category, so parameter analyses use one category at a time.
    - 57 records have no score: 52 rank-only listings (Innovation, Open University, Skill University and SDG), four 2018 Special Mentions and one withdrawn record. Two ranks are withdrawn. Null scores are excluded from score statistics.
    - The data uses 33 canonical state names. `Institution_ID` is the identity key across years and categories; never join or group institutions by name because distinct IDs can share a name.
    - Do not interpret absent category-year combinations as zero rankings.
    ''')
        quality_path=ROOT/'data'/'reports'/'data_quality_report.csv'
        if quality_path.exists():
            st.dataframe(pd.read_csv(quality_path),use_container_width=True,hide_index=True)

elif page == 'Level 1 · Sample Questions':
    st.title('Level 1 · Sample Questions')
    st.markdown('<p class="page-subtitle">Select any of the 40 questions from the project brief. Each one gives a written answer, charts and the supporting table.</p>', unsafe_allow_html=True)
    question_labels = [f"{qid} · {section} — {question}" for qid, section, question, _ in QUESTIONS]
    label = st.selectbox('Choose an analytical question', question_labels, key='level1_question')
    render_question(label.split(' · ', 1)[0])

elif page == 'Level 2 · Custom Analysis':
    st.title('Level 2 · Analysis Dashboard')
    st.markdown('<p class="page-subtitle">Answer the Annexure A questions, or build your own comparison, trend, forecast or India map from any attribute.</p>', unsafe_allow_html=True)
    workspace = st.radio('Workspace', ['Custom analysis', 'Annexure A questions'], horizontal=True, key='level2_workspace', label_visibility='collapsed')
    if workspace == 'Annexure A questions':
        sections = list(dict.fromkeys(item[1] for item in QUESTIONS))
        section = st.selectbox('Section', sections, key='level2_section')
        section_questions = {f'{item[0]} — {item[2]}': item[0] for item in QUESTIONS if item[1] == section}
        chosen = st.selectbox('Question', list(section_questions), key=f'level2_question_{sections.index(section)}')
        render_question(section_questions[chosen])
    else:
        with st.container(border=True):
            config, custom_data = build_config(filtered_df, history_filtered_df, all_categories_filtered, include_review)
        message = scope_note(custom_data, filtered_df)
        if message:
            st.caption(message)
        show_result(run_custom_analysis(custom_data, config, full_df))

elif page == 'Chatbot · AI Analysis':
    render_chat_page(
        title='NIRF Analytics Chatbot',
        subtitle='Ask natural-language questions and explore chatbot results, charts, and plotting code.',
    )
