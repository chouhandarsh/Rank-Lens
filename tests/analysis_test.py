"""Checks that all 40 questions answer correctly and every Level 2 option draws."""
from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nirf.custom_analysis import (
    COMPARE_CHARTS,
    CORRELATION_CHARTS,
    DISTRIBUTION_CHARTS,
    GROUP_FIELDS,
    MAP_COUNT_METRICS,
    MAP_VIEWS,
    MEASURES,
    NUMERIC_MEASURES,
    PROFILE_CHARTS,
    TRANSITION_CHARTS,
    TREND_CHARTS,
    _forecast_series,
    run_custom_analysis,
)
from nirf.data_loader import load_data
from nirf.filters import apply_filters, parameter_rows
from nirf.level1 import run_level1
from nirf.questions import QUESTIONS

FULL = load_data()
OVERALL = apply_filters(FULL, (2017, 2025), ['Overall'], None, None, None)
ALL_CATEGORIES = apply_filters(FULL, (2017, 2025), None, None, None, None, ignore_categories=True)
OPTIONS = {'year': 2025, 'from_year': 2017, 'to_year': 2025, 'states_pair': ['Karnataka', 'Tamil Nadu'], 'parameter': 'RPC', 'include_review': False}


def ask(qid, **changes):
    return run_level1(qid, OVERALL, ALL_CATEGORIES, FULL, {**OPTIONS, **changes})


def custom(data=OVERALL, **config):
    return run_custom_analysis(data, config, FULL)


class QuestionTests(unittest.TestCase):
    def test_every_question_has_a_written_answer_and_a_chart(self):
        for qid, *_ in QUESTIONS:
            with self.subTest(qid=qid):
                options = {'from_year': 2024, 'to_year': 2025} if qid in {'7.2', '7.3'} else {}
                result = ask(qid, **options)
                self.assertIsNotNone(result['figure'], result['note'])
                self.assertTrue(result['answer'], 'No written answer')
                self.assertNotIn('nan', result['answer'].casefold().replace('finance', ''))

    def test_every_question_handles_empty_and_narrow_selections(self):
        narrow = apply_filters(FULL, (2025, 2025), ['Overall'], ['Goa'], None, None)
        for qid, *_ in QUESTIONS:
            with self.subTest(qid=qid):
                run_level1(qid, narrow, narrow, FULL, dict(OPTIONS))
                run_level1(qid, OVERALL.iloc[0:0], ALL_CATEGORIES, FULL, dict(OPTIONS))

    def test_top_100_state_count_uses_one_year_not_nine(self):
        result = ask('2.1')
        expected = OVERALL[OVERALL.year.eq(2025) & OVERALL.rank_numeric.le(100)].groupby('state').institution_id.nunique()
        table = result['table'].set_index('state')['institutions_in_top_100']
        self.assertEqual(table.sum(), expected.sum())
        self.assertEqual(table.max(), expected.max())
        self.assertLessEqual(table.sum(), 101)

    def test_new_institutions_exclude_the_baseline_year(self):
        table = ask('2.5')['table']
        first_seen = FULL[FULL.category.eq('Overall')].groupby('institution_id').year.min()
        self.assertEqual(table.new_institutions.sum(), first_seen[first_seen.gt(2017)].shape[0])

    def test_state_comparison_title_follows_the_selection(self):
        result = ask('2.4', states_pair=['Delhi', 'Punjab'])
        self.assertIn('Delhi vs Punjab', result['figure'].layout.title.text)

    def test_score_quartiles_change_over_time_and_total_100(self):
        result = ask('3.4')
        figure = result['extra_figures'][0]
        self.assertIsNotNone(figure)
        self.assertRegex(result['answer'], r'Q4 \(highest\) \d')
        shares = [float(value) for value in result['figure'].data[0].y]
        self.assertAlmostEqual(sum(shares), 100, delta=.1)
        self.assertGreater(max(shares) - min(shares), 1, 'Shares should not all be 25%')

    def test_score_summary_matches_the_data(self):
        scores = OVERALL[OVERALL.year.eq(2025)].score.dropna()
        metrics = ask('3.3')['metrics']
        self.assertEqual(metrics['Median'], round(scores.median(), 2))
        self.assertEqual(metrics['Maximum'], round(scores.max(), 2))
        self.assertEqual(metrics['Minimum'], round(scores.min(), 2))

    def test_univariate_questions_show_trends_or_multiple_views(self):
        for qid in ('3.1', '3.2', '3.4', '3.5'):
            with self.subTest(qid=qid):
                self.assertTrue(ask(qid)['extra_figures'])
        self.assertGreater(len(ask('3.3')['figure'].data), 1)

    def test_rank_improvement_matches_the_data(self):
        left = OVERALL[OVERALL.year.eq(2017)].set_index('institution_id').rank_numeric
        right = OVERALL[OVERALL.year.eq(2025)].set_index('institution_id').rank_numeric
        gain = (left - right).dropna()
        result = ask('1.2')
        self.assertEqual(result['table'].rank_improvement.iloc[0], gain.max())
        self.assertEqual(ask('1.3')['table'].rank_improvement.iloc[0], gain.min())

    def test_top_10_entries_and_exits_include_unranked_institutions(self):
        result = ask('7.2', from_year=2024, to_year=2025)
        old = set(OVERALL[OVERALL.year.eq(2024) & OVERALL.rank_numeric.le(10)].institution_id)
        new = set(OVERALL[OVERALL.year.eq(2025) & OVERALL.rank_numeric.le(10)].institution_id)
        self.assertEqual(result['metrics']['Entered'], len(new - old))
        self.assertEqual(result['metrics']['Exited'], len(old - new))

    def test_strongest_parameter_matches_the_data(self):
        rows = parameter_rows(OVERALL)
        values = {name.upper(): rows[name].corr(rows.score) for name in ['tlr', 'rpc', 'go', 'oi', 'pr']}
        expected = max(values, key=lambda name: abs(values[name]))
        self.assertEqual(ask('4.3')['metrics']['Strongest parameter'], expected)

    def test_movement_counts_are_in_the_right_direction(self):
        table = ask('7.5')['table']
        pair = OVERALL[OVERALL.year.isin([2024, 2025])].pivot_table(index='institution_id', columns='year', values='rank_numeric').dropna()
        up = table[table.period.eq('2024–2025') & table.movement.eq('Moved up')].institutions.iloc[0]
        self.assertEqual(up, (pair[2025] < pair[2024]).sum())


class CustomAnalysisTests(unittest.TestCase):
    def assertDraws(self, result, context):
        self.assertIsNotNone(result['figure'], f'{context}: {result.get("note")}')
        result['figure'].to_dict()
        for figure in result['extra_figures']:
            figure.to_dict()

    def test_compare_every_measure_group_and_chart(self):
        for (label, measure), group, chart in product(MEASURES.items(), GROUP_FIELDS, COMPARE_CHARTS):
            data = ALL_CATEGORIES if GROUP_FIELDS[group] in {'category', 'Category_Group'} else OVERALL
            with self.subTest(measure=label, group=group, chart=chart):
                self.assertDraws(custom(data, mode='Compare groups', measure=measure, group_by=group, aggregation='Mean', chart=chart, top_n=10), (label, group, chart))

    def test_compare_with_a_split(self):
        for chart in COMPARE_CHARTS:
            with self.subTest(chart=chart):
                self.assertDraws(custom(ALL_CATEGORIES, mode='Compare groups', measure='score', group_by='State', split_by='Category', aggregation='Median', chart=chart, top_n=6), chart)

    def test_compare_answers_use_the_right_direction_for_rank(self):
        answer = custom(mode='Compare groups', measure='rank_numeric', group_by='State', aggregation='Minimum', chart='Bar', top_n=40)['answer']
        self.assertIn('best', answer)
        self.assertIn('(1.00)', answer)

    def test_grouping_by_category_compares_every_scored_category(self):
        result = custom(ALL_CATEGORIES, mode='Compare groups', measure='score', group_by='Category', aggregation='Mean', chart='Bar')
        self.assertEqual(len(result['table']), FULL[FULL.score.notna()].category.nunique())
        self.assertEqual(len(result['table']), 13)

    def test_specific_groups_can_be_chosen(self):
        result = custom(mode='Compare groups', measure='score', group_by='State', aggregation='Mean', chart='Bar', selected_groups=['Goa', 'Kerala', 'Delhi'])
        self.assertLessEqual(len(result['table']), 3)
        self.assertGreaterEqual(len(result['table']), 2)

    def test_trend_every_measure_group_and_chart(self):
        for (label, measure), group, chart in product(MEASURES.items(), ['No grouping', *GROUP_FIELDS], TREND_CHARTS):
            if group == 'Year':
                continue
            with self.subTest(measure=label, group=group, chart=chart):
                self.assertDraws(custom(mode='Trend over time', measure=measure, group_by=group, aggregation='Mean', chart=chart, top_n=5, show_trendline=True), (label, group, chart))

    def test_forecast_matches_a_separate_calculation(self):
        years = np.arange(2017, 2026)
        values = 40 + 2.0 * (years - 2017)
        values[3] += 1
        fit = _forecast_series(years, values, 2, 'score')
        slope, intercept = np.polyfit(years, values, 1)
        self.assertAlmostEqual(fit['predicted'][0], slope * 2026 + intercept)
        self.assertAlmostEqual(fit['predicted'][1], slope * 2027 + intercept)
        self.assertTrue((fit['low'] < fit['predicted']).all() and (fit['predicted'] < fit['high']).all())
        self.assertGreater(fit['high'][1] - fit['low'][1], fit['high'][0] - fit['low'][0])
        self.assertGreater(fit['r_squared'], .95)

    def test_forecast_stays_within_valid_bounds(self):
        fit = _forecast_series(np.arange(2017, 2026), np.linspace(50, 3, 9), 5, 'rank_numeric')
        self.assertGreaterEqual(fit['low'].min(), 1)
        fit = _forecast_series(np.arange(2017, 2026), np.linspace(70, 99, 9), 5, 'score')
        self.assertLessEqual(fit['high'].max(), 100)

    def test_forecast_every_measure_and_group(self):
        for (label, measure), group in product(MEASURES.items(), ['No grouping', *GROUP_FIELDS]):
            if group == 'Year':
                continue
            with self.subTest(measure=label, group=group):
                self.assertDraws(custom(mode='Forecast / prediction', measure=measure, group_by=group, aggregation='Mean', horizon=3, top_n=5), (label, group))

    def test_forecast_of_one_institution(self):
        institution = OVERALL[OVERALL.institution.str.contains('Technology Madras')].institution_id.iloc[0]
        result = custom(mode='Forecast / prediction', measure='score', group_by='Institution', aggregation='Mean', horizon=3, selected_groups=[institution])
        self.assertEqual(sorted(result['table'].year), [2026, 2027, 2028])
        self.assertIn('2028', result['answer'])

    def test_forecast_needs_three_years(self):
        result = custom(apply_filters(FULL, (2024, 2025), ['Overall']), mode='Forecast / prediction', measure='score', group_by='No grouping', aggregation='Mean', horizon=2)
        self.assertIsNone(result['figure'])
        self.assertIn('three years', result['note'])

    def test_map_every_metric_and_view(self):
        metrics = {**MAP_COUNT_METRICS, **MEASURES}
        for (label, metric), view in product(metrics.items(), MAP_VIEWS):
            with self.subTest(metric=label, view=view):
                result = custom(mode='India map', metric=metric, view=view, category='Overall', aggregation='Mean', top_n=100, year=2025, from_year=2017, to_year=2025)
                self.assertDraws(result, (label, view))
                self.assertTrue(result['is_map'])

    def test_map_counts_match_the_data(self):
        result = custom(mode='India map', metric='top_n_count', view='Single year', category='Overall', top_n=100, year=2025)
        expected = OVERALL[OVERALL.year.eq(2025) & OVERALL.rank_numeric.le(100)].groupby('state').institution_id.nunique()
        table = result['table'].set_index('state')['Institutions in top 100']
        self.assertEqual(table[table > 0].sort_index().tolist(), expected.sort_index().tolist())

    def test_map_change_is_last_year_minus_first(self):
        result = custom(mode='India map', metric='top_n_count', view='Change between two years', category='Overall', top_n=100, from_year=2017, to_year=2025)
        counts = OVERALL[OVERALL.rank_numeric.le(100)].groupby(['year', 'state']).institution_id.nunique().unstack(0).fillna(0)
        table = result['table'].set_index('state')['change']
        for state in ('Delhi', 'Jammu and Kashmir', 'Tamil Nadu'):
            self.assertEqual(table[state], counts.loc[state, 2025] - counts.loc[state, 2017])

    def test_map_share_totals_100(self):
        result = custom(mode='India map', metric='top_n_share', view='Single year', category='Engineering', top_n=100, year=2025) if False else custom(mode='India map', metric='top_n_share', view='Single year', category='Overall', top_n=100, year=2025)
        self.assertAlmostEqual(result['table']['Share of top 100 (%)'].sum(), 100, delta=.2)

    def test_score_map_works_in_every_scored_category(self):
        scored = set(FULL[FULL.score.notna()].category)
        for category in sorted(FULL.category.unique()):
            with self.subTest(category=category):
                data = apply_filters(FULL, (2017, 2025), [category])
                result = custom(data, mode='India map', metric='score', view='Single year', category=category, aggregation='Mean', year=2025)
                if category in scored:
                    self.assertDraws(result, category)
                else:
                    self.assertIsNone(result['figure'])
                    self.assertIn('Rank-only categories have no scores', result['note'])

    def test_forecast_skips_groups_with_short_histories(self):
        result = custom(mode='Forecast / prediction', measure='oi', group_by='Institution', aggregation='Mean', horizon=3, top_n=5)
        self.assertDraws(result, 'OI by institution')
        self.assertEqual(sorted(result['table'].year.unique()), [2026, 2027, 2028])

    def test_distribution_every_measure_group_and_chart(self):
        for (label, measure), group, chart in product(NUMERIC_MEASURES.items(), ['No grouping', *GROUP_FIELDS], DISTRIBUTION_CHARTS):
            with self.subTest(measure=label, group=group, chart=chart):
                self.assertDraws(custom(mode='Distribution', measure=measure, color_by=group, chart=chart, top_n=5), (label, group, chart))

    def test_scatter_every_pair(self):
        for (x_label, x), (y_label, y) in product(NUMERIC_MEASURES.items(), NUMERIC_MEASURES.items()):
            with self.subTest(x=x_label, y=y_label):
                result = custom(mode='Relationship / scatter', x=x, y=y, color_by='No grouping')
                if x == y:
                    self.assertIn('two different', result['note'])
                else:
                    self.assertDraws(result, (x, y))

    def test_scatter_options(self):
        for group in GROUP_FIELDS:
            for animate in (False, True):
                with self.subTest(group=group, animate=animate):
                    self.assertDraws(custom(mode='Relationship / scatter', x='tlr', y='score', size='pr', color_by=group, animate=animate, top_n=6), group)

    def test_scatter_correlation_matches_the_data(self):
        rows = parameter_rows(OVERALL)
        metrics = custom(mode='Relationship / scatter', x='tlr', y='score')['metrics']
        self.assertEqual(metrics['Pearson correlation'], round(rows.tlr.corr(rows.score), 3))

    def test_correlation_views(self):
        for chart in CORRELATION_CHARTS:
            with self.subTest(chart=chart):
                self.assertDraws(custom(mode='Correlation matrix', variables=list(NUMERIC_MEASURES.values()), chart=chart), chart)

    def test_top_and_bottom(self):
        for (label, measure), direction in product(NUMERIC_MEASURES.items(), ['Best', 'Worst']):
            with self.subTest(measure=label, direction=direction):
                self.assertDraws(custom(mode='Top / bottom institutions', measure=measure, direction=direction, year=2025, top_n=10), label)
        best_rank = custom(mode='Top / bottom institutions', measure='rank_numeric', direction='Best', year=2025, top_n=3)['table']
        self.assertEqual(best_rank['Rank'].iloc[0], 1)

    def test_profiles(self):
        ids = OVERALL[OVERALL.year.eq(2025)].nsmallest(3, 'rank_numeric').institution_id.tolist()
        for chart in PROFILE_CHARTS:
            with self.subTest(chart=chart):
                self.assertDraws(custom(mode='Institution parameter profile', institution_ids=ids, chart=chart, year=2025), chart)

    def test_transitions(self):
        for chart in TRANSITION_CHARTS:
            with self.subTest(chart=chart):
                self.assertDraws(custom(mode='Rank-band transitions', from_year=2017, to_year=2025, chart=chart), chart)

    def test_summary_every_measure_and_group(self):
        for (label, measure), group in product(MEASURES.items(), GROUP_FIELDS):
            data = ALL_CATEGORIES if GROUP_FIELDS[group] in {'category', 'Category_Group'} else OVERALL
            with self.subTest(measure=label, group=group):
                self.assertDraws(custom(data, mode='Summary statistics', measure=measure, group_by=group, aggregation='Mean', top_n=10), (label, group))

    def test_parameters_work_outside_overall(self):
        engineering = apply_filters(FULL, (2017, 2025), ['Engineering'])
        result = custom(engineering, mode='Compare groups', measure='tlr', group_by='State', aggregation='Mean', chart='Bar')
        self.assertIsNotNone(result['figure'])

    def test_categories_without_parameters_explain_themselves(self):
        for category in ('Research', 'Innovation', 'Open University'):
            with self.subTest(category=category):
                rows = apply_filters(FULL, (2017, 2025), [category])
                result = custom(rows, mode='Compare groups', measure='tlr', group_by='State', aggregation='Mean', chart='Bar')
                self.assertIsNone(result['figure'])
                self.assertIn('no standard parameter set', result['note'])

    def test_parameter_rows_never_pool_categories(self):
        both = apply_filters(FULL, (2025, 2025), ['Engineering', 'Overall'])
        self.assertEqual(set(parameter_rows(both).category), {'Overall'})
        neither = apply_filters(FULL, (2025, 2025), ['Law', 'Engineering'])
        self.assertEqual(set(parameter_rows(neither).category), {'Engineering'})

    def test_derived_measures_cover_consecutive_years_only(self):
        rows = FULL[FULL.category.eq('Overall')].sort_values(['institution_id', 'year'])
        madras = rows[rows.institution.eq('Indian Institute of Technology Madras')]
        self.assertEqual(madras[madras.year.eq(2025)].score_change.iloc[0], round(madras[madras.year.eq(2025)].score.iloc[0] - madras[madras.year.eq(2024)].score.iloc[0], 10) if False else madras[madras.year.eq(2025)].score.iloc[0] - madras[madras.year.eq(2024)].score.iloc[0])
        self.assertTrue(pd.isna(madras[madras.year.eq(2017)].score_change.iloc[0]))
        self.assertEqual(len(FULL), 6980)


if __name__ == '__main__':
    unittest.main()
