"""Regression checks for the clean master integration."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nirf.data_loader import load_data
from nirf.custom_analysis import run_custom_analysis
from nirf.score_chart import top_score_chart_data
from scripts.build_data import (
    CANONICAL_STATES, OVERALL_WEIGHTS, load_parameter_source, load_rankings,
    overall_score_residual,
)


data = load_data()
rankings = load_rankings()
parameters_source, parameters = load_parameter_source(rankings)

assert not data.duplicated(['year', 'category', 'institution_id']).any()
assert len(CANONICAL_STATES) == 33
assert set(data['state'].dropna()).issubset(CANONICAL_STATES)
assert data['state'].dropna().str.casefold().nunique() == len(data['state'].dropna().unique())
assert data.groupby('institution_id')['institution'].nunique().max() == 1
scores = data['score'].dropna()
assert scores.between(0, 100).all()

overall_counts = data[data['category'].eq('Overall')].groupby('year').size().to_dict()
assert overall_counts == {year: (101 if year == 2020 else 100) for year in range(2017, 2026)}

parameter_columns = [
    'year', 'institution_id', 'institution_type', 'tlr', 'rpc', 'go',
    'oi', 'pr', 'score_legacy', 'score_discrepancy', 'rank_discrepancy',
    'parameter_review_required', 'parameter_quality_note',
]
joined = rankings.merge(
    parameters[parameter_columns],
    on=['year', 'institution_id'],
    how='left',
    validate='many_to_one',
)
assert len(joined) == len(rankings) == 6734
assert len(data) == 6980
assert len(parameters_source) == 900
assert int(parameters_source['matched_to_primary'].sum()) == 894

# Master rows survive the merge with their scores and ranks unchanged.
master_rows = data[data['source_rankings'].eq('nirf_rankings_2017_2025_clean.csv')]
keys = ['year', 'category', 'institution_id']
compared = rankings.merge(master_rows, on=keys, suffixes=('_master', '_final'), validate='one_to_one')
assert len(compared) == len(rankings)
for field in ['score', 'rank_numeric']:
    left, right = compared[f'{field}_master'], compared[f'{field}_final']
    assert (left.isna() == right.isna()).all() and ((left - right).abs().dropna() < 1e-9).all()

# Every category has the parameter set its structure promises.
overall = data[data['category'].eq('Overall')]
assert overall['parameters_available'].all() and len(OVERALL_WEIGHTS) == 5
assert overall_score_residual(data) <= 0.05
research = data[data['category'].eq('Research')]
assert research[['tlr', 'rpc', 'go']].isna().all().all()
assert research[['research_sfc', 'research_qnr', 'research_qlr', 'oi', 'pr']].notna().all().all()
assert research['parameter_gap'].isna().all()
no_score = data[data['score'].isna()]
assert len(no_score) == 57 and no_score[['tlr', 'rpc', 'go', 'oi', 'pr', 'research_sfc']].isna().all().all()

# Distinct universities the master merges under one ID stay separate.
agriculture = data[data['category'].eq('Agriculture & Allied Sectors') & data['year'].eq(2025)]
agri_sciences = agriculture[agriculture['institution'].str.startswith('University of Agricultural Sciences')]
assert agri_sciences['institution_id'].is_unique and len(agri_sciences) >= 2
assert set(agri_sciences.loc[agri_sciences['score'].eq(48.79), 'city']) == {'Dharwad'}

shared_name = (
    data.groupby('institution')['institution_id'].nunique()
    .loc[lambda values: values.gt(1)].index[0]
)
shared_name_rows = data[data['institution'].eq(shared_name)]
institution_grouping = run_custom_analysis(
    shared_name_rows,
    {
        'mode': 'Compare groups',
        'measure': 'score',
        'group_by': 'Institution',
        'aggregation': 'Mean',
        'chart': 'Bar',
    },
)
assert len(institution_grouping['table']) == shared_name_rows['institution_id'].nunique()

all_chart_rows = top_score_chart_data(data, data, single_year_selected=False)
assert all_chart_rows['score'].between(0, 100).all()

iit_patna = data[
    data['institution'].eq('Indian Institute of Technology Patna')
    & data['category'].eq('Overall')
    & data['year'].between(2017, 2025)
]
multi_year = top_score_chart_data(iit_patna, data, single_year_selected=False)
assert len(multi_year) == 1
assert multi_year.iloc[0]['years_averaged'] == 9
assert multi_year.iloc[0]['score'] == iit_patna['score'].mean()

for year, expected_score in [(2017, 39.87), (2025, 57.63)]:
    one_year = iit_patna[iit_patna['year'].eq(year)]
    chart_row = top_score_chart_data(one_year, data, single_year_selected=True)
    assert len(chart_row) == 1
    assert chart_row.iloc[0]['score'] == expected_score

print('Clean master, parameter join, and score-chart regression tests passed.')
