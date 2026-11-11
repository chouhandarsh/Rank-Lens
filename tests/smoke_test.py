"""Data and analytics smoke tests; run from project root with `python tests/smoke_test.py`."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nirf.data_loader import load_data
from nirf.filters import apply_filters, parameter_rows
from nirf.questions import QUESTIONS
from nirf.level1 import run_level1
from nirf.custom_analysis import run_custom_analysis

full = load_data()
assert len(full) == 6980, f'Unexpected ranking row count: {len(full)}'
assert full['category'].nunique() == 18
assert int(full['parameters_available'].sum()) == 6673
assert int(full['research_sfc'].notna().sum()) == 250
assert int(full['parameter_review_required'].sum()) == 0

base = apply_filters(full, (2017, 2025), ['Overall'], None, None, (1, 200))
all_categories = apply_filters(full, (2017, 2025), None, None, None, (1, 200), ignore_categories=True)
assert len(parameter_rows(base)) == 901

base_options = {
    'include_review': False,
    'year': 2025,
    'from_year': 2017,
    'to_year': 2025,
    'institution': 'Indian Institute of Technology Madras',
    'states_pair': ['Karnataka', 'Tamil Nadu'],
    'parameter': 'RPC',
}
for qid, section, question, requires_parameters in QUESTIONS:
    options = dict(base_options)
    if qid in {'7.2', '7.3'}:
        options.update({'from_year': 2024, 'to_year': 2025})
    result = run_level1(qid, base, all_categories, full, options)
    assert result['figure'] is not None, f'Level 1 question {qid} did not return a Plotly figure: {result.get("note")}'
print(f'Level 1: {len(QUESTIONS)}/{len(QUESTIONS)} questions returned Plotly figures.')

# Parameter questions work in other categories, and explain themselves where no parameters exist.
parameter_questions = [item[0] for item in QUESTIONS if item[3]]
for category, expect_figure in [('Engineering', True), ('State Public University', True),
                                ('Research', False), ('Open University', False),
                                ('Special Mention (2018)', False)]:
    rows = apply_filters(full, (2017, 2025), [category])
    for qid in parameter_questions:
        options = dict(base_options)
        options.pop('institution', None)
        options['institution_id'] = rows['institution_id'].iloc[0]
        result = run_level1(qid, rows, all_categories, full, options)
        if not expect_figure:
            assert result['figure'] is None, f'{category} {qid} drew a chart without parameters'
print(f'Level 1: {len(parameter_questions)} parameter questions checked across five categories.')

configs = [
    {'mode':'Compare groups','measure':'score','group_by':'category','aggregation':'Mean','chart':'Bar'},
    {'mode':'Trend over time','measure':'score','group_by':'No grouping','aggregation':'Mean','chart':'Line'},
    {'mode':'Distribution','measure':'score','chart':'Histogram'},
    {'mode':'Relationship / scatter','x':'tlr','y':'rpc'},
    {'mode':'Correlation matrix','variables':['score','tlr','rpc','go','oi','pr']},
    {'mode':'Institution parameter profile','institution':'Indian Institute of Technology Madras','year':2025,'chart':'Radar'},
    {'mode':'Rank-band transitions','from_year':2017,'to_year':2025},
    {'mode':'Summary statistics','measure':'score','group_by':'category','aggregation':'Mean'},
    {'mode':'Trend over time','measure':'institution_id','group_by':'No grouping','aggregation':'Count','chart':'Line'},
    {'mode':'Compare groups','measure':'institution_id','group_by':'state','aggregation':'Count','chart':'Bar'},
]
for cfg in configs:
    cfg['include_review'] = False
    result = run_custom_analysis(base if cfg['mode'] != 'Rank-band transitions' else apply_filters(full,(2017,2025),['Overall'],None,None,None), cfg)
    assert result['figure'] is not None, f'Level 2 mode failed: {cfg["mode"]}: {result.get("note")}'
print(f'Level 2: {len(configs)} custom-analysis configurations returned Plotly figures.')
print('Data integration and analytics smoke tests passed.')
