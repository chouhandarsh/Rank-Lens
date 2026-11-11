"""Drives the real Streamlit app through every page, question and Level 2 option."""
from pathlib import Path
import sys
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nirf.custom_analysis import MAP_VIEWS, MODES
from nirf.questions import QUESTIONS

LEVEL1 = 'Level 1 · Sample Questions'
LEVEL2 = 'Level 2 · Custom Analysis'


def open_app(page):
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=300).run()
    app.radio[0].set_value(page).run()
    return app


def box(app, label):
    return next(item for item in app.selectbox if item.label == label)


class AppPageTests(unittest.TestCase):
    def assertClean(self, app, context):
        self.assertEqual([item.value[:300] for item in app.exception], [], context)
        self.assertEqual([item.value[:300] for item in app.error], [], context)

    def test_all_40_questions_on_level_1(self):
        app = open_app(LEVEL1)
        for option in box(app, 'Choose an analytical question').options:
            with self.subTest(question=option[:40]):
                box(app, 'Choose an analytical question').set_value(option).run()
                self.assertClean(app, option)
                self.assertTrue(any(item.value.startswith('**Answer:**') for item in app.success), f'No answer: {option}')

    def test_all_40_questions_on_level_2(self):
        app = open_app(LEVEL2)
        next(item for item in app.radio if item.label == 'Workspace').set_value('Annexure A questions').run()
        answered = 0
        for section in box(app, 'Section').options:
            box(app, 'Section').set_value(section).run()
            for question in box(app, 'Question').options:
                with self.subTest(question=question[:40]):
                    box(app, 'Question').set_value(question).run()
                    self.assertClean(app, question)
                    self.assertTrue(any(item.value.startswith('**Answer:**') for item in app.success), f'No answer: {question}')
                    answered += 1
        self.assertEqual(answered, len(QUESTIONS))

    def test_every_level_2_mode_and_chart(self):
        app = open_app(LEVEL2)
        for mode in MODES:
            box(app, 'Analysis type').set_value(mode).run()
            self.assertClean(app, mode)
            chart_boxes = [item for item in app.selectbox if item.label in {'Chart type', 'Map view'}]
            if not chart_boxes:
                continue
            label = chart_boxes[0].label
            for chart in chart_boxes[0].options:
                with self.subTest(mode=mode, chart=chart):
                    box(app, label).set_value(chart).run()
                    self.assertClean(app, (mode, chart))

    def test_every_map_view_in_the_app(self):
        app = open_app(LEVEL2)
        box(app, 'Analysis type').set_value('India map').run()
        for view in MAP_VIEWS:
            with self.subTest(view=view):
                box(app, 'Map view').set_value(view).run()
                self.assertClean(app, view)

    def test_compare_across_categories_ignores_the_sidebar_filter(self):
        app = open_app(LEVEL2)
        box(app, 'Across').set_value('Category').run()
        self.assertClean(app, 'Category compare')
        self.assertTrue(any('ignores the sidebar category filter' in item.value for item in app.caption))
        self.assertEqual(len(app.dataframe[0].value), 13)  # Categories with scores.

    def test_overview_and_chatbot_pages_still_load(self):
        for page in ('Overview', 'Chatbot · AI Analysis'):
            with self.subTest(page=page):
                self.assertClean(open_app(page), page)


if __name__ == '__main__':
    unittest.main()
