"""Tests for chatbot chart defaults and validated Plotly specifications."""
import base64
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "chatbot" / "src"))

from nirf_chatbot.visualization import (
    STATE_TO_GEOJSON,
    build_plotly_figure,
    combine_state_rows,
    infer_chart_spec,
    _load_india_geojson,
    state_map_labels,
    validate_chart_spec,
)


def _chart_app(root, rows):
    """Streamlit script for AppTest; it runs apart from this module's imports."""
    import sys

    sys.path[:0] = [root, root + "/chatbot/src"]
    import pandas as pd
    from nirf_chatbot.streamlit_ui import render_results_chart

    render_results_chart(pd.DataFrame(rows), "chat", "plot by state")


class ChatChartTests(unittest.TestCase):
    def test_top_score_defaults_to_sorted_horizontal_institution_bar(self):
        frame = pd.DataFrame(
            [
                {
                    "Institution_ID": "id-b",
                    "Institution_Name": "Institute B",
                    "Year": 2025,
                    "Category": "Overall",
                    "Score": 82.4,
                    "Rank_Numeric": 2,
                    "State": "Delhi",
                    "City": "Delhi",
                },
                {
                    "Institution_ID": "id-a",
                    "Institution_Name": "Institute A",
                    "Year": 2025,
                    "Category": "Overall",
                    "Score": 91.8,
                    "Rank_Numeric": 1,
                    "State": "Tamil Nadu",
                    "City": "Chennai",
                },
            ]
        )

        spec = infer_chart_spec(frame, "Plot the top institutes by score")
        self.assertEqual(spec["chart_type"], "Bar")
        self.assertEqual(spec["x"], "Score")
        self.assertEqual(spec["y"], "Institution_Name")
        self.assertEqual(spec["orientation"], "h")
        self.assertIsNone(spec["color"])

        figure, code = build_plotly_figure(frame, spec)
        self.assertEqual(list(figure.data[0].x), [91.8, 82.4])
        self.assertIn("fig = px.bar(chart_data, x='Score', y='Institution_Name'", code)
        self.assertIn("fig.update_yaxes(autorange='reversed')", code)

    def test_rank_trend_defaults_to_line_and_reversed_rank_axis(self):
        frame = pd.DataFrame(
            [
                {
                    "Institution_ID": "iitm",
                    "Institution_Name": "Indian Institute of Technology Madras",
                    "Year": 2025,
                    "Category": "Overall",
                    "Score": 91.8,
                    "Rank_Numeric": 1,
                    "State": "Tamil Nadu",
                    "City": "Chennai",
                },
                {
                    "Institution_ID": "iitm",
                    "Institution_Name": "Indian Institute of Technology Madras",
                    "Year": 2024,
                    "Category": "Overall",
                    "Score": 89.5,
                    "Rank_Numeric": 2,
                    "State": "Tamil Nadu",
                    "City": "Chennai",
                },
            ]
        )

        spec = infer_chart_spec(frame, "Rank trend of IIT Madras")
        self.assertEqual(spec["chart_type"], "Line")
        self.assertEqual(spec["x"], "Year")
        self.assertEqual(spec["y"], "Rank_Numeric")
        self.assertEqual(spec["color"], "Institution_Name")
        figure, code = build_plotly_figure(frame, spec)
        self.assertEqual(figure.layout.yaxis.autorange, "reversed")
        self.assertIn("markers=True", code)
        self.assertIn("autorange='reversed'", code)

    def test_multiple_categories_default_to_category_color(self):
        frame = pd.DataFrame(
            [
                {"Institution_ID": "id-a", "Institution_Name": "A", "Year": 2025,
                 "Category": "Overall", "Score": 90},
                {"Institution_ID": "id-b", "Institution_Name": "B", "Year": 2025,
                 "Category": "Engineering", "Score": 80},
            ]
        )

        spec = infer_chart_spec(frame, "Plot top institutes by score")

        self.assertEqual(spec["color"], "Category")

    def test_invalid_spec_falls_back_to_result_based_spec(self):
        frame = pd.DataFrame(
            {"Institution_Name": ["A"], "Year": [2025], "Category": ["Overall"],
             "Score": [90]}
        )
        valid = validate_chart_spec(
            frame,
            {"chart_type": "Unknown", "x": "missing", "y": "missing",
             "color": "missing", "orientation": "x"},
        )

        self.assertEqual(valid["chart_type"], "Bar")
        self.assertEqual(valid["x"], "Score")
        self.assertEqual(valid["y"], "Institution_Name")
        self.assertEqual(valid["orientation"], "h")

    def test_state_results_default_to_offline_india_map_and_match_all_data_states(self):
        project = ROOT
        rankings = pd.read_csv(
            project / "data" / "processed" / "nirf_analysis_dataset.csv",
            usecols=["state"],
        )
        canonical_states = set(rankings["state"].dropna().str.strip())
        geojson = _load_india_geojson()
        geojson_names = {
            feature["properties"]["name"]
            for feature in geojson["features"]
        }
        unmatched = {
            state
            for state in canonical_states
            if STATE_TO_GEOJSON.get(state, state) not in geojson_names
        }
        self.assertEqual(unmatched, set())

        frame = pd.DataFrame(
            {
                "State": sorted(canonical_states),
                "Year": [2025] * len(canonical_states),
                "Category": ["Overall"] * len(canonical_states),
                "institution_count": list(range(len(canonical_states))),
            }
        )
        spec = infer_chart_spec(
            frame,
            "Which states have the most institutions in the top 100?",
        )
        self.assertEqual(spec["chart_type"], "India map")
        self.assertEqual(spec["x"], "State")
        self.assertEqual(spec["y"], "institution_count")
        self.assertEqual(spec["metric_label"], "Institutions in top 100")

        figure, code = build_plotly_figure(frame, spec)
        self.assertEqual(len(figure.data), 2)
        self.assertEqual(figure.layout.geo.fitbounds, "locations")
        self.assertFalse(figure.layout.geo.visible)
        self.assertEqual(figure.data[1].featureidkey, "properties.name")
        self.assertIn("from nirf.maps import make_choropleth", code)
        compile(code, "<generated map code>", "exec")

    def test_map_combines_repeated_state_rows(self):
        cases = {
            "Score": ([80.0, 60.0, 50.0], {"Delhi": 70.0, "Kerala": 50.0}),
            "Rank_Numeric": ([7, 2, 30], {"Delhi": 2, "Kerala": 30}),
            "institution_count": ([5, 2, 3], {"Delhi": 7, "Kerala": 3}),
        }
        for metric, (values, expected) in cases.items():
            with self.subTest(metric=metric):
                frame = pd.DataFrame(
                    {
                        "Institution_Name": ["A", "B", "C"],
                        "State": ["Delhi", "Delhi", "Kerala"],
                        metric: values,
                    }
                )
                spec = {
                    "chart_type": "India map",
                    "x": "State",
                    "y": metric,
                    "color": None,
                    "orientation": "v",
                }
                figure, _ = build_plotly_figure(frame, spec)
                trace = figure.data[1]
                self.assertEqual(
                    dict(zip(trace.locations, trace.z)),
                    expected,
                )

    def test_combined_rows_count_distinct_institutions(self):
        frame = pd.DataFrame(
            {
                "Institution_Name": ["A", "A", "B"],
                "State": ["Delhi", "Delhi", "Delhi"],
                "Score": [80.0, 60.0, 40.0],
            }
        )
        combined = combine_state_rows(frame, "Score")
        self.assertEqual(combined["institution_count"].tolist(), [2])

    def test_combining_leaves_unique_states_untouched(self):
        frame = pd.DataFrame(
            {"State": ["Delhi", "Kerala"], "City": ["X", "Y"], "Score": [1, 2]}
        )
        self.assertIs(combine_state_rows(frame, "Score"), frame)

    def test_map_labels_describe_the_metric(self):
        cases = {
            "institution_count": "Institutions in top 50",
            "Rank_Numeric": "Best rank",
            "avg_rank": "Avg rank",
            "avg_score": "Average score",
            "total_institutions": "Total institutions",
        }
        for metric, label in cases.items():
            with self.subTest(metric=metric):
                self.assertEqual(
                    state_map_labels(metric, "top 50 states"),
                    (label, f"{label} by State"),
                )

    def test_average_rank_is_averaged_when_combining(self):
        frame = pd.DataFrame({"State": ["Delhi", "Delhi"], "avg_rank": [2, 4]})
        combined = combine_state_rows(frame, "avg_rank")
        self.assertEqual(combined["avg_rank"].tolist(), [3])

    def _run_chart(self, rows):
        at = AppTest.from_function(
            _chart_app,
            kwargs={"root": str(ROOT), "rows": rows},
            default_timeout=60,
        )
        at.run()
        self.assertFalse(at.exception)
        return at

    @staticmethod
    def _figure(at):
        return json.loads(at.get("plotly_chart")[0].proto.spec)

    def _state_values(self, at):
        trace = self._figure(at)["data"][1]
        # Plotly stores numeric arrays in its JSON as base64-encoded binary.
        values = np.frombuffer(
            base64.b64decode(trace["z"]["bdata"]), dtype=trace["z"]["dtype"]
        )
        return dict(zip(trace["locations"], values.tolist()))

    def test_map_category_picker_separates_categories(self):
        rows = {
            "State": ["Delhi", "Delhi", "Kerala", "Kerala"],
            "Category": ["Law", "Overall", "Law", "Overall"],
            "institution_count": [2, 5, 1, 3],
        }
        at = self._run_chart(rows)
        picker = at.selectbox(key="chat_map_category")
        self.assertEqual(picker.value, "Overall")
        self.assertEqual(self._state_values(at), {"Delhi": 5, "Kerala": 3})

        picker.set_value("Law").run()
        self.assertFalse(at.exception)
        self.assertEqual(self._state_values(at), {"Delhi": 2, "Kerala": 1})
        self.assertIn("(Law)", self._figure(at)["layout"]["title"]["text"])

    def test_single_category_has_no_picker(self):
        at = self._run_chart(
            {"State": ["Delhi"], "Category": ["Law"], "institution_count": [2]}
        )
        self.assertNotIn(
            "chat_map_category", [box.key for box in at.selectbox]
        )

    def test_map_title_follows_selected_metric(self):
        rows = {
            "State": ["Delhi", "Delhi", "Kerala"],
            "Score": [80.0, 60.0, 50.0],
            "Rank_Numeric": [7, 2, 30],
        }
        at = self._run_chart(rows)
        self.assertEqual(
            self._figure(at)["layout"]["title"]["text"],
            "Average score by State",
        )
        at.selectbox(key="chat_y_axis").set_value("Rank_Numeric").run()
        self.assertFalse(at.exception)
        self.assertEqual(
            self._figure(at)["layout"]["title"]["text"],
            "Best rank by State",
        )
        self.assertEqual(self._state_values(at), {"Delhi": 2, "Kerala": 30})


if __name__ == "__main__":
    unittest.main()
