"""Tests for shared state-map metrics and GeoJSON coverage."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nirf.data_loader import load_data
from nirf.maps import (
    STATE_NAME_MAP,
    build_state_metrics,
    load_geojson,
    make_choropleth,
    render_choropleth,
    state_name_coverage,
)


class StateMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_all_master_states_match_geojson_and_unmatched_features_are_reported(self):
        states = set(self.data["state"].dropna().str.strip())
        geojson = load_geojson()
        unmatched, without_data = state_name_coverage(states, geojson)

        self.assertEqual(len(states), 33)
        self.assertEqual(set(STATE_NAME_MAP), states)
        self.assertEqual(unmatched, [])
        self.assertEqual(
            without_data,
            [
                "Dadra and Nagar Haveli",
                "Daman and Diu",
                "Ladakh",
                "Lakshadweep",
            ],
        )

    def test_2025_overall_top_100_counts_cover_100_institutions(self):
        result = build_state_metrics(
            self.data,
            "institutions_in_top_n",
            year=2025,
            category="Overall",
            top_n=100,
        )

        self.assertEqual(int(result["institutions_in_top_n"].sum()), 100)

    def test_first_observed_year_is_not_reported_as_new_institutions(self):
        result = build_state_metrics(
            self.data,
            "new_institutions",
            year=2017,
            category="Overall",
            history_df=self.data,
        )

        self.assertEqual(int(result["new_institutions"].sum()), 0)

    def test_change_in_count_is_difference_between_single_year_counts(self):
        first = build_state_metrics(
            self.data,
            "institutions_in_top_n",
            year=2017,
            category="Overall",
            top_n=100,
        ).set_index("state")["institutions_in_top_n"]
        last = build_state_metrics(
            self.data,
            "institutions_in_top_n",
            year=2025,
            category="Overall",
            top_n=100,
        ).set_index("state")["institutions_in_top_n"]
        change = build_state_metrics(
            self.data,
            "change_in_count",
            year=(2017, 2025),
            category="Overall",
            top_n=100,
        ).set_index("state")["change_in_count"]

        expected = last.sub(first, fill_value=0).astype(int).sort_index()
        pd.testing.assert_series_equal(
            change.sort_index(), expected, check_names=False
        )

    def test_average_score_uses_mean_not_sum(self):
        frame = pd.DataFrame(
            {
                "year": [2025, 2025],
                "category": ["Overall", "Overall"],
                "institution_id": ["a", "b"],
                "state": ["Delhi", "Delhi"],
                "score": [80.0, 60.0],
                "rank_numeric": [1, 2],
            }
        )

        result = build_state_metrics(
            frame,
            "average_score",
            year=2025,
            category="Overall",
        )

        self.assertEqual(float(result.loc[0, "average_score"]), 70.0)
        self.assertEqual(int(result.loc[0, "institution_count"]), 2)

    def test_choropleth_uses_state_name_property_and_greys_unmatched_features(self):
        frame = pd.DataFrame(
            {
                "state": ["Delhi"],
                "institutions_in_top_n": [5],
                "institution_count": [5],
            }
        )

        figure = make_choropleth(
            frame,
            "institutions_in_top_n",
            "Institutions by state",
        )

        self.assertEqual(figure.data[0].featureidkey, "properties.name")
        self.assertEqual(figure.layout.geo.fitbounds, "locations")
        self.assertFalse(figure.layout.geo.visible)
        self.assertEqual(len(figure.data), 2)

    def test_empty_choropleth_reports_info_without_plotting(self):
        with patch("nirf.maps.st.info") as info, patch(
            "nirf.maps.st.plotly_chart"
        ) as plotly_chart:
            result = render_choropleth(
                pd.DataFrame(
                    columns=["state", "average_score", "institution_count"]
                ),
                "average_score",
                "Average score by state",
                key="empty-map-test",
            )

        self.assertIsNone(result)
        info.assert_called_once()
        plotly_chart.assert_not_called()


if __name__ == "__main__":
    unittest.main()
