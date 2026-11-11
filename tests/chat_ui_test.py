"""Focused tests for parsing chatbot answers and shaping result tables."""
from pathlib import Path
import sys
import unittest
from contextlib import nullcontext
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CHATBOT_SOURCE = ROOT / "chatbot" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CHATBOT_SOURCE))

import nirf_chatbot.streamlit_ui as streamlit_ui
from nirf_chatbot.streamlit_ui import parse_answer, results_frame


class ChatUiTests(unittest.TestCase):
    def test_chart_answer_extracts_plot_code_and_clean_answer(self):
        answer = (
            "Visualization: CHART\n"
            "## Answer\nTop institute: Example University.\n"
            "## Chart code\n```python\nimport pandas as pd\n```"
        )

        wants_chart, answer_text, chart_code = parse_answer(answer)

        self.assertTrue(wants_chart)
        self.assertEqual(answer_text, "Top institute: Example University.")
        self.assertEqual(chart_code, "import pandas as pd")

    def test_text_answer_does_not_expose_unrequested_code(self):
        answer = (
            "Visualization: TEXT\n## Answer\nA plain answer.\n"
            "```python\nprint('unexpected')\n```"
        )

        wants_chart, answer_text, chart_code = parse_answer(answer)

        self.assertFalse(wants_chart)
        self.assertEqual(answer_text, "A plain answer.")
        self.assertIsNone(chart_code)

    def test_results_frame_uses_database_column_order(self):
        frame = results_frame(
            {
                "columns": ["Institute Name", "Rank"],
                "rows": [{"Institute Name": "Example University", "Rank": 1}],
            }
        )

        self.assertEqual(list(frame.columns), ["Institute Name", "Rank"])
        self.assertEqual(frame.iloc[0].to_dict(), {
            "Institute Name": "Example University",
            "Rank": 1,
        })

    def test_empty_results_keep_column_headers(self):
        frame = results_frame({"columns": ["Institute Name", "Rank"], "rows": []})

        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns), ["Institute Name", "Rank"])

    def test_render_turn_shows_chart_and_plot_code_without_stale_code_variable(self):
        turn = {
            "question": "Plot the top institutes by score",
            "response": {
                "answer": (
                    "Visualization: CHART\n## Answer\nTop Overall institutes."
                ),
                "is_chart_request": True,
                "generated_sql": 'SELECT "Institution_Name", "Score" FROM public."NIRF" LIMIT 10',
                "results": {
                    "columns": [
                        "Institution_ID",
                        "Institution_Name",
                        "Year",
                        "Category",
                        "Score",
                        "Rank_Numeric",
                        "State",
                        "City",
                    ],
                    "rows": [
                        {
                            "Institution_ID": "id-a",
                            "Institution_Name": "Example Institute",
                            "Year": 2025,
                            "Category": "Overall",
                            "Score": 90,
                            "Rank_Numeric": 1,
                            "State": "Delhi",
                            "City": "Delhi",
                        }
                    ],
                },
            },
        }

        with (
            patch.object(streamlit_ui.st, "chat_message", side_effect=lambda *args, **kwargs: nullcontext()),
            patch.object(streamlit_ui.st, "columns", side_effect=lambda count: [nullcontext() for _ in range(count)]),
            patch.object(streamlit_ui.st, "tabs", return_value=[nullcontext() for _ in range(4)]),
            patch.object(streamlit_ui.st, "selectbox", side_effect=lambda label, options, index=0, **kwargs: options[index]),
            patch.object(streamlit_ui.st, "slider", side_effect=lambda *args, **kwargs: kwargs["value"]),
            patch.object(streamlit_ui.st, "plotly_chart") as plotly_chart,
            patch.object(streamlit_ui.st, "code") as code,
            patch.object(streamlit_ui.st, "markdown"),
            patch.object(streamlit_ui.st, "dataframe"),
            patch.object(streamlit_ui.st, "download_button"),
            patch.object(streamlit_ui.st, "caption"),
            patch.object(streamlit_ui.st, "info"),
        ):
            streamlit_ui._render_turn(turn, 0)

        plotly_chart.assert_called_once()
        self.assertTrue(
            any(
                call.kwargs.get("language") == "python"
                for call in code.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
