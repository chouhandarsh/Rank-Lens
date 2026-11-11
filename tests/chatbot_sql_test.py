"""Tests for safe validation of chatbot-generated PostgreSQL queries."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "chatbot" / "src"))

from nirf_chatbot.sql_validator import normalize_sql, validate_sql
from nirf_chatbot.sql_engine import sql_prompt
from nirf_chatbot.sql_executor import DatabaseConfigurationError, execute_sql
from nirf_chatbot.chat_service import (
    INDIAN_STATES_AND_UTS,
    DATABASE_CONFIGURATION_MESSAGE,
    QUERY_FAILURE_MESSAGE,
    _dataset_overview_answer,
    _dataset_overview_query,
    _clean_model_sql,
    _execute_with_retry,
    _needs_generated_sql,
    _top_score_query,
    _state_metric_answer,
    _state_metric_query,
    _institution_question_answer,
    _institution_question_query,
    _state_rank_count_answer,
    _state_rank_count_query,
)
from nirf_chatbot.answer_engine import _fallback_answer
from nirf_chatbot.answer_engine import (
    AnswerGenerationTimeout,
    generate_answer,
)
from nirf_chatbot.visualization import (
    build_plotly_figure,
    ensure_chart_columns,
    infer_chart_spec,
)
from unittest.mock import patch

import pandas as pd


class ChatbotSqlTests(unittest.TestCase):
    def test_dataset_overview_questions_use_database_summary_query(self):
        for question in (
            "What is this dataset about?",
            "How many rows does it have?",
            "How many years of data does the dataset have?",
        ):
            with self.subTest(question=question):
                query = _dataset_overview_query(question)

                self.assertIsNotNone(query)
                normalized = normalize_sql(query)
                self.assertIn("COUNT(*) AS row_count", normalized)
                self.assertIn('COUNT(DISTINCT "Year") AS year_count', normalized)
                self.assertIn('MIN("Year") AS first_year', normalized)
                self.assertTrue(validate_sql(normalized))

    def test_dataset_overview_answer_reports_counts_and_coverage(self):
        answer = _dataset_overview_answer(
            {
                "rows": [
                    {
                        "row_count": 6734,
                        "year_count": 9,
                        "first_year": 2017,
                        "last_year": 2025,
                        "category_count": 12,
                        "institution_count": 1180,
                        "state_count": 33,
                    }
                ]
            }
        )

        self.assertIn("6,734 NIRF ranking records", answer)
        self.assertIn("9 distinct years, from 2017 through 2025", answer)
        self.assertIn("12 categories", answer)
        self.assertIn("1,180 distinct institutions", answer)
        self.assertIn("33 states and union territories", answer)

    def test_dataset_question_returns_summary_without_invoking_sql_llm(self):
        results = {
            "columns": ["row_count", "year_count", "first_year", "last_year",
                        "category_count", "institution_count", "state_count"],
            "rows": [
                {
                    "row_count": 10,
                    "year_count": 2,
                    "first_year": 2024,
                    "last_year": 2025,
                    "category_count": 1,
                    "institution_count": 5,
                    "state_count": 3,
                }
            ],
        }
        with patch(
            "nirf_chatbot.chat_service.execute_sql", return_value=results
        ) as execute_sql, patch(
            "nirf_chatbot.chat_service.sql_chain"
        ) as sql_chain:
            from nirf_chatbot.chat_service import process_question

            response = process_question("How many rows does it have?")

        execute_sql.assert_called_once()
        sql_chain.invoke.assert_not_called()
        self.assertIn("10 NIRF ranking records", response["answer"])
        self.assertFalse(response["is_chart_request"])

    def test_non_dataset_question_does_not_match_dataset_summary(self):
        self.assertIsNone(_dataset_overview_query("Top Engineering colleges"))

    def test_top_score_query_defaults_to_overall_latest_year_and_names(self):
        query = _top_score_query("Plot the top institutes by score")

        self.assertIsNotNone(query)
        normalized = normalize_sql(query)
        self.assertIn('"Category" = \'Overall\'', normalized)
        self.assertIn('MAX("Year")', normalized)
        self.assertIn('"Institution_Name"', normalized)
        self.assertIn('"Score" IS NOT NULL', normalized)
        self.assertIn('"Score" DESC', normalized)
        self.assertTrue(normalized.endswith("LIMIT 10"))
        self.assertTrue(validate_sql(normalized))

    def test_top_score_query_uses_named_category_year_and_capped_limit(self):
        query = _top_score_query(
            "Top 1000 Engineering institutions by score in 2024"
        )

        self.assertIsNotNone(query)
        normalized = normalize_sql(query)
        self.assertIn('"Category" = \'Engineering\'', normalized)
        self.assertIn('"Year" = 2024', normalized)
        self.assertTrue(normalized.endswith("LIMIT 500"))
        self.assertTrue(validate_sql(normalized))

    def test_state_top_100_query_returns_all_state_metrics_for_latest_overall(self):
        query_details = _state_metric_query(
            "Which states have the most institutions in the top 100?"
        )

        self.assertIsNotNone(query_details)
        query, metric = query_details
        normalized = normalize_sql(query)
        self.assertEqual(metric, "institution_count")
        self.assertIn('"Category" = \'Overall\'', normalized)
        self.assertIn('MAX("Year")', normalized)
        self.assertIn('"State"', normalized)
        self.assertIn('COUNT(DISTINCT "Institution_ID")', normalized)
        self.assertIn('GROUP BY "State", "Year", "Category"', normalized)
        self.assertTrue(validate_sql(normalized))

    def test_state_metric_answer_is_grounded_in_query_rows(self):
        answer = _state_metric_answer(
            {
                "rows": [
                    {
                        "State": "Tamil Nadu",
                        "Year": 2025,
                        "Category": "Overall",
                        "institution_count": 12,
                    }
                ]
            },
            "institution_count",
        )

        self.assertIn("Overall in 2025", answer)
        self.assertIn("Tamil Nadu: 12", answer)

    def test_iit_madras_score_trend_has_direct_validated_query(self):
        match = _institution_question_query(
            "plot the tend of the score of iit madras"
        )

        self.assertIsNotNone(match)
        query, institution, answer_kind = match
        self.assertEqual(institution, "Indian Institute of Technology Madras")
        self.assertEqual(answer_kind, "score_trend")
        self.assertIn('"Category" = \'Overall\'', query)
        self.assertIn('"Year" ASC', query)
        self.assertTrue(validate_sql(query))

    def test_iit_madras_rank_trend_uses_overall_across_years(self):
        match = _institution_question_query(
            "Rank trend of IIT Madras"
        )

        self.assertIsNotNone(match)
        query, institution, answer_kind = match
        self.assertEqual(institution, "Indian Institute of Technology Madras")
        self.assertEqual(answer_kind, "rank_trend")
        self.assertIn('"Category" = \'Overall\'', query)
        self.assertIn('"Rank_Numeric" IS NOT NULL', query)
        self.assertIn('"Year" ASC', query)
        self.assertTrue(validate_sql(query))

    def test_iit_rank_question_defaults_to_overall_category(self):
        match = _institution_question_query(
            "What is the rank of IIT Madras in 2025?"
        )

        self.assertIsNotNone(match)
        query, _, answer_kind = match
        self.assertEqual(answer_kind, "rank")
        self.assertIn('"Year" = 2025', query)
        self.assertIn('"Category" = \'Overall\'', query)
        self.assertTrue(validate_sql(query))

    def test_institution_question_answer_uses_result_rows(self):
        answer = _institution_question_answer(
            {
                "rows": [
                    {"Year": 2017, "Score": 87.96},
                    {"Year": 2025, "Score": 91.8},
                ]
            },
            "IIT Madras",
            "score_trend",
        )

        self.assertIn("2 yearly Overall score records", answer)
        self.assertIn("87.96 to 91.8", answer)

    def test_rank_trend_answer_lists_result_ranks_by_year(self):
        answer = _institution_question_answer(
            {
                "rows": [
                    {"Year": 2024, "Rank_Numeric": 2},
                    {"Year": 2025, "Rank_Numeric": 1},
                ]
            },
            "IIT Madras",
            "rank_trend",
        )

        self.assertIn("2024: rank 2", answer)
        self.assertIn("2025: rank 1", answer)

    def test_rank_trend_request_is_marked_for_chart_rendering(self):
        with patch(
            "nirf_chatbot.chat_service.execute_sql",
            return_value={
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
                        "Institution_ID": "iitm",
                        "Institution_Name": "Indian Institute of Technology Madras",
                        "Year": 2025,
                        "Category": "Overall",
                        "Score": 91.8,
                        "Rank_Numeric": 1,
                        "State": "Tamil Nadu",
                        "City": "Chennai",
                    }
                ],
            },
        ):
            from nirf_chatbot.chat_service import process_question

            response = process_question("Rank trend of IIT Madras")

        self.assertIn("Visualization: CHART", response["answer"])
        self.assertTrue(response["is_chart_request"])

    def test_answer_timeout_is_identified_and_does_not_hide_results(self):
        with patch(
            "nirf_chatbot.answer_engine.answer_chain"
        ) as answer_chain:
            answer_chain.invoke.side_effect = TimeoutError(
                "The read operation timed out"
            )
            with self.assertRaisesRegex(
                AnswerGenerationTimeout, "results and chart are still available"
            ):
                generate_answer(
                    "plot score trend",
                    {"rows": [{"Year": 2025, "Score": 90.0}]},
                )

    def test_empty_model_answer_gets_results_fallback(self):
        fallback = _fallback_answer(
            "Visualization: CHART\n## Answer\n",
            {
                "rows": [
                    {"Year": 2017, "Score": 87.96},
                    {"Year": 2025, "Score": 91.8},
                ]
            },
        )

        self.assertIn("Visualization: CHART", fallback)
        self.assertIn("2 matching records", fallback)
        self.assertIn("2017 through 2025", fallback)

    def test_empty_results_get_fallback_answer(self):
        fallback = _fallback_answer("Visualization: TEXT", {"rows": []})

        self.assertIn("No matching records", fallback)

    def test_top_score_answer_names_institutes_and_grounded_values(self):
        answer = generate_answer(
            "Plot the top institutes by score",
            {
                "rows": [
                    {
                        "Institution_ID": "id-2",
                        "Institution_Name": "Example Institute B",
                        "Year": 2025,
                        "Category": "Overall",
                        "Score": 90.2,
                    },
                    {
                        "Institution_ID": "id-1",
                        "Institution_Name": "Example Institute A",
                        "Year": 2025,
                        "Category": "Overall",
                        "Score": 95.0,
                    },
                ]
            },
        )

        self.assertIn("Top 2 Overall institutes in 2025 by score", answer)
        self.assertIn("1. Example Institute A (95)", answer)
        self.assertIn("2. Example Institute B (90.2)", answer)

    def test_empty_results_suggest_a_corrected_question(self):
        answer = generate_answer("show top hospitals", {"rows": []})

        self.assertIn("No matching records were returned", answer)
        self.assertIn("Try specifying the Overall category", answer)

    def test_normalize_adds_limit_without_rewriting_identifiers(self):
        query = """
            SELECT Year, Category, City, Score, Institution_Name, Rank_Numeric
            FROM public."NIRF"
            WHERE Institution_Name LIKE '%IIT Madras%'
            ORDER BY Year, Category
        """

        normalized = normalize_sql(query)

        self.assertIn("SELECT Year, Category, City, Score", normalized)
        self.assertIn("Institution_Name LIKE", normalized)
        self.assertIn("Rank_Numeric", normalized)
        self.assertIn("LIMIT 1000", normalized)
        self.assertTrue(validate_sql(normalized))

    def test_normalize_preserves_quoted_column_case(self):
        normalized = normalize_sql(
            'SELECT "year" FROM public."NIRF" ORDER BY "year"'
        )

        self.assertIn('"year"', normalized)
        self.assertTrue(validate_sql(normalized))

    def test_normalize_preserves_unquoted_table_name(self):
        normalized = normalize_sql(
            "SELECT State FROM public.NIRF WHERE Rank_Numeric <= 100"
        )

        self.assertIn("FROM public.NIRF", normalized)
        self.assertIn("SELECT State", normalized)
        self.assertIn("Rank_Numeric", normalized)
        self.assertTrue(validate_sql(normalized))

    def test_database_error_retries_once_with_failed_sql_and_error(self):
        repaired = 'SELECT "State" FROM public."NIRF" LIMIT 10'
        with patch(
            "nirf_chatbot.chat_service.execute_sql",
            side_effect=[RuntimeError("permission denied"), {"rows": []}],
        ) as execute, patch(
            "nirf_chatbot.chat_service.sql_chain"
        ) as chain:
            chain.invoke.return_value = repaired
            result = _execute_with_retry(
                "List states", 'SELECT "State" FROM public."NIRF"'
            )

        self.assertEqual(result[1], {"rows": []})
        self.assertEqual(execute.call_count, 2)
        repair_prompt = chain.invoke.call_args.args[0]["question"]
        self.assertIn('SELECT "State" FROM public."NIRF"', repair_prompt)
        self.assertIn("permission denied", repair_prompt)

    def test_database_error_after_retry_returns_friendly_message(self):
        with patch(
            "nirf_chatbot.chat_service.execute_sql",
            side_effect=RuntimeError("sensitive database detail"),
        ) as execute, patch(
            "nirf_chatbot.chat_service.sql_chain"
        ) as chain:
            chain.invoke.return_value = 'SELECT "State" FROM public."NIRF"'
            result = _execute_with_retry(
                "List states", 'SELECT "State" FROM public."NIRF"'
            )

        self.assertEqual(execute.call_count, 3)
        self.assertEqual(result["answer"], QUERY_FAILURE_MESSAGE)
        self.assertNotIn("sensitive database detail", result["answer"])

    def test_missing_database_configuration_fails_closed(self):
        with patch("nirf_chatbot.sql_executor.engine", None):
            with self.assertRaisesRegex(
                RuntimeError, "Set DATABASE_URL"
            ):
                execute_sql('SELECT "State" FROM public."NIRF"')

    def test_executor_sets_timeout_and_caps_rows_at_1000(self):
        from unittest.mock import MagicMock

        engine = MagicMock()
        connection = engine.connect.return_value.__enter__.return_value
        result = MagicMock()
        result.keys.return_value = ["institution_id"]
        result.mappings.return_value.fetchmany.return_value = []
        connection.execute.side_effect = [None, result]

        with patch("nirf_chatbot.sql_executor.engine", engine):
            execute_sql(
                "WITH ranked AS (SELECT 1 AS institution_id) "
                "SELECT institution_id FROM ranked"
            )

        executed = [str(call.args[0]) for call in connection.execute.call_args_list]
        self.assertIn("60s", executed[0])
        self.assertIn("LIMIT 1000", executed[1])
        result.mappings.return_value.fetchmany.assert_called_once_with(1000)

    def test_executor_handles_statements_that_return_no_rows(self):
        from unittest.mock import MagicMock

        engine = MagicMock()
        connection = engine.connect.return_value.__enter__.return_value
        result = MagicMock()
        result.returns_rows = False
        result.rowcount = 3
        connection.execute.side_effect = [None, result]

        with patch("nirf_chatbot.sql_executor.engine", engine):
            response = execute_sql("NOTIFY example")

        self.assertEqual(response["rows"], [])
        self.assertEqual(response["rows_affected"], 3)

    def test_complex_questions_skip_the_fixed_shortcuts(self):
        for question in (
            "Which institutions improved their rank the most between 2020 and 2025?",
            "Top 5 institutes by score in Karnataka",
            "Compare the score trend of IIT Bombay and IIT Delhi",
            "Rank of IIT Madras in 2024 and 2025",
            "Which states have more than 5 institutions in the top 100?",
            "Top IITs by score",
        ):
            with self.subTest(question=question):
                self.assertTrue(_needs_generated_sql(question))

    def test_simple_questions_keep_the_fixed_shortcuts(self):
        for question in (
            "What is this dataset about?",
            "Plot the top institutes by score",
            "Rank trend of IIT Madras",
            "What is the rank of IIT Madras in 2025?",
            "Which states have the most institutions in the top 100?",
        ):
            with self.subTest(question=question):
                self.assertFalse(_needs_generated_sql(question))

    def test_filtered_count_does_not_get_the_whole_dataset_summary(self):
        self.assertIsNone(_dataset_overview_query("How many records in 2025?"))
        self.assertIsNone(
            _dataset_overview_query("How many Engineering records are there?")
        )

    def test_model_reasoning_and_markdown_are_removed_from_sql(self):
        expected = 'SELECT "State" FROM public."NIRF"'
        for output in (
            expected,
            f"```sql\n{expected}\n```",
            f"<think>Which table?</think>\n{expected}",
            f"SQL: {expected}",
        ):
            with self.subTest(output=output):
                self.assertEqual(_clean_model_sql(output), expected)

    def test_chart_columns_are_not_forced_onto_analytical_queries(self):
        question = "Compare average scores"
        for query in (
            'WITH y AS (SELECT "Year" FROM public."NIRF") SELECT "Year" FROM y',
            'SELECT a."Year" FROM public."NIRF" AS a '
            'JOIN public."NIRF" AS b ON a."Year" = b."Year"',
            'SELECT "Year", LAG("Score") OVER (ORDER BY "Year") AS previous_score '
            'FROM public."NIRF"',
        ):
            with self.subTest(query=query):
                self.assertEqual(ensure_chart_columns(query, question), query)

    def test_all_numeric_yearly_results_chart_the_requested_metric(self):
        frame = pd.DataFrame(
            {
                "Year": [2024, 2025],
                "avg_score": [50.0, 51.0],
                "score_change": [None, 1.0],
            }
        )

        spec = infer_chart_spec(frame, "Plot the year-over-year score change")

        self.assertEqual(spec["chart_type"], "Line")
        self.assertEqual(spec["x"], "Year")
        self.assertEqual(spec["y"], "score_change")
        build_plotly_figure(frame, spec)

    def test_single_numeric_column_has_a_chart_axis_from_the_results(self):
        frame = pd.DataFrame({"score_rank_correlation": [-0.89]})

        spec = infer_chart_spec(frame, "Plot the correlation")

        self.assertIn(spec["x"], frame.columns)
        self.assertIn(spec["y"], frame.columns)
        build_plotly_figure(frame, spec)

    def test_unparseable_sql_is_left_for_the_repair_step(self):
        self.assertEqual(
            ensure_chart_columns("SELECT FROM (", "plot it"),
            "SELECT FROM (",
        )

    def test_computed_columns_are_answered_by_the_model(self):
        with patch("nirf_chatbot.answer_engine.answer_chain") as answer_chain:
            answer_chain.invoke.return_value = (
                "Visualization: TEXT\n## Answer\nInstitute A improved by 5."
            )
            answer = generate_answer(
                "Who improved the most?",
                {
                    "rows": [
                        {
                            "Institution_Name": "Institute A",
                            "Year": 2025,
                            "Category": "Overall",
                            "Score": 90,
                            "rank_improvement": 5,
                        }
                    ]
                },
            )

        answer_chain.invoke.assert_called_once()
        self.assertIn("improved by 5", answer)

    def test_large_results_are_trimmed_before_reaching_the_model(self):
        rows = [{"State": "Delhi", "value": index} for index in range(150)]
        with patch("nirf_chatbot.answer_engine.answer_chain") as answer_chain:
            answer_chain.invoke.return_value = (
                "Visualization: TEXT\n## Answer\nDone."
            )
            generate_answer("Summarize", {"columns": [], "rows": rows})

        sent = answer_chain.invoke.call_args.args[0]["results"]
        self.assertEqual(len(sent["rows"]), 100)
        self.assertIn("first 100 of 150", sent["note"])

    def test_answer_failure_keeps_query_results(self):
        results = {
            "columns": ["State", "value"],
            "rows": [{"State": "Delhi", "value": 1}],
        }
        with patch(
            "nirf_chatbot.chat_service.execute_sql", return_value=results
        ), patch(
            "nirf_chatbot.chat_service.sql_chain"
        ) as chain, patch(
            "nirf_chatbot.chat_service.generate_answer",
            side_effect=RuntimeError("model unavailable"),
        ):
            chain.invoke.return_value = 'SELECT "State" FROM public."NIRF"'
            from nirf_chatbot.chat_service import process_question

            response = process_question("Which state has a gap in coverage?")

        self.assertEqual(response["results"], results)
        self.assertIn("results and chart", response["answer_warning"])

    def test_executor_error_does_not_expose_connection_credentials(self):
        from unittest.mock import MagicMock
        from nirf_chatbot.sql_executor import execute_sql as execute_readonly

        engine = MagicMock()
        connection = engine.connect.return_value.__enter__.return_value
        connection.execute.side_effect = RuntimeError(
            "postgresql://readonly:secret-password@db.example/NIRF"
        )

        with patch("nirf_chatbot.sql_executor.engine", engine):
            with self.assertRaises(RuntimeError) as raised:
                execute_readonly('SELECT "State" FROM public."NIRF"')

        self.assertNotIn("secret-password", str(raised.exception))
        self.assertNotIn("db.example", str(raised.exception))

    def test_missing_database_configuration_returns_a_friendly_chat_answer(self):
        with patch(
            "nirf_chatbot.chat_service.execute_sql",
            side_effect=DatabaseConfigurationError(
                "Set DATABASE_URL"
            ),
        ), patch("nirf_chatbot.chat_service.sql_chain") as chain:
            from nirf_chatbot.chat_service import process_question

            response = process_question("Plot the top institutes by score")

        chain.invoke.assert_not_called()
        self.assertEqual(response["answer"], DATABASE_CONFIGURATION_MESSAGE)
        self.assertIn("SELECT", response["generated_sql"])

    def test_state_top_ten_question_uses_direct_query(self):
        query_details = _state_rank_count_query(
            "Which state has the most institutions ranked in the top 10 in 2025?"
        )

        self.assertIsNotNone(query_details)
        query, year, rank_limit = query_details
        self.assertEqual((year, rank_limit), (2025, 10))
        self.assertIn('COUNT(DISTINCT "Institution_ID")', query)
        self.assertIn('FROM public."NIRF"', query)
        self.assertTrue(validate_sql(query))

    def test_state_rank_question_accepts_common_wording(self):
        query_details = _state_rank_count_query(
            "Which state has the fewest colleges in top 5 for 2024?"
        )

        self.assertIsNotNone(query_details)
        self.assertIn("ORDER BY COALESCE(counts.", query_details[0])
        self.assertIn("LEFT JOIN", query_details[0])
        self.assertIn("Andaman and Nicobar Islands", query_details[0])
        self.assertTrue(validate_sql(query_details[0]))

    def test_state_rank_question_needs_a_year_and_top_limit(self):
        self.assertIsNone(
            _state_rank_count_query("Which state has the most institutions?")
        )

    def test_state_rank_answer_uses_database_result(self):
        answer = _state_rank_count_answer(
            {"rows": [{"State": "Delhi", "institution_count": 15}]},
            2025,
            10,
        )

        self.assertIn("Delhi", answer)
        self.assertIn("15 institutions", answer)

    def test_least_answer_includes_states_with_zero_top_ranked_institutions(self):
        query_details = _state_rank_count_query(
            "Which state has the least institutions ranked in the top 10 in 2025?"
        )
        self.assertIsNotNone(query_details)
        query, year, rank_limit = query_details
        results = {
            "columns": ["State", "institution_count"],
            "rows": [
                {
                    "State": "Andaman and Nicobar Islands",
                    "institution_count": 0,
                },
                {"State": "Delhi", "institution_count": 0},
            ],
        }
        answer = _state_rank_count_answer(
            results, year, rank_limit, is_least=True
        )

        self.assertEqual(results["rows"][0]["State"], "Andaman and Nicobar Islands")
        self.assertEqual(results["rows"][0]["institution_count"], 0)
        self.assertIn("tied", answer)

    def test_state_universe_contains_all_36_states_and_union_territories(self):
        self.assertEqual(len(INDIAN_STATES_AND_UTS), 36)
        self.assertIn("Andaman and Nicobar Islands", INDIAN_STATES_AND_UTS)

    def test_prompt_accepts_informal_questions_and_common_aggregates(self):
        rendered = sql_prompt.invoke(
            {"question": "what school got the best score"}
        ).messages[0].content

        self.assertIn("Year", rendered)
        self.assertIn("Institution_ID", rendered)
        self.assertIn("Category = 'Overall'", rendered)
        self.assertIn("latest year", rendered)
        self.assertIn("Never SUM scores", rendered)
        self.assertIn('public."NIRF"', rendered)
        self.assertIn("use ILIKE for name search", rendered)
        self.assertIn("Question: Plot the top institutes by score", rendered)

    def test_prompt_explains_top_ten_state_count(self):
        rendered = sql_prompt.invoke(
            {"question": "which state has most colleges"}
        ).messages[0].content

        self.assertIn('"Rank_Numeric" BETWEEN 1 AND 100', rendered)
        self.assertIn('GROUP BY "State"', rendered)

    def test_state_top_ten_aggregate_is_allowed(self):
        query = """
            SELECT "State", COUNT(*) AS institute_count
            FROM public."NIRF"
            WHERE "Rank_Numeric" BETWEEN 1 AND 10
            GROUP BY "State"
            ORDER BY institute_count DESC, "State" ASC
            LIMIT 1
        """

        self.assertTrue(validate_sql(query))

    def test_set_operations_are_rejected(self):
        query = """
            SELECT State FROM public."NIRF"
            EXCEPT
            SELECT State FROM public."NIRF"
            WHERE Rank_Numeric BETWEEN 1 AND 100
        """
        normalized = normalize_sql(query)
        normalized = normalize_sql(query)

        self.assertTrue(validate_sql(normalized))

    def test_validator_leaves_permissions_to_the_database_role(self):
        for query in (
            'DELETE FROM public."NIRF"',
            'UPDATE public."NIRF" SET "Category" = \'test\'',
            'INSERT INTO public."NIRF" ("Category") VALUES (\'test\')',
            'TRUNCATE public."NIRF"',
            'DROP TABLE public."NIRF"',
        ):
            with self.subTest(query=query):
                self.assertTrue(validate_sql(query))

    def test_rejects_multiple_statements(self):
        with self.assertRaisesRegex(ValueError, "multiple SQL statements"):
            validate_sql(
                'SELECT "State" FROM public."NIRF"; '
                'SELECT "Rank_Numeric" FROM public."NIRF"'
            )

    def test_strips_comments_and_one_trailing_semicolon(self):
        query = """
            -- initial comment with ; not a statement
            SELECT ';' AS marker FROM public."NIRF" /* inline ; comment */;
        """

        normalized = normalize_sql(query)

        self.assertTrue(validate_sql(normalized))
        self.assertNotIn("--", normalized)
        self.assertNotIn("/*", normalized)
        self.assertFalse(normalized.endswith(";"))

    def test_rejects_empty_or_comment_only_query(self):
        for query in ("", "  ", "-- comment only", "/* comment only */"):
            with self.subTest(query=query):
                with self.assertRaisesRegex(ValueError, "output is empty"):
                    validate_sql(query)

    def test_with_select_window_functions_joins_and_limit(self):
        query = """
            WITH ranked AS (
                SELECT "Institution_ID", "Year", "Rank_Numeric",
                       LAG("Rank_Numeric") OVER (
                           PARTITION BY "Institution_ID" ORDER BY "Year"
                       ) AS previous_rank
                FROM public."NIRF"
            )
            SELECT current_row."Institution_ID", current_row."Year",
                   RANK() OVER (ORDER BY current_row."Rank_Numeric") AS rank_order
            FROM ranked AS current_row
            JOIN public."NIRF" AS n
              ON n."Institution_ID" = current_row."Institution_ID"
             AND n."Year" = current_row."Year"
        """

        normalized = normalize_sql(query)

        self.assertTrue(normalized.endswith("LIMIT 1000"))
        self.assertIn("LAG(", normalized)
        self.assertIn("RANK() OVER", normalized)
        self.assertIn("JOIN", normalized)
        self.assertTrue(validate_sql(normalized))

    def test_preserves_existing_limit(self):
        query = 'SELECT "State" FROM public."NIRF" LIMIT 12'
        self.assertTrue(normalize_sql(query).endswith("LIMIT 12"))

    def test_multiple_statement_check_ignores_semicolons_in_literals_and_comments(self):
        self.assertTrue(
            validate_sql(
                "SELECT ';' AS semicolon FROM public.\"NIRF\" "
                "/* ; ignored */ -- ; ignored\n;"
            )
        )

    def test_accepts_new_columns_and_aggregate_aliases(self):
        query = """
            SELECT "Year", COUNT(*) AS institution_count
            FROM public."NIRF"
            WHERE "Rank_Numeric" <= 10
            GROUP BY "Year"
            ORDER BY institution_count DESC
        """

        self.assertTrue(validate_sql(query))

    def test_normalization_adds_limit_when_missing(self):
        base = 'SELECT "State" FROM public."NIRF"'
        self.assertTrue(normalize_sql(base).endswith("LIMIT 1000"))
        self.assertTrue(normalize_sql(base + " LIMIT 1000").endswith("LIMIT 1000"))
        self.assertTrue(
            normalize_sql(base + " LIMIT 12").endswith("LIMIT 12")
        )

    def test_accepts_other_valid_read_tables_and_columns(self):
        self.assertTrue(validate_sql('SELECT * FROM public."other_table"'))


if __name__ == "__main__":
    unittest.main()
