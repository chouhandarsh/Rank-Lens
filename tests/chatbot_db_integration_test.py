"""Opt-in live checks against the chatbot database.

The chatbot's connection can write, so these tests must only ever run SELECT
queries. Never add a statement here that changes data or schema.
"""
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "chatbot" / "src"))

from nirf_chatbot.database import settings
from nirf_chatbot.sql_executor import execute_sql


RUN_INTEGRATION_TESTS = (
    bool(settings.DATABASE_URL)
    and os.getenv("NIRF_RUN_DB_TESTS") == "1"
)


@unittest.skipUnless(
    RUN_INTEGRATION_TESTS,
    "Set DATABASE_URL and NIRF_RUN_DB_TESTS=1 to run.",
)
class DatabaseIntegrationTests(unittest.TestCase):
    def test_table_is_populated(self):
        results = execute_sql('SELECT COUNT(*) AS row_count FROM public."NIRF"')
        self.assertGreater(results["rows"][0]["row_count"], 0)

    def test_cte_window_join_and_trailing_semicolon_query(self):
        results = execute_sql(
            """
            WITH ranked AS (
                SELECT "Institution_ID", "Year", "Rank_Numeric",
                       LAG("Rank_Numeric") OVER (
                           PARTITION BY "Institution_ID" ORDER BY "Year"
                       ) AS previous_rank
                FROM public."NIRF"
                WHERE "Category" = 'Overall'
            )
            SELECT r."Institution_ID", r."Year", r.previous_rank,
                   RANK() OVER (ORDER BY n."Rank_Numeric") AS rank_order
            FROM ranked AS r
            JOIN public."NIRF" AS n
              ON n."Institution_ID" = r."Institution_ID"
             AND n."Year" = r."Year"
            ORDER BY r."Year" DESC
            LIMIT 3;
            """
        )

        self.assertIn("previous_rank", results["columns"])
        self.assertIn("rank_order", results["columns"])
        self.assertEqual(len(results["rows"]), 3)


if __name__ == "__main__":
    unittest.main()
