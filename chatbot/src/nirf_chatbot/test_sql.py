import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a SQL query against the chatbot database."
    )
    parser.add_argument(
        "query",
        nargs="+",
        help='SQL query to run, for example: SELECT "Year" FROM public."NIRF" LIMIT 5',
    )
    args = parser.parse_args(argv)
    query = " ".join(args.query)

    try:
        from nirf_chatbot.sql_executor import execute_sql

        result = execute_sql(query)
    except (ImportError, RuntimeError, ValueError):
        print(
            "Database test failed. Check DATABASE_URL.",
            file=sys.stderr,
        )
        return 1

    columns = result["columns"]
    rows = result["rows"]
    print(f"Database test succeeded: {len(rows)} row(s) returned.")
    if columns:
        print(" | ".join(columns))
        for row in rows:
            print(" | ".join(str(row.get(column, "")) for column in columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())