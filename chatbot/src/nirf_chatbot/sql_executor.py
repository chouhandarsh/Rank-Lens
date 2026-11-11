from sqlalchemy import text
from nirf_chatbot.database import engine
from nirf_chatbot.sql_validator import normalize_sql, validate_sql


STATEMENT_TIMEOUT = "60s"
MAX_ROWS = 1000


class DatabaseConfigurationError(RuntimeError):
    pass


def execute_sql(query: str):
    query = normalize_sql(query)
    validate_sql(query)
    if engine is None:
        raise DatabaseConfigurationError(
            "The chatbot database is not configured. Set DATABASE_URL."
        )
    try:
        with engine.connect() as connection:
            with connection.begin():
                connection.execute(
                    text(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                )
                result = connection.execute(text(query))
                if not result.returns_rows:
                    return {
                        "columns": [],
                        "rows": [],
                        "rows_affected": result.rowcount,
                    }
                columns = list(result.keys())
                rows=result.mappings().fetchmany(MAX_ROWS)
                return {
                    "columns": columns,
                    "rows": [dict(row) for row in rows]
                }
    except Exception as error:
        original_error = getattr(error, "orig", error)
        diagnostic = getattr(original_error, "diag", None)
        message = getattr(diagnostic, "message_primary", None)
        if not message:
            message = type(original_error).__name__
        raise RuntimeError(f"SQL execution failed: {message}") from None
