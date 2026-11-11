import re

from nirf_chatbot.llm import llm
from nirf_chatbot.sql_engine import sql_prompt
from nirf_chatbot.sql_executor import (
    DatabaseConfigurationError,
    execute_sql,
)
from nirf_chatbot.answer_engine import (
    AnswerGenerationTimeout,
    _fallback_answer,
    generate_answer,
)
from nirf_chatbot.sql_validator import normalize_sql
from nirf_chatbot.visualization import ensure_chart_columns, is_chart_request

from langchain_core.output_parsers import StrOutputParser


sql_chain = sql_prompt | llm | StrOutputParser()
EMPTY_RESULTS = {"columns": [], "rows": []}
MAX_SQL_REPAIRS = 2
QUERY_FAILURE_MESSAGE = (
    "I couldn't run that query, even after trying to repair it. Please try "
    "rephrasing it as a question about NIRF rankings, scores, or institutions."
)
DATABASE_CONFIGURATION_MESSAGE = (
    "The chatbot database is not configured. Set DATABASE_URL in the "
    "project's .env file."
)
_COMPLEX_QUESTION = re.compile(
    r"\b(?:improv\w*|declin\w*|drop\w*|fell|fall\w*|gain\w*|chang\w*|grow\w*"
    r"|increas\w*|decreas\w*|jump\w*|climb\w*|consistent\w*|every\s+year"
    r"|each\s+year|per\s+year|year[-\s]+(?:over|on)[-\s]+year|versus|vs"
    r"|compar\w*|more\s+than|less\s+than|fewer\s+than|at\s+least|at\s+most"
    r"|greater|above|below|over\s+\d|under\s+\d|percent\w*|share|proportion"
    r"|ratio|median|varianc\w*|stddev|deviation|correlat\w*|volatil\w*|both"
    r"|but\s+not|exclud\w*|except|without|never|always|first\s+time|newly"
    r"|entered|rolling|cumulative|gap|difference|outlier\w*|quartile"
    r"|percentile|across\s+categor\w*|multiple\s+categor\w*|each\s+state"
    r"|IITs|NITs|IIMs|IIITs|cit(?:y|ies)|government|private)\b",
    re.IGNORECASE,
)


def _clean_model_sql(output: str) -> str:
    """Reduce a model reply to its SQL, dropping reasoning and markdown."""
    cleaned = re.sub(
        r"<think>.*?(?:</think>|$)", "", output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    fenced = re.search(
        r"```(?:sql)?\s*(.*?)```", cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        cleaned = fenced.group(1)
    cleaned = cleaned.strip()
    return re.sub(r"^SQL:\s*", "", cleaned, flags=re.IGNORECASE)


def _execution_failure_response(question: str, query: str, message: str) -> dict:
    return {
        "question": question,
        "generated_sql": query,
        "results": EMPTY_RESULTS,
        "answer": message,
        "is_chart_request": False,
    }


def _repair_sql(question: str, query: str, error: Exception) -> str:
    repair_question = (
        "Repair the failed PostgreSQL query for the user's original "
        "request. Return exactly one SQL query and no markdown.\n"
        f"Original question:\n{question}\n\n"
        f"Failed SQL:\n{query}\n\n"
        f"Database error:\n{error}"
    )
    repaired_query = _clean_model_sql(
        sql_chain.invoke({"question": repair_question})
    )
    if repaired_query.casefold() == "unsupported_question":
        raise ValueError("The question is not supported.")
    repaired_query = ensure_chart_columns(repaired_query, question)
    return normalize_sql(repaired_query)


def _execute_with_retry(question: str, query: str) -> tuple[str, dict] | dict:
    query = normalize_sql(query)
    last_error = None
    for attempt in range(MAX_SQL_REPAIRS + 1):
        try:
            if attempt:
                query = _repair_sql(question, query, last_error)
            return query, execute_sql(query)
        except DatabaseConfigurationError:
            return _execution_failure_response(
                question, query, DATABASE_CONFIGURATION_MESSAGE
            )
        except Exception as error:
            last_error = error
    return _execution_failure_response(question, query, QUERY_FAILURE_MESSAGE)

INDIAN_STATES_AND_UTS = (
    "Andaman and Nicobar Islands",
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chandigarh",
    "Chhattisgarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu and Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Ladakh",
    "Lakshadweep",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Puducherry",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
)

INSTITUTION_ALIASES = {
    "IIT Bombay": "Indian Institute of Technology Bombay",
    "IIT Delhi": "Indian Institute of Technology Delhi",
    "IIT Guwahati": "Indian Institute of Technology Guwahati",
    "IIT Hyderabad": "Indian Institute of Technology Hyderabad",
    "IIT Kanpur": "Indian Institute of Technology Kanpur",
    "IIT Kharagpur": "Indian Institute of Technology Kharagpur",
    "IIT Madras": "Indian Institute of Technology Madras",
    "IIT Roorkee": "Indian Institute of Technology Roorkee",
}

RANKING_CATEGORIES = (
    "Overall",
    "Engineering",
    "University",
    "College",
    "Management",
    "Pharmacy",
    "Medical",
    "Law",
    "Architecture",
    "Dental",
    "Research",
    "Innovation",
)


def _mentions_any(question: str, names: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"\b{re.escape(name)}\b", question, re.IGNORECASE)
        for name in names
    )


def _needs_generated_sql(question: str) -> bool:
    """True when the fixed shortcut queries would answer a simpler question."""
    if _COMPLEX_QUESTION.search(question):
        return True
    if len(set(re.findall(r"\b20\d{2}\b", question))) > 1:
        return True
    if len(
        [alias for alias in INSTITUTION_ALIASES
         if _mentions_any(question, (alias,))]
    ) > 1:
        return True
    return _mentions_any(question, INDIAN_STATES_AND_UTS)


def _dataset_overview_query(question: str) -> str | None:
    if re.search(r"\b20\d{2}\b", question) or _mentions_any(
        question, RANKING_CATEGORIES
    ):
        return None

    asks_about_dataset = bool(
        re.search(r"\b(dataset|data|table|records?)\b", question, re.IGNORECASE)
    )
    asks_for_overview = bool(
        re.search(
            r"\b(what|describe|overview|summary|about|contain|include)\b"
            r"|\bhow many\b|\b(count|number|total)\b",
            question,
            re.IGNORECASE,
        )
    )
    asks_for_dataset_fact = bool(
        re.search(
            r"\b(rows?|records?|entries|years?|categories|institutions?|"
            r"states?|coverage|period)\b",
            question,
            re.IGNORECASE,
        )
    )
    asks_simple_count = bool(
        re.search(
            r"\b(how many|count|number of|total)\b.*\b(rows?|records?|years?)\b",
            question,
            re.IGNORECASE,
        )
    )
    if not (
        (asks_about_dataset and asks_for_overview)
        or (asks_about_dataset and asks_for_dataset_fact)
        or asks_simple_count
    ):
        return None

    return """
        SELECT COUNT(*) AS row_count,
               COUNT(DISTINCT "Year") AS year_count,
               MIN("Year") AS first_year,
               MAX("Year") AS last_year,
               COUNT(DISTINCT "Category") AS category_count,
               COUNT(DISTINCT "Institution_ID") AS institution_count,
               COUNT(DISTINCT "State") AS state_count
        FROM public."NIRF"
        LIMIT 1
    """


def _dataset_overview_answer(results: dict) -> str:
    rows = results.get("rows") or []
    if not rows:
        return "The NIRF dataset did not return any summary information."

    summary = rows[0]
    row_count = int(summary.get("row_count") or 0)
    year_count = int(summary.get("year_count") or 0)
    category_count = int(summary.get("category_count") or 0)
    institution_count = int(summary.get("institution_count") or 0)
    state_count = int(summary.get("state_count") or 0)
    first_year = summary.get("first_year")
    last_year = summary.get("last_year")

    if not row_count:
        return "The NIRF dataset is currently empty."

    year_coverage = (
        f"{year_count} distinct years, from {first_year} through {last_year}"
        if first_year is not None and last_year is not None
        else f"{year_count} distinct years"
    )
    return (
        f"The dataset contains {row_count:,} NIRF ranking records about "
        f"institutions evaluated across ranking categories. It covers "
        f"{year_coverage} and {category_count} categories, with "
        f"{institution_count:,} distinct institutions across {state_count} "
        "states and union territories. Records include institution names, "
        "categories, scores, ranks, locations, and score/rank status fields."
    )


def _top_score_query(question: str) -> str | None:
    if not re.search(
        r"\b(top|highest|best)\b.*\b(score|scoring)\b"
        r"|\b(score|scoring)\b.*\b(top|highest|best)\b",
        question,
        re.IGNORECASE,
    ):
        return None

    category = next(
        (
            value
            for value in RANKING_CATEGORIES
            if re.search(rf"\b{re.escape(value)}\b", question, re.IGNORECASE)
        ),
        "Overall",
    )
    year_match = re.search(r"\b(20\d{2})\b", question)
    year_filter = (
        f'"Year" = {int(year_match.group(1))}'
        if year_match
        else (
            f'"Year" = (SELECT MAX("Year") FROM public."NIRF" '
            f'WHERE "Category" = \'{category}\')'
        )
    )
    count_match = re.search(r"\btop\s*(\d+)\b", question, re.IGNORECASE)
    limit = min(int(count_match.group(1)), 500) if count_match else 10
    return f"""
        SELECT "Institution_ID", "Institution_Name", "Year", "Category",
               "Score", "Rank_Numeric", "State", "City"
        FROM public."NIRF"
        WHERE "Category" = '{category}'
          AND {year_filter}
          AND "Score" IS NOT NULL
        ORDER BY "Score" DESC, "Institution_ID" ASC
        LIMIT {limit}
    """


def _state_metric_query(question: str) -> tuple[str, str] | None:
    has_state = bool(re.search(r"\bstates?\b", question, re.IGNORECASE))
    asks_map = bool(re.search(r"\bmap\b", question, re.IGNORECASE))
    rank_match = re.search(r"\btop\s*(\d+)\b", question, re.IGNORECASE)
    if not has_state or not (rank_match or asks_map):
        return None

    rank_limit = min(int(rank_match.group(1)), 500) if rank_match else 100
    category = next(
        (
            value
            for value in RANKING_CATEGORIES
            if re.search(rf"\b{re.escape(value)}\b", question, re.IGNORECASE)
        ),
        "Overall",
    )
    year_match = re.search(r"\b(20\d{2})\b", question)
    year_filter = (
        f'"Year" = {int(year_match.group(1))}'
        if year_match
        else (
            f'"Year" = (SELECT MAX("Year") FROM public."NIRF" '
            f'WHERE "Category" = \'{category}\')'
        )
    )
    if re.search(r"\baverage|mean\b.*\bscore\b", question, re.IGNORECASE):
        aggregate = 'AVG("Score") AS avg_score'
        valid_metric = '"Score" IS NOT NULL'
    elif re.search(r"\b(best|highest)\b.*\brank\b", question, re.IGNORECASE):
        aggregate = 'MIN("Rank_Numeric") AS best_rank'
        valid_metric = '"Rank_Numeric" IS NOT NULL'
    else:
        aggregate = 'COUNT(DISTINCT "Institution_ID") AS institution_count'
        valid_metric = f'"Rank_Numeric" BETWEEN 1 AND {rank_limit}'
    order_direction = "ASC" if "best_rank" in aggregate else "DESC"

    query = f"""
        SELECT "State", "Year", "Category", {aggregate}
        FROM public."NIRF"
        WHERE "Category" = '{category}'
          AND {year_filter}
          AND "State" IS NOT NULL
          AND {valid_metric}
        GROUP BY "State", "Year", "Category"
        ORDER BY {aggregate.split(" AS ")[1]} {order_direction}, "State" ASC
        LIMIT 500
    """
    metric_name = (
        "avg_score" if "avg_score" in aggregate
        else "best_rank" if "best_rank" in aggregate
        else "institution_count"
    )
    return query, metric_name


def _state_metric_answer(results: dict, metric: str) -> str:
    rows = results.get("rows", [])
    if not rows:
        return (
            "Visualization: CHART\n## Answer\nNo state-level results were "
            "returned. Try another category or year."
        )
    first = rows[0]
    year = first.get("Year")
    category = first.get("Category", "Overall")
    label = {
        "institution_count": "institutions",
        "avg_score": "average score",
        "best_rank": "best rank",
    }[metric]
    summaries = [
        f'{row["State"]}: {row[metric]}'
        for row in rows[:5]
    ]
    return (
        f"Visualization: CHART\n## Answer\nState {label} for {category} "
        f"in {year}: " + ", ".join(summaries) + "."
    )


def _state_rank_count_query(question: str) -> tuple[str, int, int] | None:
    has_state = re.search(r"\bstate\b", question, re.IGNORECASE)
    asks_extreme = re.search(r"\b(most|fewest|least)\b", question, re.IGNORECASE)
    rank_match = re.search(r"\btop\s*(\d+)\b", question, re.IGNORECASE)
    year_match = re.search(r"\b(20\d{2})\b", question)
    if not (has_state and asks_extreme and rank_match and year_match):
        return None

    rank_limit = int(rank_match.group(1))
    year = int(year_match.group(1))
    if rank_limit < 1 or rank_limit > 1000:
        return None

    is_least = bool(re.search(r"\b(fewest|least)\b", question, re.IGNORECASE))
    direction = "ASC" if is_least else "DESC"
    if is_least:
        state_values = ", ".join(
            f"('{state}')"
            for state in INDIAN_STATES_AND_UTS
        )
        query = f"""
            SELECT states."State",
                   COALESCE(counts."Institution_ID", 0) AS institution_count
            FROM (VALUES {state_values}) AS states("State")
            LEFT JOIN (
                SELECT "State",
                       COUNT(DISTINCT "Institution_ID") AS "Institution_ID"
                FROM public."NIRF"
                WHERE "Year" = {year}
                  AND "Rank_Numeric" BETWEEN 1 AND {rank_limit}
                  AND "State" IS NOT NULL
                GROUP BY "State"
            ) AS counts
              ON counts."State" = states."State"
            ORDER BY COALESCE(counts."Institution_ID", 0) {direction},
                     states."State" ASC
            LIMIT 1
        """
        return query, year, rank_limit

    query = f"""
        SELECT "State", COUNT(DISTINCT "Institution_ID") AS institution_count
        FROM public."NIRF"
        WHERE "Year" = {year}
          AND "Rank_Numeric" BETWEEN 1 AND {rank_limit}
          AND "State" IS NOT NULL
        GROUP BY "State"
        ORDER BY institution_count {direction}, "State" ASC
        LIMIT 1
    """
    return query, year, rank_limit


def _institution_question_query(
    question: str,
) -> tuple[str, str, str] | None:
    normalized_question = question.casefold()
    institution = next(
        (
            canonical
            for alias, canonical in INSTITUTION_ALIASES.items()
            if re.search(rf"\b{re.escape(alias)}\b", question, re.IGNORECASE)
            or canonical.casefold() in normalized_question
        ),
        None,
    )
    if not institution:
        return None

    asks_score = bool(re.search(r"\bscore\b", question, re.IGNORECASE))
    asks_rank = bool(re.search(r"\brank\b", question, re.IGNORECASE))
    asks_trend = bool(
        re.search(
            r"\b(trend|tend|over\s+time|historical|history)\b",
            question,
            re.IGNORECASE,
        )
    )
    asks_chart = is_chart_request(question)
    year_match = re.search(r"\b(20\d{2})\b", question)
    year_filter = (
        f' AND "Year" = {int(year_match.group(1))}'
        if year_match
        else ""
    )
    escaped_name = institution.replace("'", "''")

    if asks_trend and asks_rank:
        query = f"""
            SELECT "Institution_ID", "Institution_Name", "Year", "Category",
                   "Score", "Rank_Numeric", "State", "City"
            FROM public."NIRF"
            WHERE "Category" = 'Overall'
              AND "Rank_Numeric" IS NOT NULL
              AND "Institution_Name" ILIKE '%{escaped_name}%'
              {year_filter}
            ORDER BY "Year" ASC
        """
        return query, institution, "rank_trend"

    if asks_score and (asks_trend or asks_chart):
        query = f"""
            SELECT "Institution_ID", "Institution_Name", "Year", "Category",
                   "Score", "Rank_Numeric", "State", "City"
            FROM public."NIRF"
            WHERE "Category" = 'Overall'
              AND "Score" IS NOT NULL
              AND "Institution_Name" ILIKE '%{escaped_name}%'
              {year_filter}
            ORDER BY "Year" ASC
        """
        return query, institution, "score_trend"

    if re.search(r"\brank\b", question, re.IGNORECASE) and year_match:
        query = f"""
            SELECT "Institution_ID", "Institution_Name", "Year", "Category",
                   "Score", "Rank_Numeric", "State", "City"
            FROM public."NIRF"
            WHERE "Category" = 'Overall'
              AND "Year" = {int(year_match.group(1))}
              AND "Institution_Name" ILIKE '%{escaped_name}%'
            ORDER BY "Rank_Numeric" ASC
            LIMIT 10
        """
        return query, institution, "rank"

    return None


def _institution_question_answer(
    results: dict, institution: str, answer_kind: str
) -> str:
    rows = results["rows"]
    if not rows:
        return f"No matching Overall ranking records were found for {institution}."

    if answer_kind == "rank":
        row = rows[0]
        rank = row.get("Rank_Numeric")
        rank_text = "not ranked" if rank is None else f"ranked #{rank}"
        return (
            f'{institution} was {rank_text} in NIRF Overall for {row["Year"]}. '
            f'Score: {row.get("Score", "not available")}.'
        )

    if answer_kind == "rank_trend":
        points = [
            f'{row["Year"]}: rank {row["Rank_Numeric"]}'
            for row in rows
            if row.get("Rank_Numeric") is not None
        ]
        return (
            f'{institution} Overall rank trend: '
            + "; ".join(points)
            + "."
        )

    first, last = rows[0], rows[-1]
    return (
        f'{institution} has {len(rows)} yearly Overall score records from '
        f'{first["Year"]} to {last["Year"]}. Its score changed from '
        f'{first["Score"]} to {last["Score"]}.'
    )


def _state_rank_count_answer(
    results: dict, year: int, rank_limit: int, is_least: bool = False
) -> str:
    rows = results["rows"]
    extreme = "fewest" if is_least else "most"
    if not rows:
        return (
            f"No state or union territory has institutions ranked in the top {rank_limit} "
            f"for {year} in the available data."
        )
    row = rows[0]
    if is_least and row["institution_count"] == 0:
        return (
            f'{row["State"]} has the fewest institutions ranked in the top '
            f'{rank_limit} in {year}: 0. It is tied with other states and '
            "union territories that also have no institutions in this rank range."
        )
    return (
        f'{row["State"]} has the {extreme} unique institutions in the top '
        f'{rank_limit} for {year}, with {row["institution_count"]} institutions.'
    )


def process_question(question: str):
    chart_requested = is_chart_request(question)
    use_shortcuts = not _needs_generated_sql(question)
    dataset_overview_query = (
        _dataset_overview_query(question) if use_shortcuts else None
    )
    if dataset_overview_query:
        execution = _execute_with_retry(question, dataset_overview_query)
        if isinstance(execution, dict):
            return execution
        generated_sql, results = execution
        answer = _dataset_overview_answer(results)
        return {
            "question": question,
            "generated_sql": generated_sql,
            "results": results,
            "answer": f"Visualization: TEXT\n## Answer\n{answer}",
            "is_chart_request": False,
        }

    state_metric_query = _state_metric_query(question) if use_shortcuts else None
    if state_metric_query:
        query, metric = state_metric_query
        execution = _execute_with_retry(question, query)
        if isinstance(execution, dict):
            return execution
        generated_sql, results = execution
        return {
            "question": question,
            "generated_sql": generated_sql,
            "results": results,
            "answer": _state_metric_answer(results, metric),
            "is_chart_request": True,
        }

    top_score_query = _top_score_query(question) if use_shortcuts else None
    if top_score_query:
        execution = _execute_with_retry(question, top_score_query)
        if isinstance(execution, dict):
            return execution
        generated_sql, results = execution
        answer = generate_answer(question=question, results=results)
        return {
            "question": question,
            "generated_sql": generated_sql,
            "results": results,
            "answer": answer,
            "is_chart_request": chart_requested,
        }

    institution_query = (
        _institution_question_query(question) if use_shortcuts else None
    )
    if institution_query:
        query, institution, answer_kind = institution_query
        execution = _execute_with_retry(question, query)
        if isinstance(execution, dict):
            return execution
        generated_sql, results = execution
        answer = _institution_question_answer(results, institution, answer_kind)
        visualization = (
            "CHART"
            if answer_kind in {"score_trend", "rank_trend"}
            else "TEXT"
        )
        return {
            "question": question,
            "generated_sql": generated_sql,
            "results": results,
            "answer": f"Visualization: {visualization}\n## Answer\n{answer}",
            "is_chart_request": visualization == "CHART",
        }

    state_rank_query = _state_rank_count_query(question) if use_shortcuts else None
    if state_rank_query:
        generated_sql, year, rank_limit = state_rank_query
        is_least = bool(re.search(r"\b(fewest|least)\b", question, re.IGNORECASE))
        execution = _execute_with_retry(question, generated_sql)
        if isinstance(execution, dict):
            return execution
        generated_sql, results = execution
        answer = _state_rank_count_answer(results, year, rank_limit, is_least)
        return {
            "question": question,
            "generated_sql": generated_sql,
            "results": results,
            "answer": f"Visualization: TEXT\n## Answer\n{answer}",
            "is_chart_request": chart_requested,
        }

    try:
        generated_sql = _clean_model_sql(
            sql_chain.invoke({"question": question})
        )
    except Exception as exc:
        raise RuntimeError(
            "Sarvam timed out or failed while converting the question to SQL. "
            "Please try again with a shorter or more specific question."
        ) from exc
    if generated_sql.casefold() == "unsupported_question":
        raise ValueError(
            "This question cannot be answered using the available NIRF columns."
        )
    generated_sql = ensure_chart_columns(generated_sql, question)
    execution = _execute_with_retry(question, generated_sql)
    if isinstance(execution, dict):
        return execution
    generated_sql, results = execution
    answer_warning = None
    try:
        answer = generate_answer(question=question, results=results)
    except AnswerGenerationTimeout as exc:
        answer = _fallback_answer("", results)
        answer_warning = str(exc)
    except Exception:
        answer = _fallback_answer("", results)
        answer_warning = (
            "The database query succeeded, but Sarvam could not write the "
            "answer. The results and chart are still available."
        )

    response = {
        "question": question,
        "generated_sql": generated_sql,
        "results": results,
        "answer": answer,
        "is_chart_request": chart_requested,
    }
    if answer_warning:
        response["answer_warning"] = answer_warning
    return response