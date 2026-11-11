import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from nirf_chatbot.llm import llm
from nirf_chatbot.prompts import system_prompt
from nirf_chatbot.visualization import is_chart_request


answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (
        "human",
        """
        User Question:
        {question}

        Database Results:
        {results}

        Follow the visualization decision and response format in the system
        instructions. Use only these results in your answer or chart code.
        """
    )
])

answer_chain = answer_prompt | llm | StrOutputParser()

MAX_ROWS_FOR_MODEL = 100
# Results with any other column are analytical, so the fixed templates below
# would ignore the computed values and the model writes the answer instead.
_TEMPLATE_COLUMNS = {
    "Institution_ID",
    "Institution_Name",
    "Year",
    "Category",
    "Score",
    "avg_score",
    "Rank_Numeric",
    "State",
    "City",
    "Band",
    "Rank_Status",
    "Score_Status",
}


def _results_for_model(results: dict) -> dict:
    rows = results.get("rows", [])
    if len(rows) <= MAX_ROWS_FOR_MODEL:
        return results
    return {
        **results,
        "rows": rows[:MAX_ROWS_FOR_MODEL],
        "note": (
            f"Showing the first {MAX_ROWS_FOR_MODEL} of {len(rows)} rows "
            "returned, in query order."
        ),
    }


class AnswerGenerationTimeout(RuntimeError):
    """Raised when the model times out while composing a natural-language answer."""


def _is_timeout_error(error: Exception) -> bool:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, TimeoutError):
            return True
        if "timeout" in type(current).__name__.casefold():
            return True
        if "timed out" in str(current).casefold():
            return True
        current = current.__cause__ or current.__context__
    return False


def _has_answer_body(answer: str) -> bool:
    body = re.sub(
        r"(?im)^\s*Visualization:\s*(?:CHART|TEXT)\s*$",
        "",
        answer,
    )
    body = re.sub(r"(?m)^\s*##\s*(?:Answer|Chart code)\s*$", "", body)
    body = re.sub(r"```python\s*.*?```", "", body, flags=re.IGNORECASE | re.DOTALL)
    return bool(body.strip())


def _fallback_answer(answer: str, results: dict) -> str:
    if _has_answer_body(answer):
        return answer

    chart_match = re.search(
        r"(?im)^\s*Visualization:\s*(CHART|TEXT)\s*$",
        answer,
    )
    visualization = chart_match.group(1).upper() if chart_match else "TEXT"
    rows = results.get("rows", [])
    if not rows:
        message = "No matching records were returned."
    else:
        years = sorted(
            {
                row["Year"]
                for row in rows
                if row.get("Year") is not None
            }
        )
        if years:
            message = (
                f"The results contain {len(rows)} matching records "
                f"covering {years[0]} through {years[-1]}. See the Results and "
                "Chart tabs for details."
            )
        else:
            message = (
                f"The query returned {len(rows)} matching records. "
                "See the Results and Chart tabs for details."
            )
    return f"Visualization: {visualization}\n## Answer\n{message}"


def _format_score(value) -> str:
    try:
        return f"{float(value):.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _grounded_answer(question: str, results: dict) -> str | None:
    rows = results.get("rows", [])
    visualization = "CHART" if is_chart_request(question) else "TEXT"
    if not rows:
        return (
            f"Visualization: {visualization}\n## Answer\n"
            "No matching records were returned. Try specifying the Overall "
            "category and a year from 2017–2025, or ask for the top 10 "
            "institutions by score."
        )

    fields = rows[0].keys()
    if "Institution_Name" not in fields or "Year" not in fields or "Category" not in fields:
        return None
    if set(fields) - _TEMPLATE_COLUMNS:
        return None

    score_field = next(
        (field for field in ("Score", "avg_score") if field in fields),
        None,
    )
    rank_field = "Rank_Numeric" if "Rank_Numeric" in fields else None
    years = sorted(
        {row["Year"] for row in rows if row.get("Year") is not None}
    )
    categories = sorted(
        {row["Category"] for row in rows if row.get("Category") is not None}
    )

    if score_field:
        if len(years) > 1:
            is_rank_trend = bool(
                re.search(r"\brank\b", question, re.IGNORECASE)
                and "Rank_Numeric" in fields
                and not re.search(r"\bscore\b", question, re.IGNORECASE)
            )
            ordered = sorted(
                rows,
                key=lambda row: (
                    row.get("Year") is None,
                    row.get("Year"),
                    row.get("Institution_Name") or "",
                ),
            )
            if is_rank_trend:
                points = [
                    f'{row["Year"]}: {row["Institution_Name"]} '
                    f'(rank {row["Rank_Numeric"]})'
                    for row in ordered
                    if row.get("Rank_Numeric") is not None
                ]
                summary = (
                    f"Rank trend for {', '.join(categories) or 'the selected category'} "
                    f"({years[0]}–{years[-1]}): " + "; ".join(points[:30])
                )
            else:
                points = [
                    f'{row["Year"]}: {row["Institution_Name"]} '
                    f'({_format_score(row.get(score_field))})'
                    for row in ordered
                    if row.get(score_field) is not None
                ]
                summary = (
                    f"Scores by year for {', '.join(categories) or 'the selected category'} "
                    f"({years[0]}–{years[-1]}): " + "; ".join(points[:30])
                )
        else:
            ordered = sorted(
                rows,
                key=lambda row: (
                    row.get(score_field) is None,
                    -float(row[score_field])
                    if row.get(score_field) is not None
                    else 0,
                    row.get("Institution_ID") or "",
                ),
            )
            is_score_top = bool(
                re.search(r"\b(top|highest|best)\b.*\b(score|scoring)\b"
                          r"|\b(score|scoring)\b.*\b(top|highest|best)\b",
                          question, re.IGNORECASE)
            )
            is_rank_top = bool(
                re.search(r"\b(top|best)\b.*\b(rank|ranked)\b", question, re.IGNORECASE)
            )
            if is_rank_top and rank_field:
                ordered.sort(
                    key=lambda row: (
                        row.get(rank_field) is None,
                        row.get(rank_field) if row.get(rank_field) is not None else 0,
                    )
                )
            details = []
            for index, row in enumerate(ordered, start=1):
                value = row.get(score_field)
                if value is None:
                    continue
                details.append(
                    f'{index}. {row["Institution_Name"]} '
                    f'({_format_score(value)})'
                )
            category = categories[0] if categories else "selected"
            year = years[0] if years else "latest year"
            if is_score_top or is_rank_top:
                metric = "by rank" if is_rank_top else "by score"
                if score_field == "avg_score":
                    metric = "by average score"
                summary = (
                    f"Top {len(details)} {category} institutes in {year} {metric}: "
                    + ", ".join(details[:10])
                )
                if len(details) > 10:
                    summary += f" (showing the first 10 of {len(details)} results)"
            elif len(details) == 1:
                row = ordered[0]
                summary = (
                    f'{row["Institution_Name"]} scored {_format_score(row[score_field])} '
                    f"in {category} in {year}."
                )
            else:
                metric = "average score" if score_field == "avg_score" else "score"
                summary = (
                    f"{category} {metric} results for {year}: "
                    + ", ".join(details[:10])
                )
    elif rank_field:
        ordered = sorted(
            rows,
            key=lambda row: (
                row.get(rank_field) is None,
                row.get(rank_field) if row.get(rank_field) is not None else 0,
            ),
        )
        category = categories[0] if categories else "selected"
        year = years[0] if years else "latest year"
        details = [
            f'{index}. {row["Institution_Name"]} '
            f'(rank {row[rank_field] if row.get(rank_field) is not None else "not ranked"})'
            for index, row in enumerate(ordered, start=1)
        ]
        summary = (
            f"Institutions in {category} for {year} by rank: "
            + ", ".join(details[:10])
        )
    else:
        return None

    return f"Visualization: {visualization}\n## Answer\n{summary}"


def generate_answer(question: str, results: dict):
    grounded = _grounded_answer(question, results)
    if grounded is not None:
        return grounded

    try:
        answer = answer_chain.invoke({
            "question": question,
            "results": _results_for_model(results)
        })
    except Exception as exc:
        if _is_timeout_error(exc):
            raise AnswerGenerationTimeout(
                "The database query succeeded, but Sarvam timed out while "
                "writing the answer. The results and chart are still available."
            ) from exc
        raise
    return _fallback_answer(answer or "", results)