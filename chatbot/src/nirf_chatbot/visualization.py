import re
import pandas as pd
import plotly.express as px
import sqlglot
from sqlglot import exp
from nirf.maps import STATE_NAME_MAP, load_geojson, make_choropleth


_CHART_INTENT = re.compile(
    r"\b(?:graph|chart|plot|visuali[sz](?:e|ation)|draw|compare|comparison|trend)\b"
    r"|\bover\s+time\b",
    re.IGNORECASE,
)


def is_chart_request(question: str) -> bool:
    return bool(_CHART_INTENT.search(question))


def ensure_chart_columns(query: str, question: str) -> str:
    if not is_chart_request(question):
        return query

    try:
        statement = sqlglot.parse_one(query, read="postgres")
    except sqlglot.errors.SqlglotError:
        return query
    if not isinstance(statement, exp.Select):
        return query
    if statement.args.get("group") or statement.args.get("distinct"):
        return query
    if statement.find(exp.AggFunc):
        return query
    # Only a plain read of the base table is guaranteed to expose these
    # columns; a CTE, join, or derived table may not have them.
    if statement.find(exp.With, exp.Join, exp.Subquery, exp.Window):
        return query

    projections = statement.expressions
    if any(isinstance(item, exp.Star) for item in projections):
        return query

    selected_columns = {
        item.name.casefold()
        for item in projections
        if isinstance(item, exp.Column)
    }
    required_columns = ["Institution_Name", "Score", "Rank_Numeric"]
    for column in required_columns:
        if column.casefold() not in selected_columns:
            projections.append(exp.column(column, quoted=True))

    return statement.sql(dialect="postgres")


CHART_TYPES = ["Bar", "Line", "Area", "Scatter", "India map"]
STATE_TO_GEOJSON = STATE_NAME_MAP
HOVER_FIELDS = [
    "Institution_Name",
    "Year",
    "Category",
    "Score",
    "Rank_Numeric",
    "State",
    "City",
]


_load_india_geojson = load_geojson


def _is_best_rank(metric: str) -> bool:
    name = metric.casefold()
    return "rank" in name and not re.search(r"avg|average|mean", name)


def state_map_labels(metric: str, question: str = "") -> tuple[str, str]:
    """Return the legend label and title describing ``metric`` on the map."""
    name = metric.casefold()
    if "count" in name:
        top_match = re.search(r"\btop\s*(\d+)\b", question, re.IGNORECASE)
        metric_label = f"Institutions in top {top_match.group(1) if top_match else '100'}"
    elif _is_best_rank(metric):
        metric_label = "Best rank"
    elif "score" in name:
        metric_label = "Average score"
    else:
        metric_label = metric.replace("_", " ").capitalize()
    return metric_label, f"{metric_label} by State"


def combine_state_rows(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Reduce results to the one row per state that a choropleth requires.

    Results often hold several rows per state (one per institution or
    category). Ranks keep the best (lowest) value, counts are summed, and any
    other metric is averaged, matching the labels from ``infer_chart_spec``.
    """
    if not frame["State"].duplicated().any():
        return frame

    rows = frame.assign(
        **{metric: pd.to_numeric(frame[metric], errors="coerce")}
    ).dropna(subset=["State", metric])
    metric_name = metric.casefold()
    how = "sum" if "count" in metric_name else "min" if _is_best_rank(metric) else "mean"
    grouped = rows.groupby("State")
    combined = grouped[metric].agg(how).reset_index()

    if "count" not in metric_name:
        if "institution_count" in rows.columns:
            counts = pd.to_numeric(rows["institution_count"], errors="coerce")
            combined["institution_count"] = (
                counts.groupby(rows["State"]).sum().reindex(combined["State"]).values
            )
        elif "Institution_Name" in rows.columns:
            combined["institution_count"] = (
                grouped["Institution_Name"]
                .nunique()
                .reindex(combined["State"])
                .values
            )
    return combined


def _india_state_map(
    frame: pd.DataFrame, spec: dict
) -> tuple[object, str]:
    metric = spec["y"]
    title = spec.get("title", f"{metric.replace('_', ' ').title()} by State")
    metric_label = spec.get("metric_label", metric.replace("_", " ").title())
    frame = combine_state_rows(frame, metric)
    figure = make_choropleth(
        frame,
        metric,
        title,
        metric_label=metric_label,
    )
    code = (
        "from nirf.maps import make_choropleth\n\n"
        "# chart_data holds one row per state; repeated rows are combined first:\n"
        "# best rank, summed counts, or the average of any other metric.\n"
        "fig = make_choropleth(\n"
        "    chart_data,\n"
        f"    value_col={metric!r},\n"
        f"    title={title!r},\n"
        f"    metric_label={metric_label!r},\n"
        ")\n"
    )
    return figure, code


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]


def infer_chart_spec(frame: pd.DataFrame, question: str = "") -> dict:
    columns = list(frame.columns)
    numeric = _numeric_columns(frame)
    categories = [
        column for column in ("Category", "Category_Group") if column in columns
    ]
    color = categories[0] if categories and frame[categories[0]].nunique() > 1 else None
    question_mentions_rank = bool(
        re.search(r"\brank\b", question, re.IGNORECASE)
    )

    if "Institution_Name" in columns and "Year" in columns:
        measure = (
            "Rank_Numeric"
            if question_mentions_rank and "Rank_Numeric" in numeric
            else "Score"
            if "Score" in numeric
            else "avg_score"
            if "avg_score" in numeric
            else "Rank_Numeric"
            if "Rank_Numeric" in numeric
            else None
        )
        if measure:
            years = pd.to_numeric(frame["Year"], errors="coerce").nunique()
            if years > 1:
                return {
                    "chart_type": "Line",
                    "x": "Year",
                    "y": measure,
                    "color": "Institution_Name",
                    "orientation": "v",
                    "title": f"{measure} by institution over time",
                    "markers": True,
                }
            if measure in {"Score", "avg_score"}:
                return {
                    "chart_type": "Bar",
                    "x": measure,
                    "y": "Institution_Name",
                    "color": color,
                    "orientation": "h",
                    "title": f"Top {frame['Category'].iloc[0] if 'Category' in columns else ''} institutes by {measure}",
                    "markers": False,
                }

    if "State" in columns:
        metric = next(
            (column for column in numeric if column not in {"Year"}),
            None,
        )
        if metric:
            metric_label, title = state_map_labels(metric, question)
            return {
                "chart_type": "India map",
                "x": "State",
                "y": metric,
                "color": None,
                "orientation": "v",
                "title": title,
                "metric_label": metric_label,
                "markers": False,
            }

    if "Year" in columns and "Year" in numeric:
        metrics = [column for column in numeric if column != "Year"]
        if metrics and all(
            column in numeric or column in {"Category", "Category_Group"}
            for column in columns
        ):
            question_text = question.casefold()
            metric = max(
                metrics,
                key=lambda column: sum(
                    token in question_text
                    for token in column.casefold().split("_")
                ),
            )
            return {
                "chart_type": "Line",
                "x": "Year",
                "y": metric,
                "color": color,
                "orientation": "v",
                "title": f"{metric} by year",
                "markers": True,
            }

    if "Year" not in columns and len(numeric) >= 2:
        name = "Institution_Name" if "Institution_Name" in columns else None
        return {
            "chart_type": "Scatter",
            "x": numeric[0],
            "y": numeric[1],
            "color": None,
            "orientation": "v",
            "title": f"{numeric[1]} vs {numeric[0]}",
            "markers": False,
            "hover_name": name,
        }

    if numeric:
        category = next((column for column in columns if column not in numeric), None)
        value = numeric[0]
        if category:
            return {
                "chart_type": "Bar",
                "x": value if category == "Institution_Name" else category,
                "y": category if category == "Institution_Name" else value,
                "color": color,
                "orientation": "h" if category == "Institution_Name" else "v",
                "title": f"{value} by {category}",
                "markers": False,
            }
        return {
            "chart_type": "Bar",
            "x": value,
            "y": value,
            "color": None,
            "orientation": "v",
            "title": value,
            "markers": False,
        }

    return {
        "chart_type": "Bar",
        "x": columns[0] if columns else "",
        "y": columns[0] if columns else "",
        "color": None,
        "orientation": "v",
        "title": "Query results",
        "markers": False,
    }


def validate_chart_spec(frame: pd.DataFrame, spec: dict) -> dict:
    columns = set(frame.columns)
    valid = (
        spec.get("chart_type") in CHART_TYPES
        and (
            spec.get("chart_type") != "India map"
            or "State" in columns
        )
        and spec.get("x") in columns
        and spec.get("y") in columns
        and spec.get("orientation") in {"h", "v"}
        and (spec.get("color") is None or spec.get("color") in columns)
    )
    return spec if valid else infer_chart_spec(frame)


def build_plotly_figure(frame: pd.DataFrame, spec: dict) -> tuple[object, str]:
    spec = validate_chart_spec(frame, spec)
    chart_type = spec["chart_type"]
    x_axis = spec["x"]
    y_axis = spec["y"]
    color = spec.get("color")
    orientation = spec.get("orientation", "v")
    title = spec.get("title", "Query results")
    if chart_type == "India map":
        return _india_state_map(frame, spec)

    hover_fields = [
        column
        for column in HOVER_FIELDS
        if column in frame.columns and column not in {x_axis, y_axis, color}
    ]
    included = list(dict.fromkeys([x_axis, y_axis, *([color] if color else []), *hover_fields]))
    chart_data = frame[included].copy()

    for column in included:
        if column in _numeric_columns(frame):
            chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    chart_data = chart_data.dropna(subset=[x_axis, y_axis])

    if chart_type == "Bar" and orientation == "h":
        sort_column = x_axis
        chart_data = chart_data.sort_values(sort_column, ascending=False)
    elif chart_type in {"Line", "Area"} and x_axis == "Year":
        chart_data = chart_data.sort_values(x_axis)

    kwargs = {
        "x": x_axis,
        "y": y_axis,
        "color": color,
        "title": title,
        "hover_data": hover_fields,
    }
    if chart_type == "Bar":
        kwargs["orientation"] = orientation
        figure = px.bar(chart_data, **kwargs)
        figure.update_layout(barmode="group")
        if orientation == "h" and y_axis == "Institution_Name":
            figure.update_yaxes(autorange="reversed")
    elif chart_type == "Line":
        figure = px.line(chart_data, markers=True, **kwargs)
        if y_axis == "Rank_Numeric":
            figure.update_yaxes(autorange="reversed")
    elif chart_type == "Area":
        figure = px.area(chart_data, **kwargs)
    else:
        figure = px.scatter(
            chart_data,
            hover_name=spec.get("hover_name")
            if spec.get("hover_name") in chart_data.columns
            else None,
            **kwargs,
        )

    code = (
        "import plotly.express as px\n\n"
        "# chart_data contains the selected result rows after sorting and row limiting.\n"
        f"fig = px.{chart_type.casefold()}(chart_data, x={x_axis!r}, y={y_axis!r}, "
        f"color={color!r}, title={title!r}, hover_data={hover_fields!r}"
    )
    if chart_type == "Bar":
        code += f", orientation={orientation!r})\nfig.update_layout(barmode='group')"
        if orientation == "h" and y_axis == "Institution_Name":
            code += "\nfig.update_yaxes(autorange='reversed')"
    elif chart_type == "Line":
        code += ", markers=True)"
        if y_axis == "Rank_Numeric":
            code += "\nfig.update_yaxes(autorange='reversed')"
    elif chart_type == "Scatter":
        hover_name = (
            spec.get("hover_name")
            if spec.get("hover_name") in chart_data.columns
            else None
        )
        if hover_name:
            code += f", hover_name={hover_name!r})"
        else:
            code += ")"
    else:
        code += ")"
    return figure, code
