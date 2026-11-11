import re

import pandas as pd
import streamlit as st

from nirf.maps import show_map_attribution
from nirf_chatbot.visualization import (
    CHART_TYPES,
    HOVER_FIELDS,
    build_plotly_figure,
    combine_state_rows,
    infer_chart_spec,
    state_map_labels,
    validate_chart_spec,
)

_VISUALIZATION_MARKER = re.compile(
    r"(?im)^\s*Visualization:\s*(CHART|TEXT)\s*$"
)
_PYTHON_CODE_BLOCK = re.compile(
    r"```python\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL
)


def parse_answer(answer: str) -> tuple[bool, str, str | None]:
    marker = _VISUALIZATION_MARKER.search(answer)
    wants_chart = bool(marker and marker.group(1).upper() == "CHART")
    answer_body = answer[marker.end() :] if marker else answer
    code_match = _PYTHON_CODE_BLOCK.search(answer_body)
    chart_code = (
        code_match.group(1).strip() if wants_chart and code_match else None
    )
    if code_match:
        answer_body = answer_body[: code_match.start()]
    answer_body = re.sub(
        r"(?m)^\s*##\s*(Answer|Chart code)\s*$", "", answer_body
    )
    return wants_chart, answer_body.strip(), chart_code


def results_frame(results: dict) -> pd.DataFrame:
    columns = results.get("columns") or None
    rows = results.get("rows") or []
    return pd.DataFrame(rows, columns=columns)


def render_results_chart(
    frame: pd.DataFrame,
    key_prefix: str,
    question: str = "",
) -> str | None:
    spec = validate_chart_spec(frame, infer_chart_spec(frame, question))
    numeric_columns = [
        column
        for column in frame.columns
        if pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    if not numeric_columns:
        st.info("These results do not contain numeric values to chart.")
        return None

    chart_types = (
        CHART_TYPES
        if "State" in frame.columns
        else [chart for chart in CHART_TYPES if chart != "India map"]
    )
    controls = st.columns(4)
    with controls[0]:
        chart_type = st.selectbox(
            "Chart type",
            chart_types,
            index=chart_types.index(spec["chart_type"]),
            key=f"{key_prefix}_chart_type",
        )
    x_options = ["State"] if chart_type == "India map" else list(frame.columns)
    y_options = (
        numeric_columns
        if chart_type == "India map"
        or chart_type != "Bar"
        or spec["orientation"] != "h"
        else list(frame.columns)
    )
    with controls[1]:
        x_axis = st.selectbox(
            "Value / X axis"
            if chart_type == "Bar" and spec["orientation"] == "h"
            else "Category / X axis",
            x_options,
            index=x_options.index(spec["x"]),
            key=f"{key_prefix}_x_axis",
        )
    with controls[2]:
        y_axis = st.selectbox(
            "Category / Y axis"
            if chart_type == "Bar" and spec["orientation"] == "h"
            else "Value / Y axis",
            y_options,
            index=y_options.index(spec["y"]),
            key=f"{key_prefix}_y_axis",
        )
    group_options = ["None"] + [
        column
        for column in frame.columns
        if column not in numeric_columns and column not in {x_axis, y_axis}
    ]
    default_group = spec.get("color")
    if default_group not in group_options:
        default_group = "None"
    with controls[3]:
        group_by = st.selectbox(
            "Color / group by",
            group_options,
            index=group_options.index(default_group),
            key=f"{key_prefix}_color_group",
        )

    map_frame = frame
    selected_category = None
    if chart_type == "India map":
        # Rankings are per category, so a map shows one category at a time.
        category_column = next(
            (
                column
                for column in ("Category", "Category_Group")
                if column in frame.columns
            ),
            None,
        )
        if category_column:
            categories = sorted(frame[category_column].dropna().astype(str).unique())
            if len(categories) > 1:
                selected_category = st.selectbox(
                    "Map category",
                    categories,
                    index=(
                        categories.index("Overall")
                        if "Overall" in categories
                        else 0
                    ),
                    key=f"{key_prefix}_map_category",
                )
                map_frame = map_frame[
                    map_frame[category_column].astype(str).eq(selected_category)
                ]
    if chart_type == "India map" and "Year" in map_frame.columns:
        years = sorted(
            pd.to_numeric(map_frame["Year"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if len(years) > 1:
            selected_year = st.selectbox(
                "Map year",
                years,
                index=len(years) - 1,
                key=f"{key_prefix}_map_year",
            )
            map_frame = map_frame[
                pd.to_numeric(map_frame["Year"], errors="coerce").eq(selected_year)
            ]

    orientation = "v"
    if chart_type == "Bar":
        x_is_numeric = x_axis in numeric_columns
        y_is_numeric = y_axis in numeric_columns
        if x_is_numeric and not y_is_numeric:
            orientation = "h"
        elif not x_is_numeric and y_is_numeric:
            orientation = "v"
        else:
            orientation = spec["orientation"]
    # A map colors states by the metric, and combined rows drop other columns.
    color = None if group_by == "None" or chart_type == "India map" else group_by
    map_labels = {}
    if chart_type == "India map":
        metric_label, map_title = state_map_labels(y_axis, question)
        if selected_category:
            map_title = f"{map_title} ({selected_category})"
        map_labels = {"metric_label": metric_label, "title": map_title}
    spec = validate_chart_spec(
        map_frame,
        {
            **spec,
            "chart_type": chart_type,
            "x": x_axis,
            "y": y_axis,
            "color": color,
            "orientation": orientation,
            **map_labels,
        },
    )

    columns = list(dict.fromkeys(
        [
            x_axis,
            y_axis,
            *([color] if color else []),
            *[column for column in HOVER_FIELDS if column in map_frame.columns],
        ]
    ))
    chart_data = map_frame[columns].copy()
    for column in columns:
        if column in numeric_columns or column in {"Year", "Score", "Rank_Numeric"}:
            chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    chart_data = chart_data.dropna(subset=[x_axis, y_axis])
    if chart_type == "India map":
        chart_data = combine_state_rows(chart_data, y_axis)
    if chart_type == "Bar" and orientation == "h" and x_axis in numeric_columns:
        chart_data = chart_data.sort_values(x_axis, ascending=False)
    elif chart_type in {"Line", "Area"} and x_axis == "Year":
        chart_data = chart_data.sort_values(x_axis)
    if chart_data.empty:
        st.info("There are no numeric values available for this chart.")
        return None

    if len(chart_data) > 1:
        row_limit = st.slider(
            "Rows to display",
            min_value=1,
            max_value=len(chart_data),
            value=(
                len(chart_data)
                if chart_type == "India map"
                else min(20, len(chart_data))
            ),
            key=f"{key_prefix}_row_limit",
        )
        chart_data = chart_data.head(row_limit)
    figure, plot_code = build_plotly_figure(chart_data, spec)
    figure.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
    st.plotly_chart(
        figure,
        use_container_width=True,
        theme="streamlit",
        key=f"{key_prefix}_chart",
    )
    if chart_type == "India map":
        show_map_attribution()
    return plot_code


def _render_turn(turn: dict, index: int) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])

    response = turn["response"]
    with st.chat_message("assistant"):
        if response.get("answer_warning"):
            st.warning(response["answer_warning"])
        wants_chart, answer_text, _ = parse_answer(response["answer"])
        wants_chart = (
            wants_chart
            or response.get("is_chart_request", False)
        )
        frame = results_frame(response.get("results", {}))
        if not answer_text:
            answer_text = (
                f"Found {len(frame)} matching records. See the Results and "
                "Chart tabs for the returned data."
                if not frame.empty
                else "No matching records were found."
            )
        st.markdown(answer_text)
        tabs = st.tabs(["Results", "Chart", "Generated SQL", "Plot code"])
        plot_code = None

        with tabs[0]:
            if frame.empty:
                st.info("No matching database rows were returned.")
            else:
                st.dataframe(frame, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download results as CSV",
                    data=frame.to_csv(index=False),
                    file_name="nirf_results.csv",
                    mime="text/csv",
                    key=f"chat_csv_{index}",
                )

        with tabs[1]:
            if not wants_chart:
                st.info("Ask for a graph, chart, plot, or trend to visualize these results.")
            elif frame.empty:
                st.info("No matching database rows were returned to plot.")
            else:
                plot_code = render_results_chart(
                    frame,
                    f"chat_{index}",
                    turn["question"],
                )

        with tabs[2]:
            st.code(response["generated_sql"], language="sql")

        with tabs[3]:
            if plot_code:
                st.code(plot_code, language="python")
                st.download_button(
                    "Download chart code",
                    data=plot_code,
                    file_name="nirf_chart.py",
                    mime="text/x-python",
                    key=f"chat_code_{index}",
                )
                st.caption("This is the Plotly code used by the chart renderer.")
            else:
                st.info("No plotting code was returned for this response.")


def render_chat_page(
    title: str = "NIRF Analytics Chatbot",
    subtitle: str = "Ask questions about NIRF rankings and explore the results.",
) -> None:
    st.title(title)
    st.caption(subtitle)
    st.caption('Connected data source: PostgreSQL `public."NIRF"`')

    history = st.session_state.setdefault("nirf_chat_history", [])
    if st.button("Clear conversation", key="clear_nirf_chat"):
        history.clear()

    if not history:
        st.info(
            "Try asking: “Show the top 10 institutes by rank” or "
            "“Plot the top institutes by score”."
        )
    for index, turn in enumerate(history):
        _render_turn(turn, index)

    question = st.chat_input("Ask about NIRF rankings, scores, or institutions")
    if question and question.strip():
        try:
            with st.spinner("Querying NIRF data and preparing your results..."):
                from nirf_chatbot.chat_service import process_question

                response = process_question(question.strip())
            history.append({"question": question.strip(), "response": response})
            st.rerun()
        except Exception as exc:
            st.error(f"Unable to answer this question: {exc}")
