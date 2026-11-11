from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .theme import (
    CARD,
    DIVERGING_SCALE,
    NO_DATA,
    SEQUENTIAL_SCALE,
    SEQUENTIAL_SCALE_REVERSED,
)


CANONICAL_STATES = (
    "Andaman and Nicobar Islands",
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chandigarh",
    "Chhattisgarh",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu and Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
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
STATE_NAME_MAP = {state: state for state in CANONICAL_STATES}
GEOJSON_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "geo"
    / "india_states_osm.geojson"
)

_METRIC_LABELS = {
    "institutions_in_top_n": "Institutions in top N",
    "share_of_top_n": "Share of top N (%)",
    "average_score": "Average score",
    "best_rank": "Best rank",
    "new_institutions": "New institutions",
    "change_in_count": "Change in institutions",
}
_COLUMNS = (
    "year",
    "category",
    "institution_id",
    "state",
    "score",
    "rank_numeric",
)


@st.cache_resource
def load_geojson() -> dict[str, Any]:
    if not GEOJSON_PATH.is_file():
        raise FileNotFoundError(f"India state GeoJSON is missing: {GEOJSON_PATH}")
    with GEOJSON_PATH.open(encoding="utf-8") as geojson_file:
        geojson = json.load(geojson_file)
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("India state GeoJSON must be a FeatureCollection.")
    features = geojson.get("features", [])
    names = [
        feature.get("properties", {}).get("name")
        for feature in features
    ]
    if not features or any(not name for name in names):
        raise ValueError(
            "India state GeoJSON features must include properties.name."
        )
    if len(names) != len(set(names)):
        raise ValueError("India state GeoJSON must contain unique state names.")
    return geojson


def state_name_coverage(
    data_states: list[str] | set[str] | tuple[str, ...],
    geojson: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    source = geojson if geojson is not None else load_geojson()
    geojson_names = {
        feature["properties"]["name"] for feature in source["features"]
    }
    mapped_names = {
        STATE_NAME_MAP.get(state, state) for state in data_states
    }
    unmatched_data = sorted(
        state
        for state in data_states
        if STATE_NAME_MAP.get(state, state) not in geojson_names
    )
    geojson_without_data = sorted(geojson_names - mapped_names)
    return unmatched_data, geojson_without_data


def report_state_coverage(df: pd.DataFrame) -> None:
    columns = {column.casefold(): column for column in df.columns}
    state_column = columns.get("state")
    if state_column is None:
        raise ValueError("State coverage requires a state column.")
    states = set(df[state_column].dropna().astype(str).str.strip())
    unmatched, without_data = state_name_coverage(states)
    if unmatched:
        st.warning(f"GeoJSON features missing for data states: {', '.join(unmatched)}")
    if without_data:
        st.caption(
            "Boundary areas with no NIRF state records (shown without data): "
            + ", ".join(without_data)
        )


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = {column.casefold(): column for column in df.columns}
    rename = {
        columns[name]: name
        for name in _COLUMNS
        if name in columns and columns[name] != name
    }
    return df.rename(columns=rename).copy()


def _scope(
    df: pd.DataFrame,
    category: str | None,
    year: int | None = None,
) -> pd.DataFrame:
    work = _standardize_columns(df)
    if "state" not in work:
        raise ValueError("State metrics require a state column.")
    if category is not None:
        if "category" not in work:
            raise ValueError("Category filtering requires a category column.")
        work = work[work["category"].eq(category)]
    if year is not None:
        if "year" not in work:
            raise ValueError("Year filtering requires a year column.")
        work = work[pd.to_numeric(work["year"], errors="coerce").eq(int(year))]
    return work.dropna(subset=["state"]).copy()


def _ranked_top_n(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if "rank_numeric" not in df or "institution_id" not in df:
        raise ValueError("Top-N metrics require rank_numeric and institution_id.")
    ranked = df.copy()
    ranked["rank_numeric"] = pd.to_numeric(
        ranked["rank_numeric"], errors="coerce"
    )
    return ranked[
        ranked["rank_numeric"].notna()
        & ranked["rank_numeric"].le(top_n)
        & ranked["institution_id"].notna()
    ]


def _counts_by_state(df: pd.DataFrame, top_n: int) -> pd.Series:
    ranked = _ranked_top_n(df, top_n)
    return ranked.groupby("state")["institution_id"].nunique()


def _state_rows(
    states: pd.Series | list[str] | set[str],
    values: pd.Series,
    metric: str,
    institution_counts: pd.Series | None = None,
    fill_value: float | None = 0,
) -> pd.DataFrame:
    state_names = sorted(set(states))
    result = pd.DataFrame({"state": state_names})
    result[metric] = result["state"].map(values)
    if fill_value is not None:
        result[metric] = result[metric].fillna(fill_value)
    if institution_counts is None:
        result["institution_count"] = result[metric]
    else:
        result["institution_count"] = (
            result["state"].map(institution_counts).fillna(0)
        )
    return result


def build_state_metrics(
    df: pd.DataFrame,
    metric: str,
    year: int | tuple[int, int] | None,
    category: str | None,
    top_n: int = 100,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if metric not in _METRIC_LABELS:
        raise ValueError(f"Unsupported state metric: {metric}")
    if top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    if metric == "change_in_count":
        if not isinstance(year, tuple) or len(year) != 2:
            raise ValueError("change_in_count requires a (from_year, to_year) pair.")
        from_year, to_year = map(int, year)
        if from_year >= to_year:
            raise ValueError("The change comparison requires from_year < to_year.")
        work = _scope(df, category)
        work["year"] = pd.to_numeric(work["year"], errors="coerce")
        pair = work[work["year"].isin([from_year, to_year])]
        present_years = set(pair["year"].dropna().astype(int))
        if present_years != {from_year, to_year}:
            return pd.DataFrame(
                columns=["state", metric, "institution_count"]
            )
        by_year = {
            selected_year: _counts_by_state(
                pair[pair["year"].eq(selected_year)], top_n
            )
            for selected_year in (from_year, to_year)
        }
        states = set(pair["state"].dropna().astype(str))
        state_names = sorted(states)
        changes = pd.Series(
            {
                state: by_year[to_year].get(state, 0)
                - by_year[from_year].get(state, 0)
                for state in state_names
            }
        )
        return _state_rows(
            state_names,
            changes,
            metric,
            by_year[to_year],
        )

    if isinstance(year, tuple):
        raise ValueError(f"{metric} accepts one year, not a year pair.")
    work = _scope(df, category, year)
    if work.empty:
        return pd.DataFrame(columns=["state", metric, "institution_count"])
    states = set(work["state"].dropna().astype(str))

    if metric in {"institutions_in_top_n", "share_of_top_n"}:
        counts = _counts_by_state(work, top_n)
        if metric == "share_of_top_n":
            total = int(counts.sum())
            values = counts.astype(float).mul(100 / total) if total else counts.astype(float)
        else:
            values = counts
        return _state_rows(states, values, metric, counts)

    if metric == "average_score":
        if "score" not in work or "institution_id" not in work:
            raise ValueError("Average score requires score and institution_id.")
        work["score"] = pd.to_numeric(work["score"], errors="coerce")
        scored = work.dropna(subset=["score", "institution_id"])
        values = scored.groupby("state")["score"].mean()
        counts = scored.groupby("state")["institution_id"].nunique()
        return _state_rows(states, values, metric, counts, fill_value=None)

    if metric == "best_rank":
        if "rank_numeric" not in work or "institution_id" not in work:
            raise ValueError("Best rank requires rank_numeric and institution_id.")
        work["rank_numeric"] = pd.to_numeric(
            work["rank_numeric"], errors="coerce"
        )
        ranked = work.dropna(subset=["rank_numeric", "institution_id"])
        values = ranked.groupby("state")["rank_numeric"].min()
        counts = ranked.groupby("state")["institution_id"].nunique()
        return _state_rows(states, values, metric, counts, fill_value=None)

    selected_year = int(year) if year is not None else None
    if selected_year is None or "year" not in work or "institution_id" not in work:
        raise ValueError("New institution metrics require a selected year and institution_id.")
    year_values = pd.to_numeric(work["year"], errors="coerce")
    current = work[year_values.eq(selected_year)].dropna(
        subset=["institution_id"]
    )
    source = _standardize_columns(history_df if history_df is not None else df)
    if "year" not in source or "institution_id" not in source:
        raise ValueError("First-observed year metrics require year and institution_id.")
    if category is not None:
        if "category" not in source:
            raise ValueError("Category filtering requires a category column.")
        source = source[source["category"].eq(category)]
    source["year"] = pd.to_numeric(source["year"], errors="coerce")
    source = source.dropna(subset=["year", "institution_id"])
    first_year = source["year"].min()
    if pd.isna(first_year) or selected_year <= int(first_year):
        return _state_rows(states, pd.Series(dtype=float), metric)
    first_seen = source.groupby("institution_id")["year"].min()
    current = current[
        current["institution_id"].map(first_seen).eq(selected_year)
    ]
    counts = current.groupby("state")["institution_id"].nunique()
    return _state_rows(states, counts, metric)


def make_choropleth(
    state_df: pd.DataFrame,
    value_col: str,
    title: str,
    colorscale: str | list | None = None,
    diverging: bool = False,
    color_range: tuple[float, float] | None = None,
    animation_frame: str | None = None,
    metric_label: str | None = None,
) -> go.Figure:
    if colorscale is None:
        # Lower ranks are better, so they take the brightest color.
        colorscale = (
            DIVERGING_SCALE
            if diverging
            else SEQUENTIAL_SCALE_REVERSED
            if "rank" in value_col.casefold()
            else SEQUENTIAL_SCALE
        )
    geojson = load_geojson()
    features = geojson["features"]
    feature_names = [feature["properties"]["name"] for feature in features]
    background = go.Choropleth(
        geojson=geojson,
        locations=feature_names,
        z=[0] * len(feature_names),
        featureidkey="properties.name",
        colorscale=[[0, NO_DATA], [1, NO_DATA]],
        zmin=0,
        zmax=1,
        showscale=False,
        marker={"line": {"color": CARD, "width": 0.5}},
        hoverinfo="skip",
        name="No data",
    )
    if state_df.empty:
        figure = go.Figure(data=[background])
        figure.update_layout(title=title)
        figure.update_geos(
            fitbounds="locations",
            visible=False,
            bgcolor="rgba(0,0,0,0)",
        )
        return figure

    columns = {column.casefold(): column for column in state_df.columns}
    state_column = columns.get("state")
    metric_column = columns.get(value_col.casefold())
    if state_column is None or metric_column is None:
        raise ValueError("Choropleth data requires State and the selected metric.")
    count_column = columns.get("institution_count")
    map_data = pd.DataFrame(
        {
            "state": state_df[state_column].astype(str),
            "metric_value": pd.to_numeric(
                state_df[metric_column], errors="coerce"
            ),
        }
    )
    if count_column is None:
        map_data["institution_count"] = (
            map_data["metric_value"]
            if "count" in value_col.casefold()
            else pd.NA
        )
    else:
        map_data["institution_count"] = pd.to_numeric(
            state_df[count_column], errors="coerce"
        )
    frame_column = None
    if animation_frame is not None:
        frame_column = columns.get(animation_frame.casefold())
        if frame_column is None:
            raise ValueError(
                f"Animation requires the {animation_frame!r} column."
            )
        map_data[animation_frame] = state_df[frame_column]

    map_data["geojson_name"] = map_data["state"].map(
        lambda state: STATE_NAME_MAP.get(state, state)
    )
    unmatched = sorted(
        set(map_data.loc[
            ~map_data["geojson_name"].isin(feature_names), "state"
        ])
    )
    if unmatched:
        raise ValueError(f"India GeoJSON has no feature for data states: {unmatched}")

    key_columns = ["geojson_name"]
    if frame_column is not None:
        key_columns.append(animation_frame)
    if map_data.duplicated(key_columns).any():
        raise ValueError("Choropleth data must have one row per state and frame.")
    map_data = map_data.dropna(subset=["metric_value"])
    if map_data.empty:
        return make_choropleth(
            pd.DataFrame(),
            value_col,
            title,
            colorscale,
            diverging,
            color_range,
            metric_label=value_col.replace("_", " ").title(),
        )

    values = map_data["metric_value"]
    if diverging:
        bound = max(abs(float(values.min())), abs(float(values.max())))
        color_range = color_range or (-bound, bound)
        if color_range[0] == color_range[1]:
            color_range = (-1, 1)
    elif color_range is None and values.min() == values.max():
        padding = max(abs(float(values.iloc[0])) * 0.01, 0.5)
        color_range = (
            float(values.iloc[0]) - padding,
            float(values.iloc[0]) + padding,
        )

    metric_label = metric_label or _METRIC_LABELS.get(
        value_col, value_col.replace("_", " ").title()
    )
    hover_data: dict[str, Any] = {
        "state": True,
        "metric_value": True,
        "institution_count": True,
        "geojson_name": False,
    }
    if frame_column is not None:
        hover_data[animation_frame] = True
    value_figure = px.choropleth(
        map_data,
        geojson=geojson,
        locations="geojson_name",
        featureidkey="properties.name",
        color="metric_value",
        color_continuous_scale=colorscale,
        color_continuous_midpoint=0 if diverging else None,
        range_color=color_range,
        hover_name="state",
        hover_data=hover_data,
        animation_frame=animation_frame,
        labels={
            "state": "State",
            "metric_value": metric_label,
            "institution_count": "Institutions",
        },
        title=title,
    )
    frames = []
    for frame in value_figure.frames:
        trace_indexes = frame.traces or tuple(range(len(frame.data)))
        frame.traces = tuple(index + 1 for index in trace_indexes)
        frames.append(frame)
    figure = go.Figure(
        data=[background, *value_figure.data],
        frames=frames,
    )
    figure.update_layout(value_figure.layout)
    figure.update_layout(height=600, margin={"l": 0, "r": 0, "t": 60, "b": 0})
    figure.update_traces(
        marker_line_color=CARD,
        marker_line_width=0.5,
        selector={"name": ""},
    )
    figure.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
    )
    return figure


def render_choropleth(
    state_df: pd.DataFrame,
    value_col: str,
    title: str,
    key: str,
    colorscale: str | list | None = None,
    diverging: bool = False,
    color_range: tuple[float, float] | None = None,
    animation_frame: str | None = None,
) -> go.Figure | None:
    if state_df.empty or not state_df[value_col].notna().any():
        st.info("No geographic data remains after applying the selected filters.")
        return None
    figure = make_choropleth(
        state_df,
        value_col,
        title,
        colorscale,
        diverging,
        color_range,
        animation_frame,
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        theme="streamlit",
        key=key,
    )
    return figure


def show_map_attribution() -> None:
    st.caption(
        "Boundary data © OpenStreetMap contributors, available under ODbL 1.0. "
        "Community-mapped; not an official Survey of India boundary source. "
        "See https://www.openstreetmap.org/copyright."
    )
