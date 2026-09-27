"""Interactive map of NC high schools. Run with: streamlit run map.py"""

from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st


DATA_FILE = Path(__file__).with_name("map_data.csv")
PERCENTILE = "Composite Score Percentile Rank"
COMPOSITE_SCORE = "Composite Score"
GROWTH_OPTIONS = ["Exceeded", "Met", "NotMet", "Unavailable"]
POINT_LAYER_ID = "schools"


@st.cache_data
def load_schools(path: str, modified_at: float) -> pd.DataFrame:
    """Read the notebook export; modified_at refreshes the cache after an export."""
    schools = pd.read_csv(path, dtype={"agency_code": "string", "nces_ncessch": "string"})
    required = {
        "agency_code", "Name", "County", COMPOSITE_SCORE, PERCENTILE,
        "graduation_pct", "graduation_display", "growth_status",
        "english_ii_pct", "english_ii_display", "math_1_pct", "math_1_display",
        "act_unc_pct", "act_unc_display",
        "latitude", "longitude", "meets_academic_criteria",
    }
    missing = required.difference(schools.columns)
    if missing:
        raise ValueError(f"map_data.csv is missing columns: {', '.join(sorted(missing))}")

    numeric = [COMPOSITE_SCORE, PERCENTILE, "graduation_pct", "english_ii_pct",
               "math_1_pct", "act_unc_pct", "latitude", "longitude"]
    for column in numeric:
        schools[column] = pd.to_numeric(schools[column], errors="coerce")
    schools["meets_academic_criteria"] = (
        schools["meets_academic_criteria"].astype("string").str.lower().eq("true")
    )
    schools["growth_status"] = schools["growth_status"].fillna("Unavailable")
    return schools.dropna(subset=["latitude", "longitude"]).copy()


def meets_minimum(schools: pd.DataFrame, prefix: str, minimum: int) -> pd.Series:
    if minimum == 0:
        return pd.Series(True, index=schools.index)
    exact = schools[f"{prefix}_pct"].ge(minimum).fillna(False)
    above_95 = schools[f"{prefix}_display"].eq(">95%").fillna(False)
    return exact | (above_95 & (minimum <= 95))


def filter_schools(
    schools: pd.DataFrame,
    barrier_column: str,
    barrier_range: tuple[int, int],
    graduation_min: int,
    english_min: int,
    math_min: int,
    growth_choices: list[str],
    counties: list[str],
    shortlist_only: bool,
    act_min: int = 0,
) -> pd.DataFrame:
    mask = schools[barrier_column].between(*barrier_range).fillna(False)
    mask &= meets_minimum(schools, "graduation", graduation_min)
    mask &= meets_minimum(schools, "english_ii", english_min)
    mask &= meets_minimum(schools, "math_1", math_min)
    mask &= meets_minimum(schools, "act_unc", act_min)
    mask &= schools["growth_status"].isin(growth_choices)
    if counties:
        mask &= schools["County"].isin(counties)
    if shortlist_only:
        mask &= schools["meets_academic_criteria"]
    return schools.loc[mask].copy()


def display_rate(value: object) -> str:
    return str(value) if pd.notna(value) else "Not reported"


def make_deck(schools: pd.DataFrame) -> pdk.Deck:
    points = pd.DataFrame({
        "agency_code": schools["agency_code"],
        "school": schools["Name"],
        "county": schools["County"],
        "latitude": schools["latitude"],
        "longitude": schools["longitude"],
        "barrier_score": schools[COMPOSITE_SCORE],
        "barrier_percentile": schools[PERCENTILE],
        "growth": schools["growth_status"],
        "graduation": schools["graduation_display"].map(display_rate),
        "english_ii": schools["english_ii_display"].map(display_rate),
        "math_1": schools["math_1_display"].map(display_rate),
        "act_unc": schools["act_unc_display"].map(display_rate),
    })
    points["color"] = [
        [17, 138, 112, 220] if shortlisted else
        [229, 134, 49, 210] if percentile >= 80 else
        [55, 111, 175, 195]
        for shortlisted, percentile in zip(
            schools["meets_academic_criteria"], schools[PERCENTILE]
        )
    ]

    layer = pdk.Layer(
        "ScatterplotLayer", id=POINT_LAYER_ID, data=points,
        get_position="[longitude, latitude]", get_fill_color="color",
        get_radius=420, radius_min_pixels=5, radius_max_pixels=12,
        pickable=True, auto_highlight=True, stroked=True,
        get_line_color=[255, 255, 255, 220], line_width_min_pixels=1,
    )
    return pdk.Deck(
        # Streamlit uses None to choose a basemap matching the active theme.
        layers=[layer], map_style=pdk.map_styles.DARK,
        initial_view_state=pdk.ViewState(latitude=35.55, longitude=-79.75, zoom=6.3),
        tooltip={  # pyright: ignore[reportArgumentType]
            "html": (
                "<b>{school}</b><br/>{county}<br/>"
                "ODIS composite: {barrier_score} (national percentile {barrier_percentile})<br/>"
                "Graduation: {graduation}<br/>Growth: {growth}<br/>"
                "English II: {english_ii}<br/>Math 1: {math_1}<br/>"
                "ACT meeting UNC minimum composite: {act_unc}"
            ),
            "style": {"backgroundColor": "#243446", "color": "white"},
        },
    )


def main() -> None:
    st.set_page_config(page_title="NC school outcomes map", layout="wide")
    st.title("NC high schools")
    st.caption(
        "2026 ODIS community barriers + 2024–25 NC DPI outcomes; "
        "2024–25 NCES school locations. Matched non-charter schools only."
    )

    if not DATA_FILE.exists():
        st.error("map_data.csv is missing. Run the map cells in main.ipynb first.")
        st.stop()
    try:
        schools = load_schools(str(DATA_FILE), DATA_FILE.stat().st_mtime)
    except (ValueError, OSError) as error:
        st.error(str(error))
        st.stop()
    if schools.empty:
        st.error("map_data.csv has no school coordinates. Run the NCES matching cell in main.ipynb.")
        st.stop()

    st.sidebar.header("Filters")
    scale = st.sidebar.selectbox(
        "ODIS barrier measure", ["National percentile", "Raw composite score"],
        help="The percentile compares communities nationally; the raw composite is the ODIS score.",
    )
    if scale == "National percentile":
        barrier_column = PERCENTILE
        barrier_range = st.sidebar.slider(
            "National ODIS percentile", 0, 100, (80, 100),
            help="80–100 is the high-barrier group used in the analysis.",
        )
    else:
        barrier_column = COMPOSITE_SCORE
        low = int(schools[COMPOSITE_SCORE].min())
        high = int(schools[COMPOSITE_SCORE].max())
        barrier_range = st.sidebar.slider("ODIS composite score", low, high, (low, high))

    st.sidebar.subheader("Academic outcomes")
    graduation_min = st.sidebar.slider("Minimum graduation rate (%)", 0, 95, 0)
    english_min = st.sidebar.slider("Minimum English II proficiency (%)", 0, 95, 0)
    math_min = st.sidebar.slider("Minimum Math 1 proficiency (%)", 0, 95, 0)
    act_min = st.sidebar.slider(
        "Minimum ACT meeting UNC composite benchmark (%)", 0, 95, 0,
        help="Percentage of tested students meeting the UNC minimum ACT composite score; "
             "this is not an average ACT score.",
    )
    growth_choices = st.sidebar.multiselect(
        "Growth status", GROWTH_OPTIONS, default=GROWTH_OPTIONS,
    )
    shortlist_count = int(schools["meets_academic_criteria"].sum())
    shortlist_only = st.sidebar.checkbox(
        f"Only academic shortlist schools ({shortlist_count})"
    )
    counties = st.sidebar.multiselect("County", sorted(schools["County"].dropna().unique()))
    st.sidebar.caption(
        "A 0% minimum applies no rate filter. For a positive minimum, a masked "
        "‘>95%’ rate qualifies; a masked ‘<5%’ or missing rate does not."
    )

    filtered = filter_schools(
        schools, barrier_column, barrier_range, graduation_min, english_min,
        math_min, growth_choices, counties, shortlist_only, act_min=act_min,
    )
    count_col, shortlist_col, counties_col = st.columns(3)
    count_col.metric("Schools shown", f"{len(filtered)} / {len(schools)}")
    shortlist_col.metric("Academic shortlist", int(filtered["meets_academic_criteria"].sum()))
    counties_col.metric("Counties shown", filtered["County"].nunique())
    st.caption(
        "Green: academic shortlist · Orange: other high-barrier schools "
        "(national percentile ≥80) · Blue: other schools. Click a point for details."
    )

    if filtered.empty:
        st.info("No schools meet these filters. Lower a minimum or widen the barrier range.")
        return

    event = st.pydeck_chart(
        make_deck(filtered), height=610, on_select="rerun",
        selection_mode="single-object", key="school_map",
    )
    selected_objects = event.selection.objects.get(POINT_LAYER_ID, [])
    if selected_objects:
        selected_code = str(selected_objects[0]["agency_code"])
        selected = filtered.loc[filtered["agency_code"].eq(selected_code)]
        if not selected.empty:
            school = selected.iloc[0]
            st.subheader(f"{school['Name']} · {school['County']}")
            first, second, third, fourth, fifth = st.columns(5)
            first.metric("Graduation", display_rate(school["graduation_display"]))
            second.metric("Growth", school["growth_status"])
            third.metric("English II", display_rate(school["english_ii_display"]))
            fourth.metric("Math 1", display_rate(school["math_1_display"]))
            fifth.metric("ACT: UNC composite", display_rate(school["act_unc_display"]))
            st.caption(
                f"ODIS composite {school[COMPOSITE_SCORE]:g}; "
                f"national percentile {school[PERCENTILE]:g}. "
                "English II and Math 1 are proficiency rates. ACT shows the percentage "
                "meeting the UNC minimum composite, not an average ACT score."
            )

    with st.expander("View and download the filtered schools"):
        table = filtered[[
            "Name", "County", COMPOSITE_SCORE, PERCENTILE,
            "graduation_display", "growth_status", "english_ii_display",
            "math_1_display", "act_unc_display", "meets_academic_criteria",
        ]].rename(columns={
            "Name": "School", COMPOSITE_SCORE: "ODIS composite",
            PERCENTILE: "National percentile", "graduation_display": "Graduation",
            "growth_status": "Growth", "english_ii_display": "English II",
            "math_1_display": "Math 1", "act_unc_display": "ACT: UNC composite",
            "meets_academic_criteria": "Academic shortlist",
        })
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button(
            "Download filtered CSV", table.to_csv(index=False).encode("utf-8"),
            file_name="filtered_nc_schools.csv", mime="text/csv",
        )

    st.caption(
        "The ODIS-to-DPI name/county matches are provisional. This map shows "
        "associations, not causes of school performance."
    )


if __name__ == "__main__":
    main()
