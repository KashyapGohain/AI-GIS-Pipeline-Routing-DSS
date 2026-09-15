import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ============================================================
# EXISTING PROJECT ENGINES
# ============================================================
try:
    from routing_engine_v2 import RoutingEngineV2
    from risk_engine_dynamic_v1 import DynamicRiskEngine
    from decision_engine_dynamic_v1 import DynamicDecisionEngine
except Exception as e:
    st.error("Could not import one or more project engines.")
    st.exception(e)
    st.stop()

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="AI-GIS Pipeline Routing DSS",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {font-size:38px;font-weight:700;margin-bottom:0}
.subtitle {font-size:17px;color:#666;margin-top:0;margin-bottom:20px}
</style>
""", unsafe_allow_html=True)

# ============================================================
# ENGINE CACHE
# ============================================================
@st.cache_resource(show_spinner="Loading GIS routing engines...")
def load_engines():
    return (
        RoutingEngineV2(),
        DynamicRiskEngine(),
        DynamicDecisionEngine()
    )

try:
    routing_engine, risk_engine, decision_engine = load_engines()
except Exception as e:
    st.error("GIS engine initialization failed.")
    st.exception(e)
    st.stop()

# ============================================================
# HELPERS
# ============================================================
def val(d, keys, default=np.nan):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d[k]
    return default

def rname(d, key):
    return d.get("route_name", d.get("name", key))

def df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if obj is None:
        return pd.DataFrame()
    return pd.DataFrame(obj)

def risk_call(engine, routes):
    """
    Use the DynamicRiskEngine's validated multi-route interface.

    The engine's working test interface is:
        analyze_routes(routing_result["routes"])

    Do NOT call analyze_route() here because that method in the local
    risk-engine version expects internal risk surfaces/attributes that
    are not initialized by the current engine configuration.
    """
    if hasattr(engine, "analyze_routes"):
        return engine.analyze_routes(routes)

    # Compatibility fallback for a future engine version.
    if hasattr(engine, "analyze"):
        return engine.analyze(routes)

    raise AttributeError(
        "DynamicRiskEngine has no supported multi-route analysis method."
    )

def decision_call(engine, routing, risk):
    """
    Use the validated DecisionEngineDynamicV1 interface.
    """
    if hasattr(engine, "run_decision_analysis"):
        return engine.run_decision_analysis(routing, risk)

    # Compatibility fallbacks.
    if hasattr(engine, "analyze"):
        return engine.analyze(routing, risk)
    if hasattr(engine, "run_analysis"):
        return engine.run_analysis(routing, risk)
    if hasattr(engine, "recommend"):
        return engine.recommend(routing, risk)

    raise AttributeError(
        "DecisionEngineDynamicV1 has no supported analysis method."
    )

def profile_table(decision, profile):
    return df(decision.get("profile_scores", {}).get(profile))

def profile_best(decision, profile):
    t = profile_table(decision, profile)
    if t.empty:
        return None, np.nan
    if "rank" in t.columns:
        t = t.sort_values("rank")
    elif "dss_score" in t.columns:
        t = t.sort_values("dss_score", ascending=False)
    row = t.iloc[0]
    name = row.get("route_name", row.get("route", row.get("name")))
    score = row.get("dss_score", np.nan)
    return name, score

def overall_best(decision):
    rec = decision.get("recommendation", {})
    key = rec.get("recommended_route_key")
    name = rec.get("recommended_route")
    score = rec.get("overall_dss_score", np.nan)

    if name is None:
        t = df(decision.get("overall_ranking"))
        if not t.empty:
            if "overall_rank" in t.columns:
                t = t.sort_values("overall_rank")
            elif "overall_dss_score" in t.columns:
                t = t.sort_values("overall_dss_score", ascending=False)
            row = t.iloc[0]
            key = row.get("route_key", key)
            name = row.get("route_name", row.get("route", name))
            score = row.get("overall_dss_score", row.get("dss_score", score))
    return key, name, score

def key_by_name(routing, name):
    for k, d in routing["routes"].items():
        if rname(d, k) == name:
            return k
    return None

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">🗺️ AI-GIS Pipeline Routing DSS</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Dynamic multi-criteria pipeline route optimization '
    'and decision support system for Assam</div>',
    unsafe_allow_html=True
)

with st.expander("ℹ️ About this system"):
    st.markdown("""
This GIS-based Decision Support System generates four alternative pipeline
routes and evaluates them using physical distance, economic cost,
Environmental Impact Index, terrain/LULC risk and decision-preference weights.

**Routes**
- Route 1 — Minimum Cost
- Route 2 — Environmental Impact
- Route 3 — Balanced
- Route 4 — Shortest Feasible

The Environmental Impact Index is a comparative index, not a physical
environmental measurement.
""")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("📍 Route Configuration")

    st.subheader("Start Location")
    start_lat = st.number_input("Start Latitude", -90.0, 90.0, 27.561900,
                                step=0.000001, format="%.6f")
    start_lon = st.number_input("Start Longitude", -180.0, 180.0, 96.037400,
                                step=0.000001, format="%.6f")

    st.subheader("Destination")
    end_lat = st.number_input("Destination Latitude", -90.0, 90.0, 27.383898,
                              step=0.000001, format="%.6f")
    end_lon = st.number_input("Destination Longitude", -180.0, 180.0, 95.333229,
                              step=0.000001, format="%.6f")

    st.divider()
    st.subheader("Decision Preference")
    profile = st.selectbox(
        "Select routing preference",
        ["Balanced", "Cost Priority", "Environmental Priority", "Shortest Feasible"]
    )

    st.divider()
    run = st.button("🚀 ANALYZE ROUTES", type="primary", use_container_width=True)
    reset = st.button("🔄 Reset", use_container_width=True)

if reset:
    st.session_state.pop("analysis", None)
    st.rerun()

# ============================================================
# ANALYSIS
# ============================================================
if run:
    try:
        progress = st.progress(0)
        status = st.empty()

        status.info("Step 1/3 — Generating alternative GIS routes...")
        progress.progress(10)
        routing = routing_engine.run_route_analysis(
            start_lat=start_lat, start_lon=start_lon,
            end_lat=end_lat, end_lon=end_lon
        )

        status.info("Step 2/3 — Calculating dynamic route risk...")
        progress.progress(55)
        risk = risk_call(
            risk_engine,
            routing["routes"]
        )

        status.info("Step 3/3 — Ranking routes using DSS...")
        progress.progress(75)
        decision = decision_call(decision_engine, routing, risk)

        progress.progress(100)
        status.success("Analysis completed successfully.")

        st.session_state.analysis = {
            "routing": routing,
            "risk": risk,
            "decision": decision,
            "inputs": {
                "start_lat": start_lat, "start_lon": start_lon,
                "end_lat": end_lat, "end_lon": end_lon
            },
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        st.error("Route analysis failed.")
        st.exception(e)
        st.stop()

if "analysis" not in st.session_state:
    st.info("Enter coordinates, select a preference, and click **ANALYZE ROUTES**.")
    st.stop()

A = st.session_state.analysis
routing = A["routing"]
risk = A["risk"]
decision = A["decision"]
inp = A["inputs"]

overall_key, overall_name, overall_score = overall_best(decision)
profile_name, profile_score = profile_best(decision, profile)
selected_name = profile_name or overall_name
selected_score = profile_score if pd.notna(profile_score) else overall_score
selected_key = key_by_name(routing, selected_name) or overall_key
selected_route = routing["routes"].get(selected_key, {})
selected_risk = risk.get(selected_key, {})

# ============================================================
# RECOMMENDATION
# ============================================================
st.header("⭐ Recommended Route")
c1, c2, c3, c4 = st.columns(4)

distance = val(selected_route, ["geodesic_distance_km", "physical_distance_km", "distance_km", "distance"], 0)
risk_value = val(selected_risk, ["combined_risk", "risk"], 0)

with c1:
    st.metric("Recommended Route", selected_name or "Unknown")
with c2:
    st.metric(f"DSS Score ({profile})",
              f"{float(selected_score):.3f}" if pd.notna(selected_score) else "N/A")
with c3:
    st.metric("Distance", f"{float(distance):.2f} km")
with c4:
    st.metric("Risk", f"{float(risk_value):.3f}")

if selected_name == overall_name:
    st.success(f"**{selected_name}** is the preferred route under **{profile}** "
               f"and is also the overall DSS recommendation.")
else:
    st.info(f"Under **{profile}**, the preferred route is **{selected_name}**. "
            f"The overall DSS recommendation is **{overall_name}**.")

# ============================================================
# COMPARISON TABLE
# ============================================================
st.header("📊 Route Comparison")

rows = []
for k, rd in routing["routes"].items():
    rr = risk.get(k, {})
    rows.append({
        "Route": rname(rd, k),
        "Distance (km)": val(rd, ["geodesic_distance_km", "physical_distance_km", "distance_km", "distance"]),
        "Economic Cost": val(rd, ["economic_cost", "total_economic_cost", "base_cost"]),
        "Environmental Impact Index": val(
            rd, ["environmental_impact", "environmental_impact_index", "environmental_cost"]
        ),
        "Risk": val(rr, ["combined_risk", "risk"]),
        "Risk Category": val(rr, ["risk_category", "status"], "Unknown"),
        "_key": k
    })

table = pd.DataFrame(rows)
pt = profile_table(decision, profile)
table["DSS Score"] = np.nan
table["Rank"] = np.nan

if not pt.empty:
    nc = next((x for x in ["route_name", "route", "name"] if x in pt.columns), None)
    if nc:
        for i in table.index:
            m = pt[pt[nc] == table.loc[i, "Route"]]
            if not m.empty:
                table.loc[i, "DSS Score"] = m.iloc[0].get("dss_score", np.nan)
                table.loc[i, "Rank"] = m.iloc[0].get("rank", np.nan)

table = table.sort_values("DSS Score", ascending=False, na_position="last")
show = table.drop(columns=["_key"]).copy()

for col in ["Distance (km)", "Economic Cost", "Environmental Impact Index", "Risk", "DSS Score"]:
    show[col] = pd.to_numeric(show[col], errors="coerce").round(3)
show["Rank"] = pd.to_numeric(show["Rank"], errors="coerce").fillna(0).astype(int)

st.dataframe(
    show[["Route", "Distance (km)", "Economic Cost",
          "Environmental Impact Index", "Risk", "Risk Category",
          "DSS Score", "Rank"]],
    use_container_width=True,
    hide_index=True
)

st.caption(
    "DSS scores are relative to the four candidate routes generated for "
    "the current origin-destination query. Higher DSS score = better "
    "performance under the selected decision profile."
)

# ============================================================
# MAP
# ============================================================
st.header("🗺️ Interactive Route Map")

m = folium.Map(
    location=[(inp["start_lat"] + inp["end_lat"]) / 2,
              (inp["start_lon"] + inp["end_lon"]) / 2],
    zoom_start=9,
    tiles="OpenStreetMap",
    control_scale=True
)

colors = {
    "Route 1": "#1f77b4",
    "Route 2": "#2ca02c",
    "Route 3": "#ff7f0e",
    "Route 4": "#d62728"
}
all_points = []

for k, rd in routing["routes"].items():
    pixels = rd.get("route", [])
    if not pixels:
        continue

    name = rname(rd, k)
    color = next((v for p, v in colors.items() if p in name), "#555555")
    weight = 7 if k == selected_key else 3
    opacity = 1.0 if k == selected_key else 0.65

    step = max(1, int(np.ceil(len(pixels) / 1800)))
    display_pixels = list(pixels[::step])
    if pixels[-1] != display_pixels[-1]:
        display_pixels.append(pixels[-1])

    coords = []
    for row, col in display_pixels:
        try:
            # IMPORTANT: coordinate conversion is a method of the engine.
            lat, lon = routing_engine.pixel_to_coordinate(row, col)
            coords.append([lat, lon])
            all_points.append([lat, lon])
        except Exception:
            continue

    if len(coords) >= 2:
        d = val(rd, ["geodesic_distance_km", "physical_distance_km", "distance_km", "distance"], 0)
        folium.PolyLine(
            coords, color=color, weight=weight, opacity=opacity,
            popup=folium.Popup(f"<b>{name}</b><br>Distance: {float(d):.2f} km"),
            tooltip=name
        ).add_to(m)

folium.Marker(
    [inp["start_lat"], inp["start_lon"]],
    tooltip="START", popup="Start Location",
    icon=folium.Icon(color="green", icon="play")
).add_to(m)

folium.Marker(
    [inp["end_lat"], inp["end_lon"]],
    tooltip="DESTINATION", popup="Destination",
    icon=folium.Icon(color="red", icon="flag")
).add_to(m)

if all_points:
    lats = [x[0] for x in all_points]
    lons = [x[1] for x in all_points]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]],
                 padding=(20, 20))

legend = """
<div style="position:fixed;bottom:30px;left:30px;width:255px;
background:white;border:2px solid grey;z-index:9999;font-size:13px;padding:10px">
<b>Route Alternatives</b><br><br>
<span style="color:#1f77b4">━━</span> Route 1 — Minimum Cost<br>
<span style="color:#2ca02c">━━</span> Route 2 — Environmental Impact<br>
<span style="color:#ff7f0e">━━</span> Route 3 — Balanced<br>
<span style="color:#d62728">━━</span> Route 4 — Shortest Feasible<br><br>
<b>Thicker line = selected route</b>
</div>
"""
m.get_root().html.add_child(folium.Element(legend))
st_folium(m, height=650, use_container_width=True, returned_objects=[])

# ============================================================
# PERFORMANCE CHARTS
# ============================================================
st.header("📈 Route Performance")

a, b = st.columns(2)
with a:
    fig = px.bar(table, x="Route", y="Distance (km)", text="Distance (km)",
                 title="Physical Route Distance")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(xaxis_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with b:
    fig = px.bar(table, x="Route", y="Economic Cost", text="Economic Cost",
                 title="Economic Cost")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(xaxis_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

a, b = st.columns(2)
with a:
    fig = px.bar(table, x="Route", y="Environmental Impact Index",
                 text="Environmental Impact Index",
                 title="Environmental Impact Index")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(xaxis_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with b:
    fig = px.bar(table, x="Route", y="Risk", text="Risk",
                 title="Dynamic Route Risk")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(xaxis_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.subheader(f"🎯 DSS Ranking — {profile}")
dss = table.dropna(subset=["DSS Score"])
if not dss.empty:
    fig = px.bar(dss, x="Route", y="DSS Score", text="DSS Score",
                 title=f"Decision Support Score — {profile}")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(xaxis_title="", yaxis_range=[0, 1.1], showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# RISK
# ============================================================
st.header("⚠️ Dynamic Risk Assessment")

risk_rows = []
for k, rd in routing["routes"].items():
    rr = risk.get(k, {})
    risk_rows.append({
        "Route": rname(rd, k),
        "Mean Slope (°)": val(rr, ["mean_slope_deg", "mean_slope"]),
        "Maximum Slope (°)": val(rr, ["max_slope_deg", "max_slope"]),
        "Terrain Risk": val(rr, ["terrain_risk"]),
        "LULC Risk": val(rr, ["lulc_risk"]),
        "Combined Risk": val(rr, ["combined_risk", "risk"]),
        "Risk Category": val(rr, ["risk_category", "status"], "Unknown"),
        "High-Risk Distance (km)": val(rr, ["high_risk_distance_km"], 0),
        "High-Risk Sections": val(rr, ["high_risk_sections"], 0)
    })

st.dataframe(pd.DataFrame(risk_rows).round(3),
             use_container_width=True, hide_index=True)

st.info(
    "Zero high-risk sections does not mean zero risk. The continuous "
    "Combined Risk value still distinguishes routes."
)

# ============================================================
# PROFILE SUMMARY
# ============================================================
st.header("🎯 Decision Profile Summary")

profiles = []
for p in ["Cost Priority", "Environmental Priority", "Balanced", "Shortest Feasible"]:
    n, s = profile_best(decision, p)
    profiles.append({
        "Decision Profile": p,
        "Preferred Route": n or "N/A",
        "DSS Score": s
    })

st.dataframe(pd.DataFrame(profiles).round(3),
             use_container_width=True, hide_index=True)

# ============================================================
# ROUTE DETAILS
# ============================================================
st.header("🔎 Route Details")

options = [rname(d, k) for k, d in routing["routes"].items()]
detail = st.selectbox("Select a route", options)
dk = key_by_name(routing, detail)

if dk is not None:
    rd = routing["routes"][dk]
    rr = risk.get(dk, {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Distance",
                  f"{float(val(rd, ['physical_distance_km','distance_km','distance'], 0)):.2f} km")
    with c2:
        st.metric("Economic Cost",
                  f"{float(val(rd, ['economic_cost','total_economic_cost','base_cost'], 0)):.2f}")
    with c3:
        st.metric("Environmental Index",
                  f"{float(val(rd, ['environmental_impact','environmental_impact_index','environmental_cost'], 0)):.2f}")
    with c4:
        st.metric("Combined Risk",
                  f"{float(val(rr, ['combined_risk','risk'], 0)):.3f}")

    st.write("**Risk Category:**", val(rr, ["risk_category", "status"], "Unknown"))
    st.write("**Route Cells:**", f"{len(rd.get('route', [])):,}")
    st.write("**High-Risk Distance:**", f"{float(val(rr, ['high_risk_distance_km'], 0)):.2f} km")
    st.write("**High-Risk Sections:**", f"{int(float(val(rr, ['high_risk_sections'], 0)))}")
    st.write("**Straightness:**",
             f"{float(val(rd, ['straightness'], 0)):.3f}")

# ============================================================
# SUMMARY
# ============================================================
st.header("📋 Analysis Summary")

x, y = st.columns(2)
with x:
    st.subheader("Input Coordinates")
    st.write(
        f"**Start:** {inp['start_lat']:.6f}, {inp['start_lon']:.6f}\n\n"
        f"**Destination:** {inp['end_lat']:.6f}, {inp['end_lon']:.6f}"
    )

with y:
    st.subheader("Decision")
    st.write(
        f"**Overall Recommendation:** {overall_name}\n\n"
        f"**Selected Preference:** {profile}\n\n"
        f"**Preference Recommendation:** {selected_name}\n\n"
        f"**DSS Score:** {float(selected_score):.3f}"
    )

with st.expander("🧠 Methodology & Interpretation"):
    st.markdown("""
**Routing:** Four GIS-derived multi-criteria cost surfaces generate
alternative feasible routes.

**Risk:** Dynamic route risk is calculated from terrain/slope and
LULC characteristics along each route.

**DSS:** Economic cost, Environmental Impact Index, combined risk and
physical distance are normalized and combined using the selected
decision profile.

**Score interpretation:** DSS scores are relative to the alternatives
evaluated for the current origin-destination query. A score of 1.0
represents the best alternative in that comparison.

**Environmental Impact Index:** This is a derived comparative index,
not a physical environmental unit.
""")

st.divider()
st.markdown(
    '<div style="text-align:center;color:#777;font-size:13px">'
    'AI-GIS Pipeline Routing Decision Support System<br>'
    'Multi-Criteria Spatial Optimization • Dynamic Risk Assessment • GIS DSS'
    '</div>',
    unsafe_allow_html=True
)
