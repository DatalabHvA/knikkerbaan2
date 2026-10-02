import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import base64

st.set_page_config(page_title="Knikkerbaan Demo", layout="wide")
github_token = st.secrets["api_keys"]["gh_token"]

# ============================================================================
# CONSTANTEN
# ============================================================================
KENMERKEN = ["aantal_kort", "aantal_recht", "aantal_gebogen"]
DOEL = "tijd_s"
KENMERK_LABELS = {
    "aantal_kort": "aantal korte stukken",
    "aantal_recht": "aantal rechte stukken",
    "aantal_gebogen": "aantal gebogen stukken",
}
DATA_BESTAND = "knikkerbaan_data.csv"
MODEL_BESTANDEN = {
    "lm": "knikkerbaan_model_lm.pkl",
    "rf": "knikkerbaan_model_rf.pkl",
    "dt": "knikkerbaan_model_dt.pkl",
}

# ============================================================================
# HELPERS
# ============================================================================
def _is_fitted_lm(m): return hasattr(m, "coef_") and hasattr(m, "intercept_")
def _is_fitted_dt(m): return hasattr(m, "tree_") and getattr(m.tree_, "node_count", 0) > 0
def _is_fitted_rf(m): return hasattr(m, "estimators_") and len(getattr(m, "estimators_", [])) > 0

def _nieuwe_modellen():
    return (
        LinearRegression(),
        RandomForestRegressor(n_estimators=200, random_state=42),
        DecisionTreeRegressor(max_depth=4, random_state=42),
    )

# ============================================================================
# FORMULES
# ============================================================================
def lineaire_formule_coef(model_lm):
    if not _is_fitted_lm(model_lm): return None
    b0 = model_lm.intercept_
    b_kort, b_recht, b_gebogen = model_lm.coef_
    return (b0, b_kort, b_recht, b_gebogen)

# ============================================================================
# BESLISBOOM
# ============================================================================
def _num_leaves(tree, nid):
    l, r = tree.children_left[nid], tree.children_right[nid]
    if l == r == -1: return 1
    return (_num_leaves(tree, l) if l != -1 else 0) + (_num_leaves(tree, r) if r != -1 else 0)

def _assign_positions(tree, nid, depth, x_min, x_max, pos):
    l, r = tree.children_left[nid], tree.children_right[nid]
    y = -depth
    if l == r == -1:
        pos[nid] = ((x_min + x_max) / 2, y)
        return
    nL = _num_leaves(tree, l) if l != -1 else 0
    nR = _num_leaves(tree, r) if r != -1 else 0
    mid = x_min + (x_max - x_min) * (nL / max(1, nL + nR))
    if l != -1: _assign_positions(tree, l, depth + 1, x_min, mid, pos)
    if r != -1: _assign_positions(tree, r, depth + 1, mid, x_max, pos)
    if l != -1 and r != -1:
        x = (pos[l][0] + pos[r][0]) / 2
    elif l != -1:
        x = pos[l][0]
    elif r != -1:
        x = pos[r][0]
    else:
        x = (x_min + x_max) / 2
    pos[nid] = (x, y)

def draw_tree_with_path(model, features, x_row):
    tree = model.tree_
    pos = {}
    _assign_positions(tree, 0, 0, 0.0, 1.0, pos)

    node_indicator = model.decision_path(x_row)
    path_nodes = list(node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()

    # Edges
    for nid, (x, y) in pos.items():
        l, r = tree.children_left[nid], tree.children_right[nid]
        if l != -1:
            x2, y2 = pos[l]
            ax.plot([x, x2], [y - 0.025, y2 + 0.025], color="0.7", lw=1.5, zorder=1)
            ax.text((x + x2)/2, (y + y2)/2 - 0.05, "Nee", fontsize=9, ha="center", va="top", color="0.3")
        if r != -1:
            x2, y2 = pos[r]
            ax.plot([x, x2], [y - 0.025, y2 + 0.025], color="0.7", lw=1.5, zorder=1)
            ax.text((x + x2)/2, (y + y2)/2 - 0.05, "Ja", fontsize=9, ha="center", va="top", color="0.3")

    # Path
    for i in range(len(path_nodes) - 1):
        x1, y1 = pos[path_nodes[i]]
        x2, y2 = pos[path_nodes[i+1]]
        ax.plot([x1, x2], [y1 - 0.025, y2 + 0.025], color="red", lw=3.2, zorder=3)

    # Nodes
    for nid, (x, y) in pos.items():
        l, r = tree.children_left[nid], tree.children_right[nid]
        if l == r == -1:
            label = f"{tree.value[nid][0][0]:.0f}s"
        else:
            feat = features[tree.feature[nid]]
            label = f"{KENMERK_LABELS.get(feat, feat)} ≥ {tree.threshold[nid]:.1f}"
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFD6D6" if nid in path_nodes else "#E6F0FE",
                         edgecolor="black", linewidth=1.0), zorder=4)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(min(y for _, y in pos.values()) - 0.08, 0.08)
    return fig

# ============================================================================
# STATE MANAGEMENT
# ============================================================================
def reset_alles():
    Path(DATA_BESTAND).unlink(missing_ok=True)
    for f in MODEL_BESTANDEN.values():
        Path(f).unlink(missing_ok=True)
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def _train_modellen():
    X, y = st.session_state.data[KENMERKEN], st.session_state.data[DOEL]
    st.session_state.model_lm.fit(X, y)
    st.session_state.model_rf.fit(X, y)
    st.session_state.model_dt.fit(X, y)
    joblib.dump(st.session_state.model_lm, MODEL_BESTANDEN["lm"])
    joblib.dump(st.session_state.model_rf, MODEL_BESTANDEN["rf"])
    joblib.dump(st.session_state.model_dt, MODEL_BESTANDEN["dt"])

def verwijder_rij(idx):
    st.session_state.data = st.session_state.data.drop(idx).reset_index(drop=True)
    if len(st.session_state.data) >= 2:
        _train_modellen()
    else:
        st.session_state.model_lm, st.session_state.model_rf, st.session_state.model_dt = _nieuwe_modellen()
        for f in MODEL_BESTANDEN.values():
            Path(f).unlink(missing_ok=True)
    st.session_state.data.to_csv(DATA_BESTAND, index=False)

def werk_modellen_bij(aantal_kort, aantal_recht, aantal_gebogen, tijd_s):
    nieuwe_rij = pd.DataFrame([[aantal_kort, aantal_recht, aantal_gebogen, tijd_s]],
                              columns=KENMERKEN + [DOEL])
    st.session_state.data = pd.concat([st.session_state.data, nieuwe_rij], ignore_index=True)
    _train_modellen()
    st.session_state.data.to_csv(DATA_BESTAND, index=False)

def laad_of_init_state():
    if "model_version" not in st.session_state:
        st.session_state.model_version = 0

    if "last_update" not in st.session_state:
        st.session_state.last_update = datetime.now()

    if "data" not in st.session_state:
        if Path(DATA_BESTAND).exists():
            df = pd.read_csv(DATA_BESTAND)
            for col in KENMERKEN:
                if col not in df.columns: df[col] = 0
            st.session_state.data = df[KENMERKEN + [DOEL]] if DOEL in df.columns else pd.DataFrame(columns=KENMERKEN + [DOEL])
        else:
            st.session_state.data = pd.DataFrame(columns=KENMERKEN + [DOEL])

    model_lm = joblib.load(MODEL_BESTANDEN["lm"]) if Path(MODEL_BESTANDEN["lm"]).exists() else LinearRegression()
    model_rf = joblib.load(MODEL_BESTANDEN["rf"]) if Path(MODEL_BESTANDEN["rf"]).exists() else RandomForestRegressor(n_estimators=200, random_state=42)
    model_dt = joblib.load(MODEL_BESTANDEN["dt"]) if Path(MODEL_BESTANDEN["dt"]).exists() else DecisionTreeRegressor(max_depth=4, random_state=42)

    df = st.session_state.data
    if len(df) >= 2:
        X, y = df[KENMERKEN], df[DOEL]
        if not _is_fitted_lm(model_lm): model_lm.fit(X, y)
        if not _is_fitted_rf(model_rf): model_rf.fit(X, y)
        if not _is_fitted_dt(model_dt): model_dt.fit(X, y)

    st.session_state.model_lm = model_lm
    st.session_state.model_rf = model_rf
    st.session_state.model_dt = model_dt

def upload_knikkerbaan_data(api_token, df):
    OWNER = "DatalabHvA"
    REPO = "knikkerbaan2"
    FILE_PATH = DATA_BESTAND

    csv_text = df.to_csv(index=False)
    csv_bytes = csv_text.encode("utf-8")
    content_b64 = base64.b64encode(csv_bytes).decode()

    headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/vnd.github+json",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
            }

    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{FILE_PATH}"

    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        sha = r.json()["sha"]
    else:
        sha = None  # nieuw bestand

    payload = {
        "message": "Update CSV via API " + datetime.now().strftime("%d-%m-%Y, %H:%M:%S"),
        "content": content_b64,
    }

    if sha:
        payload["sha"] = sha

    response = requests.put(url, headers=headers, json=payload)
    return response.status_code

# ============================================================================
# APP
# ============================================================================


laad_of_init_state()

col1, col2 = st.columns(2)
with col1:
    st.title("Hoe lang duurt de knikkerbaan?")
    st.write("Bouw je knikkerbaan. Vul in hoeveel korte stukken, rechte stukken en gebogen stukken je hebt gebruikt. De computer voorspelt hoe lang de knikkerbaan duurt.")
    st.write("Laat dan de knikker lopen en meet hoe lang hij erover doet. Vul de gemeten tijd in als nieuwe meting.")

with col2:
    st.title("Voorspelling")

    # Alle beschikbare modellen met hun "is getraind"-check
    modellen = {
        "Lineair model": (st.session_state.model_lm, _is_fitted_lm),
        "Beslisboom": (st.session_state.model_dt, _is_fitted_dt),
        "Random Forest": (st.session_state.model_rf, _is_fitted_rf),
    }

    keuze = st.radio(
        "Kies een model",
        options=list(modellen.keys()),
        horizontal=True,
    )

    model, is_fitted = modellen[keuze]


st.divider()

col1, col2 = st.columns([1,1])

with col1:

    # Invoer
    st.subheader("Jouw experiment")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1: aantal_kort = st.slider("Aantal korte stukken", 0, 15, 1, 1)
    with c2: aantal_recht = st.slider("Aantal rechte stukken", 0, 15, 1, 1)
    with c3: aantal_gebogen = st.slider("Aantal gebogen stukken", 0, 15, 1, 1)

    x_row = np.array([[aantal_kort, aantal_recht, aantal_gebogen]], dtype=float)

    # Nieuwe meting
    st.divider()
    st.subheader("Nieuwe meting toevoegen")
    tijd_s = st.number_input("Gemeten knikkertijd in seconden", min_value=0.0, step=0.1, format="%.1f", value=0.0)

    if st.button("Model bijwerken", type="primary"):
        werk_modellen_bij(aantal_kort, aantal_recht, aantal_gebogen, tijd_s)
        st.session_state.model_version += 1
        st.success("✅ Modellen bijgewerkt!")
        if ((datetime.now() - st.session_state.last_update).total_seconds() > 60):
            upload_knikkerbaan_data(github_token, st.session_state.data)
            st.session_state.last_update = datetime.now()

        st.rerun()

with col2:

    st.header(f"Voorspelling van {keuze}")
    if is_fitted(model):
        y_pred = model.predict(x_row)[0]
        st.metric(keuze, f"{y_pred:.1f} seconden")
    else:
        st.warning("⚠️ Model nog niet getraind. Voeg minimaal 2 metingen toe.")

    if keuze == "Beslisboom":
        st.header("🌳 Beslisboom")
        if is_fitted(model):
            st.pyplot(draw_tree_with_path(model, KENMERKEN, x_row), use_container_width=True)
            leaf_id = model.apply(x_row)[0]
            st.caption(f"🎯 Pad eindigt bij blad {leaf_id} ({model.tree_.value[leaf_id][0][0]:.1f} seconden)")
        else:
            st.info("ℹ️ De beslisboom is nog niet getraind.")

    elif keuze == "Lineair model":
        st.header("📐 Formule lineaire regressie")
        if is_fitted(model):
            coef = lineaire_formule_coef(model)
            st.write(f"We beginnen met {coef[0]:.1f} seconden knikkertijd")

            if coef[1] >= 0:
                st.write(f"Voor ieder extra kort stuk is er {coef[1]:.1f} seconden extra knikkertijd")
            else:
                st.write(f"Voor ieder extra kort stuk is er {-coef[1]:.1f} seconden minder knikkertijd")

            if coef[2] >= 0:
                st.write(f"Voor ieder extra recht stuk is er {coef[2]:.1f} seconden extra knikkertijd")
            else:
                st.write(f"Voor ieder extra recht stuk is er {-coef[2]:.1f} seconden minder knikkertijd")

            if coef[3] >= 0:
                st.write(f"Voor ieder extra gebogen stuk is er {coef[3]:.1f} seconden extra knikkertijd")
            else:
                st.write(f"Voor ieder extra gebogen stuk is er {-coef[3]:.1f} seconden minder knikkertijd")
        else:
            st.info("ℹ️ Het lineair model is nog niet getraind.")

    elif keuze == "Random Forest":
        st.header("🌲 Random Forest")
        st.write("Een random forest bestaat uit een groot aantal beslisbomen.")
        st.write("Elke boom wordt getraind op een willekeurige selectie van de metingen.")
        st.write("De uiteindelijke voorspelling is het gemiddelde van alle bomen.")

# Data met verwijder-knoppen
st.divider()
st.header("📊 Gegevens (training set)")
st.caption(f"Er zijn {len(st.session_state.data)} metingen gedaan.")

if st.session_state.data.empty:
    st.write("Nog geen metingen opgeslagen.")
else:
    for idx, row in st.session_state.data.iterrows():
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
        col1.write(f"**{int(row['aantal_kort'])}** kort")
        col2.write(f"**{int(row['aantal_recht'])}** recht")
        col3.write(f"**{int(row['aantal_gebogen'])}** gebogen")
        col4.write(f"{row[DOEL]:.1f} seconden")
        if col5.button("🗑️", key=f"del_{idx}"):
            verwijder_rij(idx)
            st.rerun()

with st.sidebar:
    st.header("⚙️ Instellingen")
    if st.button("🗑️ Reset alles", type="secondary", use_container_width=True):
        reset_alles()
        st.success("✅ Alles verwijderd!")
        st.rerun()

    st.header("📊 Samenvatting")

    df = st.session_state.data
    if df.empty:
        st.info("Nog geen metingen. Voeg er een paar toe!")
    else:
        st.metric("Aantal metingen", len(df))
        st.metric("Langste knikkertijd", f"{df[DOEL].max():.1f}")
        st.metric("Gemiddelde knikkertijd", f"{df[DOEL].mean():.1f}")

st.caption(f"📁 {DATA_BESTAND}, {', '.join(MODEL_BESTANDEN.values())}")
