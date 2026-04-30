import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from world import World
np.set_printoptions(
    linewidth=200,       # Längere Zeilen
    precision=3,         # Nur 3 Nachkommastellen anzeigen
    suppress=True        # Verhindert die wissenschaftliche Notation (z.B. 1e-05 wird zu 0.000)
)

# ==========================================
# 1. INITIALISIERUNG
# ==========================================
# Hilfsfunktion global definieren, damit sie überall (Init + Buttons) verfügbar ist
def update_history_all(res, is_append=True):
    (s, u, o, total_F, total_E, total_H, F_A, F_B, F_C, F_D, F_U, 
     E_A, E_B, E_C, E_D, E_U, 
     H_A, H_B, H_s, H_o, H_u) = res
    
    bel_map = {"s_0":s[0],"s_1":s[1],"s_2":s[2],"u_0":u[0],"u_1":u[1],"o_0":o[0],"o_1":o[1],"o_2":o[2]}
    met_map = {
        "A0_metrics": [E_A[0], H_A[0], F_A[0]], "A1_metrics": [E_A[1], H_A[1], F_A[1]], "A2_metrics": [E_A[2], H_A[2], F_A[2]],
        "B0_metrics": [E_B[0], H_B[0], F_B[0]], "B1_metrics": [E_B[1], H_B[1], F_B[1]],
        "U0_metrics": [E_U[0], H_u[0], F_U[0]], "U1_metrics": [E_U[1], H_u[1], F_U[1]],
        "C0_metrics": [E_C[0], H_o[0], F_C[0]], "C1_metrics": [E_C[1], H_o[1], F_C[1]], "C2_metrics": [E_C[2], H_o[2], F_C[2]],
        "D_metrics":  [E_D, H_s[0], F_D],
        "total_metrics": [total_E, total_H, total_F]
    }
    
    combined = {**bel_map, **met_map}
    for k, v in combined.items():
        if k not in st.session_state.history:
            st.session_state.history[k] = []
        if is_append:
            st.session_state.history[k].append(v)
        else:
            st.session_state.history[k][-1] = v

if 'world' not in st.session_state:
    world_instance = World()
    st.session_state.world = world_instance
    st.session_state.step_counter = 0
    st.session_state.inference_done = False
    st.session_state.history = {} # Sauber starten
    
    # Sofort die Startwerte laden
    initial_res = world_instance.dashboard_get_beliefs()
    update_history_all(initial_res, is_append=True)
    
if 'history' not in st.session_state:
    st.session_state.history = {
        # Beliefs (wie bisher)
        "s_0": [], "s_1": [], "s_2": [], "u_0": [], "u_1": [], "o_0": [], "o_1": [], "o_2": [],
        # NEU: Energie-Metriken für die Knoten (Tripel: [E, H, F])
        "A0_m": [], "A1_m": [], "A2_m": [],
        "B0_m": [], "B1_m": [],
        "C0_m": [], "C1_m": [], "C2_m": [],
        "D_m":  [],
        "U0_m": [], "U1_m": [], "total_FE": []
    }

if 'inference_done' not in st.session_state:
    st.session_state.inference_done = False

# ==========================================
# 2. VARIABLEN & KNOTEN-DEFINITIONEN
# ==========================================
name_d = "<b>p(s<sub>0</sub>)</b>"
name_B0 = "<b>p(s<sub>1</sub>|s<sub>0</sub>, a<sub>0</sub>)</b>"
name_B1 = "<b>p(s<sub>2</sub>|s<sub>1</sub>, a<sub>1</sub>)</b>"
name_u0 = "<b>p(a<sub>0</sub>)</b>"
name_u1 = "<b>p(a<sub>1</sub>)</b>"
name_A0 = "<b>p(o<sub>0</sub>|s<sub>0</sub>)</b>"
name_A1 = "<b>p(o<sub>1</sub>|s<sub>1</sub>)</b>"
name_A2 = "<b>p(o<sub>2</sub>|s<sub>2</sub>)</b>"
name_C0 = "<b>~p(o<sub>0</sub>)</b>"
name_C1 = "<b>~p(o<sub>1</sub>)</b>"
name_C2 = "<b>~p(o<sub>2</sub>)</b>"
name_equal0 = "equal0"
name_equal1 = "equal1"
name_corner = "corner"

# ==========================================
# 3. FUNKTIONEN
# ==========================================

def create_cffg_blueprint():
    G = nx.Graph()
    G.add_node(name_d, pos=(0, 0), type="factor")
    G.add_node(name_B0, pos=(3, 0), type="factor")
    G.add_node(name_B1, pos=(6, 0), type="factor")
    G.add_node(name_u0, pos=(3, 1), type="factor")
    G.add_node(name_u1, pos=(6, 1), type="factor")
    G.add_node(name_A0, pos=(1.5, -1), type="factor")
    G.add_node(name_A1, pos=(4.5, -1), type="factor")
    G.add_node(name_A2, pos=(7.5, -1), type="factor")
    G.add_node(name_C0, pos=(1.5, -2), type="factor")
    G.add_node(name_C1, pos=(4.5, -2), type="factor")
    G.add_node(name_C2, pos=(7.5, -2), type="factor")
    G.add_node(name_equal0, pos=(1.5, 0), type="junction")
    G.add_node(name_equal1, pos=(4.5, 0), type="junction")
    G.add_node(name_corner, pos=(7.5, 0), type="corner")

    G.add_edge(name_d, name_equal0, var_name="s_0")
    G.add_edge(name_equal0, name_B0, var_name="s_0")
    G.add_edge(name_equal0, name_A0, var_name="s_0")
    G.add_edge(name_B0, name_equal1, var_name="s_1")
    G.add_edge(name_equal1, name_B1, var_name="s_1")
    G.add_edge(name_equal1, name_A1, var_name="s_1")
    G.add_edge(name_B1, name_corner, var_name="s_2")
    G.add_edge(name_corner, name_A2, var_name="s_2")
    G.add_edge(name_u0, name_B0, var_name="u_0")
    G.add_edge(name_u1, name_B1, var_name="u_1")
    G.add_edge(name_C0, name_A0, var_name="o_0")
    G.add_edge(name_C1, name_A1, var_name="o_1")
    G.add_edge(name_C2, name_A2, var_name="o_2")
    return G

def plot_cffg(G, edge_beliefs=None):
    pos = nx.get_node_attributes(G, 'pos')
    current_idx = st.session_state.step_counter

    # --- KANTEN ---
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='black'), mode='lines', hoverinfo='none')

    # --- KNOTEN (Punkte unsichtbar, wir nutzen nur die Annotations) ---
    node_trace = go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()], 
                            mode='markers', marker=dict(size=1, color='rgba(0,0,0,0)'), hoverinfo='none')

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(showlegend=False, margin=dict(b=40,l=20,r=20,t=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
                plot_bgcolor='white', height=650))

    # --- BOXEN (Immer weiß) ---
    annotations = []
    for node in G.nodes():
        x, y = pos[node]
        node_type = G.nodes[node].get("type", "factor")
        if node_type == "junction" or node == "=":
            annotations.append(dict(x=x, y=y, text="<b>=</b>", showarrow=False, font=dict(size=14),
                    bgcolor='white', bordercolor="black", borderwidth=1, width=20, height=20, xref="x", yref="y"))
        elif node_type == "corner": continue 
        else:
            annotations.append(dict(x=x, y=y, text=node, showarrow=False, font=dict(size=14),
                    bgcolor='white', bordercolor="black", borderwidth=1, borderpad=10, xref="x", yref="y"))
    fig.update_layout(annotations=annotations)

    def draw_mini_chart(var_key, x_pos, y_pos, n_states_default=8, is_obs=False):
        probs = None
        if edge_beliefs and var_key in edge_beliefs:
            data = edge_beliefs[var_key]
            # Sicherstellen, dass wir nicht außerhalb der History zugreifen
            idx = min(current_idx, len(data) - 1)
            probs = data[idx]
        if probs is None: probs = np.zeros(n_states_default)
            
        width, height = 0.5, 0.4 
        N = len(probs)
        bar_w = width / N
        
        # Farblogik (wie besprochen)
        bar_color = "#1f77b4" # Standard Blau
        if is_obs:
            obs_id = int(var_key.split("_")[1])
            # o0 -> step 1, o1 -> step 3, o2 -> step 5
            is_currently_observed = (current_idx == (obs_id * 2 + 1))
            if is_currently_observed:
                bar_color = "#ff7f0e" # Orange
            elif current_idx > (obs_id * 2 + 1):
                bar_color = "darkgrey" # Vergangenheit
        
        # 1. HINTERGRUND-BOX
        fig.add_shape(type="rect", x0=x_pos, y0=y_pos, x1=x_pos+width, y1=y_pos+height, 
                       fillcolor="rgba(245,245,245,0.95)", line=dict(width=0.5))
        
        # 2. BASIS-LINIE
        fig.add_shape(type="line", x0=x_pos, y0=y_pos, x1=x_pos+width, y1=y_pos, 
                       line=dict(color="black", width=1))
        
        # 3. Y-ACHSEN BESCHRIFTUNG (0 und 1)
        # Wir rücken sie minimal nach links (x_pos - 0.03)
        fig.add_annotation(
            x=x_pos - 0.03, y=y_pos,
            text="0", showarrow=False, font=dict(size=9, color="gray"),
            xanchor="right", xref="x", yref="y"
        )
        fig.add_annotation(
            x=x_pos - 0.03, y=y_pos + height,
            text="1", showarrow=False, font=dict(size=9, color="gray"),
            xanchor="right", xref="x", yref="y"
        )
        
        # 4. BALKEN ZEICHNEN
        for i, p in enumerate(probs):
            bx0 = x_pos + i * bar_w + (bar_w * 0.1)
            bx1 = x_pos + (i + 1) * bar_w - (bar_w * 0.1)
            by_top = y_pos + (float(p) * height)
            
            # Nur zeichnen wenn Wahrscheinlichkeit > 0 (sieht cleaner aus)
            if p > 0.001:
                fig.add_shape(type="rect", x0=bx0, y0=y_pos, x1=bx1, y1=by_top, 
                               fillcolor=bar_color, line=dict(width=0))

    def draw_metrics_chart(key, x_pos, y_pos):
        vals = [0, 0, 0]
        if key in st.session_state.history and len(st.session_state.history[key]) > 0:
            idx = min(st.session_state.step_counter, len(st.session_state.history[key]) - 1)
            vals = st.session_state.history[key][idx]

        # 1. Dynamisches Maximum finden
        abs_vals = [abs(v.item() if hasattr(v, 'item') else float(v)) for v in vals]
        max_val = max(abs_vals) if max(abs_vals) > 0 else 1.0
        
        width, height = 0.3, 0.2
        bar_w = width / 3
        colors = ["gray", "gray", "#9b59b6"] # Rot (E), Grün (H), Lila (F)
        labels = ["E", "H", "F"]

        # Hintergrund-Box
        fig.add_shape(type="rect", x0=x_pos, y0=y_pos, x1=x_pos+width, y1=y_pos+height, 
                      fillcolor="rgba(245,245,245,0.95)", line=dict(color="gray", width=0.5))

        # 2. Pseudo-Y-Achse rechts
        fig.add_annotation(
            x=x_pos + width + 0.02, y=y_pos + height,
            text=f"{max_val:.2f}", showarrow=False, font=dict(size=9, color="gray"),
            xanchor="left", yanchor="middle"
        )
        fig.add_annotation(
            x=x_pos + width + 0.02, y=y_pos,
            text="0", showarrow=False, font=dict(size=9, color="gray"),
            xanchor="left", yanchor="middle"
        )

        # 3. Balken und Buchstaben-Labels
        for i, v in enumerate(vals):
            # Balken-Koordinaten
            bx0 = x_pos + i * bar_w + 0.02
            bx1 = x_pos + (i + 1) * bar_w - 0.02
            h = (abs(v.item() if hasattr(v, 'item') else float(v)) / max_val) * height
            
            # Balken zeichnen
            fig.add_shape(type="rect", x0=bx0, y0=y_pos, x1=bx1, y1=y_pos + h,
                          fillcolor=colors[i], line=dict(width=0))
            
            # Buchstabe (E, H, F) unter dem Balken
            fig.add_annotation(
                x=(bx0 + bx1) / 2, # Zentriert unter dem Balken
                y=y_pos - 0.00,    # Kurz unter der Basis-Linie
                text=labels[i],
                showarrow=False,
                font=dict(size=10, color=colors[i], weight="bold"), # In Balkenfarbe für Zuordnung
                yanchor="top"
            )

    # Charts tatsächlich zeichnen
    # charts_to_draw = [
    #     ("s_0", 0.95, -0.7, 8), ("s_1", 3.95, -0.7, 8), ("s_2", 6.95, -0.7, 8), 
    #     ("u_0", 2.45, 0.3, 4), ("u_1", 5.45, 0.3, 4), 
    #     ("o_0", 0.95, -1.7, 16), ("o_1", 3.95, -1.7, 16), ("o_2", 6.95, -1.7, 16)
    # ]
    charts_to_draw = [
        ("s_0", 0.95, -0.7, 8, False), ("s_1", 3.95, -0.7, 8, False), ("s_2", 6.95, -0.7, 8, False), 
        ("u_0", 2.45, 0.3, 4, False), ("u_1", 5.45, 0.3, 4, False), 
        # Hier is_obs=True setzen:
        ("o_0", 0.95, -1.7, 16, True), ("o_1", 3.95, -1.7, 16, True), ("o_2", 6.95, -1.7, 16, True)
    ]
    # 1. Die Wahrscheinlichkeits-Charts (Beliefs) zeichnen
    for c_args in charts_to_draw:
        draw_mini_chart(*c_args)

    # 2. NEU: Die Metriken-Charts (E, H, F) zeichnen
    # Wir platzieren sie jeweils leicht versetzt zu den Faktoren
    metrics_to_draw = [
        ("A0_metrics", 1.83, -1.1), # Unter/neben p(o0|s0)
        ("A1_metrics", 4.83, -1.1), # Unter/neben p(o1|s1)
        ("A2_metrics", 7.83, -1.1), # Unter/neben p(o2|s2)
        ("C0_metrics", 1.83, -2.1), # Unter/neben p(o0|s0)
        ("C1_metrics", 4.83, -2.1), # Unter/neben p(o1|s1)
        ("C2_metrics", 7.83, -2.1),
        ("B0_metrics", 2.85, -0.43),  # Über p(s1|s0,a0)
        ("B1_metrics", 5.85, -0.43),  # Über p(s2|s1,a1)
        ("U0_metrics", 3.26, 0.9),  # Über p(s1|s0,a0)
        ("U1_metrics", 6.26, 0.9),
        ("D_metrics",  -0.14, -0.43),  # Über p(s0)
    ]
    
    for key, x, y in metrics_to_draw:
        draw_metrics_chart(key, x, y)

    # --- KANTEN-BESCHRIFTUNGEN (Variablennamen) ---
    # Wir definieren, welche Kante welches Label bekommt
    # Format: (Knoten_A, Knoten_B, Label_Text, Offset_X)
    edge_labels = [
        (name_A0, name_C0, "q(o₀)", 0.05),
        (name_A1, name_C1, "q(o₁)", 0.05),
        (name_A2, name_C2, "q(o₂)", 0.05),
        (name_equal0, name_A0, "q(s₀)", 0.05),
        (name_equal1, name_A1, "q(s₁)", 0.05),
        (name_corner, name_A2, "q(s₂)", 0.05),
        (name_B0, name_u0, "q(a₀)", 0.05),
        (name_B1, name_u1, "q(a₁)", 0.05)
    ]

    for n1, n2, txt, offset_x in edge_labels:
        # Berechne die Mitte der Kante
        x_mid = (pos[n1][0] + pos[n2][0]) / 2
        y_mid = (pos[n1][1] + pos[n2][1]) / 2
        
        fig.add_annotation(
            x=x_mid + offset_x, # Leicht nach rechts verschoben
            y=y_mid,
            text=f"<i>{txt}</i>",
            showarrow=False,
            font=dict(size=14, color="black"),
            xanchor="left"
        )
    
    # --- MANUELLER ZEITSTRAHL ---
    timeline_y = -2.4
    curr_step = st.session_state.step_counter
    timeline_start_x = 0
    timeline_end_x = 8

    # 1. Die Basis-Linie
    fig.add_shape(type="line", x0=timeline_start_x, y0=timeline_y, x1=timeline_end_x, y1=timeline_y,
                  line=dict(color="darkgrey", width=1.5), layer="below")

    # 2. DER RECHTSPFEIL (NEU)
    # Wir fügen eine Annotation am Ende der Linie hinzu
    fig.add_annotation(
        x=timeline_end_x, # Pfeilspitze an der letzten Perle (t=2)
        y=timeline_y,
        ax=timeline_end_x - 0.1, # Startpunkt des Pfeilschafts (leicht versetzt)
        ay=timeline_y,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=2, # Klassische Dreieckspitze
        arrowsize=1.8,
        arrowwidth=1.5,
        arrowcolor="darkgrey" # Gleiche Farbe wie die Linie
    )

    # 2. Die manuellen "Perlen" definieren
    # Format: (x_position, label_text, schritt_index)
    perlen = [
        (0, "Start", 0),
        (1.5, '"observe o₀"', 1),
        (3, '"act a₀"', 2),
        (4.5, '"observe o₁"', 3),
        (6, '"act a₁"', 4),
        (7.5, '"observe o₂"', 5)
    ]

    for x_pos, label, s_idx in perlen:
        # Bestimme, ob diese Perle gerade aktiv ist
        is_active = (s_idx == curr_step)
        active_color = "#ff7f0e"  # Das Orange deiner Perlen

        # 1. Perle zeichnen (unverändert, nutzt is_active)
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[timeline_y],
            mode="markers",
            marker=dict(
                size=14,
                color=active_color if is_active else "white",
                line=dict(color="black", width=1)
            ),
            hoverinfo="none", showlegend=False
        ))

        # 2. Text-Annotation zeichnen (FARBE ANGEPASST)
        fig.add_annotation(
            x=x_pos,
            y=timeline_y,
            text=f"<b>{label}</b>" if is_active else label, # Optional: Aktiv fett drucken
            showarrow=False,
            yshift=-35,
            xref="x", 
            yref="y",
            font=dict(
                size=14,
                # Hier passiert die Magie:
                color=active_color if is_active else "black"
            )
        )

    # --- LAYOUT-FINETUNING ---
    # Wir begrenzen die Y-Achse, damit der Abstand nach unten minimal ist
    fig.update_layout(
        yaxis=dict(range=[-2.8, 1.3], autorange=False)
    )

    return fig

def run_inference():
    (s, u, o) = st.session_state.world.dashboard_infer()
    mapping = {"s_0":s[0],"s_1":s[1],"s_2":s[2],"u_0":u[0],"u_1":u[1],"o_0":o[0],"o_1":o[1],"o_2":o[2]}
    for k, v in mapping.items():
        st.session_state.history[k][-1] = v

def run_next_simulation_step():
    (s, u, o) = st.session_state.world.run_step() 
    mapping = {"s_0":s[0],"s_1":s[1],"s_2":s[2],"u_0":u[0],"u_1":u[1],"o_0":o[0],"o_1":o[1],"o_2":o[2]}
    for k, v in mapping.items():
        st.session_state.history[k].append(v)
    st.session_state.step_counter += 1

# ==========================================
# 4. APP LOGIK & LAYOUT
# ==========================================
st.set_page_config(page_title="CFFG Agent Debugger", layout="wide")
st.markdown("<style>.block-container { padding-top: 2rem; }</style>", unsafe_allow_html=True)
st.title("Active Inference - Demonstration")
st.markdown("""
    <style>
    /* 1. Abstände zwischen den Elementen in der Sidebar verringern */
    [data-testid="stVerticalBlock"] > div {
        gap: 0.3rem !important;
    }

    /* 2. Die Buttons selbst flacher machen */
    div.stButton > button {
        padding-top: 0px !important;
        padding-bottom: 0px !important;
        height: 2rem !important; /* Standard ist oft ~3rem */
        min-height: 2rem !important;
        line-height: 2rem !important;
    }

    /* 3. Den Header-Abstand verringern */
    h3 {
        margin-bottom: 0.5rem !important;
        padding-bottom: 5px !important;
    }
            
    /* 1. Den Abstand unter dem Button-Container minimieren */
    div.stButton {
        margin-bottom: 5px !important;
    }

    /* 2. Den Abstand der horizontalen Linie (st.divider) nach oben verringern */
    hr {
        margin-top: 1px !important;
        margin-bottom: 1rem !important;
    }

    /* 3. Falls du st.markdown("---") nutzt, ist das oft ein 'hr' innerhalb eines Blocks */
    [data-testid="stVerticalBlock"] hr {
        margin-top: -10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

col_controls, col_graph = st.columns([1, 7])

current_step = st.session_state.step_counter
status = st.session_state.inference_done

# --- ERWEITERTE LOGIK FÜR BUTTON-ZUSTÄNDE ---

# Observe ist nur am Anfang eines Zyklus erlaubt, wenn noch nicht beobachtet wurde
disable_observe = (status is not False) or (current_step >= 5)

# Infer ist nur erlaubt, wenn gerade beobachtet wurde
disable_infer   = (status != "ready_for_infer")

# Act ist nur erlaubt, wenn Infer fertig ist UND wir noch nicht an der letzten Perle sind
# Wenn current_step == 5 ist, bleibt dieser Button dauerhaft deaktiviert
disable_act     = (status != "ready_for_act") or (current_step >= 5)

with col_controls:
    st.subheader("Controls")
    
    # Hilfsfunktion zum Zuweisen der 19 Werte in die History
    def update_history_all(res, is_append=True):
        # res entpacken (19 Werte laut deiner Beschreibung)
        (s, u, o, total_F, total_E, total_H, F_A, F_B, F_C, F_D, F_U, 
         E_A, E_B, E_C, E_D, E_U, 
         H_A, H_B, H_s, H_o, H_u) = res
        
        # 1. Beliefs Mapping
        bel_map = {"s_0":s[0],"s_1":s[1],"s_2":s[2],"u_0":u[0],"u_1":u[1],"o_0":o[0],"o_1":o[1],"o_2":o[2]}
        
        # 2. Metriken Mapping (Beispielhaft für Faktoren A, B, C und Prior D)
        # Wir speichern Tripel: [Energy, Entropy, FreeEnergy]
        met_map = {
            "A0_metrics": [E_A[0], H_A[0], F_A[0]],
            "A1_metrics": [E_A[1], H_A[1], F_A[1]],
            "A2_metrics": [E_A[2], H_A[2], F_A[2]],
            "B0_metrics": [E_B[0], H_B[0], F_B[0]],
            "B1_metrics": [E_B[1], H_B[1], F_B[1]],
            "U0_metrics": [E_U[0], H_u[0], F_U[0]],
            "U1_metrics": [E_U[1], H_u[1], F_U[1]],
            "C0_metrics": [E_C[0], H_o[0], F_C[0]],
            "C1_metrics": [E_C[1], H_o[1], F_C[1]],
            "C2_metrics": [E_C[2], H_o[2], F_C[2]], # C hat oft keine Entropie-Definition
            "D_metrics":  [E_D, H_s[0], F_D],   # Prior auf s_0
            "total_metrics":   [total_E, total_H, total_F]      # Nur FE Wert
        }
        
        # In History schreiben
        for k, v in {**bel_map, **met_map}.items():
            if k not in st.session_state.history:
                st.session_state.history[k] = []
            if is_append:
                st.session_state.history[k].append(v)
            else:
                st.session_state.history[k][-1] = v

    # --- 1. OBSERVE ---
    if st.button("Observe", width='stretch', disabled=disable_observe):
        res = st.session_state.world.dashboard_observe() 
        update_history_all(res, is_append=True)
        st.session_state.step_counter += 1 
        st.session_state.inference_done = "ready_for_infer" 
        st.rerun()

    # --- 2. INFER ---
    if st.button("Infer Beliefs", width='stretch', disabled=disable_infer):
        res = st.session_state.world.dashboard_infer()
        update_history_all(res, is_append=False) # Update des letzten Eintrags
        st.session_state.inference_done = "ready_for_act"
        if st.session_state.step_counter >= 5:
             st.session_state.inference_done = "finished"
        st.rerun()

    # --- 3. ACT ---
    if st.button("Act", width='stretch', disabled=disable_act):
        res = st.session_state.world.dashboard_act()
        update_history_all(res, is_append=True)
        st.session_state.step_counter += 1 
        st.session_state.inference_done = False 
        st.rerun()
        
    # --- 4. RESET ---
    if st.button("Reset", width='stretch'):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("Global Metrics")
    
    if "total_metrics" in st.session_state.history and len(st.session_state.history["total_metrics"]) > 0:
        # Den aktuellen Stand abgreifen
        idx = min(st.session_state.step_counter, len(st.session_state.history["total_metrics"]) - 1)
        raw_vals = st.session_state.history["total_metrics"][idx]
        
        # Sicherstellen, dass es einfache Floats sind (falls es numpy arrays sind)
        # Ändere diese Zeilen:
        e_total = raw_vals[0].item()
        h_total = raw_vals[1].item()
        f_total = raw_vals[2].item()
        
        # Plotly Bar Chart
        fig_total = go.Figure(data=[
            go.Bar(
                x=["Energy (E)", "Entropy (H)", "Free Energy (F)"],
                y=[e_total, h_total, f_total],
                marker_color=["#666666", "#999999", "#9b59b6"],
                # Jetzt funktioniert die Formatierung, da es Floats sind
                text=[f"{e_total:.2f}", f"{h_total:.2f}", f"{f_total:.2f}"],
                textposition='outside',
            )
        ])
        
        fig_total.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=30, b=40),
            xaxis=dict(tickfont=dict(size=11)),
            yaxis=dict(title="Value", showgrid=True, gridcolor="lightgrey"),
            plot_bgcolor='white',
            showlegend=False
        )
        
        st.plotly_chart(fig_total, use_container_width=True, config={'displayModeBar': False})
        
        # Optionale Text-Zusammenfassung für den schnellen Check
        st.caption(f"Aktuelle Optimierung: **F = {f_total:.4f}**")

with col_graph:
    cffg = create_cffg_blueprint()
    
    # Statt np.array(v) für alles, lassen wir die Listen einfach als Listen.
    # Die draw-Funktionen ziehen sich das raus, was sie brauchen.
    real_beliefs = st.session_state.history 
    
    fig_graph = plot_cffg(cffg, edge_beliefs=real_beliefs)
    st.plotly_chart(fig_graph, width='stretch')