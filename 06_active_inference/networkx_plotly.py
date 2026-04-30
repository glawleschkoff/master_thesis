import networkx as nx
import plotly.graph_objects as go

# ==========================================
# 1. Die Single Source of Truth (Topologie)
# ==========================================
def create_cffg_blueprint():
    """Erstellt den statischen Bauplan des CFFG."""
    G = nx.Graph()
    
    # Knoten (Faktoren) mit fixen X/Y-Koordinaten aus deiner Bleistiftzeichnung
    G.add_node("Prior_Normal", pos=(0, 1), type="factor")
    G.add_node("Transition_Matrix", pos=(2, 1), type="factor")
    G.add_node("Likelihood", pos=(4, 1), type="factor")
    
    # Kanten (Variablen/Zustände)
    G.add_edge("Prior_Normal", "Transition_Matrix", var_name="s_t-1")
    G.add_edge("Transition_Matrix", "Likelihood", var_name="s_t")
    
    return G

# ==========================================
# 2. Die Rendering-Logik (Visualisierung)
# ==========================================
def plot_cffg(G, edge_beliefs=None, active_node=None):
    """
    Zeichnet den Graphen. 
    Nimmt optionale Parameter für den dynamischen State (Debugging/Dashboard).
    """
    pos = nx.get_node_attributes(G, 'pos')
    
    # --- Kanten (Variablen) zeichnen ---
    edge_x = []
    edge_y = []
    edge_text = [] # Für das Hover-Debugging
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Debugging-Infos für Hover aufbereiten
        var_name = edge[2].get('var_name', 'unknown')
        if edge_beliefs and var_name in edge_beliefs:
            belief = edge_beliefs[var_name]
            hover_info = f"Variable: {var_name}<br>Mean: {belief['mean']:.2f}<br>Var: {belief['var']:.2f}"
        else:
            hover_info = f"Variable: {var_name}<br>No messages yet"
        
        # Wir hängen die Info an die Mitte der Kante an
        edge_text.append(hover_info)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='#888'),
        hoverinfo='text',
        mode='lines'
    )

    # --- Knoten (Faktoren) zeichnen ---
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Factor: {node}")
        
        # Dynamische Farbe: Highlighte den Knoten, der gerade rechnet
        if node == active_node:
            node_colors.append('#ff7f0e') # Orange für "Active"
        else:
            node_colors.append('#1f77b4') # Blau für "Idle"

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n for n in G.nodes()], # Beschriftung direkt am Knoten
        textposition="top center",
        marker=dict(
            symbol='square', # Faktoren sind in CFFGs meist Quadrate
            size=30,
            color=node_colors,
            line=dict(width=2, color='DarkSlateGrey')
        )
    )

    # --- Layout zusammensetzen ---
    # --- Layout zusammensetzen ---
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(text='Active Inference CFFG Blueprint', font=dict(size=16)), # <-- SAUBERE SYNTAX
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
             ))
    return fig

# ==========================================
# 3. Nutzung (Dein Workflow)
# ==========================================
# Bauplan einmal initialisieren
cffg = create_cffg_blueprint()

# BEISPIEL A: Statisch für die Thesis (Einfach als PDF speichern)
fig_static = plot_cffg(cffg)
# fig_static.write_image("cffg_thesis.pdf") # (Erfordert das 'kaleido' Paket)

# BEISPIEL B: Dynamisch für Debugging / Dashboard
# Hier würdest du die echten Werte aus deinem Algorithmus übergeben
current_beliefs = {
    "s_t-1": {"mean": 0.5, "var": 0.1},
    "s_t": {"mean": 1.2, "var": 0.05}
}
fig_dynamic = plot_cffg(cffg, edge_beliefs=current_beliefs, active_node="Transition_Matrix")

# In der Entwicklungsumgebung anzeigen
fig_dynamic.show()