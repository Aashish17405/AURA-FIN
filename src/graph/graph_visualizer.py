"""
Dynamic Stock Graph Visualizer using NetworkX and Matplotlib.
Renders inter-stock correlation clusters, sector node colors, and edge weights.
"""

from typing import List, Dict, Optional
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

SECTOR_COLORS = {
    "Technology": "#3b82f6",       # Blue
    "Consumer/Tech": "#10b981",    # Emerald
    "Semiconductors": "#8b5cf6",   # Purple
    "Financials": "#f59e0b",       # Amber
    "Communication/Tech": "#ec4899",# Pink
    "Automotive/Tech": "#ef4444",   # Red
    "Unknown": "#64748b"           # Slate
}

def plot_stock_graph(
    adjacency_matrix: np.ndarray,
    tickers: List[str],
    sector_map: Dict[str, str],
    title: str = "Dynamic Inter-Stock Correlation Graph",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders an inter-stock network graph with sector-based node coloring
    and correlation-weighted edge lines.
    """
    G = nx.Graph()
    n = len(tickers)
    
    for i, ticker in enumerate(tickers):
        sector = sector_map.get(ticker, "Unknown")
        G.add_node(ticker, sector=sector, color=SECTOR_COLORS.get(sector, "#64748b"))
        
    for i in range(n):
        for j in range(i + 1, n):
            weight = float(adjacency_matrix[i, j])
            if weight > 0.05:  # filter negligible weights for clarity
                G.add_edge(tickers[i], tickers[j], weight=weight)
                
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    fig.patch.set_facecolor("#0f172a") # dark slate background
    ax.set_facecolor("#0f172a")
    
    pos = nx.spring_layout(G, seed=42, k=1.2)
    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    
    # Draw edges with varying thickness
    edges = G.edges()
    weights = [G[u][v]["weight"] * 3.5 for u, v in edges]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#475569",
        width=weights,
        alpha=0.6,
        style="solid"
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=1200,
        edgecolors="#ffffff",
        linewidths=1.5
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_color="#ffffff",
        font_size=9,
        font_weight="bold",
        font_family="sans-serif"
    )
    
    ax.set_title(title, color="#f8fafc", fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        
    return fig
