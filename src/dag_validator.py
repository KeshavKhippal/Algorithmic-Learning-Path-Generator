"""
DAG validation: cycle detection, greedy cycle-breaking, and graph metrics.

Uses ``networkx.DiGraph`` for cycle detection and topological-sort
validation.
"""

import logging
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


# =========================================================================
# Cycle-breaking
# =========================================================================


def break_cycles(
    edges: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Remove minimum-confidence edges to make the graph acyclic.

    Uses ``networkx.find_cycle`` to detect one cycle at a time,
    removes the edge with the lowest confidence within that cycle,
    and repeats until acyclic.

    Args:
        edges: List of dicts with keys ``source_id``, ``target_id``,
               ``similarity``, ``confidence``.

    Returns:
        Tuple of ``(acyclic_edges, removed_edges)``.
    """
    G = nx.DiGraph()
    edge_key = {}  # (src, tgt) → dict
    for e in edges:
        src, tgt = e["source_id"], e["target_id"]
        G.add_edge(src, tgt, confidence=e["confidence"])
        edge_key[(src, tgt)] = e

    removed: List[Dict[str, Any]] = []

    while True:
        try:
            cycle = nx.find_cycle(G, orientation="original")
        except nx.NetworkXNoCycle:
            break

        # cycle is list of (u, v, direction)
        cycle_edges = [(u, v) for u, v, _ in cycle]

        # Find the edge with minimum confidence in this cycle
        min_conf = float("inf")
        min_edge = cycle_edges[0]
        for u, v in cycle_edges:
            conf = G[u][v].get("confidence", 0.0)
            if conf < min_conf:
                min_conf = conf
                min_edge = (u, v)

        # Remove it
        G.remove_edge(*min_edge)
        removed_entry = edge_key.pop(min_edge, {
            "source_id": min_edge[0],
            "target_id": min_edge[1],
            "similarity": 0.0,
            "confidence": min_conf,
        })
        removed_entry = dict(removed_entry)
        removed_entry["reason"] = "cycle_break"
        removed.append(removed_entry)
        logger.debug(
            "Cycle-break: removed edge %d → %d (confidence=%.4f).",
            min_edge[0], min_edge[1], min_conf,
        )

    logger.info(
        "Cycle-breaking complete: removed %d edge(s).", len(removed)
    )

    # Rebuild acyclic edge list
    acyclic_edges = [
        edge_key[(u, v)]
        for u, v in G.edges()
        if (u, v) in edge_key
    ]
    return acyclic_edges, removed


# =========================================================================
# Validation
# =========================================================================


def validate_dag(edges: List[Dict[str, Any]]) -> bool:
    """Verify that edges form a DAG (topological sort succeeds)."""
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e["source_id"], e["target_id"])
    try:
        list(nx.topological_sort(G))
        return True
    except nx.NetworkXUnfeasible:
        return False


# =========================================================================
# Metrics
# =========================================================================


def compute_metrics(
    edges: List[Dict[str, Any]],
    n_concepts: int,
) -> Dict[str, Any]:
    """Compute graph summary metrics.

    Returns dict with: total_concepts, total_edges, avg_out_degree,
    max_depth, isolated_nodes_count.
    """
    G = nx.DiGraph()
    # Add all concept nodes (even isolated ones)
    for i in range(n_concepts):
        G.add_node(i)  # placeholder — real IDs added via edges

    # Rebuild with real IDs
    G = nx.DiGraph()
    all_node_ids: Set[int] = set()

    for e in edges:
        G.add_edge(e["source_id"], e["target_id"])
        all_node_ids.add(e["source_id"])
        all_node_ids.add(e["target_id"])

    total_edges = G.number_of_edges()
    nodes_in_graph = G.number_of_nodes()

    # Isolated = concepts not appearing in any edge
    isolated_count = max(0, n_concepts - nodes_in_graph)

    # Average out-degree
    avg_out = total_edges / nodes_in_graph if nodes_in_graph > 0 else 0.0

    # Max depth (longest path)
    if total_edges > 0 and nx.is_directed_acyclic_graph(G):
        max_depth = nx.dag_longest_path_length(G)
    else:
        max_depth = 0

    return {
        "total_concepts": n_concepts,
        "total_edges": total_edges,
        "avg_out_degree": round(avg_out, 4),
        "max_depth": max_depth,
        "isolated_nodes_count": isolated_count,
    }
