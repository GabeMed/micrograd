import pytest

from neural_network import Value
from neural_network.visualize import draw, draw_matplotlib, trace


def small_graph():
    a, b = Value(2.0, label="a"), Value(-3.0, label="b")
    out = (a * b + a).tanh()
    out.backward_prop()
    return out


def test_trace_collects_nodes_and_edges():
    nodes, edges = trace(small_graph())
    assert len(nodes) == 5 and len(edges) == 5  # `a` feeds both * and +


def test_draw_matplotlib_writes_png(tmp_path):
    path = draw_matplotlib(small_graph(), tmp_path / "graph.png", title="t")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_graphviz_source_builds_without_rendering():
    pytest.importorskip("graphviz")
    dot = draw(small_graph())
    assert "tanh" in dot.source and "data 2.0000" in dot.source
