"""Draw a ``Value`` computation graph, with graphviz or with matplotlib only."""


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw(root, format="svg", rankdir="LR", path=None):
    """Build a graphviz ``Digraph`` of the graph; render it to ``path`` if given.

    Rendering needs the graphviz ``dot`` binary; building the graph does not.
    """
    from graphviz import Digraph

    assert rankdir in ["LR", "TB"]
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label=f"{{ {n.label} | data {n.data:.4f} | grad {n.grad:.4f} }}",
            shape="record",
        )
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    if path is not None:
        dot.render(path, format=format, cleanup=True)
        print(f"Graph saved to {path}.{format}")
    return dot


# Colours for the matplotlib renderer (light surface, muted chrome)
_INK, _MUTED, _EDGE, _SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
_BOX, _OP, _HIGHLIGHT = "#f0efec", "#ffffff", "#cde2fb"


def _layout(root):
    """Assign each node a column (longest path from the leaves) and a row."""
    nodes, edges = trace(root)
    parents = {n: [] for n in nodes}
    for child, parent in edges:
        parents[child].append(parent)

    depth = {}

    def d(v):
        if v not in depth:
            depth[v] = 1 + max((d(c) for c in v._prev), default=-1)
        return depth[v]

    for n in nodes:
        d(n)
    # Pull each leaf right, next to its earliest consumer, to keep edges short
    for n in nodes:
        if not n._prev and parents[n]:
            depth[n] = min(depth[p] for p in parents[n]) - 1

    columns = {}
    for n in sorted(nodes, key=lambda v: (depth[v], v.label, v.data)):
        columns.setdefault(depth[n], []).append(n)

    rows = {}
    for col in sorted(columns):
        members = columns[col]
        if col > min(columns):
            # Order by the mean row of the inputs (barycentre heuristic)
            def key(v):
                ys = [rows[c] for c in v._prev if c in rows]
                return sum(ys) / len(ys) if ys else 0.0

            members.sort(key=key)
        offset = (len(members) - 1) / 2
        for i, n in enumerate(members):
            rows[n] = i - offset
    return nodes, edges, depth, rows


def draw_matplotlib(root, path, title=None, highlight_ops=(), dpi=110):
    """Save a PNG of the graph using only matplotlib (no graphviz binary needed).

    Each value is a ``label | data | grad`` box; each operation is a small node
    between its inputs and its output. Ops whose name starts with a prefix in
    ``highlight_ops`` are shaded, e.g. ``("<Z",)`` for quantum expectation nodes.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    nodes, edges, depth, rows = _layout(root)
    box_w, box_h, op_gap, col_w, row_h = 2.3, 0.62, 1.8, 3.6, 1.05
    n_cols = max(depth.values()) + 1
    n_rows = max(max(rows.values()) - min(rows.values()) + 1, 1)

    fig = Figure(figsize=(0.62 * col_w * n_cols + 0.4, 0.75 * row_h * n_rows + 0.8))
    fig.patch.set_facecolor(_SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    pos = {n: (depth[n] * col_w, -rows[n] * row_h) for n in nodes}
    op_pos = {n: (pos[n][0] - op_gap, pos[n][1]) for n in nodes if n._op}

    def arrow(p, q):
        ax.add_patch(
            FancyArrowPatch(
                p,
                q,
                arrowstyle="-|>",
                mutation_scale=8,
                lw=0.8,
                color=_EDGE,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )

    for n in nodes:
        x, y = pos[n]
        ax.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0,rounding_size=0.08",
                fc=_BOX,
                ec=_EDGE,
                lw=0.6,
                zorder=2,
            )
        )
        name = f"{n.label}\n" if n.label else ""
        ax.text(
            x,
            y,
            f"{name}data {n.data:.4f}   grad {n.grad:.4f}",
            ha="center",
            va="center",
            fontsize=6.5,
            color=_INK,
            zorder=3,
            linespacing=1.4,
        )
        if n._op:
            ox, oy = op_pos[n]
            shaded = any(n._op.startswith(h) for h in highlight_ops)
            ax.text(
                ox,
                oy,
                n._op,
                ha="center",
                va="center",
                fontsize=7,
                color=_INK,
                zorder=3,
                bbox=dict(
                    boxstyle="round,pad=0.3", fc=_HIGHLIGHT if shaded else _OP, ec=_EDGE, lw=0.6
                ),
            )
            arrow((ox + 0.3, oy), (x - box_w / 2, y))

    for child, parent in edges:
        cx, cy = pos[child]
        ox, oy = op_pos[parent]
        arrow((cx + box_w / 2, cy), (ox - 0.3, oy))

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - box_w / 2 - 0.2, max(xs) + box_w / 2 + 0.2)
    ax.set_ylim(min(ys) - row_h * 0.6, max(ys) + row_h * (1.0 if title else 0.6))
    if title:
        ax.text(
            min(xs) - box_w / 2,
            max(ys) + row_h * 0.75,
            title,
            fontsize=9,
            color=_MUTED,
            va="center",
        )
    fig.savefig(path, dpi=dpi, facecolor=_SURFACE)
    return path
