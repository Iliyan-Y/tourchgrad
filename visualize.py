from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()

    def build(fn):
        if fn is None or fn in nodes:
            return

        nodes.add(fn)

        for child, _ in fn.next_functions:
            if child is not None:
                edges.add((child, fn))
                build(child)

    build(root.grad_fn)
    return nodes, edges


def draw_dot(root, named_tensors):
    nodes, edges = trace(root)

    tensors = dict(named_tensors)

    if not any(t is root for t in tensors.values()):
        tensors["output"] = root

    # Discover any unnamed leaf tensors.
    known = {id(t) for t in tensors.values()}

    for i, fn in enumerate(nodes):
        if hasattr(fn, "variable"):
            t = fn.variable
            if id(t) not in known:
                tensors[f"input_{i}"] = t
                known.add(id(t))

    # Match operation nodes to their resulting tensors.
    fn_to_tensor = {
        t.grad_fn: (name, t)
        for name, t in tensors.items()
        if t.grad_fn is not None
    }

    fn_ids = {
        fn: f"op_{i}"
        for i, fn in enumerate(nodes)
        if not hasattr(fn, "variable")
    }

    def tensor_id(t):
        return f"tensor_{id(t)}"

    def output_node(fn):
        if hasattr(fn, "variable"):
            return tensor_id(fn.variable)

        mapped = fn_to_tensor.get(fn)

        if mapped is not None:
            return tensor_id(mapped[1])

        return fn_ids[fn]

    dot = Digraph(
        format="svg",
        graph_attr={
            "rankdir": "LR",
            "nodesep": "0.35",
            "ranksep": "0.55",
        },
    )

    # Draw tensor/value rectangles.
    for name, t in tensors.items():
        value = f"{t.detach().item():.4f}" if t.numel() == 1 else str(tuple(t.shape))

        fields = [name, f"data: {value}"]

        if t.is_leaf and t.grad is not None:
            fields.append(f"grad: {t.grad.item():.4f}")

        dot.node(
            tensor_id(t),
            label="{" + "|".join(fields) + "}",
            shape="record",
            style="filled",
            fillcolor="#eaf2f8",
        )

    # Draw operation nodes and connect each to its result tensor.
    for fn in nodes:
        if hasattr(fn, "variable"):
            continue

        label = fn.name().replace("Backward0", "").replace("Backward", "")

        dot.node(
            fn_ids[fn],
            label=label,
            shape="ellipse",
            style="filled",
            fillcolor="#f4f4f4",
        )

        mapped = fn_to_tensor.get(fn)

        if mapped is not None:
            dot.edge(fn_ids[fn], tensor_id(mapped[1]))

    # Connect input tensors to the operations that consume them.
    for child, parent in edges:
        if not hasattr(parent, "variable"):
            dot.edge(output_node(child), fn_ids[parent])

    return dot
