import torch

from visualize import draw_dot


def createTwoLayerNeuron():
    # Simple expression of neuron with 2 axons and bias
    # Cell exprsion:
    # x1 * w1  + x2 * w2  + b = cell ->  activation is tanh -> out
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = torch.tensor(2.0, device=device, dtype=torch.float32, requires_grad=True)
    w = torch.tensor(-3.0, device=device, dtype=torch.float32, requires_grad=True)
    x2 = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
    w2 = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.tensor(6.88, device=device, dtype=torch.float32, requires_grad=True)

    # Forward computation
    cell = x * w + x2 * w2 + b
    o = torch.tanh(cell)

    # Calculate gradients using backpropagation.
    o.backward()

    # Inspect the gradients.
    print(f"o     = {o.item():.6f}")
    print(f"x = {x.grad.item():.6f}")
    print(f"w = {w.grad.item():.6f}")
    print(f"x2 = {x2.grad.item():.6f}")
    print(f"w2 = {w2.grad.item():.6f}")

    # Inspect the gradients.
    # Draw the graph
    dot = draw_dot(
        o,
        {
            "x": x,
            "w": w,
            "x2": x2,
            "w2": w2,
            "b": b,
            "cell": cell,
            "o": o,
        },
    )

    dot.render("computation_graph", cleanup=True)
    print("Saved computation_graph.svg")
