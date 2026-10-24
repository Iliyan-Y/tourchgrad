import torch

from nn import MLP
from predict import predict
from visualize import draw_dot


def print_hi(name):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = torch.tensor(2.0, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(3.0, device=device, dtype=torch.float32, requires_grad=True)

    # Forward computation
    x_squared = x ** 2
    product = x_squared * y
    sine = torch.sin(y)
    z = product + sine

    # Evaluate the function.
    print(f"z = {z.item():.6f}")

    # Calculate gradients using backpropagation.
    z.backward()

    # Inspect the gradients.
    print(f"z     = {z.item():.6f}")
    print(f"dz/dx = {x.grad.item():.6f}")
    print(f"dz/dy = {y.grad.item():.6f}")

    # Draw the graph
    dot = draw_dot(
        z,
        {
            "x": x,
            "y": y,
            "x**2": x_squared,
            "x**2 * y": product,
            "sin(y)": sine,
            "z": z,
        },
    )

    dot.render("computation_graph", cleanup=True)
    print("Saved computation_graph.svg")


if __name__ == '__main__':
    # createTwoLayerNeuron()
    # single_pass()
    # multi_pass_dataset()
    model = MLP(3, [4, 4, 1])
    predict(model)
