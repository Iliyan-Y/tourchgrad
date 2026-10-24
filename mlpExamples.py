import torch

from nn import MLP, device


def single_pass():
    torch.manual_seed(42)

    # Architecture: 3 -> 4 -> 4 -> 1
    n = MLP(3, [4, 4, 1])
    print("Number of parameters:", len(n.parameters()))

    # Three input features.
    # x = [2.0, 3.0, -1.0]
    x = [
        torch.tensor(v, dtype=torch.float32, device=device,
                     requires_grad=True)
        for v in [2.0, 3.0, -1.0]
    ]

    target = torch.tensor(0.5, device=device)

    # Forward pass.
    prediction = n(x)[0]  # target the only one item in the out we created MLP with 3 4 4 1 <- [0]
    # we can have multy level prediction
    # predictions = torch.stack(n(x))
    # loss = ((predictions - target) ** 2).mean()

    loss = (prediction - target) ** 2

    # Backpropagate before drawing to populate gradients.
    loss.backward()

    print("\nOutput:")
    print("Prediction:", prediction.item())
    print("Loss:", loss.item())

    # 4. Manually update parameters using gradient descent.
    learning_rate = 0.01

    with torch.no_grad():
        for p in n.parameters():
            p -= learning_rate * p.grad

    # 5. Clear gradients before the next training iteration.
    for p in n.parameters():
        p.grad = None

    # Check the new prediction.
    new_prediction = n(x)[0]
    new_loss = (new_prediction - target) ** 2

    print("\nAfter one update:")
    print("Prediction:", new_prediction.item())
    print("Loss:", new_loss.item())

    # named_tensors = {
    #     "x0": x[0],
    #     "x1": x[1],
    #     "x2": x[2],
    #     "predictions": prediction,
    #     "loss": loss,
    # }
    # for i, p in enumerate(n.parameters()):
    #     named_tensors[f"p{i}"] = p
    # # Draw the graph leading to the loss.
    # dot = draw_dot(loss, named_tensors)
    # dot.render("computation_graph", cleanup=True)


def multi_pass_dataset():
    # seed makes the random number generator reproducible: with the same seed,
    # code, and execution conditions, you can generally recreate the same initialisation.
    # 42 has no special mathematical significance. It's just a popular example.
    # it is setting a fixed starting point
    torch.manual_seed(42)
    # Four examples, each with three input features.
    xs = [
        [2.0, 3.0, -1.0],  # 1
        [3.0, -1.0, 0.5],  # -1
        [0.5, 1.0, 1.0],  # -1
        [1.0, 1.0, -1.0],  # 1
    ]

    # Desired outputs.
    ys = [1.0, -1.0, -1.0, 1.0]

    # Our network: 3 inputs -> 4 -> 4 -> 1 output.
    model = MLP(3, [4, 4, 1])
    parameters = model.parameters()

    targets = torch.tensor(ys, dtype=torch.float32, device=device)

    learning_rate = 0.05
    epochs = 3000
    target_loss = 1e-5

    for epoch in range(epochs):

        # 1. Forward pass: run all four examples through the MLP.
        predictions = torch.stack([
            model(x)[0] for x in xs
        ])

        # 2. Calculate the mean squared error across all examples.
        loss = ((predictions - targets) ** 2).mean()

        # 3. Clear gradients from the previous iteration.
        for p in parameters:
            p.grad = None

        # 4. Backpropagation: calculate gradients for all parameters.
        loss.backward()
        # print(f"gradient {parameters[0].grad}")

        # 5. Update all weights and biases using gradient descent.
        with torch.no_grad():
            for p in parameters:
                p -= learning_rate * p.grad

        # Check the loss after the parameter update.
        with torch.no_grad():
            updated_predictions = torch.stack([
                model(x)[0] for x in xs
            ])
            updated_loss = (
                    (updated_predictions - targets) ** 2
            ).mean()

        # print(f"learning: Loss: {loss.item():.6f}")
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

        # Brake if we achieve the target early
        if updated_loss.item() < target_loss:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Evaluate the trained network.
    print("\nFinal predictions:")

    with torch.no_grad():
        for x, target in zip(xs, ys):
            prediction = model(x)[0].item()
            print(
                f"Input: {x} | "
                f"Target: {target:+.1f} | "
                f"Prediction: {prediction:+.4f}"
            )
    # save(model)


def save(model):
    # Save learned parameters.
    torch.save(
        [p.detach().cpu() for p in model.parameters()],
        "model.pt",
    )

    print("Model saved to model.pt")
