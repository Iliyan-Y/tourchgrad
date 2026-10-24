import torch

from nn import device


# Create the same architecture.
# model = MLP(3, [4, 4, 1])

def predict(model):
    # Load saved parameters.
    saved_parameters = torch.load(
        "model.pt",
        map_location="cpu",
        weights_only=True,
    )

    # Copy learned values into the new network.
    with torch.no_grad():
        for parameter, saved in zip(
                model.parameters(), saved_parameters
        ):
            parameter.copy_(saved.to(device))

    print("\nInteractive prediction (type 'q' to quit)")

    while True:
        raw = input("Enter 3 numbers separated by spaces: ")

        if raw.lower() == "q":
            break

        try:
            values = [float(v) for v in raw.split()]

            if len(values) != 3:
                print("Please enter exactly 3 numbers.")
                continue

            with torch.no_grad():
                prediction = model(values)[0].item()

            # Our training targets were -1 and +1.
            predicted_class = 1 if prediction >= 0 else -1

            print(f"Raw output: {prediction:.4f}")
            print(f"Predicted class: {predicted_class:+d}")

        except ValueError:
            print("Enter valid numbers, e.g. 2 3 -1")
