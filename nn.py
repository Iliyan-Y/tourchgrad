import torch

# device = torch.device(
#     "mps" if torch.backends.mps.is_available() else "cpu"
# )


device = torch.device("cpu")


class Neuron:
    def __init__(self, number_of_inputs):
        # One independent scalar weight per input.
        self.w = [
            torch.empty((), device=device, dtype=torch.float32).uniform_(-1, 1)
            for _ in range(number_of_inputs)
        ]
        # Tell autograd to track each parameter.
        for w in self.w:
            w.requires_grad_()

        # Once scaler bias
        self.b = torch.empty((), device=device, dtype=torch.float32).uniform_(-1, 1)
        self.b.requires_grad_()

    def __call__(self, x):
        # w * x + b
        raw_cell = sum(
            (wi * xi for wi, xi in zip(self.w, x)),
            start=torch.zeros((), device=device),
        ) + self.b
        activation = torch.tanh(raw_cell)
        return activation

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [
            Neuron(nin) for _ in range(nout)
        ]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [
            p
            for neuron in self.neurons
            for p in neuron.parameters()
        ]


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts

        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [
            p
            for layer in self.layers
            for p in layer.parameters()
        ]
