import torch

class EventSNN():
    def __init__(self, height, width, device = "cpu"):
        
        self.height = height
        self.width = width
        self.device = device

        self.n = height * width * 2

        self.input_layer = Layer(self.n, self.n // 4, device=self.device)
        self.hidden1 = Layer(self.n // 4, self.n // 4, device = self.device)
        self.out_layer = Layer(self.n//4 , self.n, device = self.device)

    def events_to_spikes(self):
        pass


class LIFNeurons:
    def __init__(self, n_neurons, tau=10.0, threshold=1.0, reset=0.0, device="cpu"):
        self.n = n_neurons
        self.tau = tau
        self.threshold = threshold
        self.reset = reset
        self.device = device

        self.neurons = []

        self.V = torch.zeros(self.n, device = self.device)   

        self.last_t = None

    def decay(self, t):
        if self.last_t is None:
            self.last_t = t
            return

        dt = t - self.last_t
        decay_factor = torch.exp(torch.tensor(-dt / self.tau, device=self.device))
        self.V *= decay_factor
        self.last_t = t


class Layer:
    def __init__(self, n_input, n_output, tau=10.0, threshold=1.0, device="cpu"):
        self.n_input = n_input
        self.n_output = n_output
        self.device = device

        # weights: [input → output]
        self.W = torch.randn(n_input, n_output, device=device) * 0.1

        self.neurons = LIFNeurons(n_output, tau=tau, threshold=threshold, device=device)

    def get_neuron_idx(self, x, y, pol):
        pol = int(pol)
        pol = 1 if pol > 1 else pol
        pol = 0 if pol < 0 else pol

        return 2 * (x + self.width * y) + pol

    def forward_event(self, input_index, t):
        """
        input_index: int (which input neuron spiked)
        t: timestamp
        """

        # 1. decay all neurons
        self.neurons.decay(t)

        # 2. get synaptic weights for this input spike
        # this is the key vectorized trick
        current = self.W[input_index]   # shape: [n_output]

        # 3. integrate
        self.neurons.V += current

        # 4. spike generation (vectorized)
        spikes = (self.neurons.V >= self.neurons.threshold)

        # 5. reset spiking neurons
        self.neurons.V[spikes] = self.neurons.reset

        return spikes.float()