import torch
from torch import nn
import numpy as np

class ClusterFriendlyLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear that transforms the weight
    matrix to make it more amenable to cluster-based quantization.
    """

    def __init__(self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        scale_output=True,
        shift_input=True,
        record_input_stats=True, # Faster with this turned off.
    ):
        """
        scale_output: Apply a per-channel scale factor to the output post bias addition. Scale factor is chosen
                    to make cluster quantization of the weight matrix easier, e.g. by modifying the variance.
        shift_input: Apply a per-channel shift factor to the input prior to matrix multiplication. Shift factor
                    is chosen to make incoming activations zero-mean and thereby reduce the impact of quantization error.
        """
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.output_scale = None
        self.input_shift = None
        if scale_output:
            self.output_scale = nn.Parameter(torch.ones(out_features, dtype=dtype, device=device))
        if shift_input:
            self.input_shift = nn.Parameter(torch.zeros((1, in_features), dtype=dtype, device=device))

        self.record_input_stats = record_input_stats
        if self.record_input_stats:
            self.input_means = [] # [torch.Tensor(b, in_features)]
            self.input_stds = [] # [torch.Tensor(b, in_features)]

    def forward(self, x):
        if self.record_input_stats:
            self.input_means.append(x.mean(dim=-2))
            self.input_stds.append(x.std(dim=-2))
        if self.input_shift is not None:
            # Center the input activations prior to the actual linear layer.
            x = x - self.input_shift
        x = super().forward(x)
        if self.output_scale is not None:
            x = x * self.output_scale
        return x

    @torch.no_grad()
    def _cluster_based_scales_parallel(self, num_clusters, sensitivities):
        """
        Determine the scales based on how the row clusters on its own, in parallel.
        """
        assert num_clusters is not None

        def find_col_scales(inputs):
            import kmeans1d
            col, sense, num_clusters = inputs
            kmeans = kmeans1d.cluster(col.detach().numpy(), num_clusters, weights=sense.detach().numpy())
            cs = torch.tensor(kmeans.centroids)
            # Empirically, the relative distribution is usually the same but sometimes flipped across the y-axis.
            # There is probably a faster way to do this.
            flip = -1 if cs.min().abs() > cs.max() else 1
            scale = flip * col.std()
            return scale

        tasks = []
        for col, sense in zip(self.weight, sensitivities):
            tasks.append((col, sense, num_clusters))

        scales = []
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor() as executor:
            for result in executor.map(find_col_scales, tasks):
                scales.append(result)

        return torch.tensor(scales).unsqueeze(-1)

    @torch.no_grad()
    def center_activations(self, activation_means: torch.Tensor):
        assert activation_means is not None
        assert activation_means.shape[-1] == self.weight.shape[1], "Activation shifts are applied to the input, shapes must match."
        assert len(activation_means.shape) == 2, "Activation means must be 2D (samples, means)."
        assert self.input_shift is not None, "Input shift has not been initialized."
        assert torch.all(self.input_shift == 0), "Input shift has already been set."

        if self.weight.unique().shape[0] <= 2**8:
            print("WARNING: Applying activation shifts on a tensor that looks like it was already quantized. \
You will probably get better results by centering activations prior to quantization.")

        shifts = activation_means.mean(dim=0)
        self.input_shift.copy_(shifts.unsqueeze(0))

        # Compensate in the bias.
        bias_adjustment = shifts @ self.weight.t()
        self.bias.add_(bias_adjustment)

    @torch.no_grad()
    def _update_scales(self, sensitivities, num_clusters=None):
        """
        Update the output scale based on the sensitivities.

        Exposed for testing.
        """
        assert self.output_scale is None or torch.all(self.output_scale == 1), "Scales have already been set."

        if sensitivities is not None:
            sensitivities = sensitivities
            assert torch.all(sensitivities >= 0), "Sensitivities must be positive."
        else:
            sensitivities = torch.ones_like(self.weight)

        output_scale = self._cluster_based_scales_parallel(num_clusters, sensitivities)
        assert list(output_scale.shape) == [self.weight.shape[0], 1], f"{list(output_scale.shape)} != {[self.weight.shape[0], 1]}"
        assert output_scale.squeeze().shape == self.output_scale.squeeze().shape, f"{output_scale.squeeze().shape} != {self.output_scale.shape}"

        # Normalizing the weight to 1 standard deviation improves K-means clustering.
        self.weight.div_(output_scale)
        if self.bias is not None:
            self.bias.div_(output_scale.squeeze())
        self.output_scale.copy_(output_scale.squeeze())
        assert torch.equal(self.output_scale, output_scale.squeeze())

    @torch.no_grad()
    def quantize_args(self, nbits: int, sensitivities):
        """
        Generate arguments for parallel quantization.
        """
        num_clusters = 2 ** nbits
        assert self.weight.unique().shape[0] > num_clusters, "Module has already been quantized."

        if sensitivities is None:
            sensitivities = torch.ones_like(self.weight)

        # Must be done here since it modifies weight.
        self._update_scales(sensitivities, num_clusters=num_clusters)

        weight = self.weight

        return {"num_clusters": num_clusters, "weight": weight.detach(), "sensitivities": sensitivities.detach()}

    @staticmethod
    def quantize_func():
        """
        Return a function to quantize the dict returned by `quantize_args`.
        """
        return run_quantize

    @torch.no_grad()
    def apply_quantize(self, result: dict):
        """
        Update the layer with the quantized weights returned by `run_quantize`.
        """
        quantized_weight = result["weight"]
        self.weight.copy_(quantized_weight)

    @torch.no_grad()
    def quantize(self, nbits: int, sensitivities):
        """
        Quantize this layer in-place without changing the precision.
        """
        # Split up so the same code can be run synchronously or in parallel.
        args = self.quantize_args(nbits, sensitivities)
        res = run_quantize(args)
        self.apply_quantize(res)

def run_quantize(args: dict):
    """
    Static quantize method which can be run in parallel.
    """
    import kmeans1d

    num_clusters = args["num_clusters"]
    weight = args["weight"]
    sensitivities = args["sensitivities"]

    # kmeans1d finds the globally optimal solution.
    # sklearn's KMeans is susceptible to finding local minima.
    weight1d = weight.reshape(-1)
    sensitivities1d = sensitivities.reshape(-1)

    kmeans = kmeans1d.cluster(weight1d.numpy(), num_clusters, weights=sensitivities1d.numpy())

    wq = np.array(kmeans.centroids)[np.array(kmeans.clusters)]
    assert weight1d.shape[0] == wq.shape[0], f"{weight1d.shape[0]} != {wq.shape[0]}"
    wq = wq.reshape(weight.shape)
    # .float() because kmeans1d returns doubles
    wq = torch.from_numpy(wq).contiguous().float()

    assert wq.unique().shape[0] == num_clusters, f"{wq.unique().shape[0]} != {num_clusters}"
    return {"weight": wq, "sensitivities": sensitivities}