import torch
from torch import nn
import numpy as np
import kmeans1d

from line_profiler import profile

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
        record_input_stats=False, # Faster with this turned off.
    ):
        """
        scale_output: Apply a per-channel scale factor to the output post bias addition. Scale factor is chosen
                    to make cluster quantization of the weight matrix easier, e.g. by modifying the variance.
        shift_input: Apply a per-channel shift factor to the input prior to matrix multiplication. Shift factor
                    is chosen to make incoming activations zero-mean and thereby reduce the impact of quantization error.
        """
        super().__init__(in_features, out_features, bias=shift_input or bias, device=device, dtype=dtype)
        self.output_scale = None
        self.input_shift = None
        self.scale_output = scale_output
        if scale_output:
            self.output_scale = nn.Parameter(torch.ones(out_features, dtype=dtype, device=device))
        if shift_input:
            self.input_shift = nn.Parameter(torch.zeros((1, in_features), dtype=dtype, device=device))

        self.record_input_stats = record_input_stats
        self.input_means = [] # [torch.Tensor(b, in_features)]
        self.input_stds = [] # [torch.Tensor(b, in_features)]

    def forward(self, x):
        if self.record_input_stats:
            # If collected during a backwards pass, we don't need gradients.
            self.input_means.append(x.mean(dim=-2).values.detach())
            self.input_stds.append(x.std(dim=-2).detach())
        if self.input_shift is not None:
            # Center the input activations prior to the actual linear layer.
            x = x - self.input_shift
        x = super().forward(x)
        if self.output_scale is not None:
            x = x * self.output_scale
        return x

    @torch.no_grad()
    @profile
    def _cluster_based_scales(self, num_clusters, sensitivities):
        """
        Determine the scales based on an approximation of how the row clusters on its own.
        """
        assert num_clusters is not None

        # Empirically, the relative distribution is usually the same but sometimes flipped across the y-axis.
        # It would be nice if there was a way to do this without clustering, but all the heuristics
        # I've tried so far are lacking (ie result in worse perplexity).
        def compute_column_centroids(inputs):
            col, sense, num_clusters = inputs
            kmeans = kmeans1d.cluster(col, num_clusters, weights=sense)
            return np.array(kmeans.centroids)

        # Do this in two steps since it is a bottleneck.
        # First compute all the per-column clusters.
        weight_np = self.weight.detach().cpu().numpy()
        sensitivities_np = sensitivities.cpu().numpy()

        tasks = []
        for col, sense in zip(weight_np, sensitivities_np):
            tasks.append((col, sense, num_clusters))

        centroids = np.zeros((len(tasks), num_clusters))
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        with ThreadPoolExecutor() as executor:
            # for i, result in tqdm(enumerate(executor.map(compute_column_centroids, tasks)), desc="centroids"):
            for i, result in enumerate(executor.map(compute_column_centroids, tasks)):
                centroids[i] = result

        # Then compute the scales based on them.
        def compute_all_scales(centroids, stds):
            scales = np.zeros_like(stds)
            assert len(centroids) == len(stds)
            for i, (cs, std) in enumerate(zip(centroids, stds)):
                flip = -1 if np.abs(cs.min()) > cs.max() else 1
                scale = flip * std
                scales[i] = scale
            return scales

        stds = self.weight.std(dim=1).cpu().numpy()
        return torch.from_numpy(compute_all_scales(centroids, stds)).to(self.weight.device).unsqueeze(-1)

        # Ideally we would look at the weighted clusters and
        # decide to flip if the magnitude of the smallest one
        # is greater than that of the largest one.
        # Unfortunately, this means computing clusters twice
        # which is quite slow.
        # This is close, but not it.
        # stds = self.weight.std(dim=1)
        # weighted_means = (self.weight * sensitivities).sum(dim=1) / sensitivities.sum(dim=1)
        # flips = torch.where(weighted_means < 0, -1, 1)
        # return (stds * flips).unsqueeze(-1)

    @torch.no_grad()
    @profile
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
    @profile
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

        output_scale = self._cluster_based_scales(num_clusters, sensitivities)

        assert list(output_scale.shape) == [self.weight.shape[0], 1], f"{list(output_scale.shape)} != {[self.weight.shape[0], 1]}"
        assert output_scale.squeeze().shape == self.output_scale.squeeze().shape, f"{output_scale.squeeze().shape} != {self.output_scale.shape}"

        # Normalizing the weight to 1 standard deviation improves K-means clustering.
        self.weight.div_(output_scale)
        if self.bias is not None:
            self.bias.div_(output_scale.squeeze())
        self.output_scale.copy_(output_scale.squeeze())
        assert torch.equal(self.output_scale, output_scale.squeeze())

    @torch.no_grad()
    @profile
    def quantize_args(self, nbits: int, sensitivities):
        """
        Generate arguments for parallel quantization.
        """
        num_clusters = 2 ** nbits
        assert self.weight.unique().shape[0] > num_clusters, "Module has already been quantized."

        if sensitivities is None:
            sensitivities = torch.ones_like(self.weight)

        # Must be done here since it modifies weight.
        if self.scale_output:
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
        self.weight.copy_(quantized_weight.to(self.weight.device))

    @torch.no_grad()
    def quantize(self, nbits: int, sensitivities):
        """
        Quantize this layer in-place without changing the precision.
        """
        # Split up so the same code can be run synchronously or in parallel.
        args = self.quantize_args(nbits, sensitivities)
        res = run_quantize(args)
        self.apply_quantize(res)

    @staticmethod
    def update_stat_collection(model, collect_stats: bool):
        for module in model.modules():
            if isinstance(module, ClusterFriendlyLinear):
                module.record_input_stats = collect_stats

    @staticmethod
    def update_scaling(model, apply_scaling: bool):
        for module in model.modules():
            if isinstance(module, ClusterFriendlyLinear):
                module.scale_output = apply_scaling

@profile
def run_quantize(args: dict):
    """
    Static quantize method which can be run in parallel.
    """
    num_clusters = args["num_clusters"]
    weight = args["weight"]
    sensitivities = args["sensitivities"]
    name = args["name"]

    # Interestingly there are often repeated values, so we can sum their sensitivities for a faster kmeans.
    unique_weight, unique_inverse = np.unique(weight.cpu().numpy(), return_inverse=True)
    unique_sensitivities = np.bincount(unique_inverse, weights=sensitivities.cpu().numpy().flatten())

    eps = 0
    while True:
        if eps != 0:
            print(f"wow numerical instability in {name} eps {eps}")

        kmeans = kmeans1d.cluster(unique_weight, num_clusters, weights=unique_sensitivities + eps)
        if not np.isnan(kmeans.centroids).any():
            break
        elif eps == 0:
            eps = 1e-12
        elif eps > 1e-4:
            print(f"WARNING: KMeans significantly interrupted by numerical instability for {name}.")
        else:
            eps *= 100

    wq = np.array(kmeans.centroids, dtype=np.float32)[np.array(kmeans.clusters)]
    assert np.unique(wq).shape[0] <= num_clusters, f"{np.unique(wq).shape[0]} > {num_clusters} for {name}"

    wq = wq[unique_inverse].reshape(weight.shape)
    wq = torch.from_numpy(wq).contiguous()

    assert wq.unique().shape[0] <= num_clusters, f"{wq.unique().shape[0]} > {num_clusters} for {name}"
    return {"weight": wq}