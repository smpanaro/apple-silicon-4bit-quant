import os
import torch
from torch import nn
from transformers.models.gpt2 import modeling_gpt2
from cluster_friendly_linear import ClusterFriendlyLinear

"""
Modify huggingface's GPT2 model to split the Q,K, and V projections
and to apply the weighting, scaling and shifting.
"""

class CAttn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = ClusterFriendlyLinear(config.hidden_size, config.hidden_size)
        self.k_proj = ClusterFriendlyLinear(config.hidden_size, config.hidden_size)
        self.v_proj = ClusterFriendlyLinear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return torch.cat([q, k, v], dim=-1)

class GPT2Attention(modeling_gpt2.GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)
        setattr(self, 'c_attn', CAttn(config))
        setattr(self, 'c_proj', ClusterFriendlyLinear(config.hidden_size, config.hidden_size))

class GPT2MLP(modeling_gpt2.GPT2MLP):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        setattr(self, 'c_fc', ClusterFriendlyLinear(config.hidden_size, intermediate_size))
        setattr(self, 'c_proj', ClusterFriendlyLinear(intermediate_size, config.hidden_size))

class GPT2Block(modeling_gpt2.GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
        setattr(self, 'attn', GPT2Attention(config))
        setattr(self, 'mlp', GPT2MLP(inner_dim, config))

class GPT2Model(modeling_gpt2.GPT2Model):
    _keys_to_ignore_on_load_missing = ["c_attn.weight", "c_attn.bias"] # split and handled in CAttn
    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'h',
                nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]))
        self._register_load_state_dict_pre_hook(pre_load_hook)

class GPT2LMHeadModel(modeling_gpt2.GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [
        # From superclass.
        r"attn.masked_bias", r"attn.bias", r"lm_head.weight",
        # c_attn is split into these.
        r"q_proj.weight",
        r"q_proj.bias",
        r"k_proj.weight",
        r"k_proj.bias",
        r"v_proj.weight",
        r"v_proj.bias",
        # New parameter for easier quantization, initialized to have no impact.
        r"output_scale",
        r"input_scale",
        r"input_shift",
    ]
    # c_attn is split into q,k,v projections.
    _keys_to_ignore_on_load_unexpected = [ r"c_attn.weight", r"c_attn.bias"]

    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'transformer', GPT2Model(config))

    @torch.no_grad()
    def center_activations(self, activation_stats: dict[str, torch.Tensor]):
        """
        Use the provided activation statistics to zero mean the inputs of
        linear layers to minimize quantization error.
        """
        if isinstance(activation_stats, str):
            from safetensors import safe_open
            d = {}
            with safe_open(activation_stats, "pt", "cpu") as f:
                for k in f.keys():
                    d[k] = f.get_tensor(k)
            assert len(d) > 0, f"empty sensitivities file at path {activation_stats}"
            activation_stats = d
        assert activation_stats is not None, "no activation stats provided"
        assert len(activation_stats) > 0, "no activation stats provided"

        try:
            from tqdm import tqdm
            HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False
            def tqdm(iterator, *args, **kwargs):
                return iterator

        for name, module in (pbar := tqdm(self.named_modules())):
            if HAS_TQDM:
                pbar.set_description("Centering linear layer input activations")
            if isinstance(module, ClusterFriendlyLinear):
                module_stats = activation_stats.get(f"{name}.input_means", None)
                assert module_stats is not None, f"no stats found for {name}"
                module.center_activations(module_stats.squeeze(1)) # remove the batch singleton in [nsamples, 1, hidden dim]

    @torch.no_grad()
    def quantize(self, nbits: int, sensitivities: dict[str, torch.Tensor]={}, parallel=True):
        """
        Quantize the model in place. If sensitivites are provided, they are used
        to improve quantization accuracies.
        """

        if isinstance(sensitivities, str):
            from safetensors import safe_open
            d = {}
            with safe_open(sensitivities, "pt", "cpu") as f:
                for k in f.keys():
                    d[k] = f.get_tensor(k)
            assert len(d) > 0, f"empty sensitivities file at path {sensitivities}"
            sensitivities = d
        elif sensitivities is None:
            sensitivities = {}

        if len(sensitivities) > 0:
            pre_load_hook(sensitivities, None, None, None, None, None, None)

        try:
            from tqdm import tqdm
            HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False
            def tqdm(iterator, *args, **kwargs):
                return iterator

        modules = []
        tasks = []
        for name, module in (pbar := tqdm(self.named_modules())):
            if HAS_TQDM:
                pbar.set_description("Preparing for parallel quantization")
            if isinstance(module, ClusterFriendlyLinear):
                module_sensitivity = sensitivities.get(f"{name}.weight", None)
                if len(sensitivities) > 0:
                    assert module_sensitivity is not None, f"no sensitivity found for {name}"
                    assert module_sensitivity.shape == module.weight.shape, f"mismatched sensitivity shape for {name}"

                if parallel:
                    modules.append(module)
                    tasks.append(module.quantize_args(nbits, module_sensitivity))
                else:
                    module.quantize(nbits, module_sensitivity)

        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor() as executor:
                func = ClusterFriendlyLinear.quantize_func()
                def wrapped_func(i):
                    def inner(args):
                        return (i, func(args))
                    return inner
                futures = {executor.submit(wrapped_func(i), task) for i, task in enumerate(tasks)}

                for future in (pbar := tqdm(as_completed(futures), total=len(futures))):
                    if HAS_TQDM:
                        pbar.set_description(f"Quantizing to {nbits} bits")
                    (i, result) = future.result()
                    modules[i].apply_quantize(result)

    def assert_quantized(self, nbits: int):
        num_clusters = 2 ** nbits
        total = 0
        for name, module in self.named_modules():
            if isinstance(module, ClusterFriendlyLinear):
                distinct_weight_values = module.weight.unique().shape[0]
                assert distinct_weight_values <= num_clusters, f"{name} has {distinct_weight_values} unique values, expected {num_clusters} or fewer."
                total += 1
        print(f"Validated that all {total} linear layers have <= {num_clusters} unique values.")

    def convert_sensitivities(self, sensitivities, just_attn=False):
        """
        If HF sensitivities are provided, convert them so they match this model.
        """
        if len(sensitivities) > 0:
            pre_load_hook(sensitivities, None, None, None, None, None, None, just_attn=just_attn)
        return sensitivities

def pre_load_hook(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs, just_attn=False):
    """
    Convert the conv1d weights to linear.
    """
    keys = list(state_dict.keys())

    if any('output_scale' in k for k in keys):
        # Checkpoint was already converted.
        return

    for name in keys:
        if "c_attn" in name:
            q,k,v = state_dict[name].t().chunk(3, dim=0)
            state_dict[name.replace("c_attn", "c_attn.q_proj")] = q
            state_dict[name.replace("c_attn", "c_attn.k_proj")] = k
            state_dict[name.replace("c_attn", "c_attn.v_proj")] = v
            del state_dict[name]
            # print('split', name)
            # if len(q.shape) > 1:
            #     print(f"{name}.q", q[:3, :3].flatten())
            #     print(f"{name}.k", k[:3, :3].flatten())
            #     print(f"{name}.v", v[:3, :3].flatten())
        # elif just_attn:
        #     print("skipping")
        #     continue

        elif not name.endswith('bias') and any([x in name for x in ['c_attn', 'c_fc', 'c_proj']]):
            state_dict[name] = state_dict[name].t()
            # print("transposed", name)