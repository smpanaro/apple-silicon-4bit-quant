import torch
import gc
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from cluster_friendly_linear import ClusterFriendlyLinear
from line_profiler import profile
from safetensors import safe_open
from tqdm import tqdm

class Quantizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def center_activations(self, activation_stats: str):
        """
        Use the provided activation statistics to zero mean the inputs of
        linear layers to minimize quantization error.
        """
        from safetensors import safe_open
        activation_stats_path = activation_stats
        activation_stats = safe_open(activation_stats, "pt", self.device)

        assert len(activation_stats.keys()) > 0, f"empty sensitivities file at path {activation_stats_path}"
        assert activation_stats is not None, "no activation stats provided"

        for i, (name, module) in (pbar := tqdm(enumerate(self._named_custom_linears()))):
            pbar.set_description("Centering linear layer input activations")
            if isinstance(module, ClusterFriendlyLinear):
                module_stats = activation_stats.get_tensor(f"{name}.input_means")
                assert module_stats is not None, f"no stats found for {name}"
                module.center_activations(module_stats.squeeze(1)) # remove the batch singleton in [nsamples, 1, hidden dim]

            if i % 10 == 0:
                gc.collect()
                torch.mps.empty_cache()

    @torch.no_grad()
    @profile
    def quantize(self, nbits: int, sensitivities: dict[str, torch.Tensor]={}, parallel=True):
        """
        Quantize the model in place. If sensitivites are provided, they are used
        to improve quantization accuracies (substantially).
        """

        def make_chunks(worklist, batchsize):
            for i in range(0, len(worklist), batchsize):
                yield worklist[i:i+batchsize]

        # Always open on CPU since these are only used in kmeans which is CPU-only.
        if isinstance(sensitivities, str):
            sensitivities_path = sensitivities
            sensitivities = safe_open(sensitivities, "pt", "cpu")
            assert len(sensitivities.keys()) > 0, f"empty sensitivities file at path {sensitivities_path}"

        # Chunk so we can reduce memory usage for large models.
        chunks = list(make_chunks(self._named_custom_linears(), 30))
        print(f"Quantizing model in {len(chunks)} chunks.")
        # for chunk in tqdm(chunks, desc="Quantizing chunks", leave=True):
        for i, chunk in enumerate(chunks):
            # Load the sensitivities for this chunk.
            chunk_sensitivities = {}
            if sensitivities is not None:
                for name, module in chunk:
                    key_name = name
                    # TEMPORARY FOR LLAMA and GEMMA (and anything else that comes in from mlx)
                    # if f"{key_name}.weight" not in sensitivities.keys():
                    #     key_name = key_name\
                    #         .replace("attn", "self_attn")\
                    #         .replace("self_attn.proj", "self_attn.o_proj")\
                    #         .replace("transformer.h", "model.layers")\
                    #         .replace("mlp.fc_1", "mlp.gate_proj")\
                    #         .replace("mlp.fc_2", "mlp.up_proj")\
                    #         .replace("mlp.proj", "mlp.down_proj")
                    # GPT ONLY:
                    key_name = key_name.replace(".q_proj", "")
                    key_name = key_name.replace(".k_proj", "")
                    key_name = key_name.replace(".v_proj", "")
                    # GEMMA:
                    # key_name = key_name.replace("transformer", "model.layers")\
                    #         .replace("mlp.fc_1", "mlp.gate_proj")\
                    #         .replace("mlp.fc_2", "mlp.up_proj")\
                    #         .replace("mlp.proj", "mlp.down_proj")
                    # print(sensitivities.keys())
                    # print(key_name  )

                    # Erroring here? You probably need to manipulate your key names, similar to above.
                    chunk_sensitivities[key_name] = sensitivities.get_tensor(f"{key_name}.weight")

            if hasattr(self.model, "convert_sensitivities"):
                chunk_sensitivities = self.model.convert_sensitivities(chunk_sensitivities, just_attn=True)

            modules = []
            tasks = []
            # Manual progress for chunks since nested tqdm doesn't seem to work.
            for name, module in tqdm(chunk, desc=f"Preparing chunk {i+1}/{len(chunks)} for quantization"):
                module_sensitivity = chunk_sensitivities.get(name, None)
                if len(chunk_sensitivities) > 0:
                    assert module_sensitivity is not None, f"no sensitivity found for {name}"
                    assert module_sensitivity.shape == module.weight.shape, f"mismatched sensitivity shape for {name}, expected {module.weight.shape}, got {module_sensitivity.shape}"

                if parallel:
                    modules.append(module)
                    tasks.append({"name": name} | module.quantize_args(nbits, module_sensitivity))
                else:
                    module.quantize(nbits, module_sensitivity)

            max_inflight_elements = 4096*4096*9 # ~45GB for pythia-6.9b -- tune per model (seems memory use is not linear with # paramss)
            with BoundedExecutor(max_inflight_elements, max_workers=os.cpu_count()) as executor:
                func = ClusterFriendlyLinear.quantize_func()
                futures = []
                for task in tqdm(tasks, desc=f"Quantizing to {nbits} bits"):
                # for task in tasks:
                    future = executor.submit(task["weight"].numel(), func, task)
                    futures.append(future)

                for i, future in tqdm(enumerate(futures), desc=f"Applying quantized results", total=len(futures)):
                # for i, future in enumerate(futures):
                    result = future.result()
                    modules[i].apply_quantize(result)

            del chunk_sensitivities, modules, tasks
            gc.collect()
            torch.mps.empty_cache()

    def assert_quantized(self, nbits: int):
        num_clusters = 2 ** nbits
        total = 0
        for name, module in self.model.named_modules():
            if isinstance(module, ClusterFriendlyLinear):
                distinct_weight_values = module.weight.unique().shape[0]
                assert distinct_weight_values <= num_clusters, f"{name} has {distinct_weight_values} unique values, expected {num_clusters} or fewer."
                total += 1
        assert total > 0, "No quantized linear layers found in the model."
        print(f"Validated that all {total} linear layers have <= {num_clusters} unique values.")

    def _named_custom_linears(self):
        # Return a list for better progress bars.
        return [(name, module)
                for (name, module) in self.model.named_modules()
                if isinstance(module, ClusterFriendlyLinear)]

from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
class BoundedExecutor:
    """BoundedExecutor behaves as a ThreadPoolExecutor which will block on
    calls to submit() once the limit given as "bound" work items are queued for
    execution.
    :param bound: Integer - the maximum number of items in the work queue
    :param max_workers: Integer - the size of the thread pool

    https://gist.github.com/frankcleary/f97fe244ef54cd75278e521ea52a697a
    """
    def __init__(self, bound, max_workers):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = BoundedSemaphore(bound)

    """See concurrent.futures.Executor#submit"""
    def submit(self, cost, fn, *args, **kwargs):
        # :|
        for _ in range(cost):
            self.semaphore.acquire()

        try:
            future = self.executor.submit(fn, *args, **kwargs)
        except:
            self.semaphore.release(n=cost)
            raise
        else:
            future.add_done_callback(lambda x: self.semaphore.release(n=cost))
        return future

    """See concurrent.futures.Executor#shutdown"""
    def shutdown(self, wait=True):
        self.executor.shutdown(wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False