import sys
import os
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Set

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from gpt2 import GPT2LMHeadModel

from litgpt import GPT as LitGPT, Config as LitConfig, Tokenizer as LitTokenizer
from litgpt.utils import  check_valid_checkpoint_dir, lazy_load

from cluster_friendly_linear import ClusterFriendlyLinear
from quantizer import Quantizer

from line_profiler import profile
from filprofiler.api import profile as filprofile

from jsonargparse import CLI

class WrappedLitTokenizer:
    """
    Make the lit-gpt tokenizer have the same interface as HF.
    """

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, x, return_tensors='pt'):
        return WrappedLitTokenizer.Outputs(input_ids=self.tok.encode(x).unsqueeze(0))

    @dataclass
    class Outputs:
        input_ids: torch.Tensor

        def to(self, device):
            self.input_ids.to(device)
            return self

class WrappedLitModel(nn.Module):
    """
    Make the lit-gpt models have the same interface as HF.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.transformer = model.transformer
        self.config = model.config

    def __call__(self, input_ids, labels=None):
        logits = self.model(input_ids)#[0]
        loss = None
        if labels is not None:
            # From lit-gpt
            # loss = torch.nn.functional.cross_entropy(
            #         logits[:-1], input_ids[0, 1:].to(dtype=torch.long), reduction="sum"
            #     )
            # From HF gpt2
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(dtype=torch.long)
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return WrappedLitModel.Outputs(logits, loss)

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        return self.model.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)

    def save_pretrained(self, name):
        # todo
        pass

    @dataclass
    class Outputs:
        logits: torch.Tensor
        loss: Optional[torch.Tensor]

    @staticmethod
    def update_state_dict(sd, config):
        q_per_kv = config.n_head // config.n_query_groups
        # TODO: This only supports MQA, not GQA

        for key in list(sd.keys()):
            # print(key)
            if "attn.attn" in key:
                qkv = sd[key]._load_tensor()
                # MHA
                q,k,v = qkv.split(qkv.shape[0] // 3, dim=0)
                # MQA (not GQA)
                # q,k,v = qkv.split((q_per_kv * config.head_size, config.head_size, config.head_size), dim=0) # MHA

                assert q_per_kv != 1 or set(q.shape) == set(k.shape) == set(v.shape), f"all shapes should be square"
                sd[key.replace("attn.attn", "attn.q_proj")] = q
                sd[key.replace("attn.attn", "attn.k_proj")] = k
                sd[key.replace("attn.attn", "attn.v_proj")] = v
                del sd[key]
        return sd

def compute_loss(logits, labels):
    # From HF gpt2
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(dtype=torch.long)
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

# From SqueezeLLM-gradients, without the validation data.
@profile
def get_c4(nsamples, seed, seqlen, model, tokenizer=None):
    import random
    random.seed(seed)

    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', token=False
    )

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        input_ids_len = trainenc.input_ids.shape[1]
        i = random.randint(0, input_ids_len - seqlen - 1) if input_ids_len != seqlen else 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        yield (inp, tar)

@profile
def collect_sensitivities(model_name_or_path, model, output_path, seqlen, device, tokenizer=None):
    # Copied and adapted from SqueezeLLM-gradients.

    def get_modules(model_name, layer):
        if "gpt2" in model_name:
            return [
                layer.attn.c_attn.q_proj,
                layer.attn.c_attn.k_proj,
                layer.attn.c_attn.v_proj,
                layer.attn.c_proj,
                layer.mlp.c_fc,
                layer.mlp.c_proj,
            ]
        # else it's lit-gpt
        elif model.config.mlp_class == "GptNeoxMLP":
            return [
                layer.attn.q_proj,
                layer.attn.k_proj,
                layer.attn.v_proj,
                layer.attn.proj,
                layer.mlp.fc,
                layer.mlp.proj,
            ]
        elif model.config.mlp_class in ["LLaMAMLP", "GemmaMLP"]:
            return [
                layer.attn.q_proj,
                layer.attn.k_proj,
                layer.attn.v_proj,
                # layer.attn.proj,
                layer.mlp.fc_1,
                layer.mlp.fc_2,
                layer.mlp.proj,
            ]

    # Use c4, like the fishers. Not wikitext since we're testing on it.
    nsamples = 100
    seed = 42
    dataloader = get_c4(nsamples=nsamples, seed=seed, model=model_name_or_path, seqlen=seqlen, tokenizer=tokenizer)

    _layers = model.transformer.h
    _num_linear_per_layer = len(get_modules(model_name_or_path, _layers[0]))

    # Broken on Colab
    # model = torch.compile(model)
    model = model.to(device)

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Diverge from the original SqueezeLLM-gradients to reduce memory usage (enough to run 6.9b @ 512 seqlen on 64GB RAM).
    # https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
    optimizer_dict = {p: optim.SGD([p], lr=0.01, momentum=0.9, foreach=False) for p in model.parameters()}

    grads = [[0.] * _num_linear_per_layer for _ in _layers]

    index_dict = {}
    for i, layer in enumerate(_layers):
        for j, module in enumerate(get_modules(model_name_or_path, layer)):
            index_dict[module.weight] = (i, j)

    def optimizer_hook(parameter) -> None:
        if parameter in index_dict:
            i,j = index_dict[parameter]
            grads[i][j] += (parameter.grad ** 2).float().cpu()
        optimizer_dict[parameter].zero_grad()

    for p in model.parameters():
        p.register_post_accumulate_grad_hook(optimizer_hook)

    for data in tqdm(dataloader, total=nsamples):
        data = data[0]
        # with torch.autocast(device_type=device, enabled=device != "mps", dtype=torch.float16):
        x = data.to(device)
        outputs = model(input_ids=x, labels=x)
        loss = outputs.loss
        loss.backward()

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # This is a hacky solution to save the gradients
    # where we overwrite all the weights in the model as the gradients
    # and use HF save_pretrained
    for i, layer in enumerate(_layers):
        for j, module in enumerate(get_modules(model_name_or_path, layer)):
            module.weight.data = grads[i][j]

    print(f"saving model gradient at {output_path}")
    from safetensors.torch import save_model
    metadata = {
        'nsamples': nsamples,
        'seed': seed,
        'seqlen': seqlen,
    }
    metadata = {k: str(v) for k,v in metadata.items()}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m = model if "gpt2" in model_name_or_path else model.model
    save_model(m, output_path, metadata)

@torch.no_grad()
@profile
def collect_activation_stats(model_name_or_path, model, output_path, seqlen, device, tokenizer=None):
    ClusterFriendlyLinear.update_stat_collection(model, True) # Enable stat collection.
    assert output_path.endswith("safetensors")
    # Use c4, like the fishers. Not wikitext since we're testing on it.
    nsamples = 100
    dataloader = get_c4(nsamples=nsamples, seed=42, model=model_name_or_path, seqlen=seqlen, tokenizer=tokenizer)

    for data in (pbar := tqdm(dataloader)):
        pbar.set_description("Collecting input activation statistics")
        data = data[0]
        x = data.to(device)
        _ = model(input_ids=x)

    d = {}
    for name, module in model.named_modules():
        if isinstance(module, ClusterFriendlyLinear):
            means = torch.stack(module.input_means)
            assert means.shape[-1] == module.weight.shape[1], f"mismatched shapes: {means.shape} and {module.weight.shape}"
            d[f"{name}.input_means"] = means
            stdevs = torch.stack(module.input_stds)
            assert stdevs.shape[-1] == module.weight.shape[1], f"mismatched shapes: {stdevs.shape} and {module.weight.shape}"
            d[f"{name}.input_stds"] = stdevs
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(d, output_path, metadata={"model": model_name_or_path, "seqlen": str(seqlen), "device": device, "nsamples": str(nsamples)})
    ClusterFriendlyLinear.update_stat_collection(model, False) # Disable stat collection for increased performance + decreased memory.

@profile
def main(
    model_name: str = "gpt2",
    nbits: int=4,
    device: str="mps",
    weighting: bool = False,
    scaling: bool = False,
    centering: bool = False,
    quantize: bool = True,
    save: bool = False,
):
    print(f"Running with model {model_name} on {device}")
    print(f"Quantizing to {nbits} bits with weighting={weighting}, scaling={scaling}, centering={centering}, quantize={quantize}")
    sensitivities = f"observed-information/{model_name}-grads.safetensors"
    activation_stats = f"observed-information/{model_name}-activation-stats.safetensors"

    if not quantize and (weighting or scaling):
        print(F"Warning: Unexpected to use weight or scaling without quantization. These should have no impact.")

    if save:
        filename = f"{model_name}_{nbits}bit"
        print("Will save model to:", filename)
    model = None

    needs_activation_stats = centering and not os.path.exists(activation_stats)
    needs_sensitivities = (weighting or scaling) and not os.path.exists(sensitivities)

    if "gpt2" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        max_length = model.config.n_positions
    else:
        # Assume lit-gpt.
        checkpoint_dir = Path("checkpoints") / model_name
        check_valid_checkpoint_dir(checkpoint_dir)

        config = LitConfig.from_file(checkpoint_dir / "model_config.yaml")
        config._linear_class = ClusterFriendlyLinear
        print("config", config)
        model = LitGPT(config)

        checkpoint_path = checkpoint_dir / "lit_model.pth"
        checkpoint = lazy_load(checkpoint_path)

        # We expect the params we will be adding to be missing.
        sd = checkpoint.get("model", checkpoint)
        sd = WrappedLitModel.update_state_dict(sd, config)
        missing_keys, _ = model.load_state_dict(sd, strict=False)
        # missing_keys, _ = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
         # TODO: bias only if centering? Maybe?
        missing_keys = [key for key in missing_keys if "input_shift" not in key and "output_scale" not in key and "bias" not in key]
        assert len(missing_keys) == 0, f"Missing key(s) in state_dict: {missing_keys}"

        model = WrappedLitModel(model)

        tokenizer = WrappedLitTokenizer(LitTokenizer(checkpoint_dir))
        max_length = min(config.block_size, 2048) # > 2048 takes too much RAM for LLaMA, < 2048 and the numbers don't match for most evals
        observed_info_max_length = max_length
        needs_observed_info = needs_activation_stats or needs_sensitivities
        if any([x in model_name for x in ["2.8b", "3b", "6.9b", "7b"]]) and needs_observed_info:
            observed_info_max_length = 512 # observed_info_max_length // 4
            # Otherwise MPS uses a ton of memory. TODO try mimalloc
            # SqueezeLLM only uses 512 seq lenth anyways.
            print(f"Shortened observed info max length from {config.block_size} to {observed_info_max_length} for large model")

    # Move the model to a faster device for inference.
    if needs_activation_stats or needs_sensitivities:
        model = model.to(device)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if needs_activation_stats:
        print(f"Activation stats not found at {activation_stats}. Generating...")
        collect_activation_stats(model_name, model, activation_stats, observed_info_max_length, device, tokenizer)
        gc.collect()

    if needs_sensitivities:
        print(f"Sensitivities not found at {sensitivities}. Generating...")
        collect_sensitivities(model_name, model, sensitivities, observed_info_max_length, device, tokenizer)
        print("Sensitivities gathered, re-run to quantize.")
        sys.exit(0)

    if centering:
        # This is faster on a faster device.
        gc.collect()
        model = model.to("mps")
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        quantizer = Quantizer(model, "mps")
        quantizer.center_activations(activation_stats) # Apply shifts.


    if quantize:
        # Quantization is CPU-heavy, so we move the model back to CPU.
        model = model.to("cpu")
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        quantizer = Quantizer(model, "cpu")
        ClusterFriendlyLinear.update_scaling(model, scaling) # Toggle scaling.
        quantizer.quantize(nbits, sensitivities if weighting else None) # Quantizing using weights and/or scales.

    if save:
        model.save_pretrained(filename)

    # Ensure that the model is actually quantized.
    if quantize:
        quantizer.assert_quantized(nbits)
    else:
        print("Quantization not requested, skipping assertion.")

    gc.collect()
    model = model.to(device)
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name) if tokenizer is None else tokenizer

    # Calculate perplexity as in https://huggingface.co/docs/transformers/en/perplexity
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt").to(device)

    stride = 512
    seq_len = encodings.input_ids.size(1)
    nlls = []
    parallel_perplexity = False # BUGGY on MPS at batch size > 1. Also, doesn't seem memory bound for ~160M models.

    if not parallel_perplexity:
    # START: original computation
        prev_end_loc = 0
        for begin_loc in (pbar := tqdm(range(0, seq_len, stride))):
            pbar.set_description("Calculating perplexity")
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.inference_mode():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
    # END: original computation
    else:
    # START: parallel computation
        batches = []
        batch_size = 8 # 8 overflows on mps @ 160M
        prev_end_loc = 0
        for begin_loc in (pbar := tqdm(range(0, seq_len, stride))):
            pbar.set_description(f"Preparing batches of {batch_size} for perplexity")
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            is_last_loop = trg_len != stride
            if len(batches) == 0 or len(batches[-1]) == batch_size or is_last_loop:
                batches.append([])
            batches[-1].append((input_ids, target_ids))

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        for batch in (pbar := tqdm(batches)):
            pbar.set_description("Calculating perplexity")
            batch_inputs = torch.cat([x[0] for x in batch])

            assert batch_inputs.shape[0] <= batch_size
            batch_targets = [x[1] for x in batch]

            with torch.no_grad():
                # HuggingFace models don't support computing unbatched loss.
                batch_outputs = model(batch_inputs).logits

                for (bo, bt) in zip(batch_outputs.split(1, dim=0), batch_targets):
                    nlls.append(compute_loss(bo, bt))
    # END: parallel computation

    assert not torch.isnan(torch.stack(nlls)).any()
    assert not torch.isinf(torch.stack(nlls)).any()
    ppl = torch.exp(torch.stack(nlls).mean())
    print("Wikitext perplexity:", ppl)

if __name__ == "__main__":
    CLI(main)