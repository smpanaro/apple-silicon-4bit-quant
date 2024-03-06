import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from gpt2 import GPT2LMHeadModel
from cluster_friendly_linear import ClusterFriendlyLinear
from transformers import AutoTokenizer


"""

"""

# From SqueezeLLM-gradients, without the validation data.
def get_c4(nsamples, seed, seqlen, model):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', token=False
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

@torch.no_grad()
def collect_activation_stats(model_name_or_path, model, seqlen, device):
    # Use c4, like the fishers. Not wikitext since we're testing on it.
    nsamples = 100
    dataloader = get_c4(nsamples=nsamples, seed=42, model=model_name_or_path, seqlen=seqlen)

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
    save_file(d, f"{model_name_or_path}-activation-stats.safetensors", metadata={"model": model_name_or_path, "seqlen": str(seqlen), "device": device, "nsamples": str(nsamples)})

model_name = "gpt2"
nbits = 4
sensitivities = f"{model_name}-grads.safetensors"
activation_stats = f"{model_name}-activation-stats.safetensors"
device = "cpu" # mps gives inaccurate results

filename = f"{model_name}_4bit"
print("Quantizing and saving model to:", filename)
model = GPT2LMHeadModel.from_pretrained(model_name)
collect_activation_stats(model_name, model, 1024, device) # Skip this if you already have the stats file. Also set record_input_stats=False to gain speed.
model.center_activations(activation_stats) # Apply shifts.
model.quantize(nbits, sensitivities) # Quantizing using weights and scales.
model.save_pretrained(filename)

# Ensure that the model is actually quantized.
model.assert_quantized(nbits)

model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Calculate perplexity as in https://huggingface.co/docs/transformers/en/perplexity
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt").to(device)

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in (pbar := tqdm(range(0, seq_len, stride))):
    pbar.set_description("Calculating perplexity")
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print("Wikitext perplexity:", ppl)