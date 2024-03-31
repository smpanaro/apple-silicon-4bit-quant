# LLMs for your iPhone: Whole-Tensor 4 Bit Quantization

Supporting code for [the blog post](https://stephenpanaro.com/blog/llm-quantization-for-iphone).

## Reproduction Steps

1. Use the `gpt2` branch of [this fork](https://github.com/smpanaro/SqueezeLLM-gradients/tree/gpt2) to generate SqueezeLLM Fisher information.
    - `python run.py --output_dir gpt2-grads --model_name gpt2 --dataset c4`
    - Save it in this repo's root as `gpt2-grads.safetensors`.
    - Alternatively, download pre-generated ones from [here](https://github.com/smpanaro/apple-silicon-4bit-quant/releases/tag/march6-2024).
1. `python -m venv env && . env/bin/activate && pip install -r requirements.txt`
1. Run `python run.py` to quantize and evaluate a model.

## Results

```
❯ python run.py --model_name gpt2 --weighting=True --scaling=True --centering=True
Running with model gpt2 on mps
Quantizing to 4 bits with weighting=True, scaling=True, centering=True, quantize=True
Centering linear layer input activations: : 72it [00:00, 100.14it/s]
Quantizing model in 3 chunks.
Preparing chunk 1/3 for quantization: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:17<00:00,  1.76it/s]
Quantizing to 4 bits: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:20<00:00,  1.47it/s]
Applying quantized results: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:06<00:00,  4.50it/s]
Preparing chunk 2/3 for quantization: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:15<00:00,  1.93it/s]
Quantizing to 4 bits: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:20<00:00,  1.46it/s]
Applying quantized results: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:06<00:00,  4.47it/s]
Preparing chunk 3/3 for quantization: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:06<00:00,  1.93it/s]
Quantizing to 4 bits: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:06<00:00,  1.79it/s]
Applying quantized results: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:06<00:00,  1.77it/s]
Validated that all 72 linear layers have <= 16 unique values.
Token indices sequence length is longer than the specified maximum sequence length for this model (287644 > 1024). Running this sequence through the model will result in indexing errors
Calculating perplexity: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 560/562 [00:51<00:00, 10.85it/s]
Wikitext perplexity: tensor(28.1254, device='mps:0')
```
