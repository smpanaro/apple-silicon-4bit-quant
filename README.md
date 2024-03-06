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
❯ python run.py
Quantizing and saving model to: gpt2_4bit
Token indices sequence length is longer than the specified maximum sequence length for this model (2173 > 1024). Running this sequence through the model will result in indexing errors
Collecting input activation statistics: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:32<00:00,  1.08it/s]
Centering linear layer input activations: : 200it [00:08, 23.07it/s]
Preparing for parallel quantization: : 200it [00:49,  4.05it/s]
Quantizing to 4 bits: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 72/72 [00:33<00:00,  2.14it/s]
Validated that all 72 linear layers have <= 16 unique values.
Token indices sequence length is longer than the specified maximum sequence length for this model (287644 > 1024). Running this sequence through the model will result in indexing errors
Calculating perplexity: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 560/562 [09:51<00:02,  1.06s/it]
Wikitext perplexity: tensor(28.1250)
```
