# Llama 3.2 in Jax 
**Why Llama 3.2?**
- According to Islam *et al.* (2025) Llama 3.2 achieves the most balanced trade-off between performance and energy efficiency. 
- Llama 3.2 is an open-weight model with a known architecture therefore allowing for weight-transfer. 

**Modifications**
- Implemented Flash Attention 
- Implemented 4bit matmul for quantisation to reduce memory overhead.
- KV-Cache for some speed up in terms of latency. 

**Weights**
- Weights come from the official [Llama 3.2 repo](https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main)
- Along with the [tokeniser model](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/original/tokenizer.model)

**Script**
- Install dependencies: `pip install -r requirements.txt` 
- Measuring latency and carbon intensity for all sections of the model.
- Run `python3 evaluate.py` after installing dependencies and the above weights. 
- Script will measure the accuracy and latency, and overall wattage draw. 
- `--component-carbon` flag will measure the carbon of each component. See *carbon counting* below.
- Profiling flags use the Jax profiler with python function calls filtered out.
- `generate.py` is an example generation script. 

**Dataset**
- A custom warehouse dataset created in a hybrid fashion.
- Dataset implemetation [here](https://github.com/MeRichard123/Dynamic_FEFO_Cold-Zone_Dispatch)
- Dataset is public on [HuggingFace](https://huggingface.co/datasets/RichardMe123/Dynamic_FEFO_Cold-Zone_Dispatch)

**Benchmark Results**

All runs used the first 300 validation examples on CUDA with KV caching. FEFO used synchronized component carbon profiling, which adds latency overhead. SST-2 ran with component profiling disabled. Accuracy measures exact matches of pallet IDs and priorities for FEFO and sentiment labels for SST-2. INT4 quantised results use a group size of 64.

| Dataset | Precision | Examples | Accuracy | Invalid responses | Mean latency (s) | P50 latency (s) | P95 latency (s) | Whole-script emissions (g CO₂e) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FEFO | INT4 | 300 | 28.33% | 7 | 69.22 | 76.02 | 77.93 | 17.973 |
| FEFO | BF16 | 300 | 14.67% | 182 | 76.23 | 68.56 | 78.52 | 18.908 |
| SST-2 | INT4| 300 | 80.67% | 0 | 0.116 | 0.097 | 0.172 | 0.287 |
| SST-2 | BF16 | 300 | 77.33% | 0 | 0.115 | 0.093 | 0.166 | 0.282 |

Use `--sst2` to evaluate sentiment on the SST-2 validation split. These commands evaluate the full split and save precision-specific reports in `evaluation_results/sst2/`.

```bash
python3 evaluate.py --sst2 --all --quantisation 4 --suffix int4 --output evaluation_results/sst2
python3 evaluate.py --sst2 --all --quantisation bf16 --suffix bf16 --output evaluation_results/sst2
```

**Carbon Counting**

![](./evaluation_results/figures/component_carbon_pie_bf16.png)
![](./evaluation_results/figures/component_carbon_pie_int4.png)

**Future**
- `lora.py` to train the model
- The rest of the quantised matmul kernels. 

