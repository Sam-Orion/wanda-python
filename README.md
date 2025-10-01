# Pruning LLMs by Weights and Activations
Unofficial PyTorch implementation of **Wanda** (Pruning by **W**eights **and a**ctivations), as presented in the paper:

**A Simple and Effective Pruning Approach for Large Language Models** </br>
*Mingjie Sun\*, Zhuang Liu\*, Anna Bair, J. Zico Kolter* (* indicates equal contribution) <br>
Carnegie Mellon University, Meta AI Research and Bosch Center for AI  <br>
[Paper](https://arxiv.org/abs/2306.11695) - [Project page](https://eric-mingjie.github.io/wanda/home.html)

```bibtex
@article{sun2023wanda,
  title={A Simple and Effective Pruning Approach for Large Language Models}, 
  author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
  year={2023},
  journal={arXiv preprint arXiv:2306.11695}
}
```

--- 
<p align="center">
<img src="https://user-images.githubusercontent.com/20168304/273351964-53c3807e-3453-49c5-b855-b620b1026466.png" width=100% height=100% 
class="center">
</p>

Compared to magnitude pruning which removes weights solely based on their magnitudes, our pruning approach **Wanda** removes weights on a *per-output* basis, by the product of weight magnitudes and input activation norms.

## Update
- [x] (9.22.2023) Add [support](https://github.com/locuslab/wanda#pruning-llama-2) for LLaMA-2.
- [x] (9.22.2023) Add [code](https://github.com/locuslab/wanda#ablation-on-obs-weight-update) to reproduce the ablation study on OBS weight update in the paper.
- [x] (10.6.2023) Add new [support](https://github.com/locuslab/wanda#ablation-on-obs-weight-update) for the weight update analysis in the ablation study. Feel free to try it out!
- [x] (10.6.2023) Add [support](https://github.com/locuslab/wanda#zero-shot-evaluation) for zero-shot evaluation.
- [x] (10.20.2023) Add code for pruning OPT models.
- [x] (10.23.2023) Add code for [LoRA fine-tuning](lora_ft).

## Setup

### Quick Installation (Recommended)
Install the package in development mode:
```bash
pip install -e .
```

This will install the package and create command-line tools `wanda` and `wanda-opt` that you can use directly.

### Manual Installation
For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Usage

### Using the Command Line Tools
After installation, you can use the provided command-line tools:

```bash
# For LLaMA models
wanda \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/wanda/

# For OPT models  
wanda-opt \
    --model facebook/opt-6.7b \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/opt_6.7b/unstructured/wanda/
```

### Using Python Scripts
The [scripts](scripts) directory contains all the bash commands to replicate the main results (Table 2) in the paper.

Below is an example command for pruning LLaMA-7B with Wanda, to achieve unstructured 50% sparsity.
```sh
python -m wanda.main \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/wanda/ 
```
Below I have provided a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: We have implemented three pruning methods, namely [`magnitude`, `wanda`, `sparsegpt`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--use_variant`: Whether to use the Wanda variant, default is `False`. 
- `--save`: Specifies the directory where the result will be stored.

For structured N:M sparsity, set the argument `--sparsity_type` to "2:4" or "4:8". An illustrative command is provided below:
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_7b/2-4/wanda/ 
```

### Pruning LLaMA-2
For [LLaMA-2](https://ai.meta.com/llama/) models, replace `--model` with `meta-llama/Llama-2-7b-hf` (take `7b` as an example):
```sh 
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama2_7b/unstructured/wanda/
```
LLaMA-2 results: (LLaMA-2-34b is not released as of 9.22.2023)
|sparsity| ppl              | llama2-7b | llama2-13b | llama2-70b |
|------|------------------|----------|------------|------------|
|-| dense            | 5.12     | 4.57       | 3.12     |
|unstructured 50%| magnitude        | 14.89    | 6.37       | 4.98     |
|unstructured 50%| sparsegpt        | 6.51     | 5.63       | **3.98**  |
|unstructured 50%| wanda            | **6.42** | **5.56**   | **3.98**  |
|4:8| magnitude        | 16.48    | 6.76       | 5.58     |
|4:8| sparsegpt        | 8.12     | 6.60      | 4.59     |
|4:8| wanda            | **7.97** | **6.55**  | **4.47**     |
|2:4| magnitude        | 54.59    | 8.33       | 6.33       |
|2:4| sparsegpt        | **10.17** | 8.32       | 5.40      |
|2:4| wanda            | 11.02    | **8.27**   | **5.16**     |

### Ablation on OBS weight update
To reproduce the analysis on weight update, I have provided my implementation for this ablation. All commands can be found in [this script](scripts/ablate_weight_update.sh).
```sh
for method in ablate_mag_seq ablate_wanda_seq ablate_mag_iter ablate_wanda_iter 
do 
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model decapoda-research/llama-7b-hf \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --prune_method ${method} \
  --save out/llama_7b_ablation/unstructured/
done 
```
Here `ablate_{mag/wanda}_{seq/iter}` means that we use magnitude pruning or wanda to obtain the pruned mask at each layer, then apply weight update procedure with either a sequential style or an iterative style every 128 input channels. For details, please see Section 5 of the [paper](https://arxiv.org/abs/2306.11695).

### Speedup Evaluation
The pruning speed for each method is evaluated by the cumulated time spent on pruning (for each layer), without the forward passes.

For inference speedup with structured sparsity, you can refer to this [blog post](https://pytorch.org/tutorials/prototype/semi_structured_sparse.html), where  structured sparsity is supported by `PyTorch >= 2.1`. You can switch between the CUTLASS or CuSPARSELt kernel [here](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/sparse/semi_structured.py#L55).

Last, for pruning image classifiers, see directory [image_classifiers](image_classifiers) for details.

## Acknowledgement
This repository is build upon the [SparseGPT](https://github.com/IST-DASLab/sparsegpt) repository.

## License
This project is released under the GNU license. Please see the [LICENSE](LICENSE) file for more information.

## Questions
Feel free to discuss code with me through issues/emails!

shubhammisra936@gmail.com 
