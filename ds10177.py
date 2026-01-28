import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

# --- Global Configuration & Hyperparameters ---
MODEL_ID = "/home/leimingyu/Qwen2.5-3B-Instruct" 
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-v1"
CALIBRATION_SAMPLES = 64
EVALUATION_SAMPLES = 128
SEQ_LENGTH = 1024
# [cite_start]Percentiles for robust outlier handling during domain clipping [cite: 122]
CLIPPING_PERCENTILES = torch.tensor([0.001, 0.999])

def final_daq_quantization(func, bits, calibration_data, n_samples=20000):
    """
    Constructs an optimized quantization grid using Derivative-Aware Quantization (DA-Q).
    [cite_start]Formulates quantization as a sensitivity-weighted reconstruction problem[cite: 8, 44].
    """
    if calibration_data is None or calibration_data.numel() == 0: return torch.tensor([])
    
    # [cite_start]1. Domain Restriction: Clipping to mitigate heavy-tail interference [cite: 122, 124]
    x_range = (calibration_data.min().item(), calibration_data.max().item())
    if x_range[0] == x_range[1]: return torch.full((2**bits,), x_range[0])
    
    # [cite_start]2. Sensitivity Modeling: Computing 1st (Gradient) and 2nd (Curvature) order derivatives [cite: 95, 101]
    x_samples = torch.linspace(x_range[0], x_range[1], n_samples, requires_grad=True)
    y = func(x_samples)
    grad1, = torch.autograd.grad(y, x_samples, torch.ones_like(y), create_graph=True)
    grad2, = torch.autograd.grad(grad1, x_samples, torch.ones_like(grad1))
    
    # [cite_start]3. Hybrid Weighting: Combining linear sensitivity and curvature distortion [cite: 101, 104]
    weights_combined = 0.7 * grad1.abs() + 0.3 * grad2.abs()
    weights_combined = weights_combined.detach().numpy() + 1e-8
    probabilities = weights_combined / weights_combined.sum()
    
    # [cite_start]4. Importance Resampling: Transforming continuous sensitivity into discrete density [cite: 130]
    np.random.seed(42)
    resampled_indices = np.random.choice(len(x_samples), size=n_samples, p=probabilities)
    resampled_x = x_samples.detach().numpy()[resampled_indices]
    
    # [cite_start]5. Centroid Optimization: K-Means clustering to minimize weighted reconstruction error [cite: 132]
    k = 2 ** bits
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(resampled_x.reshape(-1, 1))
    return torch.from_numpy(np.sort(kmeans.cluster_centers_.flatten())).float()

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True

    @triton.jit
    def _daq_kernel_linear_scan(x_ptr, output_ptr, levels_ptr, lut_ptr, n_elements, num_levels, min_val, max_val, BLOCK_SIZE: tl.constexpr):
        """
        Triton kernel for memory-efficient fused activation quantization.
        [cite_start]Minimizes global memory materialization through on-chip lookup[cite: 11, 148].
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # [cite_start]Load and clamp input to the restricted domain [cite: 122]
        x = tl.load(x_ptr + offsets, mask=mask)
        x_clamped = tl.minimum(tl.maximum(x, min_val), max_val)
        
        # [cite_start]Parallel nearest-neighbor search across quantization centroids [cite: 241]
        best_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        min_dist = tl.full((BLOCK_SIZE,), 1e10, dtype=tl.float32)
        for i in range(num_levels):
            level_val = tl.load(levels_ptr + i)
            dist = tl.abs(x_clamped - level_val)
            is_closer = dist < min_dist
            best_idx = tl.where(is_closer, i, best_idx)
            min_dist = tl.where(is_closer, dist, min_dist)
            
        # [cite_start]Materialize post-activation results directly from Shared Memory LUT [cite: 155]
        result = tl.load(lut_ptr + best_idx, mask=mask)
        tl.store(output_ptr + offsets, result, mask=mask)

except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not found. Falling back to native PyTorch implementation.")

class QuantizedActivation(nn.Module):
    """
    [cite_start]Wrapper module for non-linear operators (e.g., SiLU) with fused DA-Q logic[cite: 148].
    """
    def __init__(self, original_act_fn, quant_levels):
        super().__init__()
        self.register_buffer('quant_levels', quant_levels)
        with torch.no_grad():
            # [cite_start]Pre-compute the Activation Lookup Table (LUT) for O(1) inference [cite: 137]
            self.register_buffer('lut', original_act_fn(quant_levels))

    def forward(self, x):
        if TRITON_AVAILABLE:
            output = torch.empty_like(x)
            n_elements = x.numel()

            def next_power_of_2(n):
                n -= 1
                n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16
                return n + 1

            BLOCK_SIZE = min(next_power_of_2(n_elements if n_elements > 0 else 1), 1024)
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

            # [cite_start]Launch fused Triton kernel to bypass standard memory wall [cite: 144, 238]
            _daq_kernel_linear_scan[grid](
                x, output, self.quant_levels, self.lut,
                n_elements, self.quant_levels.numel(),
                float(self.quant_levels[0].item()), float(self.quant_levels[-1].item()),
                BLOCK_SIZE=BLOCK_SIZE
            )
            return output
        else:
            # Vectorized PyTorch fallback (Standard memory-bound execution)
            levels = self.quant_levels.to(x.device)
            lut = self.lut.to(x.device)
            x_clamped = torch.clamp(x, levels[0], levels[-1])
            dist = torch.abs(x_clamped.unsqueeze(-1) - levels.unsqueeze(0))
            indices = torch.argmin(dist, dim=-1)
            return lut[indices]

def calibrate_and_get_activations(model, tokenizer, num_samples=32):
    """
    [cite_start]Collects pre-activation statistics across the calibration set[cite: 123].
    """
    print(f"--- Calibration: Harvesting activation statistics on {num_samples} samples ---")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=False)
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(input[0].detach().cpu())
        
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.SiLU):
            hooks.append(module.register_forward_hook(hook_fn))
            
    for i in tqdm(range(num_samples), desc="Calibrating"):
        sample = dataset[i]
        text = sample['text'].strip()
        if not text: continue
        tokens = tokenizer(text, return_tensors="pt", max_length=SEQ_LENGTH, truncation=True).input_ids.to(model.device)
        with torch.no_grad(): model(tokens)
        
    for hook in hooks: hook.remove()
    return torch.cat([act.flatten() for act in activations])

def apply_quantization(model, levels):
    """
    [cite_start]Performs surgery on the model to replace standard activations with DA-Q variants[cite: 160].
    """
    model_dtype = next(model.parameters()).dtype
    target_device = next(model.parameters()).device
    levels = levels.to(device=target_device, dtype=model_dtype)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.SiLU):
            parent_name = name.rsplit('.', 1)[0]
            parent_module = model.get_submodule(parent_name)
            child_name = name.split('.')[-1]
            setattr(parent_module, child_name, QuantizedActivation(module, levels))
    return model

@torch.no_grad()
def evaluate_ppl(model, tokenizer, eval_dataset):
    """
    [cite_start]Evaluates Model Perplexity (PPL) as the primary stability metric[cite: 167].
    """
    model.eval()
    total_nll, total_tokens = 0, 0
    for sample in tqdm(eval_dataset, desc="Evaluating PPL"):
        text = sample['text'].strip()
        if not text: continue
        tokens = tokenizer(text, return_tensors="pt", max_length=SEQ_LENGTH, truncation=True).input_ids.to(model.device)
        outputs = model(tokens, labels=tokens)
        neg_log_likelihood = outputs.loss * tokens.shape[1]
        total_nll += neg_log_likelihood.item()
        total_tokens += tokens.shape[1]
    if total_tokens == 0: return float('inf')
    return np.exp(total_nll / total_tokens)

def comprehensive_daq_evaluation():
    """
    [cite_start]Main execution pipeline for evaluating the Hybrid Quantization Grid[cite: 207].
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    device_map = "cuda:0"

    # Stage 1: Calibration & Domain Clipping
    model_for_calib = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True)
    activation_data = calibrate_and_get_activations(model_for_calib, tokenizer, num_samples=CALIBRATION_SAMPLES)
    del model_for_calib
    torch.cuda.empty_cache()
    
    # Efficient quantile calculation via stochastic sampling to prevent OOM
    max_samples_for_quantile = 8_000_000
    if activation_data.numel() > max_samples_for_quantile:
        act_data_for_quantile = activation_data[torch.randperm(activation_data.numel())[:max_samples_for_quantile]].float()
    else:
        act_data_for_quantile = activation_data.float()
    
    clip_boundaries = torch.quantile(act_data_for_quantile, CLIPPING_PERCENTILES)
    clipped_act_data = torch.clamp(activation_data.float(), clip_boundaries[0], clip_boundaries[1])

    # Stage 2: Evaluation Set Pre-loading
    full_eval_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test", streaming=False)
    eval_dataset = [full_eval_dataset[i] for i in range(EVALUATION_SAMPLES)]

    # [cite_start]Stage 3: Multi-bit & Multi-alpha Sensitivity Analysis [cite: 165]
    bits_list = [2, 3, 4]
    [cite_start]alphas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0] # 0.0=Uniform, 1.0=Pure DA-Q [cite: 209, 213]
    results = []

    # BF16 Baseline
    print("\n--- Evaluating BF16 (FP16) Oracle Baseline ---")
    model_bf16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True)
    start_time = time.time()
    ppl_bf16 = evaluate_ppl(model_bf16, tokenizer, eval_dataset)
    results.append({'bits': 16, 'alpha': 'N/A', 'ppl': ppl_bf16, 'time (s)': time.time() - start_time})
    del model_bf16
    torch.cuda.empty_cache()

    # Empirical Grid Sweep
    for bits in bits_list:
        uniform_levels_base = torch.linspace(clipped_act_data.min(), clipped_act_data.max(), 2**bits)
        daq_levels_base = final_daq_quantization(F.silu, bits, clipped_act_data)

        for alpha in alphas:
            print(f"\n--- Testing Configuration: {bits}-bit, alpha={alpha} ---")
            hybrid_levels = (1 - alpha) * uniform_levels_base + alpha * daq_levels_base

            model_quant = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True)
            model_quant = apply_quantization(model_quant, hybrid_levels)
            
            # [cite_start]JIT Warm-up for accurate latency measurement [cite: 157]
            warmup_sample = eval_dataset[0]['text'].strip()
            if warmup_sample:
                tokens = tokenizer(warmup_sample, return_tensors="pt", max_length=SEQ_LENGTH, truncation=True).input_ids.to(device_map)
                with torch.no_grad(): _ = model_quant(tokens)
            torch.cuda.synchronize()

            start_time = time.time()
            ppl = evaluate_ppl(model_quant, tokenizer, eval_dataset)
            torch.cuda.synchronize()
            duration = time.time() - start_time
            
            results.append({'bits': bits, 'alpha': alpha, 'ppl': ppl, 'time (s)': duration})
            del model_quant
            torch.cuda.empty_cache()

    # Final Summary Report
    print("\n--- 🏆 DA-Q Evaluation Summary 🏆 ---")
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    comprehensive_daq_evaluation()



