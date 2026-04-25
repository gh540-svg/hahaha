"""
GradKV Core Utilities
- Gradient collection from KV-Cache
- SVD subspace extraction
- Principal angle computation
- Capability projection
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def collect_kv_gradients(
    model,
    tokenizer,
    texts: List[str],
    target_layers: List[int],
    max_length: int = 512,
    device: str = "cuda",
    label_char_spans: Optional[List[List[Tuple[int, int]]]] = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Collect gradients of task loss w.r.t. K and V projection outputs.

    Strategy: Hook into k_proj and v_proj Linear layers to capture their
    output (the K and V vectors), retain_grad, then backprop to get gradients.

    Args:
        label_char_spans: optional, one list of (start_char, end_char) per text.
            When given, the CE loss is computed ONLY on tokens whose character
            offsets fall inside at least one span; all other positions get -100.
            This lets the caller target "assertion-only" or "solution-only" loss
            for execution-aware subspace extraction.

    Returns: {layer_idx: {"K_grads": [N, d], "V_grads": [N, d]}}
    """
    use_spans = label_char_spans is not None
    model.eval()
    kv_grads = {l: {"K_grads": [], "V_grads": []} for l in target_layers}

    # Enable grad on k_proj/v_proj weights for target layers so their outputs get grad
    target_params = []
    for layer_idx in target_layers:
        attn = model.model.layers[layer_idx].self_attn
        for p in attn.k_proj.parameters():
            p.requires_grad_(True)
            target_params.append(p)
        for p in attn.v_proj.parameters():
            p.requires_grad_(True)
            target_params.append(p)

    for ti, text in enumerate(texts):
        tok_kwargs = dict(return_tensors="pt", truncation=True,
                          max_length=max_length, padding=False)
        if use_spans:
            tok_kwargs["return_offsets_mapping"] = True
        inputs = tokenizer(text, **tok_kwargs).to(device)

        if use_spans:
            offsets = inputs.pop("offset_mapping")[0].tolist()
            input_ids = inputs["input_ids"]
            labels = torch.full_like(input_ids, -100)
            for (cs, ce) in label_char_spans[ti]:
                for k, (ts, te) in enumerate(offsets):
                    if te > ts and ts >= cs and te <= ce:
                        labels[0, k] = input_ids[0, k]
            if (labels != -100).sum().item() == 0:
                continue
        else:
            labels = inputs["input_ids"]

        model.zero_grad()

        # Storage for this sample's K/V activations
        k_activations = {}
        v_activations = {}
        hook_handles = []

        for layer_idx in target_layers:
            attn = model.model.layers[layer_idx].self_attn

            # Hook k_proj output — output will have requires_grad=True because weights do
            def make_k_hook(lid):
                def hook(module, input, output):
                    output.retain_grad()
                    k_activations[lid] = output
                return hook
            h_k = attn.k_proj.register_forward_hook(make_k_hook(layer_idx))
            hook_handles.append(h_k)

            # Hook v_proj output
            def make_v_hook(lid):
                def hook(module, input, output):
                    output.retain_grad()
                    v_activations[lid] = output
                return hook
            h_v = attn.v_proj.register_forward_hook(make_v_hook(layer_idx))
            hook_handles.append(h_v)

        # Forward pass with gradient enabled
        with torch.enable_grad():
            outputs = model(**inputs, labels=labels, use_cache=False)
            loss = outputs.loss
            loss.backward()

        # Collect gradients
        for layer_idx in target_layers:
            if layer_idx in k_activations and k_activations[layer_idx].grad is not None:
                # k_proj output: [1, seq_len, kv_dim]
                g_k = k_activations[layer_idx].grad[0].detach().cpu().float()  # [seq, kv_dim]
                kv_grads[layer_idx]["K_grads"].append(g_k)
            if layer_idx in v_activations and v_activations[layer_idx].grad is not None:
                g_v = v_activations[layer_idx].grad[0].detach().cpu().float()
                kv_grads[layer_idx]["V_grads"].append(g_v)

        # Cleanup
        for h in hook_handles:
            h.remove()
        del k_activations, v_activations

    # Re-freeze target params
    for p in target_params:
        p.requires_grad_(False)
    model.eval()

    # Stack gradients: [total_tokens, d]
    for layer_idx in target_layers:
        if kv_grads[layer_idx]["K_grads"]:
            kv_grads[layer_idx]["K_grads"] = torch.cat(
                kv_grads[layer_idx]["K_grads"], dim=0
            )
            kv_grads[layer_idx]["V_grads"] = torch.cat(
                kv_grads[layer_idx]["V_grads"], dim=0
            )
        else:
            kv_grads[layer_idx]["K_grads"] = torch.zeros(1, 1)
            kv_grads[layer_idx]["V_grads"] = torch.zeros(1, 1)

    return kv_grads


def extract_subspace(
    gradient_matrix: torch.Tensor,
    rank: int = 32,
    energy_threshold: Optional[float] = None,
    half_rank: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    SVD on gradient matrix to extract capability subspace.

    Args:
        gradient_matrix: [N, d] gradient matrix
        rank: number of top singular directions to keep
        energy_threshold: if set, auto-select rank to capture this fraction of energy
        half_rank: if True, use half of the full KV dimension as rank
                   (e.g. 64 for 0.5B with kv_dim=128, 256 for 7B with kv_dim=512)

    Returns:
        V_r: [d, r] top-r right singular vectors (capability subspace basis)
        singular_values: all singular values
        effective_rank: actual rank used
    """
    # Center the gradients
    G = gradient_matrix.float()
    G = G - G.mean(dim=0, keepdim=True)

    # SVD
    U, S, Vt = torch.linalg.svd(G, full_matrices=False)

    if half_rank:
        rank = G.shape[1] // 2
    elif energy_threshold is not None:
        cumulative_energy = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
        rank = int((cumulative_energy < energy_threshold).sum().item()) + 1
        rank = min(rank, S.shape[0])

    effective_rank = min(rank, S.shape[0], Vt.shape[0])
    V_r = Vt[:effective_rank].T  # [d, r]

    return V_r, S, effective_rank


def compute_projection_matrix(V_r: torch.Tensor) -> torch.Tensor:
    """
    Compute projection matrix P = V_r @ V_r^T

    Args:
        V_r: [d, r] subspace basis
    Returns:
        P: [d, d] projection matrix
    """
    return V_r @ V_r.T


def project_kv(
    K: torch.Tensor,
    V: torch.Tensor,
    P: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project KV-Cache onto capability subspace.
    K_tilde = K @ P, V_tilde = V @ P
    """
    return K @ P, V @ P


def compute_principal_angles(
    V1: torch.Tensor,
    V2: torch.Tensor
) -> torch.Tensor:
    """
    Compute principal angles between two subspaces.

    Args:
        V1: [d, r1] basis of subspace 1
        V2: [d, r2] basis of subspace 2

    Returns:
        angles: principal angles in radians, sorted ascending
    """
    # QR decomposition to ensure orthonormal bases
    Q1, _ = torch.linalg.qr(V1.float())
    Q2, _ = torch.linalg.qr(V2.float())

    # Compute SVD of Q1^T @ Q2
    M = Q1.T @ Q2
    _, S, _ = torch.linalg.svd(M)

    # Clamp to valid range for arccos
    S = torch.clamp(S, -1.0, 1.0)

    # Principal angles
    angles = torch.arccos(S)

    return angles


def subspace_similarity(V1: torch.Tensor, V2: torch.Tensor) -> float:
    """
    Compute overall similarity between two subspaces as mean cosine of principal angles.
    Returns value in [0, 1]: 0 = orthogonal, 1 = identical.
    """
    angles = compute_principal_angles(V1, V2)
    return torch.cos(angles).mean().item()


def grassmann_distance(V1: torch.Tensor, V2: torch.Tensor) -> float:
    """
    Grassmann distance between two subspaces = ||angles||_2.
    """
    angles = compute_principal_angles(V1, V2)
    return torch.norm(angles).item()


def singular_value_analysis(S: torch.Tensor) -> Dict[str, float]:
    """
    Analyze singular value distribution for a capability's gradient matrix.
    """
    S = S.float()
    total_energy = (S ** 2).sum().item()
    cumulative = torch.cumsum(S ** 2, dim=0) / total_energy

    return {
        "total_energy": total_energy,
        "top1_ratio": (S[0] ** 2).item() / total_energy if total_energy > 0 else 0,
        "top8_ratio": cumulative[min(7, len(S)-1)].item(),
        "top16_ratio": cumulative[min(15, len(S)-1)].item(),
        "top32_ratio": cumulative[min(31, len(S)-1)].item(),
        "top64_ratio": cumulative[min(63, len(S)-1)].item(),
        "rank_for_90pct": int((cumulative < 0.9).sum().item()) + 1,
        "rank_for_95pct": int((cumulative < 0.95).sum().item()) + 1,
        "rank_for_99pct": int((cumulative < 0.99).sum().item()) + 1,
        "spectral_decay_rate": (S[0] / S[min(len(S)-1, 31)]).item() if len(S) > 1 else float('inf'),
    }
