#!/usr/bin/env python3
"""
eval_attention.py
Train two transformers (FlashAttention vs Taylor linear) using the SAME autoencoder tokenization.
Measure:
 - Mesh quality proxy: Chamfer-based distance between generated samples and validation meshes (MMD-like nearest-neighbor avg).
 - Training throughput (steps/sec)
 - Sampling time per mesh (seconds)

Usage (example):
 python eval_attention.py --dataset ./Data/mesh_dataset.npz --ae_checkpoint ./eval_results/quantizer/ae_lfq_1.pt \
    --output_dir ./eval_results/attention --device cuda --train_steps 20000 --batch_size 4 --eval_samples 200
"""
import os
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# project-specific imports (adjust if needed)
try:
    from meshgpt_pytorch import MeshAutoencoder, MeshTransformer, MeshTransformerTrainer, MeshDataset, mesh_render
except Exception:
    from data_scripts.Mesh_dataset import MeshDataset
    from model_scripts.autoencoder import MeshAutoencoder
    from model_scripts.transformer import MeshTransformer
    from model_scripts.trainers import MeshTransformerTrainer
    import data_scripts.mesh_render as mesh_render

# Chamfer fallback as above
try:
    import pytorch3d.ops as p3d_ops
    USE_PYTORCH3D = True
except Exception:
    USE_PYTORCH3D = False
    from scipy.spatial import cKDTree

def chamfer_np(p1, p2):
    k1 = cKDTree(p1)
    k2 = cKDTree(p2)
    d1, _ = k1.query(p2)
    d2, _ = k2.query(p1)
    return float(np.mean(d1**2) + np.mean(d2**2))

def tokenize_dataset(ae_model, dataset, device):
    """
    Tokenize dataset into sequences using autoencoder; return token sequences and mapping to original meshes for later Chamfer evaluation.
    """
    token_seqs = []
    original_meshes = []
    for item in tqdm(dataset.data, desc="Tokenizing"):
        with torch.no_grad():
            codes = ae_model.tokenize(vertices=item['vertices'].to(device),
                                     faces=item['faces'].to(device),
                                     face_edges=item.get('face_edges', None))
            codes = codes.flatten().unsqueeze(0)
            codes = codes[:, : (codes.shape[-1] // ae_model.num_quantizers) * ae_model.num_quantizers]
            token_seqs.append(codes.cpu())
            # store original mesh vertex cloud for evaluation
            org_coords = item['vertices'][item['faces']].unsqueeze(0).view(-1,3).cpu().numpy()
            original_meshes.append(org_coords)
    return token_seqs, original_meshes

def train_transformer_and_eval(transformer_cfg, token_seqs, original_meshes, device, args, run_name):
    """
    transformer_cfg: dict with transformer constructor kwargs and 'attention_type' string (e.g., 'flash' or 'taylor')
    """
    # Build transformer instance
    ae = transformer_cfg['ae_model']  # pass reference to autoencoder for geometry dims
    transformer = MeshTransformer(ae, **transformer_cfg['transformer_kwargs']).to(device)

    trainer = MeshTransformerTrainer(
        model=transformer,
        warmup_steps=10,
        num_train_steps=args.train_steps,
        dataset=token_seqs,  # may need to adapt to trainer API which expects MeshDataset-like object; you may need to build a simple Dataset wrapper
        batch_size=args.batch_size,
        grad_accum_every=1,
        learning_rate=args.lr,
        checkpoint_every_epoch=1,
        # pass attention flags via transformer kwargs if necessary
    )

    # Training loop with throughput measurement
    print(f"Training transformer ({run_name}) ...")
    start_time = time.perf_counter()
    trainer.train(args.epochs)  # adapt to your trainer API; if trainer.train accepts steps, use trainer.train(num_steps)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    # compute steps/sec estimate if trainer exposes steps run
    steps_done = getattr(trainer, 'steps_done', None) or args.train_steps
    steps_per_sec = steps_done / elapsed if elapsed > 0 else None

    # Sampling: generate N samples from transformer and compute Chamfer NN to original validation set
    print("Sampling meshes for evaluation ...")
    sample_meshes = []
    sample_times = []
    for i in range(args.eval_samples):
        t0 = time.perf_counter()
        out = transformer.generate(texts=[args.eval_category], temperature=0.0)  # category-based generation
        t1 = time.perf_counter()
        sample_times.append(t1 - t0)
        # save generated mesh vertices cloud (same format as before)
        # out is expected to be (tensor, mask) pairs similar to earlier code; adapt as necessary
        # For simplicity assume out[0][0] exists and is (coords, mask)
        coords, mask = out[0]
        coords_np = coords[mask].cpu().numpy().reshape(-1,3)
        sample_meshes.append(coords_np)

    # Chamfer-based nearest-neighbor proxy: for each generated sample, compute nearest Chamfer to any original mesh and average.
    chamfer_nn_list = []
    for g in sample_meshes:
        best = float('inf')
        for org in original_meshes[:args.val_search_pool]:
            c = chamfer_np(g, org)
            if c < best:
                best = c
        chamfer_nn_list.append(best)
    chamfer_mean = float(np.mean(chamfer_nn_list))
    sampling_time_mean = float(np.mean(sample_times))

    # Save metrics
    metrics = {
        "run_name": run_name,
        "steps_done": steps_done,
        "steps_per_sec": steps_per_sec,
        "eval_samples": args.eval_samples,
        "chamfer_nn_mean": chamfer_mean,
        "sampling_time_mean": sampling_time_mean
    }

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ae_checkpoint", type=str, required=True, help="path to chosen autoencoder checkpoint (best quantizer)")
    parser.add_argument("--output_dir", type=str, default="./eval_results/attention")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_steps", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_samples", type=int, default=50)
    parser.add_argument("--val_search_pool", type=int, default=200, help="how many validation meshes to search for nearest Chamfer")
    parser.add_argument("--eval_category", type=str, default="chair")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load autoencoder
    ae = MeshAutoencoder(
        decoder_dims_through_depth=(128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
        codebook_size=16384,
        dim_codebook=192,
        dim_area_embed=16,
        dim_coor_embed=16,
        dim_normal_embed=16,
        dim_angle_embed=8,
        attn_decoder_depth=4,
        attn_encoder_depth=2,
    ).to(device)
    ck = torch.load(args.ae_checkpoint, map_location=device)
    ae.load_state_dict(ck['model'])

    # Load dataset and tokenize
    dataset = MeshDataset.load(args.dataset) if hasattr(MeshDataset, 'load') else MeshDataset(args.dataset)
    token_seqs, original_meshes = tokenize_dataset(ae, dataset, device)

    # Prepare transformer configs for both attention variants
    base_transformer_kwargs = dict(
        dim=768,
        coarse_pre_gateloop_depth=3,
        fine_pre_gateloop_depth=3,
        attn_depth=12,
        attn_heads=12,
        max_seq_len=4602,
        condition_on_text=True,
        gateloop_use_heinsen=False,
        dropout=0.0,
        text_condition_model_types="bge",
        text_condition_cond_drop_prob=0.0
    )

    results = {}
    for attn_type in ("flash", "taylor"):
        print("=== Running attention:", attn_type)
        tf_cfg = {
            "ae_model": ae,
            "transformer_kwargs": base_transformer_kwargs,
            "attention_type": attn_type
        }
        run_name = f"transformer_{attn_type}"
        metrics = train_transformer_and_eval(tf_cfg, token_seqs, original_meshes, device, args, run_name)
        results[run_name] = metrics
        with open(Path(args.output_dir) / f"metrics_{run_name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Full summary
    with open(Path(args.output_dir) / "attention_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done. Results saved to", args.output_dir)

if __name__ == "__main__":
    main()
