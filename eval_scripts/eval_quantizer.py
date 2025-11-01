#!/usr/bin/env python3
"""
eval_quantizer.py
Train / evaluate two autoencoders (ResidualVQ vs ResidualLFQ) and report:
 - Vertex MSE (reconstruction)
 - Chamfer distance (reconstruction)
 - Codebook utilization (unique codewords used / entropy)

Usage (example):
 python eval_quantizer.py --dataset ./Data/mesh_dataset.npz --output_dir ./eval_results/quantizer --device cuda \
    --epochs 50 --batch_size 16 --val_samples 200 --checkpoint_folder ./checkpoints/ae
"""
import os
import json
import time
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm

# Try to import project classes - adjust these imports to match your codebase
from data_scripts.Mesh_dataset import MeshDataset
from model_scripts.autoencoder import MeshAutoencoder
from training_scripts.autoencoder_trainer import MeshAutoencoderTrainer
import data_scripts.mesh_render as mesh_render

# Chamfer utilities: prefer pytorch3d if available; else use scipy KDTree
USE_PYTORCH3D = False
try:
    import pytorch3d.ops as p3d_ops
    USE_PYTORCH3D = True
except Exception:
    from scipy.spatial import cKDTree

def chamfer_np(p1, p2):
    """Simple CPU Chamfer implementation using KDTree between two Nx3 numpy arrays.
       Returns symmetric Chamfer (mean nearest squared distance).
    """
    k1 = cKDTree(p1)
    k2 = cKDTree(p2)
    d1, _ = k1.query(p2)  # for each point in p2, nearest in p1
    d2, _ = k2.query(p1)
    return float(np.mean(d1**2) + np.mean(d2**2))

def eval_autoencoder(model, dataset, device, val_samples=200, out_dir=None):
    model.eval()
    total_mse = 0.0
    chamfer_accum = 0.0
    codeword_list = []
    samples = list(dataset.data)
    np.random.shuffle(samples)
    samples = samples[:val_samples]
    for item in tqdm(samples, desc="AE eval"):
        verts = item['vertices']  # torch tensor on cuda in your data; may be cpu
        faces = item['faces']
        # Ensure tensors on device
        if isinstance(verts, torch.Tensor):
            verts_cpu = verts.detach().cpu().numpy()
        else:
            verts_cpu = np.asarray(verts)

        # Tokenize and decode via autoencoder
        with torch.no_grad():
            # using your autoencoder.tokenize method
            codes = model.tokenize(vertices=item['vertices'].to(device),
                                   faces=item['faces'].to(device),
                                   face_edges=item.get('face_edges', None))
            codes = codes.flatten().unsqueeze(0)
            # crop to quantizer multiple if necessary
            codes = codes[:, : (codes.shape[-1] // model.num_quantizers) * model.num_quantizers]
            coords, mask = model.decode_from_codes_to_faces(codes)
            recon = coords.view(-1, 3).cpu().numpy()

            # ground-truth organization
            orgs = item['vertices'][item['faces']].unsqueeze(0).view(-1, 3).cpu().numpy()

            mse = np.mean((orgs - recon)**2)
            total_mse += mse

            # chamfer between recon and original meshes (per-face vertex clouds)
            c = chamfer_np(orgs, recon)
            chamfer_accum += c

            # collect codewords (flatten)
            # assume codes is torch tensor
            flat_codes = codes.flatten().cpu().numpy().tolist()
            codeword_list.extend(flat_codes)

    n = len(samples)
    mse_avg = total_mse / n
    chamfer_avg = chamfer_accum / n

    # codebook utilization metrics
    cw_counts = Counter(codeword_list)
    unique_used = len(cw_counts)
    codebook_size = getattr(model, 'codebook_size', None) or getattr(model, 'vq_codebook_size', None)
    util_rate = unique_used / float(codebook_size) if codebook_size else None

    # entropy
    freqs = np.array(list(cw_counts.values()), dtype=float)
    probs = freqs / freqs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    results = {
        "mse_avg": float(mse_avg),
        "chamfer_avg": float(chamfer_avg),
        "codebook_size": int(codebook_size) if codebook_size else None,
        "unique_used": int(unique_used),
        "utilization_rate": float(util_rate) if util_rate is not None else None,
        "code_entropy": float(entropy)
    }

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # Save a few reconstructions for manual inspection
        save_path = Path(out_dir) / "ae_eval_sample.obj"
        try:
            # try to write combined mesh of a few reconstructions
            mesh_render.save_rendering(str(save_path), [coords.detach().cpu().numpy()])
        except Exception:
            pass

    return results

def train_and_eval(use_residual_lfq, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # construct autoencoder
    model = MeshAutoencoder(
        decoder_dims_through_depth=(128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
        codebook_size=args.codebook_size,
        dim_codebook=192,
        dim_area_embed=16,
        dim_coor_embed=16,
        dim_normal_embed=16,
        dim_angle_embed=8,
        attn_decoder_depth=4,
        attn_encoder_depth=2,
        use_residual_lfq=use_residual_lfq
    ).to(device)

    dataset = MeshDataset.load(args.dataset) if hasattr(MeshDataset, 'load') else MeshDataset(args.dataset)
    # Trainer
    trainer = MeshAutoencoderTrainer(
        model=model,
        warmup_steps=10,
        dataset=dataset,
        num_train_steps=args.train_steps,
        batch_size=args.batch_size,
        grad_accum_every=args.grad_accum_every,
        learning_rate=args.lr,
        checkpoint_every_epoch=1,
        checkpoint_folder=args.checkpoint_folder
    )

    print(f"Training AE (LFQ={use_residual_lfq}) ...")
    trainer.train(args.epochs, stop_at_loss=args.stop_loss)

    # Save model
    out_name = f"ae_lfq_{int(use_residual_lfq)}.pt"
    save_path = Path(args.output_dir) / out_name
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict()}, str(save_path))

    # load for eval (to ensure correct state)
    trainer.load(str(save_path))
    results = eval_autoencoder(model, dataset, device, val_samples=args.val_samples, out_dir=args.output_dir)
    return results, str(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="path to MeshDataset npz")
    parser.add_argument("--output_dir", type=str, default="./eval_results/quantizer")
    parser.add_argument("--checkpoint_folder", type=str, default="./checkpoints/ae")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_every", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--codebook_size", type=int, default=16384)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--stop_loss", type=float, default=0.2)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    summary = {}
    # Run ResidualVQ (LFQ=False) and ResidualLFQ (LFQ=True)
    for use_lfq in (False, True):
        key = "ResidualLFQ" if use_lfq else "ResidualVQ"
        print("==== Running:", key)
        results, model_path = train_and_eval(use_lfq, args)
        results['model_path'] = model_path
        summary[key] = results
        # save intermediate summary
        with open(Path(args.output_dir) / f"summary_{key}.json", "w") as f:
            json.dump(results, f, indent=2)

    # write full summary
    with open(Path(args.output_dir) / "quantizer_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Done. Results saved to", args.output_dir)

if __name__ == "__main__":
    main()
