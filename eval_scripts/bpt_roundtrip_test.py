# scripts/bpt_diagnostics.py
# Diagnostic tool to analyze BPT compression / patch statistics over a MeshDataset.
# Usage: python scripts/bpt_diagnostics.py

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import trimesh

# adapt import paths to your repo
from data_scripts.Mesh_dataset import MeshDataset
# functions provided by BPT.py in your repo
from data_scripts.BPT import patchified_mesh, get_block_representation, BPT_serialize, BPT_deserialize

# ---- CONFIG ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
DATASET_PATH = DATA_DIR / "meshData_pkl.npz"   # change if different
NUM_SAMPLES = 1000          # number of meshes to sample from dataset (set None to use all)
COMBOS = [(8, 16), (4, 32), (16, 8), (8, 8), (4, 16), (16, 16)]
USE_SPECIAL_OPTIONS = [True, False]
OUTPUT_JSON = PROJECT_ROOT / "bpt_diagnostics_results.json"

# -----------------

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)

def mesh_from_item(item):
    verts = item.get('vertices', None)
    faces = item.get('faces', None)
    if verts is None or faces is None:
        return None
    verts = to_numpy(verts)
    faces = to_numpy(faces)
    # keep only valid faces
    if faces.size == 0:
        return None
    valid_mask = np.all((faces >= 0) & (faces < len(verts)), axis=1)
    faces_clean = faces[valid_mask]
    if faces_clean.size == 0:
        return None
    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces_clean, process=False)
        return mesh
    except Exception as e:
        print("[mesh_from_item] trimesh build failed:", e)
        return None

def analyze_patch_sequence(seq, special_token=-2):
    """
    seq : np.ndarray of shape (N,3) from patchified_mesh
    returns num_patches, avg_patch_len, patch_len_list
    """
    # detect special rows that mark patch boundary
    if seq.size == 0:
        return 0, 0.0, []
    is_special = (seq == special_token).all(axis=1)
    patch_lens = []
    start = 0
    for i, s in enumerate(is_special):
        if s:
            patch_lens.append(i - start)  # number of coordinate rows in this patch (center + boundary)
            start = i + 1
    # trailing block (if not terminated by special marker)
    if start < len(seq):
        patch_lens.append(len(seq) - start)
    if len(patch_lens) == 0:
        return 0, 0.0, []
    return int(len(patch_lens)), float(np.mean(patch_lens)), patch_lens

def analyze_codes_breakdown(codes, block_size, offset_size):
    """
    Given a 1D integer numpy array `codes` and bpt params,
    return counts of:
      - block tokens (normal block ids)
      - offset tokens
      - special-shifted-block tokens
      - negative special_token occurrences (if use_special_block=False)
    """
    if codes is None or len(codes) == 0:
        return dict(total=0, block_tokens=0, offset_tokens=0, special_shift_block=0, negative_specials=0)

    block_base = block_size ** 3
    offset_range = offset_size ** 3
    special_block_base = block_base + offset_range

    codes_arr = np.asarray(codes).astype(np.int64)

    negative_specials = np.sum(codes_arr < 0)
    block_tokens = np.sum((codes_arr >= 0) & (codes_arr < block_base))
    offset_tokens = np.sum((codes_arr >= block_base) & (codes_arr < block_base + offset_range))
    special_shift_block = np.sum((codes_arr >= special_block_base) & (codes_arr < special_block_base + block_base))

    return dict(
        total=int(len(codes_arr)),
        block_tokens=int(block_tokens),
        offset_tokens=int(offset_tokens),
        special_shift_block=int(special_shift_block),
        negative_specials=int(negative_specials)
    )

def combo_key(b, o, use_special):
    return f"{b}_{o}_{int(use_special)}"

def analyze_dataset(dataset_path, combos, use_special_opts, num_samples=None):
    ds = MeshDataset.load(str(dataset_path))
    n = len(ds.data)
    indices = list(range(n)) if num_samples is None else list(range(min(n, num_samples)))

    results = {
        "per_sample": {},
        "combo_stats": {}
    }

    # prepare combo keys
    combo_keys = []
    for b, o in combos:
        for use_special in use_special_opts:
            key = combo_key(b, o, use_special)
            combo_keys.append((b, o, use_special))
            results["combo_stats"][key] = {
                "codes_lengths": [],
                "naive_token_counts": [],   # triangles * 3
                "compression_ratios": [],
                "block_token_counts": [],
                "offset_token_counts": [],
                "special_shift_counts": [],
                "negative_special_counts": [],
            }

    for idx in tqdm(indices, desc="Analyzing meshes"):
        item = ds.data[idx]
        mesh = mesh_from_item(item)
        if mesh is None:
            continue
        tri_count = len(mesh.faces)
        naive_tokens = tri_count * 3

        # patchify (coordinate-level sequence)
        try:
            seq = patchified_mesh(mesh, special_token=-2)
        except Exception as e:
            print(f"[patchify] idx {idx} failed:", e)
            continue

        num_patches, avg_patch_len, patch_lens = analyze_patch_sequence(seq, special_token=-2)

        per_sample_info = {
            "tri_count": int(tri_count),
            "naive_tokens": int(naive_tokens),
            "num_patches": num_patches,
            "avg_patch_len": float(avg_patch_len),
            "patch_lens_summary": {
                "min": int(np.min(patch_lens)) if len(patch_lens) else 0,
                "max": int(np.max(patch_lens)) if len(patch_lens) else 0,
                "median": float(np.median(patch_lens)) if len(patch_lens) else 0,
            },
            "combo_results": {}
        }

        # test each combo on this patch sequence
        for b, o, use_special in combo_keys:
            key_str = f"{b}_{o}_{int(use_special)}"
            try:
                codes = get_block_representation(
                    seq,
                    block_size=b,
                    offset_size=o,
                    block_compressed=True,
                    special_token=-2,
                    use_special_block=use_special
                )
                codes_arr = np.asarray(codes).astype(np.int64)
            except Exception as e:
                print(f"[get_block_representation] idx {idx} b{b} o{o} use_special{use_special} failed:", e)
                codes_arr = np.array([], dtype=np.int64)

            breakdown = analyze_codes_breakdown(codes_arr, b, o)
            codes_len = breakdown["total"]
            compression = (naive_tokens / codes_len) if codes_len > 0 else 0.0

            # accumulate combo stats
            stats = results["combo_stats"][key]
            stats["codes_lengths"].append(codes_len)
            stats["naive_token_counts"].append(naive_tokens)
            stats["compression_ratios"].append(compression)
            stats["block_token_counts"].append(breakdown["block_tokens"])
            stats["offset_token_counts"].append(breakdown["offset_tokens"])
            stats["special_shift_counts"].append(breakdown["special_shift_block"])
            stats["negative_special_counts"].append(breakdown["negative_specials"])

            per_sample_info["combo_results"][key] = {
                "codes_len": int(codes_len),
                "compression": float(compression),
                "breakdown": breakdown
            }

        results["per_sample"][str(idx)] = per_sample_info

    # aggregate combo stats: compute mean, median, std etc
    for key, stats in results["combo_stats"].items():
        arr_lens = np.array(stats["codes_lengths"], dtype=np.float32)
        comp = np.array(stats["compression_ratios"], dtype=np.float32)
        stats_summary = {
            "n_samples": int(len(arr_lens)),
            "codes_len_mean": float(np.mean(arr_lens)) if len(arr_lens) else 0.0,
            "codes_len_median": float(np.median(arr_lens)) if len(arr_lens) else 0.0,
            "codes_len_std": float(np.std(arr_lens)) if len(arr_lens) else 0.0,
            "compression_mean": float(np.mean(comp)) if len(comp) else 0.0,
            "compression_median": float(np.median(comp)) if len(comp) else 0.0,
            "compression_std": float(np.std(comp)) if len(comp) else 0.0
        }
        results["combo_stats"][key]["summary"] = stats_summary

    return results

def parse_combo_key(key_str):
    b, o, s = key_str.split("_")
    return int(b), int(o), bool(int(s))

def recommend_best_combo(results, top_k=5):
    # rank combos by median compression (higher is better)
    ranking = []
    for key_str, stats in results["combo_stats"].items():
        b, o, s = parse_combo_key(key_str)
        median_comp = stats["summary"]["compression_median"]
        ranking.append(((b, o, s), median_comp, stats["summary"]))

    ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
    return ranking[:top_k]

if __name__ == "__main__":
    print("Loading dataset:", DATASET_PATH)
    res = analyze_dataset(DATASET_PATH, COMBOS, USE_SPECIAL_OPTIONS, num_samples=NUM_SAMPLES)
    # save results
    out = {
        "config": {
            "num_samples": NUM_SAMPLES,
            "combos": COMBOS,
            "use_special_opts": USE_SPECIAL_OPTIONS
        },
        "results": res
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote diagnostics to", OUTPUT_JSON)

    ranking = recommend_best_combo(res, top_k=10)
    print("\nTop recommended (block_size, offset_size, use_special_block) by median compression:")
    for key, median_comp, summary in ranking:
        b, o, s = key
        print(f"  {key}  median_comp={median_comp:.3f}  samples={summary['n_samples']}  codes_len_median={summary['codes_len_median']:.1f}")
    print("\nDone.")
