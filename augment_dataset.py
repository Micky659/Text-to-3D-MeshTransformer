"""
augment_mesh_dataset.py

Safe augmentation for MeshGPT-style dataset stored as an .npz array of dicts.

Rules implemented:
 - cap table and chair to 5000 each (downsample if needed)
 - ensure each other category has at least 500 examples (upsample by duplication)
 - Augment duplicates only by small vertex jitter + tiny isotropic scaling (no rotations/flips)
 - Keeps faces same (valid indices), recomputes face_edges optionally
 - Writes new dataset to an .npz
"""

import numpy as np
import random
from pathlib import Path
import torch
import argparse
import copy
import math
import time

from data_scripts.Mesh_dataset import MeshDataset


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="Data/meshData_org.npz", help="input npz dataset")
parser.add_argument("--output", type=str, default="Data/meshData_augmented_2.npz", help="output npz dataset")
parser.add_argument("--min_per_cat", type=int, default=500, help="minimum target per category for non-table/chair")
parser.add_argument("--cap_table", type=int, default=5000)
parser.add_argument("--cap_chair", type=int, default=5000)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--regenerate_face_edges", action='store_true', help="call dataset.generate_face_edges() afterwards (if available).")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

in_path = Path(args.input)
out_path = Path(args.output)
assert in_path.exists(), f"Input file {in_path} not found"

print("Loading dataset:", in_path)
loaded = np.load(str(in_path), allow_pickle=True)
raw_list = list(loaded["arr_0"])
print("Loaded items:", len(raw_list))

# group by category text
from collections import defaultdict
by_cat = defaultdict(list)
for item in raw_list:
    # item is expected to be a dict-like with 'texts','vertices','faces' etc.
    cat = item.get('texts', None)
    if cat is None:
        # fallback: try 'text' or name
        cat = item.get('text', item.get('name', 'UNKNOWN'))
    by_cat[cat].append(item)

print("Categories found:", len(by_cat))
for k in sorted(by_cat.keys()):
    print(f"  {k}: {len(by_cat[k])}")

# target plan
targets = {}
for cat, items in by_cat.items():
    if cat.lower().startswith("table"):
        targets[cat] = min(args.cap_table, len(items)) if len(items) <= args.cap_table else args.cap_table
    elif cat.lower().startswith("chair"):
        targets[cat] = min(args.cap_chair, len(items)) if len(items) <= args.cap_chair else args.cap_chair
    else:
        targets[cat] = max(len(items), args.min_per_cat)

print("\nTarget counts per category (sample):")
for c in list(targets.keys())[:20]:
    print(" ", c, "->", targets[c])

# augmentation function (no rotations)
def augment_vertices(vertices_np, jitter_frac=0.005, scale_frac=0.01):
    """
    vertices_np: (V,3) numpy float32
    jitter_frac: relative jitter magnitude fraction of bbox diagonal
    scale_frac: range for isotropic scaling multiply: (1 - scale_frac, 1 + scale_frac)
    returns new_vertices_np (same dtype)
    """
    v = vertices_np.astype(np.float64)
    # compute bbox diag
    mins = v.min(axis=0)
    maxs = v.max(axis=0)
    diag = np.linalg.norm(maxs - mins)
    if diag <= 0:
        diag = 1.0
    # jitter
    sigma = diag * jitter_frac
    noise = np.random.normal(scale=sigma, size=v.shape)
    v2 = v + noise

    # small isotropic scale about centroid
    center = v2.mean(axis=0)
    s = 1.0 + np.random.uniform(-scale_frac, scale_frac)
    v2 = (v2 - center) * s + center

    # Important: Renormalize to original bounding box scale to preserve normalization/orientation
    # Compute current bbox diag and scale to match original diag (if diag>0)
    new_mins = v2.min(axis=0)
    new_maxs = v2.max(axis=0)
    new_diag = np.linalg.norm(new_maxs - new_mins)
    if new_diag > 1e-12:
        v2 = (v2 - v2.mean(axis=0)) * (diag / new_diag) + v.mean(axis=0)

    return v2.astype(np.float32)


def augment_mesh(vertices, scale_factor):
    jitter_factor = 0.005
    possible_values = np.arange(-jitter_factor, jitter_factor, 0.0005)
    offsets = np.random.choice(possible_values, size=vertices.shape)
    vertices = vertices + offsets

    # vertices = vertices * scale_factor
    # To ensure that the mesh models are on the "ground"
    # min_y = np.min(vertices[:, 1])
    # difference = -0.95 - min_y
    # vertices[:, 1] += difference
    return vertices


# Build new data list
new_data = []

# First: downsample table/chair if needed
for cat, items in by_cat.items():
    cur = len(items)
    targ = targets[cat]
    if targ < cur:
        # downsample random subset
        chosen = random.sample(items, targ)
        new_data.extend(chosen)
    else:
        new_data.extend(items)  # keep all, augmentation later if needed

print("After downsampling - count:", len(new_data))

# Now upsample each category to its target by duplicating + jitter
final_data = []
by_cat_new = defaultdict(list)
for it in new_data:
    cat = it.get('texts', it.get('text', it.get('name', 'UNKNOWN')))
    by_cat_new[cat].append(it)

for cat, items in by_cat_new.items():
    cur = len(items)
    targ = targets[cat]
    # keep originals first
    for it in items:
        # ensure vertices/faces are numpy arrays on CPU
        # also make a shallow copy to be safe
        copy_it = dict(it) if isinstance(it, dict) else dict(it.item())
        # convert tensors to numpy if needed
        # v = copy_it.get('vertices')
        # if hasattr(v, 'cpu'):
        #     v = v.cpu().numpy()
        # copy_it['vertices'] = np.asarray(v, dtype=np.float32)
        # f = copy_it.get('faces')
        # if hasattr(f, 'cpu'):
        #     f = f.cpu().numpy()
        # copy_it['faces'] = np.asarray(f, dtype=np.int64)
        # keep face_edges if present (do not modify)
        final_data.append(copy_it)

    # create augmented duplicates if needed
    needed = targ - cur
    if needed <= 0:
        continue

    print(f"Augmenting {cat}: need {needed} examples (cur {cur} -> targ {targ})")
    # choose sources to duplicate uniformly at random
    src_indices = list(range(cur))
    for i in range(needed):
        src = random.choice(src_indices)
        src_item = items[src]
        # copy dict and augment vertices only
        new_item = dict(src_item)
        # pull vertices as numpy
        v = new_item['vertices']
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()

        possible_values = np.arange(0.75, 1.0, 0.005)
        scale_factor = np.random.choice(possible_values, size=1)[0]
        # apply augmentation: small jitter + scale, no rotation
        # v_aug = augment_vertices(v, jitter_frac=0.005, scale_frac=0.01)
        v_aug = augment_mesh(v, 0.01)
        new_item['vertices'] = torch.tensor(v_aug.tolist(), dtype=torch.float).to("cuda")
        # faces left same (indices still valid)
        # If face_edges present, keep as-is — it's safe because faces unchanged.
        # but if you want to recompute face_edges later, set to missing:
        if 'face_edges' in new_item:
            # optionally remove so user can regenerate deterministically
            # comment the next line if you want to keep same face_edges
            new_item.pop('face_edges', None)
        # name can get suffix to avoid duplicates
        base_name = new_item.get('name', None)
        if base_name:
            new_item['name'] = f"{base_name}_aug_{i}"
        # Append to final_data
        final_data.append(new_item)

print("After augmentation - total items:", len(final_data))

# Final per-category counts report
from collections import Counter
cnt = Counter([it.get('texts', it.get('text', it.get('name', 'UNKNOWN'))) for it in final_data])
print("\nFinal counts summary (category -> new count):")
for cat, c in cnt.most_common():
    print(f" {cat}: {c}")

# Save
print("Saving augmented dataset to", out_path)
new_dataset = MeshDataset(final_data)
new_dataset.generate_face_edges()
new_dataset.save(str(out_path))
# np.savez_compressed(str(out_path), final_data, allow_pickle=True)
print("Saved.")

print("Done. Recommended: load this new dataset and run dataset.generate_face_edges() if you rely on precalculated face_edges.")
