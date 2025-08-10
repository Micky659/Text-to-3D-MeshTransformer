import numpy as np
import torch
from data_scripts.Mesh_dataset import MeshDataset
from data_scripts.BPT import BPT_serialize, BPT_deserialize
import trimesh
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

working_dir = project_root / "Data"
dataset_path = working_dir / "objverse_shapenet_modelnet_max_250faces_186M_tokens.npz"

ds = MeshDataset.load(dataset_path)
item = ds.data[0]
verts = item['vertices']
faces = item['faces']
if isinstance(verts, torch.Tensor):
    verts = verts.cpu().numpy()
if isinstance(faces, torch.Tensor):
    faces = faces.cpu().numpy()

valid_mask = np.all((faces >= 0) & (faces < len(verts)), axis=1)
faces_clean = faces[valid_mask]

mesh = trimesh.Trimesh(vertices=verts, faces=faces_clean, process=False)
codes = BPT_serialize(mesh)
reconstructed = BPT_deserialize(codes)

print("Original verts:", len(verts), "faces:", len(faces_clean))
print("BPT codes length:", len(codes))
print("Reconstructed vertex triples (triangle count):", len(reconstructed) // 3)
print("Compression ratio (triangles -> tokens):", (len(faces_clean) * 3) / max(1, len(codes)))
