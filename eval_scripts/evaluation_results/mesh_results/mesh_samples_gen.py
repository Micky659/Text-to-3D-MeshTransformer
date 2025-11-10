import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

from data_scripts.Mesh_dataset import MeshDataset
from model_scripts.autoencoder import MeshAutoencoder
from model_scripts.transformer import MeshTransformer

# Import the mesh saving utilities
import sys

sys.path.append(str(Path(__file__).parent))
from data_scripts.mesh_render_v2 import extract_vertices_from_mesh


script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
working_dir = project_root / "Data"
DATASET_PATH = working_dir / "meshData_org.npz"


def save_obj_from_vertices(vertices, output_path):
    """
    Save vertices as OBJ file.

    Args:
        vertices: numpy array (N, 3) - must be multiple of 3 for triangles
        output_path: path to save OBJ file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        # Write vertices
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write faces (assuming triangles)
        num_faces = len(vertices) // 3
        for i in range(num_faces):
            idx = i * 3 + 1  # OBJ indices start at 1
            f.write(f"f {idx} {idx + 1} {idx + 2}\n")



# Load dataset
print("Loading dataset...")
dataset = MeshDataset.load(DATASET_PATH)
print(f"Dataset loaded: {len(dataset.data)} meshes")

# Initialize autoencoder
print("\nInitializing autoencoder...")
autoencoder = MeshAutoencoder(
    decoder_dims_through_depth=(128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
    codebook_size=16384,
    dim_codebook=192,
    dim_area_embed=16,
    dim_coor_embed=16,
    dim_normal_embed=16,
    dim_angle_embed=8,
    attn_decoder_depth=4,
    attn_encoder_depth=2
).to("cuda")

# Load autoencoder weights
pkg = torch.load(str(project_root / "Model" / "mesh-encoder_16k_4_2.pt"))
autoencoder.load_state_dict(pkg['model'])
print("Autoencoder loaded")

# Initialize transformer
print("\nInitializing transformer...")
max_seq = max(len(d["faces"]) for d in dataset.data if "faces" in d) * (
        autoencoder.num_vertices_per_face * autoencoder.num_quantizers)
print(f"Max token sequence: {max_seq}")

transformer = MeshTransformer(
    autoencoder,
    dim=768,
    coarse_pre_gateloop_depth=3,
    fine_pre_gateloop_depth=3,
    attn_depth=12,
    attn_heads=12,
    max_seq_len=max_seq,
    condition_on_text=True,
    gateloop_use_heinsen=False,
    dropout=0.0,
    text_condition_model_types="bge",
    text_condition_cond_drop_prob=0.0
)

total_params = sum(p.numel() for p in transformer.decoder.parameters())
print(f"Decoder total parameters: {total_params / 1000000:.1f}M")

# Load transformer weights
pkg = torch.load(str(project_root / "Model" / "mesh-transformer_16k_4_2.pt"))
transformer.load_state_dict(pkg['model'])
print("Transformer loaded")

transformer.eval()
autoencoder.eval()

output_dir = Path("mesh_samples")
output_dir.mkdir(parents=True, exist_ok=True)

categories = ['chair']
temperature = [1.0]

for category in categories:

    category_dir = output_dir / category.replace(' ', '_')
    category_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(5):
        mesh_output = transformer.generate(
            texts=[category],
            temperature=1.0,
        )

        generated_vertices = extract_vertices_from_mesh(mesh_output)

        generated_path = category_dir / f"{category}_{sample_idx}.obj"
        save_obj_from_vertices(generated_vertices, generated_path)