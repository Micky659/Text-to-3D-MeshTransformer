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
from data_scripts.mesh_render_v2 import extract_vertices_faces_from_mesh


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
pkg = torch.load(str(project_root / "Models" / "mesh-encoder_16k_4_2.pt"))
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
pkg = torch.load(str(project_root / "Models" / "mesh-transformer_16k_4_2.pt"))
transformer.load_state_dict(pkg['model'])
print("Transformer loaded")

transformer.eval()
autoencoder.eval()

output_dir = Path("mesh_samples")
output_dir.mkdir(parents=True, exist_ok=True)

# categories = ['chair']

# ['printer printing machine', 'bench', 'pistol handgun side arm shooting iron', 'earphone earpiece headphone phone',
# 'table', 'dishwasher dish washer dishwashing machine', 'knife', 'pillow', 'helmet', 'bed', 'microwave microwave oven',
# 'basket handbasket', 'mug', 'faucet spigot', 'pot flowerpot', 'rocket projectile', 'piano pianoforte forte-piano',
# 'cellular telephone cellular phone cellphone cell mobile phone', 'computer keyboard keypad', 'can tin tin can',
# 'sofa couch lounge', 'chair', 'ashcan trash can garbage can wastebin ash bin ash-bin ashbin dustbin trash barrel trash bin',
# 'cap', 'tower', 'bus autobus coach charabanc double-decker jitney motorbus motorcoach omnibus passenger vehi',
# 'skateboard', 'loudspeaker speaker speaker unit loudspeaker system speaker system', 'bag traveling bag travelling bag grip suitcase',
# 'bookshelf', 'jar', 'mailbox letter box', 'file file cabinet filing cabinet', 'guitar', 'motorcycle bike', 'remote control remote',
# 'camera photographic camera', 'microphone mike', 'bathtub bathing tub bath tub', 'bottle', 'display video display',
# 'airplane aeroplane plane', 'bowl', 'telephone phone telephone set', 'birdhouse', 'cabinet', 'clock', 'laptop laptop computer',
# 'vessel watercraft', 'train railroad train', 'washer automatic washer washing machine', 'car auto automobile machine motorcar',
# 'lamp', 'stove', 'rifle']

unique_categories = list(set([item['texts'] for item in dataset.data if 'texts' in item]))
print(unique_categories)
runs = 50
temperature = [1.0]

for category in unique_categories:
    print("Generating Category:", category)

    category_dir = output_dir / category.replace(' ', '_')
    category_dir.mkdir(parents=True, exist_ok=True)

    success = 0

    while success < runs:
        mesh_output = transformer.generate(
            texts=[category],
            temperature=1.0,
        )

        generated_vertices, faces = extract_vertices_faces_from_mesh(mesh_output)
        success += 1

        # Convert to numpy arrays
        vertices = np.asarray(generated_vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)

        face_vertices = vertices[faces]  # Shape: (num_faces, 3, 3)
        v0, v1, v2 = face_vertices[:, 0, :], face_vertices[:, 1, :], face_vertices[:, 2, :]

        edge1, edge2 = v1 - v0, v2 - v0
        normals = np.cross(edge1, edge2)
        areas = 0.5 * np.linalg.norm(normals, axis=1)

        zero_area_faces = np.sum(areas < 1e-10)
        if zero_area_faces > 10:
            print(f"Zero-area faces detected: {zero_area_faces}")
            success -= 1
            continue

        generated_path = category_dir / f"{category}_{success}.obj"
        save_obj_from_vertices(generated_vertices, generated_path)