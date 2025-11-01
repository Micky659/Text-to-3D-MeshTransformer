from data_scripts.Mesh_dataset import MeshDataset
from model_scripts.autoencoder import MeshAutoencoder
from model_scripts.transformer import MeshTransformer
import torch
import numpy as np
import json
import math
from pathlib import Path

script_dir = Path(__file__).resolve().parent
DATASET_PATH = str(script_dir.parent.parent) + "/Data/meshData_org_augmented.npz"
print(DATASET_PATH)

dataset = MeshDataset.load(DATASET_PATH)

# Initialize autoencoder
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

pkg = torch.load(str(script_dir.parent.parent) + "/Model/mesh-encoder_16k_4_2.pt")
autoencoder.load_state_dict(pkg['model'])

import gc

gc.collect()
torch.cuda.empty_cache()

# Initialize transformer
transformer = MeshTransformer(
    autoencoder,
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

total_params = sum(p.numel() for p in transformer.decoder.parameters())
total_params = f"{total_params / 1000000:.1f}M"
print(f"Decoder total parameters: {total_params}")

pkg = torch.load(str(script_dir.parent.parent) + '/checkpoints/mesh-transformer.ckpt.epoch_150_avg_loss_0.005.pt')
transformer.load_state_dict(pkg['model'])

# Create output folder
folder = 'renders'
obj_file_path = Path(folder)
obj_file_path.mkdir(exist_ok=True, parents=True)

# Get unique labels
labels = list(set(item["texts"] for item in dataset.data))
print(f"Found {len(labels)} unique labels: {labels}")

# Target dimensions for realistic scaling (in meters) - adjust based on your needs
TARGET_DIMENSIONS = {
    "loudspeaker": [0.4, 0.4, 1.0],
    "dishwasher": [0.6, 0.6, 0.85],
    "laptop": [0.35, 0.25, 0.03],
    "train": [3.2, 25.0, 4.2],
    "remote control": [0.05, 0.18, 0.02],
    "trash can": [0.45, 0.45, 0.9],
    "can": [0.07, 0.07, 0.12],
    "rocket": [2.0, 2.0, 20.0],
    "bathtub": [0.8, 1.6, 0.6],
    "microphone": [0.05, 0.05, 0.18],
    "clock": [0.3, 0.05, 0.3],
    "piano": [1.5, 2.0, 1.2],
    "telephone": [0.2, 0.2, 0.1],
    "table": [1.5, 0.8, 0.75],
    "faucet": [0.05, 0.15, 0.15],
    "skateboard": [0.25, 0.8, 0.12],
    "stove": [0.6, 0.6, 0.9],
    "jar": [0.1, 0.1, 0.15],
    "bench": [1.2, 0.4, 0.5],
    "sofa": [2.2, 1.0, 0.9],
    "earphone": [0.15, 0.15, 0.05],
    "rifle": [0.1, 1.2, 0.2],
    "bed": [1.5, 2.0, 0.6],
    "pillow": [0.6, 0.4, 0.15],
    "computer keyboard": [0.15, 0.45, 0.03],
    "bag": [0.4, 0.2, 0.5],
    "printer": [0.45, 0.45, 0.25],
    "knife": [0.05, 0.3, 0.03],
    "mailbox": [0.3, 0.3, 1.2],
    "cabinet": [0.9, 0.45, 1.2],
    "cellular telephone": [0.08, 0.15, 0.01],
    "helmet": [0.25, 0.3, 0.25],
    "bus": [2.5, 12.0, 3.2],
    "cap": [0.25, 0.25, 0.15],
    "microwave": [0.5, 0.4, 0.3],
    "pot": [0.3, 0.3, 0.4],
    "basket": [0.35, 0.35, 0.25],
    "pistol": [0.05, 0.25, 0.15],
    "chair": [0.5, 0.5, 0.95],
    "mug": [0.1, 0.1, 0.12],
    "bottle": [0.08, 0.08, 0.25],
    "camera": [0.15, 0.12, 0.1],
    "washer": [0.6, 0.6, 0.85],
    "birdhouse": [0.25, 0.25, 0.3],
    "bowl": [0.2, 0.2, 0.1],
    "bookshelf": [0.8, 0.3, 1.8],
    "lamp": [0.4, 0.4, 1.6],
    "tower": [5.0, 5.0, 30.0],
    "guitar": [0.35, 1.0, 0.1],
    "file cabinet": [0.6, 0.6, 1.2],
    "car": [1.8, 4.5, 1.5],
    "motorcycle": [0.8, 2.2, 1.2],
    "display": [0.6, 0.05, 0.4],
    "airplane": [35.0, 40.0, 12.0],
    "vessel": [6.0, 25.0, 8.0]
}


def orient_triangle_upward(v1, v2, v3):
    """Ensure triangle normal points upward"""
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = np.cross(edge1, edge2)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    up = np.array([0, 1, 0])
    if np.dot(normal, up) < 0:
        v1, v3 = v3, v1
    return v1, v2, v3


def get_angle(v1, v2, v3):
    """Calculate interior angle at v1"""
    v1, v2, v3 = orient_triangle_upward(v1, v2, v3)
    vec1 = v2 - v1
    vec2 = v3 - v1
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return math.degrees(angle_rad)


def extract_vertices_from_mesh(mesh_output):
    """Extract vertices from mesh (tensor, mask) format"""
    all_vertices = []

    mesh_list = mesh_output if isinstance(mesh_output, list) else [mesh_output]

    for mesh_item in mesh_list:
        tensor, mask = mesh_item

        for tensor_batch, mask_batch in zip(tensor, mask):
            vertices = tensor_batch[mask_batch].cpu().numpy().reshape(-1, 3)
            all_vertices.append(vertices)

    if all_vertices:
        return np.vstack(all_vertices)
    else:
        return np.array([]).reshape(0, 3)


def scale_mesh_to_target(vertices, target_bbox):
    """
    Scale mesh to match target bounding box dimensions.

    Args:
        vertices: numpy array of shape (N, 3)
        target_bbox: [width, depth, height] in meters (X, Z, Y ordering)

    Returns:
        scaled vertices
    """
    if len(vertices) == 0:
        return vertices

    # Calculate current bbox
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    current_dims = max_coords - min_coords

    # Target dimensions in [X, Z, Y] order to match vertices [X, Y, Z]
    target_x = target_bbox[0]  # width
    target_z = target_bbox[1]  # depth
    target_y = target_bbox[2]  # height

    # Calculate scale factors for each axis
    scale_x = target_x / (current_dims[0] + 1e-8)
    scale_y = target_y / (current_dims[1] + 1e-8)
    scale_z = target_z / (current_dims[2] + 1e-8)

    # Center the mesh at origin before scaling
    center = (min_coords + max_coords) / 2
    vertices_centered = vertices - center

    # Apply scaling
    vertices_scaled = vertices_centered * np.array([scale_x, scale_y, scale_z])

    # Move back to original center position (or you can keep at origin)
    # For now, keep at origin for consistency

    return vertices_scaled


def save_single_mesh(path, mesh_output, scale_to_target=None):
    """
    Save a single mesh to OBJ file with proper face orientation.
    Optionally scale to target dimensions.

    Args:
        path: output file path
        mesh_output: (tensor, mask) tuple from transformer.generate()
        scale_to_target: optional [w, d, h] to scale mesh to these dimensions

    Returns:
        numpy array of final vertices (after scaling)
    """
    # Extract vertices
    vertices = extract_vertices_from_mesh(mesh_output)

    if len(vertices) == 0:
        print(f"Warning: No vertices to save for {path}")
        return vertices

    # Scale if requested
    if scale_to_target is not None:
        vertices = scale_mesh_to_target(vertices, scale_to_target)

    # Prepare OBJ content
    all_vertices = []
    all_faces = []

    # Write vertices
    for vertex in vertices:
        all_vertices.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

    # Calculate mesh center for face orientation
    mesh_center = np.mean(vertices, axis=0)

    # Write faces with correct orientation
    for i in range(0, len(vertices), 3):
        v1 = vertices[i]
        v2 = vertices[i + 1]
        v3 = vertices[i + 2]

        normal = np.cross(v2 - v1, v3 - v1)

        # Determine direction for face orientation
        if get_angle(v1, v2, v3) > 60:
            direction_vector = mesh_center - np.mean([v1, v2, v3], axis=0)
            direction_vector = -direction_vector
        else:
            direction_vector = np.array([0, 1, 0])

        # Reorder vertices if needed
        if np.dot(normal, direction_vector) > 0:
            order = [0, 1, 2]
        else:
            order = [0, 2, 1]

        reordered_vertices = [v1, v2, v3][order[0]], [v1, v2, v3][order[1]], [v1, v2, v3][order[2]]
        indices = [np.where((vertices == vertex).all(axis=1))[0][0] + 1 for vertex in reordered_vertices]
        all_faces.append(f"f {indices[0]} {indices[1]} {indices[2]}\n")

    # Write to file
    obj_content = "".join(all_vertices) + "".join(all_faces)
    with open(path, "w") as file:
        file.write(obj_content)

    print(f"[save_single_mesh] Saved to {path}")
    return vertices


def compute_mesh_metadata(vertices):
    """
    Compute metadata from vertices.

    Args:
        vertices: numpy array of shape (N, 3)

    Returns:
        dict with bbox, anchor, bbox_center, num_faces
    """
    if len(vertices) == 0:
        return {
            "bbox": [0.0, 0.0, 0.0],
            "anchor": [0.0, 0.0, 0.0],
            "bbox_center": [0.0, 0.0, 0.0],
        }

    # Get min and max for each dimension
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    # Compute bbox in W, D, H order (X, Z, Y)
    width = float(max_coords[0] - min_coords[0])
    depth = float(max_coords[2] - min_coords[2])
    height = float(max_coords[1] - min_coords[1])

    # Compute bbox center
    center_x = float((min_coords[0] + max_coords[0]) / 2)
    center_y = float((min_coords[1] + max_coords[1]) / 2)
    center_z = float((min_coords[2] + max_coords[2]) / 2)

    # Compute anchor: centroid of floor-contact vertices
    ys = vertices[:, 1]
    miny = float(np.min(ys))
    tol = 1e-3
    contact_idx = np.where(ys <= (miny + tol))[0]

    if contact_idx.size > 0:
        anchor_pt = vertices[contact_idx].mean(axis=0)
    else:
        anchor_pt = vertices.mean(axis=0)

    anchor_x = float(anchor_pt[0])
    anchor_y = float(anchor_pt[1])
    anchor_z = float(anchor_pt[2])

    return {
        "bbox": [round(width, 2), round(depth, 2), round(height, 2)],
        "anchor": [round(anchor_x, 2), round(anchor_y, 2), round(anchor_z, 2)],
        "bbox_center": [round(center_x, 2), round(center_y, 2), round(center_z, 2)],
    }


# Generate meshes and compute metadata
metadata_dict = {}

for label in labels:
    print(f"\nGenerating mesh for: {label}")

    try:
        # Generate mesh
        mesh_output = transformer.generate(texts=[label], temperature=0.0)

        # Get target dimensions for this category
        for key in TARGET_DIMENSIONS.keys():
            if key in label:
                target_dims = TARGET_DIMENSIONS.get(key, None)

        if target_dims is None:
            print(f"  Warning: No target dimensions for '{label}', using unscaled mesh")

        # Save mesh with scaling
        out_path = f'{folder}/{label}.obj'
        final_vertices = save_single_mesh(out_path, mesh_output, scale_to_target=target_dims)

        # Compute metadata on the final scaled vertices
        metadata = compute_mesh_metadata(final_vertices)
        metadata_dict[label] = metadata

        print(f"  - BBox (W×D×H): {metadata['bbox']}")
        print(f"  - Anchor: {metadata['anchor']}")
        print(f"  - Center: {metadata['bbox_center']}")

        if target_dims:
            print(f"  - Scaled to target: {target_dims}")

    except Exception as e:
        print(f"  - Error generating {label}: {str(e)}")
        import traceback

        traceback.print_exc()
        continue

# Save metadata to JSON
metadata_file = f'{folder}/mesh_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata_dict, f, indent=4)

print(f"\n{'=' * 60}")
print(f"Metadata saved to: {metadata_file}")
print(f"Generated {len(metadata_dict)} meshes successfully")

# Print summary
print(f"\n{'=' * 60}")
print("METADATA SUMMARY")
print(f"{'=' * 60}")
for label, data in metadata_dict.items():
    print(f"\n{label}:")
    print(f"  BBox (W×D×H): {data['bbox']}")
    print(f"  Anchor: {data['anchor']}")
    print(f"  Center: {data['bbox_center']}")