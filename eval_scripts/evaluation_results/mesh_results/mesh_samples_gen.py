import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
import time

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


def compute_mesh_metrics(vertices, faces):
    """
    Compute various mesh quality metrics.

    Returns:
        dict: Dictionary containing mesh metrics
    """
    metrics = {}

    # Basic counts
    metrics['num_vertices'] = len(vertices)
    metrics['num_faces'] = len(faces)

    # Face area statistics
    face_vertices = vertices[faces]  # Shape: (num_faces, 3, 3)
    v0, v1, v2 = face_vertices[:, 0, :], face_vertices[:, 1, :], face_vertices[:, 2, :]

    edge1, edge2 = v1 - v0, v2 - v0
    normals = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(normals, axis=1)

    metrics['total_surface_area'] = float(np.sum(areas))
    metrics['zero_area_faces'] = int(np.sum(areas < 1e-10))

    # # Edge length statistics
    # edge_lengths = []
    # for i in range(3):
    #     edges = face_vertices[:, (i + 1) % 3, :] - face_vertices[:, i, :]
    #     edge_lengths.append(np.linalg.norm(edges, axis=1))
    # edge_lengths = np.concatenate(edge_lengths)
    #
    # metrics['mean_edge_length'] = float(np.mean(edge_lengths))
    # metrics['std_edge_length'] = float(np.std(edge_lengths))
    # metrics['min_edge_length'] = float(np.min(edge_lengths))
    # metrics['max_edge_length'] = float(np.max(edge_lengths))

    # Bounding box
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    bbox_size = bbox_max - bbox_min

    metrics['bbox_volume'] = float(np.prod(bbox_size))
    metrics['bbox_dimensions'] = bbox_size.tolist()

    # Aspect ratio (ratio of largest to smallest bbox dimension)
    # metrics['aspect_ratio'] = float(np.max(bbox_size) / (np.min(bbox_size) + 1e-10))

    return metrics


def save_evaluation_results(results, output_path):
    """Save evaluation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


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

category_counts = {}
for item in dataset.data:
    if 'texts' in item:
        cat = item['texts']
        category_counts[cat] = category_counts.get(cat, 0) + 1
print(f"Category counts: {category_counts}")
# Filter categories with sufficient samples for evaluation
MIN_SAMPLES_FOR_EVALUATION = 100  # Minimum reference samples for statistical validity
unique_categories = [cat for cat, count in category_counts.items() if count >= MIN_SAMPLES_FOR_EVALUATION]
unique_categories.sort(key=lambda x: category_counts[x], reverse=True)  # Sort by sample count

# ['table', 'chair', 'sofa couch lounge', 'bench', 'display video display', 'cabinet', 'lamp', 'bookshelf', 'loudspeaker speaker speaker unit loudspeaker system speaker system', 'airplane aeroplane plane', 'bathtub bathing tub bath tub', 'vessel watercraft', 'telephone phone telephone set', 'bottle', 'pot flowerpot', 'mug', 'jar', 'knife', 'cellular telephone cellular phone cellphone cell mobile phone', 'clock', 'rifle', 'laptop laptop computer', 'bed', 'faucet spigot', 'bowl', 'file file cabinet filing cabinet', 'ashcan trash can garbage can wastebin ash bin ash-bin ashbin dustbin trash barrel trash bin']

print(f"Found {len(unique_categories)} unique categories")
print(unique_categories)

# Evaluation parameters
TARGET_SAMPLES = 50
TEMPERATURE = 1.0
MAX_ATTEMPTS_PER_SAMPLE = 5  # Maximum attempts before giving up on a sample

# Load existing results if they exist
results_file = output_dir / "evaluation_results.json"
if results_file.exists():
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    print(f"Loaded existing results for {len(all_results)} categories")
else:
    all_results = {}

# Process each category
for category_idx, category in enumerate(unique_categories):
    TEMPERATURE = 1.0
    print(f"\n{'=' * 80}")
    print(f"Processing category {category_idx + 1}/{len(unique_categories)}: {category}")
    print(f"{'=' * 80}")

    # Skip if already processed
    if category in all_results and all_results[category].get('completed', False):
        print(f"Category '{category}' already completed. Skipping...")
        continue

    if category == 'cabinet':
        print(f"Category '{category}'. Skipping...")
        continue


    category_dir = output_dir / category.replace(' ', '_')
    category_dir.mkdir(parents=True, exist_ok=True)

    # Initialize category results
    category_results = {
        'category': category,
        'target_samples': TARGET_SAMPLES,
        'temperature': TEMPERATURE,
        'successful_samples': 0,
        'total_attempts': 0,
        'inference_times': [],
        'mesh_metrics': [],
        'failed_samples': 0,
        'failure_reasons': defaultdict(int),
        'completed': False
    }

    # Progress bar
    pbar = tqdm(total=TARGET_SAMPLES, desc=f"Generating {category}")

    success_count = 0
    consecutive_failures = 0

    while success_count < TARGET_SAMPLES:
        # Safety check: stop if too many consecutive failures
        if consecutive_failures >= MAX_ATTEMPTS_PER_SAMPLE:
            if TEMPERATURE >= 0.0:
                TEMPERATURE = TEMPERATURE - 0.1
            else:
                TEMPERATURE = 0
            consecutive_failures = 0
            print(f"\nWarning: {MAX_ATTEMPTS_PER_SAMPLE} consecutive failures. Lowering temperature to, {TEMPERATURE}.")

        category_results['total_attempts'] += 1

        try:
            # Time the generation
            start_time = time.time()

            mesh_output = transformer.generate(
                texts=[category],
                temperature=TEMPERATURE,
            )

            inference_time = time.time() - start_time

            # Extract mesh data
            generated_vertices, faces = extract_vertices_faces_from_mesh(mesh_output)

            # Convert to numpy arrays
            vertices = np.asarray(generated_vertices, dtype=np.float32)
            faces = np.asarray(faces, dtype=np.int32)

            # Validate mesh
            if len(vertices) == 0 or len(faces) == 0:
                category_results['failed_samples'] += 1
                category_results['failure_reasons']['empty_mesh'] += 1
                consecutive_failures += 1
                continue

            # Compute mesh metrics
            mesh_metrics = compute_mesh_metrics(vertices, faces)

            # Check for zero-area faces
            if mesh_metrics['zero_area_faces'] > 10:
                category_results['failed_samples'] += 1
                category_results['failure_reasons']['too_many_zero_area_faces'] += 1
                consecutive_failures += 1
                print("\nWarning: Too many zero areas faces.", mesh_metrics['zero_area_faces'])
                continue

            # Success! Save the mesh
            success_count += 1
            consecutive_failures = 0

            generated_path = category_dir / f"{category.replace(' ', '_')}_{success_count:03d}.obj"
            save_obj_from_vertices(generated_vertices, generated_path)

            # Record metrics
            category_results['inference_times'].append(inference_time)
            mesh_metrics['inference_time'] = inference_time
            mesh_metrics['sample_id'] = success_count
            mesh_metrics['file_path'] = str(generated_path)
            mesh_metrics['temp'] = TEMPERATURE
            category_results['mesh_metrics'].append(mesh_metrics)

            pbar.update(1)

        except Exception as e:
            category_results['failed_samples'] += 1
            category_results['failure_reasons'][f'exception_{type(e).__name__}'] += 1
            consecutive_failures += 1
            print(f"\nError generating sample: {e}")
            continue

    pbar.close()

    # Compute aggregate statistics
    category_results['successful_samples'] = success_count

    if category_results['inference_times']:
        category_results['avg_inference_time'] = float(np.mean(category_results['inference_times']))
        category_results['std_inference_time'] = float(np.std(category_results['inference_times']))
        category_results['min_inference_time'] = float(np.min(category_results['inference_times']))
        category_results['max_inference_time'] = float(np.max(category_results['inference_times']))

    category_results['success_rate'] = success_count / category_results['total_attempts'] if category_results[
                                                                                                 'total_attempts'] > 0 else 0

    # Aggregate mesh metrics
    if category_results['mesh_metrics']:
        metric_keys = ['num_vertices', 'num_faces', 'total_surface_area', 'bbox_volume']

        for key in metric_keys:
            values = [m[key] for m in category_results['mesh_metrics'] if key in m]
            if values:
                category_results[f'avg_{key}'] = float(np.mean(values))
                category_results[f'std_{key}'] = float(np.std(values))
                category_results[f'min_{key}'] = float(np.min(values))
                category_results[f'max_{key}'] = float(np.max(values))

    category_results['completed'] = True
    category_results['failure_reasons'] = dict(category_results['failure_reasons'])

    # Save results for this category
    all_results[category] = category_results
    save_evaluation_results(all_results, results_file)

    # Print summary for this category
    print(f"\n{'=' * 80}")
    print(f"Category: {category} - COMPLETED")
    print(f"{'=' * 80}")
    print(f"Successful samples: {success_count}/{TARGET_SAMPLES}")
    print(f"Total attempts: {category_results['total_attempts']}")
    print(f"Success rate: {category_results['success_rate']:.2%}")
    if category_results['inference_times']:
        print(
            f"Avg inference time: {category_results['avg_inference_time']:.3f}s ± {category_results['std_inference_time']:.3f}s")
    if category_results['failure_reasons']:
        print(f"Failure reasons: {category_results['failure_reasons']}")
    print(f"{'=' * 80}\n")

# Print final summary
print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print(f"Total categories processed: {len(all_results)}")
print(f"Results saved to: {results_file}")

# Compute overall statistics
completed_categories = [r for r in all_results.values() if r.get('completed', False)]
if completed_categories:
    total_samples = sum(r['successful_samples'] for r in completed_categories)
    total_attempts = sum(r['total_attempts'] for r in completed_categories)
    overall_success_rate = total_samples / total_attempts if total_attempts > 0 else 0

    print(f"\nOverall Statistics:")
    print(f"  Total successful samples: {total_samples}")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Overall success rate: {overall_success_rate:.2%}")

    all_inference_times = []
    for r in completed_categories:
        all_inference_times.extend(r['inference_times'])

    if all_inference_times:
        print(f"  Average inference time: {np.mean(all_inference_times):.3f}s ± {np.std(all_inference_times):.3f}s")

print("=" * 80)