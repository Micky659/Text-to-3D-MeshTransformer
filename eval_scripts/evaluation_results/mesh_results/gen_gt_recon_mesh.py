"""
Comprehensive evaluation script for mesh transformer model.
Generates meshes at varying temperatures and compares with ground truth.
"""

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


def compute_mesh_similarity(vertices1, vertices2):
    """
    Compute similarity between two meshes using MSE.
    Lower MSE means more similar.

    Args:
        vertices1: numpy array (N1, 3)
        vertices2: numpy array (N2, 3)

    Returns:
        mse: mean squared error
    """

    # Normalize both meshes to unit scale for fair comparison
    def normalize_vertices(verts):
        if len(verts) == 0:
            return verts
        center = np.mean(verts, axis=0)
        centered = verts - center
        scale = np.max(np.abs(centered)) + 1e-8
        return centered / scale

    v1_norm = normalize_vertices(vertices1)
    v2_norm = normalize_vertices(vertices2)

    # Pad to same length if needed
    max_len = max(len(v1_norm), len(v2_norm))
    if len(v1_norm) < max_len:
        v1_norm = np.pad(v1_norm, ((0, max_len - len(v1_norm)), (0, 0)), mode='constant')
    if len(v2_norm) < max_len:
        v2_norm = np.pad(v2_norm, ((0, max_len - len(v2_norm)), (0, 0)), mode='constant')

    mse = np.mean((v1_norm - v2_norm) ** 2)
    return mse


def find_closest_mesh_in_dataset(generated_vertices, dataset, category):
    """
    Find the closest mesh in the dataset to the generated mesh.

    Args:
        generated_vertices: numpy array (N, 3)
        dataset: MeshDataset object
        category: category to filter by

    Returns:
        closest_item: dictionary with mesh data
        min_mse: similarity score
        index: index in dataset
    """
    min_mse = float('inf')
    closest_item = None
    closest_idx = -1

    for idx, item in enumerate(dataset.data):
        # Filter by category
        if item.get('texts', '').lower() != category.lower():
            continue

        # Get vertices from the dataset item
        gt_vertices = item['vertices'][item['faces']].reshape(-1, 3)
        if isinstance(gt_vertices, torch.Tensor):
            gt_vertices = gt_vertices.cpu().numpy()

        # Compute similarity
        mse = compute_mesh_similarity(generated_vertices, gt_vertices)

        if mse < min_mse:
            min_mse = mse
            closest_item = item
            closest_idx = idx

    return closest_item, min_mse, closest_idx


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


def evaluate_mesh_generation(
        transformer,
        autoencoder,
        dataset,
        categories,
        temperatures=[0.0, 0.3, 0.5, 0.7, 0.9],
        num_samples_per_category=5,
        output_dir="evaluation_results"
):
    """
    Main evaluation function.

    Args:
        transformer: MeshTransformer model
        autoencoder: MeshAutoencoder model
        dataset: MeshDataset object
        categories: list of category names to evaluate
        temperatures: list of temperature values
        num_samples_per_category: number of samples per category
        output_dir: directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results tracking
    results = {
        'categories': {},
        'summary': {}
    }

    transformer.eval()
    autoencoder.eval()

    print(f"\n{'=' * 80}")
    print(f"Starting Mesh Generation Evaluation")
    print(f"Categories: {categories}")
    print(f"Temperatures: {temperatures}")
    print(f"Samples per category: {num_samples_per_category}")
    print(f"{'=' * 80}\n")

    with torch.no_grad():
        for category in categories:
            print(f"\n{'=' * 60}")
            print(f"Processing Category: {category}")
            print(f"{'=' * 60}")

            category_dir = output_dir / category.replace(' ', '_')
            category_dir.mkdir(parents=True, exist_ok=True)

            category_results = {
                'samples': [],
                'avg_similarity': 0.0,
                'temperatures_used': []
            }

            # Generate samples at different temperatures
            for sample_idx in range(num_samples_per_category):
                # Cycle through temperatures
                temp_idx = sample_idx % len(temperatures)
                temperature = temperatures[temp_idx]

                print(f"\n  Sample {sample_idx + 1}/{num_samples_per_category} (Temperature: {temperature})")

                sample_dir = category_dir / f"sample_{sample_idx + 1}_temp_{temperature}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                try:
                    # Generate mesh
                    print(f"    Generating mesh...")
                    mesh_output = transformer.generate(
                        texts=[category],
                        temperature=temperature
                    )

                    # Extract vertices from generated mesh
                    generated_vertices = extract_vertices_from_mesh(mesh_output)

                    if len(generated_vertices) == 0:
                        print(f"    Warning: Generated mesh has no vertices, skipping...")
                        continue

                    # Save generated mesh
                    generated_path = sample_dir / "generated.obj"
                    save_obj_from_vertices(generated_vertices, generated_path)
                    print(f"    Saved generated mesh: {generated_path}")

                    # Find closest mesh in dataset
                    print(f"    Finding closest ground truth mesh...")
                    closest_item, similarity_score, gt_idx = find_closest_mesh_in_dataset(
                        generated_vertices, dataset, category
                    )

                    if closest_item is None:
                        print(f"    Warning: No matching ground truth found for category '{category}'")
                        continue

                    print(f"    Found closest mesh (index: {gt_idx}, similarity: {similarity_score:.6f})")

                    # Save ground truth mesh
                    gt_vertices = closest_item['vertices'][closest_item['faces']].reshape(-1, 3)
                    if isinstance(gt_vertices, torch.Tensor):
                        gt_vertices = gt_vertices.cpu().numpy()

                    gt_path = sample_dir / "ground_truth.obj"
                    save_obj_from_vertices(gt_vertices, gt_path)
                    print(f"    Saved ground truth mesh: {gt_path}")

                    # Reconstruct mesh using autoencoder
                    print(f"    Reconstructing mesh with autoencoder...")
                    codes = autoencoder.tokenize(
                        vertices=closest_item['vertices'],
                        faces=closest_item['faces'],
                        face_edges=closest_item['face_edges']
                    )

                    codes_flat = codes.flatten().unsqueeze(0)
                    # Ensure codes are properly shaped
                    codes_flat = codes_flat[
                        :, :codes_flat.shape[-1] // autoencoder.num_quantizers * autoencoder.num_quantizers]

                    reconstructed_coords, reconstructed_mask = autoencoder.decode_from_codes_to_faces(codes_flat)
                    reconstructed_vertices = reconstructed_coords[reconstructed_mask].cpu().numpy().reshape(-1, 3)

                    # Save reconstructed mesh
                    reconstructed_path = sample_dir / "reconstructed.obj"
                    save_obj_from_vertices(reconstructed_vertices, reconstructed_path)
                    print(f"    Saved reconstructed mesh: {reconstructed_path}")

                    # Compute reconstruction error
                    recon_mse = np.mean((gt_vertices - reconstructed_vertices[:len(gt_vertices)]) ** 2)

                    # Store results
                    sample_result = {
                        'sample_idx': sample_idx + 1,
                        'temperature': temperature,
                        'similarity_to_gt': float(similarity_score),
                        'reconstruction_mse': float(recon_mse),
                        'gt_dataset_idx': gt_idx,
                        'num_vertices_generated': len(generated_vertices),
                        'num_vertices_gt': len(gt_vertices),
                        'num_vertices_reconstructed': len(reconstructed_vertices)
                    }

                    category_results['samples'].append(sample_result)
                    category_results['temperatures_used'].append(temperature)

                    print(f"    ✓ Sample completed successfully")
                    print(f"      - Similarity to GT: {similarity_score:.6f}")
                    print(f"      - Reconstruction MSE: {recon_mse:.6f}")

                except Exception as e:
                    print(f"    Error processing sample: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Compute category statistics
            if category_results['samples']:
                category_results['avg_similarity'] = np.mean([
                    s['similarity_to_gt'] for s in category_results['samples']
                ])
                category_results['avg_reconstruction_mse'] = np.mean([
                    s['reconstruction_mse'] for s in category_results['samples']
                ])
                category_results['num_successful_samples'] = len(category_results['samples'])

            results['categories'][category] = category_results

            # Save category results
            category_json_path = category_dir / "results.json"
            with open(category_json_path, 'w') as f:
                json.dump(category_results, f, indent=2)

            print(f"\n  Category Summary:")
            print(f"    Successful samples: {len(category_results['samples'])}/{num_samples_per_category}")
            if category_results['samples']:
                print(f"    Average similarity: {category_results['avg_similarity']:.6f}")
                print(f"    Average reconstruction MSE: {category_results['avg_reconstruction_mse']:.6f}")

    # Compute overall statistics
    all_similarities = []
    all_recon_mse = []
    for cat_results in results['categories'].values():
        if cat_results['samples']:
            all_similarities.extend([s['similarity_to_gt'] for s in cat_results['samples']])
            all_recon_mse.extend([s['reconstruction_mse'] for s in cat_results['samples']])

    if all_similarities:
        results['summary'] = {
            'total_samples': len(all_similarities),
            'avg_similarity': float(np.mean(all_similarities)),
            'std_similarity': float(np.std(all_similarities)),
            'avg_reconstruction_mse': float(np.mean(all_recon_mse)),
            'std_reconstruction_mse': float(np.std(all_recon_mse)),
            'categories_evaluated': len(results['categories'])
        }

    # Save overall results
    results_json_path = output_dir / "evaluation_summary.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Evaluation Complete!")
    print(f"{'=' * 80}")
    print(f"Overall Summary:")
    print(f"  Total samples: {results['summary'].get('total_samples', 0)}")
    print(f"  Average similarity: {results['summary'].get('avg_similarity', 0):.6f}")
    print(f"  Average reconstruction MSE: {results['summary'].get('avg_reconstruction_mse', 0):.6f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary saved to: {results_json_path}")

    return results


# Main execution
if __name__ == "__main__":
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    working_dir = project_root / "Data"
    DATASET_PATH = working_dir / "meshData_org.npz"

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

    # Get unique categories from dataset
    unique_categories = list(set([item['texts'] for item in dataset.data if 'texts' in item]))
    print(f"\nFound {len(unique_categories)} unique categories in dataset")

    # Select categories to evaluate (you can modify this list)
    categories_to_evaluate = unique_categories  # Evaluate first 5 categories
    # Or specify manually:
    # categories_to_evaluate = ['chair', 'table', 'car', 'bed', 'sofa']

    print(f"\nEvaluating categories: {categories_to_evaluate}")

    # Run evaluation
    results = evaluate_mesh_generation(
        transformer=transformer,
        autoencoder=autoencoder,
        dataset=dataset,
        categories=categories_to_evaluate,
        temperatures=[0.0, 0.2, 0.3, 0.5, 0.7],
        num_samples_per_category=5,
        output_dir="evaluation_results"
    )

    print("\n✓ Evaluation complete!")