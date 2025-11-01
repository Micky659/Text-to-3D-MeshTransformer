from pathlib import Path
import torch
import random
import json
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from data_scripts.Mesh_dataset import MeshDataset
from model_scripts.autoencoder import MeshAutoencoder
from training_scripts.autoencoder_trainer import MeshAutoencoderTrainer

# Assuming you have a mesh_render module for visualization
# from your_module import mesh_render

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent


# ============================================================================
# CONFIGURATION - Easily modify these constants
# ============================================================================

# Evaluation Configuration
EVALUATION_CONFIG = {
    'sample_size': 100,  # Number of samples to evaluate
    'random_visualization_samples': 30,  # Number of samples per visualization batch
    'dataset_path': str(project_root / "Data/meshData_org_onlychair.npz"),
    'results_base_folder': './evaluation_results',
    'save_individual_meshes': True,  # Save best/worst meshes as individual OBJ files
    'seed': 42  # Random seed for reproducibility
}

# Model Variants to Evaluate
MODEL_VARIANTS = [
    # {
    #     'name': 'ResidualLFQ',
    #     'model_path': project_root / 'Model/residuallfq/mesh_encoder_residuallfq_chairs_final_20251028_065213.pt',
    #     # 'model_path': project_root / 'checkpoints_chairs/residuallfq/mesh-autoencoder.ckpt.epoch_10_avg_loss_1.59365_recon_1.5353_commit_0.5835.pt',
    #     'use_residual_lfq': True,
    #     'use_residual_simvq': False,
    #     'description': 'Residual Lookup-Free Quantization'
    # },
    # {
    #     'name': 'ResidualLFQ chk 20',
    #     # 'model_path': project_root / 'Model/residuallfq/mesh_encoder_residuallfq_chairs_final_20251028_065213.pt',
    #     'model_path': project_root / 'checkpoints_chairs/residuallfq/mesh-autoencoder.ckpt.epoch_20_avg_loss_0.64722_recon_0.5865_commit_0.6074.pt',
    #     'use_residual_lfq': True,
    #     'use_residual_simvq': False,
    #     'description': 'Residual Lookup-Free Quantization'
    # },
    {
        'name': 'ResidualLFQ',
        # 'model_path': project_root / 'Model/residuallfq/mesh_encoder_residuallfq_chairs_final_20251028_065213.pt',
        'model_path': project_root / 'checkpoints_chairs/residuallfq/mesh-autoencoder.ckpt.epoch_40_avg_loss_0.41796_recon_0.3618_commit_0.5619.pt',
        'use_residual_lfq': True,
        'use_residual_simvq': False,
        'description': 'Residual Lookup-Free Quantization'
    },
    # {
    #     'name': 'ResidualLFQ chk 40',
    #     # 'model_path': project_root / 'Model/residuallfq/mesh_encoder_residuallfq_chairs_final_20251028_065213.pt',
    #     'model_path': project_root / 'checkpoints_chairs/residuallfq/mesh-autoencoder.ckpt.epoch_40_avg_loss_0.41796_recon_0.3618_commit_0.5619.pt',
    #     'use_residual_lfq': True,
    #     'use_residual_simvq': False,
    #     'description': 'Residual Lookup-Free Quantization'
    # },
    {
        'name': 'RVQ',
        # 'model_path': project_root / 'Model/rvq/mesh_encoder_rvq_chairs_final_20251028_054016.pt',
        'model_path': project_root / 'checkpoints_chairs/rvq/mesh-autoencoder.ckpt.epoch_40_avg_loss_2.33480_recon_2.3191_commit_0.1570.pt',
        'use_residual_lfq': False,
        'use_residual_simvq': False,
        'description': 'Residual Vector Quantization'
    },
    # {
    #     'name': 'ResidualSimVQ',
    #     'model_path': project_root / 'Model/residualsimvq/mesh_encoder_residualsimvq_chairs_final_20251028_073627.pt',
    #     # 'model_path': project_root / 'checkpoints_chairs/residualsimvq/mesh-autoencoder.ckpt.epoch_10_avg_loss_2.28610_recon_2.2782_commit_0.0787.pt',
    #     'use_residual_lfq': False,
    #     'use_residual_simvq': True,
    #     'description': 'Residual Similarity-based Vector Quantization'
    # },
    # {
    #     'name': 'ResidualSimVQ chk 20',
    #     # 'model_path': project_root / 'Model/residualsimvq/mesh_encoder_residualsimvq_chairs_final_20251028_073627.pt',
    #     'model_path': project_root / 'checkpoints_chairs/residualsimvq/mesh-autoencoder.ckpt.epoch_20_avg_loss_2.03218_recon_2.0223_commit_0.0984.pt',
    #     'use_residual_lfq': False,
    #     'use_residual_simvq': True,
    #     'description': 'Residual Similarity-based Vector Quantization'
    # },
    {
        'name': 'ResidualSimVQ',
        # 'model_path': project_root / 'Model/residualsimvq/mesh_encoder_residualsimvq_chairs_final_20251028_073627.pt',
        'model_path': project_root / 'checkpoints_chairs/residualsimvq/mesh-autoencoder.ckpt.epoch_40_avg_loss_0.66832_recon_0.6570_commit_0.1132.pt',
        'use_residual_lfq': False,
        'use_residual_simvq': True,
        'description': 'Residual Similarity-based Vector Quantization'
    },
    # {
    #     'name': 'ResidualSimVQ chk 40',
    #     # 'model_path': project_root / 'Model/residualsimvq/mesh_encoder_residualsimvq_chairs_final_20251028_073627.pt',
    #     'model_path': project_root / 'checkpoints_chairs/residualsimvq/mesh-autoencoder.ckpt.epoch_40_avg_loss_0.66832_recon_0.6570_commit_0.1132.pt',
    #     'use_residual_lfq': False,
    #     'use_residual_simvq': True,
    #     'description': 'Residual Similarity-based Vector Quantization'
    # }
]

# Model Architecture Configuration (should match training)
MODEL_CONFIG = {
    'decoder_dims_through_depth': (128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
    'codebook_size': 16384,
    'dim_codebook': 192,
    'dim_area_embed': 16,
    'dim_coor_embed': 16,
    'dim_normal_embed': 16,
    'dim_angle_embed': 8,
    'attn_decoder_depth': 4,
    'attn_encoder_depth': 2,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_mesh_to_obj(vertices, faces, filepath):
    """
    Save mesh data to OBJ file format.

    Args:
        vertices: Tensor of shape [n_verts, 3]
        faces: Tensor of shape [n_faces, 3] with vertex indices
        filepath: Path to save the OBJ file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True, parents=True)

    # Convert to numpy if needed
    if torch.is_tensor(vertices):
        vertices = vertices.cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.cpu().numpy()

    with open(filepath, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write faces (OBJ format is 1-indexed)
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"Saved mesh to: {filepath}")


def coords_to_mesh_data(coords, mask):
    """
    Convert face coordinates to vertices and faces format.

    Args:
        coords: Tensor of shape [batch, n_faces, 3, 3] (face coordinates)
        mask: Boolean mask for valid faces

    Returns:
        vertices: Unique vertices [n_verts, 3]
        faces: Face indices [n_faces, 3]
    """
    # Remove batch dimension and filter by mask
    coords = coords[0]  # [n_faces, 3, 3]
    mask = mask[0]  # [n_faces]

    coords = coords[mask]  # Filter valid faces

    # Reshape to [n_faces * 3, 3]
    coords_flat = coords.reshape(-1, 3)

    # Find unique vertices and create face indices
    unique_verts, inverse_indices = torch.unique(coords_flat, dim=0, return_inverse=True)
    faces = inverse_indices.reshape(-1, 3)

    return unique_verts, faces


def calculate_additional_metrics(original, reconstructed):
    """
    Calculate additional evaluation metrics beyond MSE.

    Args:
        original: Original coordinates
        reconstructed: Reconstructed coordinates

    Returns:
        Dictionary of metrics
    """
    # Flatten for computation
    orig_flat = original.view(-1, 3).cpu()
    recon_flat = reconstructed.view(-1, 3).cpu()

    # MSE
    mse = torch.mean((orig_flat - recon_flat) ** 2).item()

    # RMSE
    rmse = torch.sqrt(torch.tensor(mse)).item()

    # MAE (Mean Absolute Error)
    mae = torch.mean(torch.abs(orig_flat - recon_flat)).item()

    # Per-axis errors
    per_axis_mse = torch.mean((orig_flat - recon_flat) ** 2, dim=0).tolist()

    # Maximum error
    max_error = torch.max(torch.abs(orig_flat - recon_flat)).item()

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'per_axis_mse': {
            'x': per_axis_mse[0],
            'y': per_axis_mse[1],
            'z': per_axis_mse[2]
        }
    }


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_model_variant(
        variant_config,
        dataset,
        evaluation_config,
        model_config,
        shuffled_data
):
    """
    Evaluate a single model variant and save results.

    Args:
        variant_config: Dictionary with model configuration
        dataset: Dataset to evaluate on
        evaluation_config: Evaluation parameters
        model_config: Model architecture configuration

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'=' * 80}")
    print(f"Evaluating: {variant_config['name']}")
    print(f"Description: {variant_config['description']}")
    print(f"Model path: {variant_config['model_path']}")
    print(f"{'=' * 80}\n")

    # Create results folder for this variant
    variant_folder = Path(evaluation_config['results_base_folder']) / variant_config['name'].lower()
    variant_folder.mkdir(exist_ok=True, parents=True)

    # Initialize model
    autoencoder = MeshAutoencoder(
        **model_config,
        use_residual_lfq=variant_config['use_residual_lfq'],
        use_residual_simvq=variant_config['use_residual_simvq']
    ).to("cuda")

    # Initialize trainer and load model
    autoencoder_trainer = MeshAutoencoderTrainer(
        model=autoencoder,
        dataset=dataset,
        warmup_steps=10,
        num_train_steps=1,
        batch_size=16,
        grad_accum_every=4
    )

    try:
        autoencoder_trainer.load(variant_config['model_path'])
        print(f"✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load model: {str(e)}")
        return None

    # Set model to evaluation mode
    autoencoder.eval()

    # Initialize tracking variables
    min_mse, max_mse = float('inf'), float('-inf')
    min_data, max_data = None, None
    random_samples_pred, random_samples_recon = [], []
    all_visualization_samples = []
    total_metrics = defaultdict(float)
    per_category_metrics = defaultdict(lambda: {'count': 0, 'total_mse': 0.0})
    sample_size = evaluation_config['sample_size']

    print(f"Evaluating on {sample_size} samples...\n")

    codebook_usage = []
    num_quantizers = getattr(autoencoder, "num_quantizers", 1)
    codebook_size = getattr(autoencoder, "codebook_size", None)
    device = next(autoencoder.parameters()).device

    if codebook_size is not None:
        codebook_usage = [
            torch.zeros(codebook_size, dtype=torch.long, device=device)
            for _ in range(num_quantizers)
        ]

    # Evaluate samples
    with torch.no_grad():
        for idx, item in enumerate(tqdm(shuffled_data[:sample_size], desc="Evaluating")):
            # Tokenize
            codes = autoencoder.tokenize(
                vertices=item['vertices'],
                faces=item['faces'],
                face_edges=item['face_edges']
            )

            codes = codes.flatten().unsqueeze(0)
            codes = codes[:, :codes.shape[-1] // autoencoder.num_quantizers * autoencoder.num_quantizers]

            if codebook_usage:
                # If autoencoder.tokenize returns shape [B, num_quantizers, N]
                if codes.ndim == 3:
                    for i in range(autoencoder.num_quantizers):
                        idx = codes[:, i, :].flatten()
                        usage = torch.bincount(idx, minlength=autoencoder.codebook_size)
                        codebook_usage[i][:usage.shape[0]] += usage.to(codebook_usage[i].device)
                # Fallback if flatten() already collapsed quantizers
                else:
                    usage = torch.bincount(codes.flatten(), minlength=autoencoder.codebook_size)
                    codebook_usage[0][:usage.shape[0]] += usage.to(codebook_usage[0].device)

            # Decode
            coords, mask = autoencoder.decode_from_codes_to_faces(codes)
            orgs = item['vertices'][item['faces']].unsqueeze(0)

            # Calculate metrics
            metrics = calculate_additional_metrics(orgs, coords)
            mse = metrics['mse']

            # Update totals
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        total_metrics[f"{key}_{subkey}"] += subvalue
                else:
                    total_metrics[key] += value

            # Track per-category metrics
            category = item.get('texts', 'unknown')
            per_category_metrics[category]['count'] += 1
            per_category_metrics[category]['total_mse'] += mse

            # Track min/max MSE
            if mse < min_mse:
                min_mse = mse
                min_data = {
                    'coords': coords,
                    'orgs': orgs,
                    'mask': mask,
                    'item': item,
                    'metrics': metrics
                }

            if mse > max_mse:
                max_mse = mse
                max_data = {
                    'coords': coords,
                    'orgs': orgs,
                    'mask': mask,
                    'item': item,
                    'metrics': metrics
                }

            # Collect samples for visualization
            if len(random_samples_recon) < evaluation_config['random_visualization_samples']:
                random_samples_pred.append(orgs)
                random_samples_recon.append(coords)
            else:
                all_visualization_samples.extend([random_samples_pred, random_samples_recon])
                random_samples_pred, random_samples_recon = [], []

    # Calculate average metrics
    avg_metrics = {key: value / sample_size for key, value in total_metrics.items()}

    # Calculate per-category averages
    per_category_avg = {
        category: data['total_mse'] / data['count']
        for category, data in per_category_metrics.items()
    }

    # --- Finalize codebook usage statistics ---
    codebook_usage_stats = {}
    if codebook_usage:
        stats = []
        for i, usage in enumerate(codebook_usage):
            active = (usage > 0).sum().item()
            utilization = active / usage.numel() * 100
            stats.append({
                "quantizer_index": i,
                "num_embeddings": usage.numel(),
                "active_embeddings": active,
                "utilization_percent": round(utilization, 2)
            })
            print(f"Quantizer {i}: {active}/{usage.numel()} used ({utilization:.2f}%)")

        codebook_usage_stats = {
            "per_quantizer": stats,
            "average_utilization": round(
                sum(s["utilization_percent"] for s in stats) / len(stats), 2
            )
        }

    # Prepare results summary
    results = {
        'variant_name': variant_config['name'],
        'description': variant_config['description'],
        'model_path': str(variant_config['model_path']),
        'evaluation_date': datetime.now().isoformat(),
        'sample_size': sample_size,
        'average_metrics': avg_metrics,
        'min_mse': min_mse,
        'max_mse': max_mse,
        'per_category_mse': per_category_avg,
        'total_parameters': sum(p.numel() for p in autoencoder.parameters()),
        "codebook_usage": codebook_usage_stats
    }

    # # Save results to JSON
    # results_file = variant_folder / 'evaluation_results.json'
    # with open(results_file, 'w') as f:
    #     json.dump(results, f, indent=4)
    # print(f"\n✓ Results saved to: {results_file}")
    #
    # # Save best and worst reconstruction meshes
    # if evaluation_config['save_individual_meshes'] and min_data and max_data:
    #     # Best reconstruction
    #     best_verts_orig, best_faces_orig = coords_to_mesh_data(min_data['orgs'], min_data['mask'])
    #     best_verts_recon, best_faces_recon = coords_to_mesh_data(min_data['coords'], min_data['mask'])
    #
    #     save_mesh_to_obj(best_verts_orig, best_faces_orig, variant_folder / 'best_original.obj')
    #     save_mesh_to_obj(best_verts_recon, best_faces_recon, variant_folder / 'best_reconstructed.obj')
    #
    #     # Worst reconstruction
    #     worst_verts_orig, worst_faces_orig = coords_to_mesh_data(max_data['orgs'], max_data['mask'])
    #     worst_verts_recon, worst_faces_recon = coords_to_mesh_data(max_data['coords'], max_data['mask'])
    #
    #     save_mesh_to_obj(worst_verts_orig, worst_faces_orig, variant_folder / 'worst_original.obj')
    #     save_mesh_to_obj(worst_verts_recon, worst_faces_recon, variant_folder / 'worst_reconstructed.obj')
    #
    #     print(f"✓ Best/Worst meshes saved to: {variant_folder}")

    # Save visualization samples if available
    # if all_visualization_samples and hasattr(mesh_render, 'combind_mesh_with_rows'):
    #     viz_file = variant_folder / 'visualization_samples.obj'
    #     mesh_render.combind_mesh_with_rows(str(viz_file), all_visualization_samples)
    #     print(f"✓ Visualization samples saved to: {viz_file}")

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"Evaluation Summary - {variant_config['name']}")
    print(f"{'=' * 80}")
    print(f"Average MSE: {avg_metrics['mse']:.10f}")
    print(f"Average RMSE: {avg_metrics['rmse']:.10f}")
    print(f"Average MAE: {avg_metrics['mae']:.10f}")
    print(f"Min MSE: {min_mse:.10f}")
    print(f"Max MSE: {max_mse:.10f}")
    print(f"Max Error: {avg_metrics['max_error']:.10f}")
    print(f"\nPer-axis MSE:")
    print(f"  X: {avg_metrics['per_axis_mse_x']:.10f}")
    print(f"  Y: {avg_metrics['per_axis_mse_y']:.10f}")
    print(f"  Z: {avg_metrics['per_axis_mse_z']:.10f}")
    print(f"\nPer-category MSE:")
    for category, mse in per_category_avg.items():
        count = per_category_metrics[category]['count']
        print(f"  {category}: {mse:.10f} ({count} samples)")
    print(f"{'=' * 80}\n")

    # Clean up
    del autoencoder
    del autoencoder_trainer
    torch.cuda.empty_cache()

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main evaluation function."""

    print(f"\n{'=' * 80}")
    print("Mesh Autoencoder Quantizer Evaluation")
    print(f"{'=' * 80}\n")

    # # Set random seed
    # random.seed(EVALUATION_CONFIG['seed'])
    # torch.manual_seed(EVALUATION_CONFIG['seed'])
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(EVALUATION_CONFIG['seed'])

    # Load dataset
    print(f"Loading dataset from: {EVALUATION_CONFIG['dataset_path']}")
    dataset = MeshDataset.load(EVALUATION_CONFIG['dataset_path'])
    print(f"✓ Dataset loaded: {len(dataset)} samples\n")

    # Shuffle dataset for random sampling
    # random.seed(evaluation_config['seed'])
    shuffled_data = dataset.data.copy()
    random.shuffle(shuffled_data)

    # Create base results folder
    results_base = Path(EVALUATION_CONFIG['results_base_folder'])
    results_base.mkdir(exist_ok=True, parents=True)

    # Evaluate all variants
    all_results = []

    for variant in MODEL_VARIANTS:
        try:
            result = evaluate_model_variant(
                variant_config=variant,
                dataset=dataset,
                evaluation_config=EVALUATION_CONFIG,
                model_config=MODEL_CONFIG,
                shuffled_data=shuffled_data
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n✗ Failed to evaluate {variant['name']}: {str(e)}\n")

    # Save comparative summary
    if all_results:
        comparison_file = results_base / 'comparison_summary.json'
        comparison_data = {
            'evaluation_date': datetime.now().isoformat(),
            'configuration': EVALUATION_CONFIG,
            'results': all_results
        }

        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=4)

        print(f"\n{'=' * 80}")
        print("Comparative Summary")
        print(f"{'=' * 80}\n")

        # Sort by average MSE
        sorted_results = sorted(all_results, key=lambda x: x['average_metrics']['mse'])

        print(f"{'Rank':<6} {'Variant':<20} {'Avg MSE':<15} {'Avg RMSE':<15} {'Avg MAE':<15}")
        print("-" * 80)
        for rank, result in enumerate(sorted_results, 1):
            print(f"{rank:<6} {result['variant_name']:<20} "
                  f"{result['average_metrics']['mse']:<15.10f} "
                  f"{result['average_metrics']['rmse']:<15.10f} "
                  f"{result['average_metrics']['mae']:<15.10f}")

        print(f"\n✓ Comparison summary saved to: {comparison_file}")
        print(f"✓ All results saved to: {results_base}\n")

    print("Evaluation completed!")


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()