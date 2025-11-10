"""
Dataset Metadata Extraction Script
Extracts comprehensive statistics for thesis documentation.
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import json
import pandas as pd
from tqdm import tqdm
import hashlib

from data_scripts.Mesh_dataset import MeshDataset
from model_scripts.autoencoder import MeshAutoencoder


def compute_mesh_hash(vertices, faces):
    """
    Compute hash of mesh for duplicate detection.

    Args:
        vertices: vertex tensor or array
        faces: face tensor or array

    Returns:
        hash string
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()

    # Round to avoid floating point precision issues
    vertices_rounded = np.round(vertices, decimals=6)

    # Create hash from vertices and faces
    mesh_data = np.concatenate([vertices_rounded.flatten(), faces.flatten()])
    mesh_hash = hashlib.md5(mesh_data.tobytes()).hexdigest()

    return mesh_hash


def compute_mesh_complexity_metrics(vertices, faces):
    """
    Compute mesh complexity metrics.

    Args:
        vertices: vertex tensor (N, 3)
        faces: face tensor (F, 3)

    Returns:
        dict with complexity metrics
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()

    num_vertices = len(vertices)
    num_faces = len(faces)

    # Compute bounding box volume
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    bbox_dims = max_coords - min_coords
    bbox_volume = np.prod(bbox_dims)

    # Compute surface area (approximate)
    surface_area = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face]
        # Triangle area using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(cross)
        surface_area += area

    # Compute mesh density
    mesh_density = num_faces / (bbox_volume + 1e-8)

    # Compute average edge length
    edge_lengths = []
    for face in faces:
        v0, v1, v2 = vertices[face]
        edge_lengths.append(np.linalg.norm(v1 - v0))
        edge_lengths.append(np.linalg.norm(v2 - v1))
        edge_lengths.append(np.linalg.norm(v0 - v2))

    avg_edge_length = np.mean(edge_lengths) if edge_lengths else 0.0
    std_edge_length = np.std(edge_lengths) if edge_lengths else 0.0

    return {
        'bbox_volume': float(bbox_volume),
        'surface_area': float(surface_area),
        'mesh_density': float(mesh_density),
        'avg_edge_length': float(avg_edge_length),
        'std_edge_length': float(std_edge_length),
        'bbox_dimensions': bbox_dims.tolist()
    }


def extract_dataset_metadata(dataset, autoencoder=None, output_dir="dataset_analysis"):
    """
    Extract comprehensive metadata from dataset.

    Args:
        dataset: MeshDataset object
        autoencoder: MeshAutoencoder for token computation (optional)
        output_dir: directory to save results

    Returns:
        dict with all metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"Dataset Metadata Extraction")
    print(f"{'=' * 80}\n")

    # Initialize tracking structures
    category_stats = defaultdict(lambda: {
        'count': 0,
        'total_vertices': 0,
        'total_faces': 0,
        'total_tokens': 0,
        'vertices_list': [],
        'faces_list': [],
        'tokens_list': [],
        'mesh_hashes': set(),
        'unique_count': 0,
        'complexity_metrics': {
            'bbox_volumes': [],
            'surface_areas': [],
            'mesh_densities': [],
            'avg_edge_lengths': [],
            'std_edge_lengths': []
        }
    })

    global_mesh_hashes = set()
    total_unique_meshes = 0
    processing_errors = []

    # Process each mesh
    print("Processing dataset...")
    for idx, item in enumerate(tqdm(dataset.data)):
        try:
            category = item.get('texts', 'unknown')
            vertices = item['vertices']
            faces = item['faces']

            # Convert to numpy if tensor
            if isinstance(vertices, torch.Tensor):
                vertices_np = vertices.cpu().numpy()
            else:
                vertices_np = vertices

            if isinstance(faces, torch.Tensor):
                faces_np = faces.cpu().numpy()
            else:
                faces_np = faces

            num_vertices = len(vertices_np)
            num_faces = len(faces_np)

            # Compute mesh hash for uniqueness
            mesh_hash = compute_mesh_hash(vertices_np, faces_np)

            # Update global uniqueness
            if mesh_hash not in global_mesh_hashes:
                global_mesh_hashes.add(mesh_hash)
                total_unique_meshes += 1

            # Update category uniqueness
            if mesh_hash not in category_stats[category]['mesh_hashes']:
                category_stats[category]['mesh_hashes'].add(mesh_hash)
                category_stats[category]['unique_count'] += 1

            # Compute tokens if autoencoder provided
            num_tokens = 0
            if autoencoder is not None:
                try:
                    face_edges = item.get('face_edges', None)
                    codes = autoencoder.tokenize(
                        vertices=vertices,
                        faces=faces,
                        face_edges=face_edges
                    )
                    num_tokens = codes.numel()
                    category_stats[category]['tokens_list'].append(num_tokens)
                    category_stats[category]['total_tokens'] += num_tokens
                except Exception as e:
                    processing_errors.append({
                        'index': idx,
                        'category': category,
                        'error': f"Tokenization error: {str(e)}"
                    })

            # Update category statistics
            category_stats[category]['count'] += 1
            category_stats[category]['total_vertices'] += num_vertices
            category_stats[category]['total_faces'] += num_faces
            category_stats[category]['vertices_list'].append(num_vertices)
            category_stats[category]['faces_list'].append(num_faces)

            # Compute complexity metrics
            try:
                complexity = compute_mesh_complexity_metrics(vertices_np, faces_np)
                category_stats[category]['complexity_metrics']['bbox_volumes'].append(complexity['bbox_volume'])
                category_stats[category]['complexity_metrics']['surface_areas'].append(complexity['surface_area'])
                category_stats[category]['complexity_metrics']['mesh_densities'].append(complexity['mesh_density'])
                category_stats[category]['complexity_metrics']['avg_edge_lengths'].append(complexity['avg_edge_length'])
                category_stats[category]['complexity_metrics']['std_edge_lengths'].append(complexity['std_edge_length'])
            except Exception as e:
                processing_errors.append({
                    'index': idx,
                    'category': category,
                    'error': f"Complexity computation error: {str(e)}"
                })

        except Exception as e:
            processing_errors.append({
                'index': idx,
                'category': item.get('texts', 'unknown'),
                'error': f"General processing error: {str(e)}"
            })
            continue

    print(f"\nProcessed {len(dataset.data)} meshes")
    print(f"Errors encountered: {len(processing_errors)}")

    # Compute aggregated statistics
    print("\nComputing statistics...")
    results = {
        'dataset_summary': {
            'total_meshes': len(dataset.data),
            'total_unique_meshes': total_unique_meshes,
            'duplicate_meshes': len(dataset.data) - total_unique_meshes,
            'duplicate_percentage': ((len(dataset.data) - total_unique_meshes) / len(dataset.data) * 100) if len(
                dataset.data) > 0 else 0,
            'num_categories': len(category_stats),
            'processing_errors': len(processing_errors)
        },
        'categories': {},
        'processing_errors': processing_errors[:100]  # Store first 100 errors
    }

    # Aggregate category statistics
    category_table = []

    for category, stats in category_stats.items():
        # Remove hash sets (not JSON serializable)
        stats.pop('mesh_hashes', None)

        # Compute statistics
        avg_vertices = stats['total_vertices'] / stats['count'] if stats['count'] > 0 else 0
        avg_faces = stats['total_faces'] / stats['count'] if stats['count'] > 0 else 0
        avg_tokens = stats['total_tokens'] / stats['count'] if stats['count'] > 0 and stats['total_tokens'] > 0 else 0

        # Compute percentiles
        vertices_percentiles = np.percentile(stats['vertices_list'], [25, 50, 75]) if stats['vertices_list'] else [0, 0,
                                                                                                                   0]
        faces_percentiles = np.percentile(stats['faces_list'], [25, 50, 75]) if stats['faces_list'] else [0, 0, 0]

        # Complexity metrics
        complexity_stats = {}
        for metric_name, values in stats['complexity_metrics'].items():
            if values:
                complexity_stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
            else:
                complexity_stats[metric_name] = {
                    'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0
                }

        category_info = {
            'category_name': category,
            'total_meshes': stats['count'],
            'unique_meshes': stats['unique_count'],
            'duplicate_meshes': stats['count'] - stats['unique_count'],
            'duplicate_percentage': ((stats['count'] - stats['unique_count']) / stats['count'] * 100) if stats[
                                                                                                             'count'] > 0 else 0,

            # Vertex statistics
            'avg_vertices': round(avg_vertices, 2),
            'min_vertices': int(min(stats['vertices_list'])) if stats['vertices_list'] else 0,
            'max_vertices': int(max(stats['vertices_list'])) if stats['vertices_list'] else 0,
            'std_vertices': round(np.std(stats['vertices_list']), 2) if stats['vertices_list'] else 0,
            'median_vertices': round(vertices_percentiles[1], 2),
            'q1_vertices': round(vertices_percentiles[0], 2),
            'q3_vertices': round(vertices_percentiles[2], 2),

            # Face statistics
            'avg_faces': round(avg_faces, 2),
            'min_faces': int(min(stats['faces_list'])) if stats['faces_list'] else 0,
            'max_faces': int(max(stats['faces_list'])) if stats['faces_list'] else 0,
            'std_faces': round(np.std(stats['faces_list']), 2) if stats['faces_list'] else 0,
            'median_faces': round(faces_percentiles[1], 2),
            'q1_faces': round(faces_percentiles[0], 2),
            'q3_faces': round(faces_percentiles[2], 2),

            # Token statistics
            'avg_tokens': round(avg_tokens, 2),
            'total_tokens': stats['total_tokens'],
            'min_tokens': int(min(stats['tokens_list'])) if stats['tokens_list'] else 0,
            'max_tokens': int(max(stats['tokens_list'])) if stats['tokens_list'] else 0,
            'std_tokens': round(np.std(stats['tokens_list']), 2) if stats['tokens_list'] else 0,

            # Complexity metrics
            'complexity_metrics': complexity_stats,

            # Percentage of dataset
            'percentage_of_dataset': round((stats['count'] / len(dataset.data) * 100), 2) if len(
                dataset.data) > 0 else 0
        }

        results['categories'][category] = category_info
        category_table.append(category_info)

    # Sort categories by count
    category_table.sort(key=lambda x: x['total_meshes'], reverse=True)

    # Create summary statistics
    all_vertices = [v for stats in category_stats.values() for v in stats['vertices_list']]
    all_faces = [f for stats in category_stats.values() for f in stats['faces_list']]
    all_tokens = [t for stats in category_stats.values() for t in stats['tokens_list']]

    results['dataset_summary'].update({
        'total_vertices': sum(all_vertices),
        'avg_vertices': round(np.mean(all_vertices), 2) if all_vertices else 0,
        'median_vertices': round(np.median(all_vertices), 2) if all_vertices else 0,
        'std_vertices': round(np.std(all_vertices), 2) if all_vertices else 0,

        'total_faces': sum(all_faces),
        'avg_faces': round(np.mean(all_faces), 2) if all_faces else 0,
        'median_faces': round(np.median(all_faces), 2) if all_faces else 0,
        'std_faces': round(np.std(all_faces), 2) if all_faces else 0,

        'total_tokens': sum(all_tokens),
        'avg_tokens': round(np.mean(all_tokens), 2) if all_tokens else 0,
        'median_tokens': round(np.median(all_tokens), 2) if all_tokens else 0,
        'std_tokens': round(np.std(all_tokens), 2) if all_tokens else 0,
    })

    # Save results
    print("\nSaving results...")

    # Save JSON with full details
    json_path = output_dir / "dataset_metadata_full.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved full metadata: {json_path}")

    # Create and save category table as CSV
    df_category = pd.DataFrame(category_table)
    csv_path = output_dir / "category_statistics.csv"
    df_category.to_csv(csv_path, index=False)
    print(f"Saved category statistics: {csv_path}")

    # Create and save formatted table for thesis
    thesis_columns = [
        'category_name', 'total_meshes', 'unique_meshes', 'duplicate_percentage',
        'avg_vertices', 'avg_faces', 'avg_tokens', 'total_tokens',
        'percentage_of_dataset'
    ]
    df_thesis = df_category[thesis_columns]
    thesis_path = output_dir / "thesis_table.csv"
    df_thesis.to_csv(thesis_path, index=False)
    print(f"Saved thesis table: {thesis_path}")

    # Create LaTeX table
    latex_table = df_thesis.to_latex(
        index=False,
        float_format="%.2f",
        caption="Dataset Statistics by Category",
        label="tab:dataset_stats"
    )
    latex_path = output_dir / "thesis_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table: {latex_path}")

    # Create summary report
    report_path = output_dir / "dataset_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        for key, value in results['dataset_summary'].items():
            if key != 'processing_errors':
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("CATEGORY BREAKDOWN (Top 20)\n")
        f.write("=" * 80 + "\n\n")

        for i, cat in enumerate(category_table[:20], 1):
            f.write(f"{i}. {cat['category_name'].upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"   Total Meshes: {cat['total_meshes']} ({cat['percentage_of_dataset']:.2f}% of dataset)\n")
            f.write(
                f"   Unique Meshes: {cat['unique_meshes']} ({cat['unique_meshes'] / cat['total_meshes'] * 100:.1f}% unique)\n")
            f.write(
                f"   Vertices: avg={cat['avg_vertices']:.1f}, range=[{cat['min_vertices']}-{cat['max_vertices']}]\n")
            f.write(f"   Faces: avg={cat['avg_faces']:.1f}, range=[{cat['min_faces']}-{cat['max_faces']}]\n")
            if cat['avg_tokens'] > 0:
                f.write(f"   Tokens: avg={cat['avg_tokens']:.1f}, total={cat['total_tokens']}\n")
            f.write("\n")

        if processing_errors:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"PROCESSING ERRORS ({len(processing_errors)} total)\n")
            f.write("=" * 80 + "\n\n")
            for err in processing_errors[:20]:
                f.write(f"Index {err['index']} ({err['category']}): {err['error']}\n")

    print(f"Saved detailed report: {report_path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Total Meshes: {results['dataset_summary']['total_meshes']}")
    print(f"Unique Meshes: {results['dataset_summary']['total_unique_meshes']}")
    print(f"Duplicate Rate: {results['dataset_summary']['duplicate_percentage']:.2f}%")
    print(f"Categories: {results['dataset_summary']['num_categories']}")
    print(f"\nAverage Vertices: {results['dataset_summary']['avg_vertices']:.2f}")
    print(f"Average Faces: {results['dataset_summary']['avg_faces']:.2f}")
    if results['dataset_summary']['avg_tokens'] > 0:
        print(f"Average Tokens: {results['dataset_summary']['avg_tokens']:.2f}")
        print(f"Total Tokens: {results['dataset_summary']['total_tokens']:,}")

    print(f"\n{'=' * 80}")
    print(f"TOP 10 CATEGORIES")
    print(f"{'=' * 80}")
    for i, cat in enumerate(category_table[:10], 1):
        print(
            f"{i:2d}. {cat['category_name']:40s} - {cat['total_meshes']:5d} meshes ({cat['percentage_of_dataset']:5.2f}%)")

    print(f"\n{'=' * 80}")
    print("Analysis complete! Files saved to:", output_dir)
    print("=" * 80)

    return results


# Main execution
if __name__ == "__main__":
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir
    working_dir = project_root / "Data"
    DATASET_PATH = working_dir / "meshData_org.npz"

    # Load dataset
    print("Loading dataset...")
    dataset = MeshDataset.load(DATASET_PATH)
    print(f"Dataset loaded: {len(dataset.data)} meshes\n")

    # Initialize autoencoder for token computation
    print("Initializing autoencoder for token computation...")
    try:
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
        model_path = project_root / "Model" / "mesh-encoder_16k_4_2.pt"
        if model_path.exists():
            pkg = torch.load(str(model_path))
            autoencoder.load_state_dict(pkg['model'])
            autoencoder.eval()
            print("Autoencoder loaded successfully\n")
        else:
            print(f"Warning: Model not found at {model_path}")
            print("Proceeding without token computation\n")
            autoencoder = None
    except Exception as e:
        print(f"Warning: Could not initialize autoencoder: {e}")
        print("Proceeding without token computation\n")
        autoencoder = None

    # Extract metadata
    with torch.no_grad():
        results = extract_dataset_metadata(
            dataset=dataset,
            autoencoder=autoencoder,
            output_dir="dataset_analysis"
        )

    print("\n✓ Metadata extraction complete!")
    print("\nGenerated files:")
    print("  - dataset_metadata_full.json  (complete statistics)")
    print("  - category_statistics.csv     (detailed category stats)")
    print("  - thesis_table.csv            (formatted for thesis)")
    print("  - thesis_table.tex            (LaTeX table)")
    print("  - dataset_report.txt          (human-readable report)")