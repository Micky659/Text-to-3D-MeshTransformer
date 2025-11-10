"""
Practical inference benchmark script ready to use with your existing code.
Drop this into your project and run it.
"""

import torch
import time
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import sys
import random
import gc

# Add your project paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
working_dir = project_root / "Data"
sys.path.append(str(script_dir))

warmup = 0

# Your imports
from data_scripts.Mesh_dataset import MeshDataset
from model_scripts.autoencoder import MeshAutoencoder
from model_scripts.transformer import MeshTransformer
from data_scripts.mesh_render_v2 import extract_vertices_from_mesh, extract_vertices_faces_from_mesh


# MeshGPT imports (adjust paths as needed)
# from model.pointnet import get_pointnet_classifier
# from dataset.quantized_soup import QuantizedSoupTripletsCreator
# from dataset.triangles import TriangleNodesWithFacesAndSequenceIndices
# from trainer import get_rvqvae_v0_decoder
# from model.transformer import QuantSoupTransformer

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


class SimpleInferenceTimer:
    """Simple timer with GPU sync."""

    def __init__(self):
        self.times = []

    def time_function(self, func, warmup=3, runs=100, category='chair', model='Ours'):
        """
        Time a function with warmup.

        Args:
            category:
            func: Function to time (should take no arguments)
            warmup: Number of warmup runs
            runs: Number of timed runs

        Returns:
            dict with timing statistics
        """
        # Warmup
        print(f"Warming up ({warmup} runs)...")
        for _ in range(warmup):
            try:
                func()
            except:
                pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        # Timed runs
        times = []
        tokens_generated = []
        success = 97

        output_dir = Path(category)
        output_dir.mkdir(exist_ok=True)

        model_dir = output_dir / model.replace(' ', '_')
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running benchmark ({runs} iterations)...")
        # for i in tqdm(range(runs)):
        while success < runs:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            start = time.perf_counter()

            try:
                result = func()
                if result is not None:
                    success += 1
                else:
                    print(result, "failed")
                    continue
            except Exception as e:
                print(f"Failed: {e}")
                continue

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

            if isinstance(result, tuple):
                vertices, num_tokens, mesh, faces = result[:4]
            else:
                vertices = result
                num_tokens = None
                mesh = None
                faces = None

            # Convert to numpy arrays
            vertices = np.asarray(vertices, dtype=np.float32)
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

            if num_tokens is not None:
                tokens_generated.append(num_tokens)

            if model == 'Ours':
                generated_path = model_dir / f"{category}_{success}.obj"
                save_obj_from_vertices(vertices, generated_path)
            elif model == 'MeshGPT_beam' or model == 'MeshGPT_topp':
                generated_path = model_dir / f"{category}_{success}.obj"
                mesh.export(generated_path)

        if len(times) == 0:
            return None


        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'median': np.median(times),
            'min': np.min(times),
            'max': np.max(times),
            'success_rate': success / runs * 100,
            'total_time': np.sum(times),
            'mean_tokens': np.mean(tokens_generated),
            'std_tokens': np.std(tokens_generated),
            'mean_tokens_per_second': np.mean(tokens_generated) / np.mean(times)
        }


def benchmark_your_model(num_samples=100, category="chair", temperature=1.0):
    """
    Benchmark your model - drop-in replacement for your generation code.
    """
    print("\n" + "=" * 80)
    print(f"BENCHMARKING YOUR MODEL: {category}")
    print("=" * 80 + "\n")

    DATASET_PATH = working_dir / "meshData_org.npz"

    # Load models
    print("Loading models...")
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

    pkg = torch.load(str(project_root / "Model" / "mesh-encoder_16k_4_2.pt"))
    autoencoder.load_state_dict(pkg['model'])

    dataset = MeshDataset.load(DATASET_PATH)
    max_seq = max(len(d["faces"]) for d in dataset.data if "faces" in d) * (
            autoencoder.num_vertices_per_face * autoencoder.num_quantizers)

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

    pkg = torch.load(str(project_root / "Model" / "mesh-transformer_16k_4_2.pt"))
    transformer.load_state_dict(pkg['model'])

    transformer.eval()
    autoencoder.eval()

    print("Models loaded!\n")

    # Define generation function
    def generate_one():
        with torch.no_grad():
            mesh_output = transformer.generate(
                texts=[category],
                temperature=temperature,
                cache_kv=True,  # Important for fair comparison!
            )
            vertices, faces = extract_vertices_faces_from_mesh(mesh_output)
            num_tokens = len(vertices) // 3 * 3 * autoencoder.num_quantizers

            return vertices, num_tokens, None, faces

    # Run benchmark
    timer = SimpleInferenceTimer()
    stats = timer.time_function(generate_one, warmup=warmup, runs=num_samples, category=category)

    if stats is None:
        print("Benchmark failed!")
        return None

    # Print results
    print(f"\n{'=' * 80}")
    print(f"RESULTS: YOUR MODEL")
    print(f"{'=' * 80}")
    print(f"Mean time:      {stats['mean']:.3f}s ± {stats['std']:.3f}s")
    print(f"Median time:    {stats['median']:.3f}s")
    print(f"Min/Max:        {stats['min']:.3f}s / {stats['max']:.3f}s")
    print(f"Throughput:     {1 / stats['mean']:.2f} meshes/second")
    print(f"Success rate:   {stats['success_rate']:.1f}%")
    print(f"Total time:     {stats['total_time']:.2f}s")
    print(f"{'=' * 80}\n")

    return stats


def benchmark_meshgpt_topp(
        config,
        num_samples=100,
        category="chair",
        temperature=1.0,
        top_p=0.95
):
    """
    Benchmark MeshGPT with top-p sampling.
    Requires MeshGPT code and models to be available.
    """
    print("\n" + "=" * 80)
    print(f"BENCHMARKING MESHGPT (TOP-P): {category}")
    print("=" * 80 + "\n")

    # Import MeshGPT components
    import omegaconf
    from MeshGPT.model.pointnet import get_pointnet_classifier
    from MeshGPT.dataset.quantized_soup import QuantizedSoupTripletsCreator
    from MeshGPT.dataset.triangles import TriangleNodesWithFacesAndSequenceIndices
    from MeshGPT.trainer import get_rvqvae_v0_decoder
    from MeshGPT.trainer.train_transformer import get_qsoup_model_config
    from MeshGPT.model.transformer import QuantSoupTransformer
    from MeshGPT.util.misc import get_parameters_from_state_dict
    import typing
    import numpy
    torch.serialization.add_safe_globals([
        omegaconf.dictconfig.DictConfig,
        omegaconf.base.ContainerMetadata,
        typing.Any,
        omegaconf.nodes.AnyNode,
        omegaconf.base.Metadata,
        numpy._core.multiarray.scalar,
    ])

    device = torch.device('cuda:0')
    vq_cfg = omegaconf.OmegaConf.load(Path(config.vq_resume).parents[1] / "config.yaml")
    dataset = TriangleNodesWithFacesAndSequenceIndices(config, 'train', True, True , config.ft_category)

    model_cfg = get_qsoup_model_config(config, vq_cfg.embed_levels)
    model = QuantSoupTransformer(model_cfg, vq_cfg)
    state_dict = torch.load(config.resume)
    print(state_dict["state_dict"].keys())
    sequencer = QuantizedSoupTripletsCreator(config, vq_cfg)
    model.load_state_dict(state_dict["state_dict"]["model"])
    model = model.to(device)
    model = model.eval()
    sequencer = sequencer.to(device)
    sequencer = sequencer.eval()
    decoder = get_rvqvae_v0_decoder(vq_cfg, config.vq_resume, device)
    pnet = get_pointnet_classifier().to(device)

    print("Models loaded!\n")

    prompt_num_faces = 4
    max_val_tokens = cfg.max_val_tokens

    # Define generation function
    def generate_one():
        with torch.no_grad():
            data = dataset.get(random.randint(0, len(dataset) - 1))
            soup_sequence, face_in_idx, face_out_idx, target = sequencer.get_completion_sequence(
                data.x.to(device),
                data.edge_index.to(device),
                data.faces.to(device),
                data.num_vertices,
                1 + 6 * prompt_num_faces
            )

            y = model.generate(
                soup_sequence, face_in_idx, face_out_idx, sequencer, max_val_tokens,
                temperature=temperature, top_k=None, top_p=top_p, use_kv_cache=True
            )

            if y is None:
                return None

            gen_vertices, gen_faces = sequencer.decode(y[0], decoder)

            import trimesh
            mesh = trimesh.Trimesh(vertices=gen_vertices, faces=gen_faces, process=False)
            if not pnet.classifier_guided_filter(mesh, cfg.ft_category):
                return None

            num_tokens = y.shape[-1]

            return gen_vertices, num_tokens, mesh

    # Run benchmark
    timer = SimpleInferenceTimer()
    stats = timer.time_function(generate_one, warmup=warmup, runs=num_samples, category=category, model="MeshGPT_topp")

    if stats is None:
        print("Benchmark failed!")
        return None

    # Print results
    print(f"\n{'=' * 80}")
    print(f"RESULTS: MESHGPT (TOP-P)")
    print(f"{'=' * 80}")
    print(f"Mean time:      {stats['mean']:.3f}s ± {stats['std']:.3f}s")
    print(f"Median time:    {stats['median']:.3f}s")
    print(f"Min/Max:        {stats['min']:.3f}s / {stats['max']:.3f}s")
    print(f"Throughput:     {1 / stats['mean']:.2f} meshes/second")
    print(f"Success rate:   {stats['success_rate']:.1f}%")
    print(f"Total time:     {stats['total_time']:.2f}s")
    print(f"{'=' * 80}\n")

    return stats


def benchmark_meshgpt_beam(
        num_samples=100,
        category="chair",
        beam_width=6
):
    """
    Benchmark MeshGPT with beam search.
    Similar to top-p but uses beam search generation.
    """
    print("\n" + "=" * 80)
    print(f"BENCHMARKING MESHGPT (BEAM SEARCH): {category}")
    print("=" * 80 + "\n")

    import omegaconf
    from model.pointnet import get_pointnet_classifier
    from dataset.quantized_soup import QuantizedSoupTripletsCreator
    from dataset.triangles import TriangleNodesWithFacesAndSequenceIndices
    from trainer import get_rvqvae_v0_decoder
    from trainer.train_transformer import get_qsoup_model_config
    from model.transformer import QuantSoupTransformer
    from util.misc import get_parameters_from_state_dict

    device = torch.device('cuda:0')

    # Load config and models (adjust paths!)
    config_path = Path("path/to/meshgpt/config.yaml")  # UPDATE THIS
    resume_path = Path("path/to/meshgpt/checkpoint.ckpt")  # UPDATE THIS

    cfg = omegaconf.OmegaConf.load(config_path)
    cfg.resume = str(resume_path)
    cfg.ft_category = category

    vq_cfg = omegaconf.OmegaConf.load(Path(cfg.vq_resume).parents[1] / "config.yaml")

    # Load models
    print("Loading MeshGPT models...")
    dataset = TriangleNodesWithFacesAndSequenceIndices(cfg, 'train', True, True, cfg.ft_category)
    model_cfg = get_qsoup_model_config(cfg, vq_cfg.embed_levels)
    model = QuantSoupTransformer(model_cfg, vq_cfg)
    state_dict = torch.load(cfg.resume, map_location="cpu")["state_dict"]
    sequencer = QuantizedSoupTripletsCreator(cfg, vq_cfg)
    model.load_state_dict(get_parameters_from_state_dict(state_dict, "model"))
    model = model.to(device).eval()
    sequencer = sequencer.to(device).eval()
    decoder = get_rvqvae_v0_decoder(vq_cfg, cfg.vq_resume, device)
    pnet = get_pointnet_classifier().to(device)

    print("Models loaded!\n")

    prompt_num_faces = 4
    max_val_tokens = cfg.max_val_tokens

    # Main difference is in generation:
    def generate_one():
        with torch.no_grad():
            data = dataset.get(random.randint(0, len(dataset) - 1))
            soup_sequence, face_in_idx, face_out_idx, target = sequencer.get_completion_sequence(
                data.x.to(device),
                data.edge_index.to(device),
                data.faces.to(device),
                data.num_vertices,
                1 + 6 * prompt_num_faces
            )

            y = model.generate_with_beamsearch(
                soup_sequence, face_in_idx, face_out_idx, sequencer,
                max_val_tokens, use_kv_cache=True, beam_width=beam_width
            )

            if y is None:
                return None

            gen_vertices, gen_faces = sequencer.decode(y[0], decoder)

            import trimesh
            mesh = trimesh.Trimesh(vertices=gen_vertices, faces=gen_faces, process=False)
            if not pnet.classifier_guided_filter(mesh, cfg.ft_category):
                return None

            num_tokens = y.shape[-1]

            return gen_vertices, num_tokens, mesh

    # Run benchmark
    timer = SimpleInferenceTimer()
    stats = timer.time_function(generate_one, warmup=5, runs=num_samples, category=category, model="MeshGPT_beam")

    if stats is None:
        print("Benchmark failed!")
        return None

    # Print results
    print(f"\n{'=' * 80}")
    print(f"RESULTS: MESHGPT (Beam Search)")
    print(f"{'=' * 80}")
    print(f"Mean time:      {stats['mean']:.3f}s ± {stats['std']:.3f}s")
    print(f"Median time:    {stats['median']:.3f}s")
    print(f"Min/Max:        {stats['min']:.3f}s / {stats['max']:.3f}s")
    print(f"Throughput:     {1 / stats['mean']:.2f} meshes/second")
    print(f"Success rate:   {stats['success_rate']:.1f}%")
    print(f"Total time:     {stats['total_time']:.2f}s")
    print(f"{'=' * 80}\n")

    return stats


def compare_all_models(num_samples=100, category="chair"):
    """
    Run all benchmarks and compare results.
    """
    print("\n" + "=" * 80)
    print("COMPLETE INFERENCE COMPARISON")
    print("=" * 80 + "\n")

    results = {}

    # 1. Your model
    print("[1/3] Testing your model...")
    results['YourModel'] = benchmark_your_model(
        num_samples=num_samples,
        category=category,
        temperature=1.0
    )

    # 2. MeshGPT top-p
    print("\n[2/3] Testing MeshGPT (top-p)...")
    try:
        results['MeshGPT_topp'] = benchmark_meshgpt_topp(
            num_samples=num_samples,
            category=category,
            temperature=1.0,
            top_p=0.95
        )
    except Exception as e:
        print(f"MeshGPT top-p benchmark failed: {e}")
        results['MeshGPT_topp'] = None

    # 3. MeshGPT beam search
    print("\n[3/3] Testing MeshGPT (beam search)...")
    try:
        results['MeshGPT_beam'] = benchmark_meshgpt_beam(
            num_samples=num_samples,
            category=category,
            beam_width=6
        )
    except Exception as e:
        print(f"MeshGPT beam search benchmark failed: {e}")
        results['MeshGPT_beam'] = None

    # Comparison table
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80 + "\n")

    print(f"{'Model':<25} {'Time/Mesh':<15} {'Throughput':<20} {'Success Rate':<15}")
    print("-" * 75)

    for name, stats in results.items():
        if stats is None:
            print(f"{name:<25} {'FAILED':<15}")
        else:
            print(f"{name:<25} {stats['mean']:.3f}s ± {stats['std']:.3f}s  "
                  f"{1 / stats['mean']:.2f} mesh/s        "
                  f"{stats['success_rate']:.1f}%")

    # Speedup analysis
    times = {name: stats['mean'] for name, stats in results.items() if stats is not None}
    if len(times) > 1:
        baseline = max(times.values())

        print("\n" + "=" * 80)
        print("SPEEDUP (relative to slowest)")
        print("=" * 80)

        for name, time_val in sorted(times.items(), key=lambda x: x[1]):
            speedup = baseline / time_val
            print(f"{name:<25} {speedup:.2f}x faster")

    print("\n" + "=" * 80 + "\n")

    # Save results
    output_file = f"inference_comparison_{category}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Simple usage - just run this script!
    """
    import omegaconf
    cfg = omegaconf.OmegaConf.load(Path(sys.argv[1]).parents[1] / "config.yaml")
    cfg.resume = sys.argv[1]
    cfg.padding = 0.0
    cfg.num_val_samples = int(sys.argv[3])
    cfg.sequence_stride = cfg.block_size
    cfg.top_p = 0.95
    cfg.temperature = 1.0
    cfg.low_augment = True

    # Option 1: Test only your model
    results = benchmark_your_model(num_samples=500, category="chair")
    output_file = f"inference_comparison_chair_OUR.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    #
    # results = benchmark_meshgpt_topp(cfg, num_samples=2, category="chair")
    # output_file = f"inference_comparison_chair_MESHGPT_TOPP.json"
    # with open(output_file, 'w') as f:
    #     json.dump(results, f, indent=2)

    #  python -m eval_scripts.evaluation_results.mano_y_mano.inference_eval_specific eval_scripts/evaluation_results/mano_y_mano/MeshGPT/pretrained/transformer_ft_03001627/checkpoints/2287-0-003.ckpt topp 5
    # results = benchmark_meshgpt_beam(cfg, num_samples=500, category="chair")
    # output_file = f"inference_comparison_chair_MESHGPT_BEAM.json"
    # with open(output_file, 'w') as f:
    #     json.dump(results, f, indent=2)

    # Option 2: Test all models and compare
    # compare_all_models(num_samples=100, category="chair")

    # Option 3: Test across multiple categories
    # for category in ['chair', 'table', 'lamp']:
    #     compare_all_models(num_samples=50, category=category)