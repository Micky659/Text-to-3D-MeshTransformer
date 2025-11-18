"""
MeshGPT Evaluation Framework
Implements metrics from the MeshGPT paper for comparing 3D mesh generation models.

Metrics Computed:
1. MMD (Minimum Matching Distance): Measures how well the generated distribution matches the real distribution
2. COV (Coverage): Measures diversity - what percentage of real samples are covered by generated samples
3. 1-NNA (1-Nearest Neighbor Accuracy): Measures overfitting - should be ~50% for good generalization
4. FID (Frechet Inception Distance): Measures visual quality via rendered images
5. KID (Kernel Inception Distance): More robust visual quality metric than FID
6. Compactness: Average number of vertices and faces
"""

import numpy as np
import torch
import trimesh
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm
from pathlib import Path
import json
from tqdm import tqdm
from typing import List, Dict, Tuple
import warnings
import random
import os

warnings.filterwarnings('ignore')

# Try to import PyTorch3D for rendering
try:
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
        MeshRasterizer, SoftPhongShader, PointLights, TexturesVertex
    )
    from pytorch3d.structures import Meshes

    PYTORCH3D_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch3D not available. FID/KID metrics will be skipped.")
    PYTORCH3D_AVAILABLE = False

# Try to import torchvision for InceptionV3
try:
    from torchvision.models import inception_v3
    from torchvision.transforms import Normalize, Resize, ToTensor

    INCEPTION_AVAILABLE = True
except ImportError:
    print("Warning: torchvision not available. FID/KID metrics will be skipped.")
    INCEPTION_AVAILABLE = False


# ============================================================================
# HYPERPARAMETERS
# ============================================================================

class EvaluationConfig:
    """
    Configuration for mesh evaluation metrics.

    Hyperparameters:
    ---------------
    N_GENERATED: int (default: 1000)
        Number of meshes to generate per category.
        Reasoning: 1000 samples provides robust statistics while being computationally feasible.
        The paper likely used similar numbers based on ShapeNet subset sizes.

    N_REFERENCE: int (default: 1000)
        Number of reference meshes to sample from dataset.
        Reasoning: Should match or exceed N_GENERATED for fair distribution comparison.

    N_VIEWPOINTS: int (default: 8)
        Number of camera viewpoints for rendering meshes.
        Reasoning: Paper explicitly uses 8 viewpoints to capture geometric structures.

    IMAGE_SIZE: int (default: 256)
        Resolution of rendered images for FID/KID computation.
        Reasoning: 256x256 balances quality and computational efficiency for Inception features.

    CD_NUM_SAMPLES: int (default: 2048)
        Number of points to sample from each mesh for Chamfer Distance computation.
        Reasoning: 2048 points provides good surface coverage while being efficient.
        Standard in point cloud literature.

    BATCH_SIZE: int (default: 32)
        Batch size for pairwise distance computations.
        Reasoning: Trades off memory usage and speed. Adjust based on available GPU memory.

    COV_THRESHOLD: float (default: None)
        Distance threshold for coverage metric. If None, uses median of pairwise distances.
        Reasoning: Adaptive threshold based on data distribution ensures meaningful coverage.

    DEVICE: str (default: 'cuda' if available)
        Computation device.
    """

    N_GENERATED = 500  # Number of meshes to generate per category
    N_REFERENCE = 500  # Number of reference meshes from dataset
    N_VIEWPOINTS = 8  # Number of camera angles for rendering
    IMAGE_SIZE = 256  # Resolution for FID/KID computation
    CD_NUM_SAMPLES = 2048  # Points to sample for Chamfer Distance
    BATCH_SIZE = 32  # Batch size for distance computations
    COV_THRESHOLD = None  # Auto-compute threshold if None
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# CHAMFER DISTANCE COMPUTATION
# ============================================================================

def compute_chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    Compute Chamfer Distance between two point clouds.

    CD = mean(min_distance(p1->p2)) + mean(min_distance(p2->p1))

    Lower is better - measures geometric similarity between shapes.
    """
    # Convert to torch for efficient computation
    p1 = torch.from_numpy(points1).float().to(EvaluationConfig.DEVICE)
    p2 = torch.from_numpy(points2).float().to(EvaluationConfig.DEVICE)

    # Compute pairwise distances
    dists = torch.cdist(p1.unsqueeze(0), p2.unsqueeze(0)).squeeze(0)

    # Chamfer distance: average of nearest neighbor distances in both directions
    cd = torch.mean(torch.min(dists, dim=1)[0]) + torch.mean(torch.min(dists, dim=0)[0])

    return cd.item()


def sample_points_from_mesh(vertices: np.ndarray, faces: np.ndarray,
                            global_center, global_scale, is_meshgpt: bool = False) -> np.ndarray:
    """
    Sample points uniformly from mesh surface.

    Args:
        vertices: (V, 3) array of vertex positions
        faces: (F, 3) array of face indices
        n_samples: Number of points to sample

    Returns:
        points: (n_samples, 3) array of sampled points
    """
    n_samples = EvaluationConfig.CD_NUM_SAMPLES

    # if is_meshgpt:
    #     # MeshGPT: forward=-Y, up=+X
    #     # Reference: forward=-Z, up=+Y
    R = np.array([
        [0, 0, 1],   # Reference X = MeshGPT Z
        [1, 0, 0],   # Reference Y = MeshGPT X
        [0, 1, 0]   # Reference Z = -MeshGPT Y
    ])
    vertices = vertices @ R.T

    vertices = (vertices - global_center) / global_scale

    np.random.seed(42)

    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        points, _ = trimesh.sample.sample_surface(mesh, n_samples)
        return points
    except Exception as e:
        print(f"Warning: Failed to sample from mesh: {e}")
        # Fallback: return vertex positions with duplication if needed
        if len(vertices) >= n_samples:
            indices = np.random.choice(len(vertices), n_samples, replace=False)
        else:
            indices = np.random.choice(len(vertices), n_samples, replace=True)
        return vertices[indices]


# ============================================================================
# SHAPE QUALITY METRICS (MMD, COV, 1-NNA)
# ============================================================================

def compute_pairwise_cd_matrix(meshes1: List[Dict], meshes2: List[Dict],
                               meshes1_is_meshgpt: bool = False,
                               meshes2_is_meshgpt: bool = False) -> np.ndarray:
    """
    Compute pairwise Chamfer Distance matrix between two sets of meshes.

    Args:
        meshes1: List of mesh dictionaries with 'vertices' and 'faces'
        meshes2: List of mesh dictionaries with 'vertices' and 'faces'

    Returns:
        distance_matrix: (len(meshes1), len(meshes2)) array of Chamfer Distances
    """
    n1, n2 = len(meshes1), len(meshes2)
    distances = np.zeros((n1, n2))

    all_vertices = []

    for m in meshes1:
        verts = m['vertices'].copy()
        if meshes1_is_meshgpt:
            R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            verts = verts @ R.T
        all_vertices.append(verts)

    for m in meshes2:
        verts = m['vertices'].copy()
        if meshes2_is_meshgpt:
            R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            verts = verts @ R.T
        all_vertices.append(verts)

    all_vertices = np.concatenate(all_vertices)
    global_min = all_vertices.min(axis=0)
    global_max = all_vertices.max(axis=0)
    global_center = (global_max + global_min) / 2
    global_scale = np.max(global_max - global_min)

    print("Computing pairwise Chamfer Distance matrix...")
    for i in tqdm(range(n1)):
        points1 = sample_points_from_mesh(meshes1[i]['vertices'], meshes1[i]['faces'], global_center, global_scale, is_meshgpt=meshes1_is_meshgpt)

        for j in range(n2):
            points2 = sample_points_from_mesh(meshes2[j]['vertices'], meshes2[j]['faces'], global_center, global_scale, is_meshgpt=meshes2_is_meshgpt)
            distances[i, j] = compute_chamfer_distance(points1, points2)

    return distances


def compute_mmd(dist_matrix, generated_meshes: List[Dict], reference_meshes: List[Dict]) -> float:
    """
    Minimum Matching Distance (MMD)

    Definition: Average of minimum distances from each generated sample to reference set.

    What it measures: How well the generated distribution matches the real distribution.
    Lower is better - indicates generated samples are close to real samples.

    Formula: MMD = mean(min_distance(gen_i, all_ref))

    Interpretation:
    - Low MMD: Generated samples are similar to real samples (good)
    - High MMD: Generated samples deviate from real distribution (bad)
    """
    mmd = np.mean(np.min(dist_matrix, axis=1))  # ← axis=0!
    return mmd * 1000


# def compute_coverage(dist_matrix, generated_meshes: List[Dict], reference_meshes: List[Dict],
#                      threshold: float = None) -> float:
#     """
#     Coverage (COV)
#
#     Definition: Percentage of reference samples that are "covered" by generated samples.
#     A reference sample is covered if at least one generated sample is within threshold distance.
#
#     What it measures: Diversity and completeness of the generated distribution.
#     Higher is better - indicates the model can generate diverse shapes covering the real distribution.
#
#     Interpretation:
#     - High COV (>40%): Model generates diverse shapes covering most of the real distribution (good)
#     - Low COV (<20%): Model has mode collapse, generating limited variety (bad)
#
#     Args:
#         threshold: Distance threshold for coverage. If None, uses median of pairwise distances.
#     """
#
#     # if threshold is None:
#     #     # Use median of all pairwise distances as adaptive threshold
#     #     threshold = np.median(dist_matrix)
#     #
#     # # For each reference sample, check if any generated sample is within threshold
#     # min_distances_to_ref = np.min(dist_matrix, axis=0)
#     # coverage = np.mean(min_distances_to_ref < 0.01) * 100
#     #
#     # return coverage
#
#     # For each generated sample, find its nearest reference sample
#     nn_indices = np.argmin(dist_matrix, axis=1)
#     # Count how many unique reference meshes were matched
#     unique_refs = len(np.unique(nn_indices))
#     cov = (unique_refs / dist_matrix.shape[1]) * 100
#     return cov

def compute_coverage(dist_matrix, generated_meshes, reference_meshes, threshold=None):
    """
    Coverage: What percentage of reference samples have at least one
    generated sample as their nearest neighbor?

    dist_matrix: (n_generated, n_reference)
    """
    # For each REFERENCE (column), find which GENERATED (row) is closest
    nearest_gen_to_ref = np.argmin(dist_matrix, axis=0)  # ← axis=0, not axis=1!

    # Count how many unique generated meshes are "covering" references
    # Actually, we want to count how many references are covered
    # Paper definition: for each ref, check if ANY gen is close enough

    if threshold is None:
        # Use median of all pairwise distances as adaptive threshold
        # threshold = np.median(dist_matrix)
        print(np.percentile(dist_matrix, 5))
        threshold = np.percentile(dist_matrix, 5)
        # threshold = 0.1

    # For each reference, get minimum distance to any generated mesh
    min_dist_to_ref = np.min(dist_matrix, axis=0)  # ← axis=0!

    # Count references that have at least one generated mesh within threshold
    covered = np.sum(min_dist_to_ref < threshold)
    cov = (covered / len(reference_meshes)) * 100

    return cov


def compute_1nna(generated_meshes: List[Dict], reference_meshes: List[Dict],
                 meshes1_is_meshgpt: bool = False,
                 meshes2_is_meshgpt: bool = False) -> float:
    """
    1-Nearest Neighbor Accuracy (1-NNA)

    Definition: Train a 1-NN classifier to distinguish generated from real samples.
    Accuracy of this classifier indicates overfitting.

    What it measures: Whether the model memorizes training data vs. generalizes.
    50% is optimal - indicates generated and real distributions are indistinguishable.

    Procedure:
    1. Combine generated and reference samples
    2. For each sample, find its nearest neighbor (excluding itself)
    3. Predict: if NN is from same set (gen/ref), accuracy increases

    Interpretation:
    - ~50%: Perfect - generated distribution matches real distribution (good)
    - >80%: Overfitting - model memorizes training data (bad)
    - <20%: Underfitting - generated samples are too different from real (bad)
    """
    n_gen = len(generated_meshes)
    n_ref = len(reference_meshes)

    # Compute distance matrix for all samples
    all_meshes = generated_meshes + reference_meshes
    n_total = len(all_meshes)

    all_vertices = []

    for m in generated_meshes:
        verts = m['vertices'].copy()
        if meshes1_is_meshgpt:
            R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            verts = verts @ R.T
        all_vertices.append(verts)

    for m in reference_meshes:
        verts = m['vertices'].copy()
        if meshes2_is_meshgpt:
            R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            verts = verts @ R.T
        all_vertices.append(verts)

    all_vertices = np.concatenate(all_vertices)
    global_min = all_vertices.min(axis=0)
    global_max = all_vertices.max(axis=0)
    global_center = (global_max + global_min) / 2
    global_scale = np.max(global_max - global_min)

    # Create labels: 0 for generated, 1 for reference
    labels = np.array([0] * n_gen + [1] * n_ref)

    print("Computing 1-NNA distances...")
    distances = np.zeros((n_total, n_total))

    for i in tqdm(range(n_total)):
        points_i = sample_points_from_mesh(all_meshes[i]['vertices'], all_meshes[i]['faces'], global_center, global_scale)

        for j in range(i + 1, n_total):
            points_j = sample_points_from_mesh(all_meshes[j]['vertices'], all_meshes[j]['faces'], global_center, global_scale)
            dist = compute_chamfer_distance(points_i, points_j)
            distances[i, j] = dist
            distances[j, i] = dist

    # Set diagonal to infinity to exclude self
    np.fill_diagonal(distances, np.inf)

    # Find nearest neighbor for each sample
    nearest_neighbors = np.argmin(distances, axis=1)

    # Predict label: if NN is from same class, predict same class
    predictions = labels[nearest_neighbors]

    # Compute accuracy
    accuracy = np.mean(predictions == labels) * 100

    return accuracy


# ============================================================================
# VISUAL QUALITY METRICS (FID, KID)
# ============================================================================

class MeshRenderer:
    """
    Renders meshes from multiple viewpoints for FID/KID computation.
    Uses PyTorch3D for efficient batch rendering with metallic material.
    """

    def __init__(self, image_size: int = 256, device: str = 'cuda'):
        self.image_size = image_size
        self.device = device
        self.n_viewpoints = EvaluationConfig.N_VIEWPOINTS

        if not PYTORCH3D_AVAILABLE:
            raise ImportError("PyTorch3D required for rendering. Install with: pip install pytorch3d")

        # Setup cameras at different viewpoints (8 evenly spaced around object)
        angles = np.linspace(0, 2 * np.pi, self.n_viewpoints, endpoint=False)
        elevations = [30] * self.n_viewpoints  # Fixed elevation

        R_list = []
        T_list = []
        for angle, elev in zip(angles, elevations):
            # Convert spherical to Cartesian
            elev_rad = np.radians(elev)
            x = 2.5 * np.cos(elev_rad) * np.cos(angle)
            y = 2.5 * np.sin(elev_rad)
            z = 2.5 * np.cos(elev_rad) * np.sin(angle)

            T_list.append([x, y, z])

            # Simple look-at rotation (pointing to origin)
            camera_pos = np.array([x, y, z])
            camera_dir = -camera_pos / np.linalg.norm(camera_pos)
            up = np.array([0, 1, 0])
            right = np.cross(up, camera_dir)
            right = right / np.linalg.norm(right)
            up = np.cross(camera_dir, right)

            R = np.stack([right, up, camera_dir], axis=0)
            R_list.append(R)

        R = torch.tensor(np.stack(R_list), dtype=torch.float32, device=device)
        T = torch.tensor(T_list, dtype=torch.float32, device=device)

        self.cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        # Rasterization settings (higher quality)
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Lighting (bright white light for metallic effect)
        lights = PointLights(device=device, location=[[0, 3, 0]])

        # Renderer
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=self.cameras, lights=lights)
        )

    def render_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Render mesh from multiple viewpoints.

        Returns:
            images: (n_viewpoints, 3, H, W) array of rendered images
        """
        # Normalize mesh to unit sphere
        vertices_centered = vertices - vertices.mean(axis=0)
        scale = np.abs(vertices_centered).max()
        vertices_normalized = vertices_centered / (scale + 1e-8)

        # Convert to PyTorch
        verts = torch.tensor(vertices_normalized, dtype=torch.float32, device=self.device)
        faces_t = torch.tensor(faces, dtype=torch.int64, device=self.device)

        # Create mesh with metallic gray texture
        verts_rgb = torch.ones_like(verts) * 0.7  # Gray metallic color
        textures = TexturesVertex(verts_features=verts_rgb.unsqueeze(0).repeat(self.n_viewpoints, 1, 1))

        mesh = Meshes(
            verts=[verts] * self.n_viewpoints,
            faces=[faces_t] * self.n_viewpoints,
            textures=textures
        )

        # Render
        images = self.renderer(mesh)  # (n_viewpoints, H, W, 4)
        images = images[..., :3]  # Remove alpha channel
        images = images.permute(0, 3, 1, 2)  # (n_viewpoints, 3, H, W)

        return images.cpu().numpy()


class InceptionFeatureExtractor:
    """
    Extracts features from InceptionV3 for FID/KID computation.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device

        if not INCEPTION_AVAILABLE:
            raise ImportError("torchvision required. Install with: pip install torchvision")

        # Load pre-trained InceptionV3
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.fc = torch.nn.Identity()  # Remove final classification layer
        self.model = self.model.to(device)
        self.model.eval()

        # ImageNet normalization
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract InceptionV3 features from images.

        Args:
            images: (N, 3, H, W) array of images in [0, 1]

        Returns:
            features: (N, 2048) array of features
        """
        images_tensor = torch.from_numpy(images).float().to(self.device)

        # Normalize
        images_normalized = torch.stack([self.normalize(img) for img in images_tensor])

        # Resize to 299x299 for InceptionV3
        if images_normalized.shape[-1] != 299:
            resize = Resize((299, 299))
            images_normalized = resize(images_normalized)

        # Extract features
        with torch.no_grad():
            features = self.model(images_normalized)

        return features.cpu().numpy()


def compute_fid(generated_features: np.ndarray, reference_features: np.ndarray) -> float:
    """
    Frechet Inception Distance (FID)

    Definition: Measures distance between generated and real distributions in feature space.
    Uses mean and covariance of Inception features.

    What it measures: Visual quality and diversity of generated meshes after rendering.
    Lower is better - indicates generated renders look similar to real renders.

    Formula: FID = ||μ_gen - μ_ref||² + Tr(Σ_gen + Σ_ref - 2√(Σ_gen·Σ_ref))

    Interpretation:
    - FID < 20: Excellent visual quality (like MeshGPT)
    - FID 20-50: Good quality
    - FID 50-100: Moderate quality
    - FID > 100: Poor quality

    Note: Sensitive to sample size. Requires hundreds of samples for stability.
    """
    # Compute mean and covariance
    mu_gen = np.mean(generated_features, axis=0)
    mu_ref = np.mean(reference_features, axis=0)

    sigma_gen = np.cov(generated_features, rowvar=False)
    sigma_ref = np.cov(reference_features, rowvar=False)

    # Compute FID
    diff = mu_gen - mu_ref

    # Product of covariances
    covmean = sqrtm(sigma_gen @ sigma_ref)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean)

    return fid


def compute_kid(generated_features: np.ndarray, reference_features: np.ndarray,
                subset_size: int = 1000) -> float:
    """
    Kernel Inception Distance (KID)

    Definition: More robust alternative to FID using polynomial kernel.
    Measures distance between distributions using maximum mean discrepancy.

    What it measures: Visual quality with better statistical properties than FID.
    Lower is better - indicates generated renders match real distribution.

    Advantages over FID:
    - Less biased with small sample sizes
    - More robust to outliers
    - Unbiased estimator

    Interpretation:
    - KID < 0.01: Excellent quality (like MeshGPT)
    - KID 0.01-0.05: Good quality
    - KID > 0.05: Poor quality

    Note: Values are typically much smaller than FID.
    """
    n_gen = len(generated_features)
    n_ref = len(reference_features)

    # Subsample if needed for computational efficiency
    if n_gen > subset_size:
        indices = np.random.choice(n_gen, subset_size, replace=False)
        generated_features = generated_features[indices]
        n_gen = subset_size

    if n_ref > subset_size:
        indices = np.random.choice(n_ref, subset_size, replace=False)
        reference_features = reference_features[indices]
        n_ref = subset_size

    # Polynomial kernel (degree 3)
    def polynomial_kernel(X, Y):
        return (1 + X @ Y.T / X.shape[1]) ** 3

    # Compute kernel matrices
    K_gen_gen = polynomial_kernel(generated_features, generated_features)
    K_ref_ref = polynomial_kernel(reference_features, reference_features)
    K_gen_ref = polynomial_kernel(generated_features, reference_features)

    # Remove diagonal for unbiased estimator
    np.fill_diagonal(K_gen_gen, 0)
    np.fill_diagonal(K_ref_ref, 0)

    # Compute KID
    kid = (np.sum(K_gen_gen) / (n_gen * (n_gen - 1)) +
           np.sum(K_ref_ref) / (n_ref * (n_ref - 1)) -
           2 * np.sum(K_gen_ref) / (n_gen * n_ref))

    return kid


# ============================================================================
# COMPACTNESS METRICS
# ============================================================================

def compute_compactness(meshes: List[Dict]) -> Tuple[float, float]:
    """
    Compactness Metrics: Average number of vertices and faces

    What it measures: Model efficiency - can it represent shapes with fewer polygons?
    Lower is better (more compact) - but must balance with quality.

    Interpretation:
    - MeshGPT achieves ~100-150 vertices and ~200-300 faces
    - This is much more compact than methods like GET3D (>10k vertices)
    - Compact meshes are faster to render and easier to edit

    Trade-off: Very low polygon counts may lose geometric detail.
    """
    n_vertices = [len(mesh['vertices']) for mesh in meshes]
    n_faces = [len(mesh['faces']) for mesh in meshes]

    avg_vertices = np.mean(n_vertices)
    avg_faces = np.mean(n_faces)

    return avg_vertices, avg_faces


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def load_meshes_from_obj_files(obj_dir: str, max_meshes: int = None) -> List[Dict]:
    """
    Load meshes from directory of OBJ files.

    Args:
        obj_dir: Directory containing .obj files
        max_meshes: Maximum number of meshes to load

    Returns:
        meshes: List of dictionaries with 'vertices' and 'faces'
    """
    obj_files = list(Path(obj_dir).glob("*.obj"))

    if max_meshes is not None:
        obj_files = random.sample(obj_files, max_meshes)
        # obj_files = obj_files[:max_meshes]

    meshes = []
    print(f"Loading {len(obj_files)} meshes from {obj_dir}...")

    for obj_file in tqdm(obj_files):
        try:
            mesh = trimesh.load(obj_file, force='mesh')
            meshes.append({
                'vertices': np.array(mesh.vertices, dtype=np.float32),
                'faces': np.array(mesh.faces, dtype=np.int32),
                'filename': obj_file.name
            })
        except Exception as e:
            print(f"Warning: Failed to load {obj_file}: {e}")

    return meshes


def evaluate_mesh_generation(
        generated_obj_dir: str,
        reference_obj_dir: str,
        output_path: str = "evaluation_results.json",
        compute_visual_metrics: bool = True
) -> Dict:
    """
    Complete evaluation pipeline for mesh generation.

    Args:
        generated_obj_dir: Directory with generated OBJ files
        reference_obj_dir: Directory with reference/ground truth OBJ files
        output_path: Where to save results JSON
        compute_visual_metrics: Whether to compute FID/KID (requires rendering)

    Returns:
        results: Dictionary with all metrics
    """
    # Load meshes
    print("=" * 80)
    print("LOADING MESHES")
    print("=" * 80)
    generated_meshes = load_meshes_from_obj_files(
        generated_obj_dir,
        max_meshes=EvaluationConfig.N_GENERATED
    )
    reference_meshes = load_meshes_from_obj_files(
        reference_obj_dir,
        max_meshes=EvaluationConfig.N_REFERENCE
    )

    print(f"\nLoaded {len(generated_meshes)} generated meshes")
    print(f"Loaded {len(reference_meshes)} reference meshes")

    results = {
        'config': {
            'n_generated': len(generated_meshes),
            'n_reference': len(reference_meshes),
            'cd_num_samples': EvaluationConfig.CD_NUM_SAMPLES,
            'n_viewpoints': EvaluationConfig.N_VIEWPOINTS,
            'image_size': EvaluationConfig.IMAGE_SIZE
        }
    }

    # Compute shape quality metrics
    print("\n" + "=" * 80)
    print("COMPUTING SHAPE QUALITY METRICS")
    print("=" * 80)

    print("\n[1/3] Computing MMD (Minimum Matching Distance)...")
    mmd = compute_mmd(generated_meshes, reference_meshes)
    results['MMD'] = float(mmd * 1000)  # Multiply by 1000 as per paper
    print(f"✓ MMD: {results['MMD']:.2f} (×10³)")

    print("\n[2/3] Computing Coverage (COV)...")
    cov = compute_coverage(generated_meshes, reference_meshes)
    results['COV'] = float(cov)
    print(f"✓ Coverage: {cov:.2f}%")

    print("\n[3/3] Computing 1-NNA (1-Nearest Neighbor Accuracy)...")
    nna = compute_1nna(generated_meshes, reference_meshes)
    results['1-NNA'] = float(nna)
    print(f"✓ 1-NNA: {nna:.2f}%")

    # Compute compactness
    print("\n" + "=" * 80)
    print("COMPUTING COMPACTNESS METRICS")
    print("=" * 80)

    avg_v, avg_f = compute_compactness(generated_meshes)
    results['avg_vertices'] = float(avg_v)
    results['avg_faces'] = float(avg_f)
    print(f"✓ Average Vertices: {avg_v:.1f}")
    print(f"✓ Average Faces: {avg_f:.1f}")

    # Compute visual metrics (FID/KID)
    if compute_visual_metrics and PYTORCH3D_AVAILABLE and INCEPTION_AVAILABLE:
        print("\n" + "=" * 80)
        print("COMPUTING VISUAL QUALITY METRICS (FID/KID)")
        print("=" * 80)

        try:
            renderer = MeshRenderer(
                image_size=EvaluationConfig.IMAGE_SIZE,
                device=EvaluationConfig.DEVICE
            )
            feature_extractor = InceptionFeatureExtractor(device=EvaluationConfig.DEVICE)

            # Render and extract features for generated meshes
            print("\nRendering generated meshes...")
            gen_features_list = []
            for mesh in tqdm(generated_meshes[:100]):  # Limit to 100 for speed
                try:
                    images = renderer.render_mesh(mesh['vertices'], mesh['faces'])
                    images = images / 255.0  # Normalize to [0, 1]
                    features = feature_extractor.extract_features(images)
                    gen_features_list.append(features)
                except Exception as e:
                    print(f"Warning: Failed to render mesh: {e}")

            # Render and extract features for reference meshes
            print("\nRendering reference meshes...")
            ref_features_list = []
            for mesh in tqdm(reference_meshes[:100]):  # Limit to 100 for speed
                try:
                    images = renderer.render_mesh(mesh['vertices'], mesh['faces'])
                    images = images / 255.0
                    features = feature_extractor.extract_features(images)
                    ref_features_list.append(features)
                except Exception as e:
                    print(f"Warning: Failed to render mesh: {e}")

            if gen_features_list and ref_features_list:
                gen_features = np.concatenate(gen_features_list, axis=0)
                ref_features = np.concatenate(ref_features_list, axis=0)

                print("\nComputing FID...")
                fid = compute_fid(gen_features, ref_features)
                results['FID'] = float(fid)
                print(f"✓ FID: {fid:.2f}")

                print("\nComputing KID...")
                kid = compute_kid(gen_features, ref_features)
                results['KID'] = float(kid)
                print(f"✓ KID: {kid:.4f}")
            else:
                print("Warning: Could not render enough meshes for FID/KID")

        except Exception as e:
            print(f"Error computing visual metrics: {e}")
            results['FID'] = None
            results['KID'] = None
    else:
        print("\nSkipping visual metrics (FID/KID)")
        results['FID'] = None
        results['KID'] = None

    # Save results
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"MMD (×10³):     {results['MMD']:.2f}  (lower is better)")
    print(f"Coverage (%):   {results['COV']:.2f}  (higher is better)")
    print(f"1-NNA (%):      {results['1-NNA']:.2f}  (50% is optimal)")
    if results['FID'] is not None:
        print(f"FID:            {results['FID']:.2f}  (lower is better)")
    if results['KID'] is not None:
        print(f"KID:            {results['KID']:.4f}  (lower is better)")
    print(f"Avg Vertices:   {results['avg_vertices']:.1f}")
    print(f"Avg Faces:      {results['avg_faces']:.1f}")
    print("=" * 80)

    return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def run_comparison_evaluation():
    generated_obj_dir_ours = "Chair_samples\\Our_chair_samples"
    generated_obj_dir_meshgpt = "Chair_samples\\MeshGPT_chair_samples"
    reference_obj_dir = "C:\\Users\\koth_ak\\Akshat's Thesis\\Text-to-3D-MeshTransformer\\Data\\gt_mesh_samples\\chair"

    # # Example usage
    # results = evaluate_mesh_generation(
    #     generated_obj_dir="Ours\reference_gt",
    #     reference_obj_dir="Chair_samples\reference_gt",
    #     output_path="METRICS_EVAL_OURS.json",
    #     compute_visual_metrics=True  # Set False to skip FID/KID
    # )
    import random
    generated_meshes_ours = load_meshes_from_obj_files(
        generated_obj_dir_ours,
        # max_meshes=EvaluationConfig.N_GENERATED
        max_meshes=980
    )
    generated_meshes_meshgpt = load_meshes_from_obj_files(
        generated_obj_dir_meshgpt,
        max_meshes=1000
    )
    reference_meshes = load_meshes_from_obj_files(
        reference_obj_dir,
        max_meshes=6000
    )

    # half = len(reference_meshes) // 2
    # set_A = reference_meshes[:half]
    # set_B = reference_meshes[half:]

    import json
    import os
    import random
    import csv
    from datetime import datetime

    results = []  # store all experiments

    output_json = "mesh_metric_evaluation.json"
    output_csv = "mesh_metric_evaluation.csv"

    for ref_size in [1000]:
    # for ref_size in [115]:
        print("\n" + "=" * 80)
        print(f"Evaluating with reference subset size: {ref_size}")
        print("=" * 80)

        # Randomly sample reference subset
        ref_subset = random.sample(reference_meshes, ref_size)

        # ====================================================
        # --- MESH GPT ---
        # ====================================================
        dist_matrix = compute_pairwise_cd_matrix(generated_meshes_meshgpt, ref_subset, meshes1_is_meshgpt=False, meshes2_is_meshgpt=False)
        print(f"Min: {dist_matrix.min():.4f}")
        print(f"25th percentile: {np.percentile(dist_matrix, 25):.4f}")
        print(f"Median: {np.median(dist_matrix):.4f}")
        print(f"75th percentile: {np.percentile(dist_matrix, 75):.4f}")
        print(f"Max: {dist_matrix.max():.4f}")
        mmd = compute_mmd(dist_matrix, generated_meshes_meshgpt, ref_subset)
        print(mmd)
        cov = compute_coverage(dist_matrix, generated_meshes_meshgpt, ref_subset)
        print(cov)# using fixed version (no threshold)
        nna = compute_1nna(generated_meshes_meshgpt, ref_subset, meshes1_is_meshgpt=False, meshes2_is_meshgpt=False)
        # nna = 50

        print(f"MESHGPT output - Ref size {ref_size}: MMD={mmd:.3f}, COV={cov:.2f}, 1-NNA={nna:.2f}")

        results.append({
            "model": "MeshGPT",
            "ref_size": ref_size,
            "MMD": float(mmd),
            "COV": float(cov),
            "1-NNA": float(nna),
            "timestamp": datetime.now().isoformat()
        })

        # ====================================================
        # --- OUR MODEL ---
        # ====================================================
        dist_matrix = compute_pairwise_cd_matrix(generated_meshes_ours, ref_subset)
        print(f"Min: {dist_matrix.min():.4f}")
        print(f"25th percentile: {np.percentile(dist_matrix, 25):.4f}")
        print(f"Median: {np.median(dist_matrix):.4f}")
        print(f"75th percentile: {np.percentile(dist_matrix, 75):.4f}")
        print(f"Max: {dist_matrix.max():.4f}")
        mmd = compute_mmd(dist_matrix, generated_meshes_ours, ref_subset)
        print(mmd)
        cov = compute_coverage(dist_matrix, generated_meshes_ours, ref_subset)  # using fixed version (no threshold)
        print(cov)
        nna = compute_1nna(generated_meshes_ours, ref_subset)
        # nna = 50
        print(f"OURS output - Ref size {ref_size}: MMD={mmd:.3f}, COV={cov:.2f}, 1-NNA={nna:.2f}")

        results.append({
            "model": "Ours",
            "ref_size": ref_size,
            "MMD": float(mmd),
            "COV": float(cov),
            "1-NNA": float(nna),
            "timestamp": datetime.now().isoformat()
        })

        # ====================================================
        # --- Save partial results after each loop ---
        # ====================================================
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {output_json}")

        # Also write/update CSV file
        write_header = not os.path.exists(output_csv)
        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "ref_size", "MMD", "COV", "1-NNA", "timestamp"])
            if write_header:
                writer.writeheader()
            writer.writerow(results[-2])  # MeshGPT
            writer.writerow(results[-1])  # Ours
        print(f"✓ Results appended to {output_csv}")


# Assume these functions are imported from your evaluation module
# from evaluation_module import (
#     load_meshes_from_obj_files,
#     compute_pairwise_cd_matrix,
#     compute_mmd,
#     compute_coverage,
#     compute_1nna
# )

import os
import random
import json
import csv
import numpy as np
from datetime import datetime
from pathlib import Path

def mesh_samples_evaluation(mesh_dirs_path="mesh_samples",
                            reference_dirs_path="C:\\Users\\koth_ak\\Akshat's Thesis\\Text-to-3D-MeshTransformer\\Data\\gt_mesh_samples",
                            output_json="mesh_category_metrics.json", output_csv="mesh_category_metrics.csv",
                            samples_per_category=50, repetitions=3):
    mesh_dirs_path = Path(mesh_dirs_path)
    reference_dirs_path = Path(reference_dirs_path)

    results = []

    # Loop over categories (folders) in generated meshes directory
    for category_folder in mesh_dirs_path.iterdir():
        if not category_folder.is_dir():
            continue
        category_name = category_folder.name
        print(f"\n=== Evaluating category: {category_name} ===")

        generated_mesh_dir = category_folder
        reference_mesh_dir = reference_dirs_path / category_name

        if not reference_mesh_dir.exists():
            print(f"[WARNING] Reference directory not found for category '{category_name}', skipping.")
            continue

        # Load all meshes once per category
        generated_meshes = load_meshes_from_obj_files(str(generated_mesh_dir))
        reference_meshes = load_meshes_from_obj_files(str(reference_mesh_dir))

        if len(generated_meshes) < samples_per_category or len(reference_meshes) < samples_per_category:
            print(f"[WARNING] Not enough meshes in category '{category_name}' for sampling, skipping.")
            continue

        # Repeat evaluation multiple times for averaging
        category_metrics = []
        for rep in range(repetitions):
            print(f"\n--- Repetition {rep + 1}/{repetitions} ---")

            sampled_generated = random.sample(generated_meshes, samples_per_category)
            sampled_reference = random.sample(reference_meshes, samples_per_category)

            dist_matrix = compute_pairwise_cd_matrix(sampled_generated, sampled_reference)
            mmd = compute_mmd(dist_matrix, sampled_generated, sampled_reference)
            cov = compute_coverage(dist_matrix, sampled_generated, sampled_reference)
            nna = compute_1nna(sampled_generated, sampled_reference)

            print(f"Repetition {rep + 1} - MMD: {mmd:.3f}, COV: {cov:.2f}, 1-NNA: {nna:.2f}")
            category_metrics.append({"MMD": mmd, "COV": cov, "1-NNA": nna})

        # Compute average over repetitions
        avg_mmd = float(np.mean([m["MMD"] for m in category_metrics]))
        avg_cov = float(np.mean([m["COV"] for m in category_metrics]))
        avg_nna = float(np.mean([m["1-NNA"] for m in category_metrics]))

        results.append({
            "category": category_name,
            "samples_per_category": samples_per_category,
            "repetitions": repetitions,
            "avg_MMD": avg_mmd,
            "avg_COV": avg_cov,
            "avg_1-NNA": avg_nna,
        })

        # Save partial results after each category
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ JSON results saved to {output_json}")

        # Save/update CSV
        write_header = not os.path.exists(output_csv)
        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f,
                                    fieldnames=["category", "samples_per_category", "repetitions", "avg_MMD", "avg_COV",
                                                "avg_1-NNA", "timestamp"])
            if write_header:
                writer.writeheader()
            writer.writerow(results[-1])
        print(f"✓ CSV results appended to {output_csv}")

    print("\nAll categories processed.")


if __name__ == "__main__":
    # mesh_samples_evaluation()
    run_comparison_evaluation()
