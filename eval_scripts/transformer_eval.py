import gc
from data_scripts.Mesh_dataset import MeshDataset
from model_scripts.autoencoder import MeshAutoencoder
from model_scripts.transformer import MeshTransformer
from data_scripts import mesh_render
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import re
import time
from pathlib import Path
import os
import math
import time
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from scipy import linalg
from sklearn.neighbors import KDTree
from itertools import combinations
import multiprocessing
import trimesh
import pyrender
import warnings



script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
working_dir = project_root / "Data"
DATASET_PATH = working_dir / "meshData_org.npz"

dataset = MeshDataset.load(DATASET_PATH)
# ds.data = ds.data[:400]

autoencoder = MeshAutoencoder(
        decoder_dims_through_depth=(128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
        codebook_size=16384,
        # Smaller vocab size will speed up the transformer training, however if you are training on meshes more then 250 triangle, I'd advice to use 16384 codebook size
        dim_codebook=192,
        dim_area_embed=16,
        dim_coor_embed=16,
        dim_normal_embed=16,
        dim_angle_embed=8,

        attn_decoder_depth=4,
        attn_encoder_depth=2
    ).to("cuda")


pkg = torch.load(str(project_root/"Model/mesh-encoder_16k_4_2.pt"))
autoencoder.load_state_dict(pkg['model'])

max_seq = max(len(d["faces"]) for d in dataset if "faces" in d)  * (autoencoder.num_vertices_per_face * autoencoder.num_quantizers)
print("Max token sequence:" , max_seq)

# GPT2-Small model
transformer = MeshTransformer(
    autoencoder,
    dim = 768,
    coarse_pre_gateloop_depth = 3,
    fine_pre_gateloop_depth= 3,
    attn_depth = 12,
    attn_heads = 12,
    max_seq_len = max_seq,
    condition_on_text = True,
    gateloop_use_heinsen = False,
    dropout  = 0.0,
    text_condition_model_types = "bge",
    text_condition_cond_drop_prob = 0.0
)

total_params = sum(p.numel() for p in transformer.decoder.parameters())
total_params = f"{total_params / 1000000:.1f}M"
print(f"Decoder total parameters: {total_params}")

pkg = torch.load(str(project_root/"Model"/"mesh-transformer_16k_4_2.pt"))
transformer.load_state_dict(pkg['model'])

folder = project_root/'Data'/'renders'
obj_file_path = Path(folder)
obj_file_path.mkdir(exist_ok=True, parents=True)

# -------------------------
# Helpers: sampling on mesh
# -------------------------
import numpy as np
import torch

# --- sample directly from triangular faces given coordinates (F, 3, 3) ---
def sample_points_from_face_coords(face_coords_np, n_points=5000, device='cpu'):
    """
    face_coords_np: numpy array shape (F,3,3) or torch tensor
    returns: numpy array (n_points, 3) on CPU
    """
    if isinstance(face_coords_np, np.ndarray):
        faces = torch.from_numpy(face_coords_np).to(device=device, dtype=torch.float32)
    else:
        faces = face_coords_np.to(device=device, dtype=torch.float32)

    v0 = faces[:, 0, :]
    v1 = faces[:, 1, :]
    v2 = faces[:, 2, :]

    # triangle areas
    tri_cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    tri_areas = 0.5 * torch.norm(tri_cross, dim=1)
    tri_areas = tri_areas.clamp(min=1e-12)
    probs = tri_areas / tri_areas.sum()

    # sample triangles proportional to area
    idx = torch.multinomial(probs, n_points, replacement=True)

    # barycentric coords (uniform)
    r1 = torch.sqrt(torch.rand(n_points, device=device))
    r2 = torch.rand(n_points, device=device)
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2

    tv0 = v0[idx]
    tv1 = v1[idx]
    tv2 = v2[idx]

    samples = tv0 * a.unsqueeze(1) + tv1 * b.unsqueeze(1) + tv2 * c.unsqueeze(1)
    return samples.detach().cpu().numpy()


# --- optional: convert face_coords (F,3,3) -> unique vertices + face indices ---
def faces_coords_to_vertices_faces(face_coords_np, round_decimals=6):
    """
    face_coords_np: numpy array (F,3,3)
    returns: (vertices_np (V,3), faces_np (F,3) indices)
    """
    if isinstance(face_coords_np, torch.Tensor):
        face_coords_np = face_coords_np.cpu().numpy()
    verts_map = {}
    unique_verts = []
    faces_idx = []
    for tri in face_coords_np:              # tri is (3,3)
        face_inds = []
        for v in tri:
            key = tuple(np.round(v, round_decimals).tolist())
            if key not in verts_map:
                verts_map[key] = len(unique_verts)
                unique_verts.append(list(key))
            face_inds.append(verts_map[key])
        faces_idx.append(face_inds)
    return np.array(unique_verts, dtype=float), np.array(faces_idx, dtype=int)


# --- utility to robustly extract face_coords array from generator output ---
def extract_face_coords_from_generate_output(out):
    """
    Accepts generator output that may be:
     - (face_coords, face_mask) where face_coords is list or tensor
     - a single dict {'vertices':..., 'faces':...}
     - a list/other forms
    Returns face_coords_np (F,3,3) numpy array or None if not found.
    """
    # Case 1: tuple/list of (face_coords, face_mask)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        face_coords, face_mask = out
        # If lists (batch), take first
        if isinstance(face_coords, list):
            fc = face_coords[0]
            fm = face_mask[0] if isinstance(face_mask, list) else face_mask
        else:
            fc = face_coords
            fm = face_mask
        # apply mask if provided
        if fm is not None:
            if isinstance(fm, torch.Tensor):
                fm_cpu = fm.cpu().numpy()
            else:
                fm_cpu = np.array(fm)
            if isinstance(fc, torch.Tensor):
                fc_np = fc.cpu().numpy()
            else:
                fc_np = np.array(fc)
            # mask may be boolean per-face
            try:
                selected = fc_np[fm_cpu]
            except Exception:
                # fallback: assume mask indexes valid faces
                selected = fc_np[np.where(fm_cpu)]
            return selected

        # no mask -> return all
        return fc.cpu().numpy() if isinstance(fc, torch.Tensor) else np.array(fc)

    # Case 2: dict-like {vertices, faces}
    if isinstance(out, dict) and 'vertices' in out and 'faces' in out:
        verts = np.array(out['vertices'])
        faces_idx = np.array(out['faces'])
        face_coords = verts[faces_idx]   # (F,3,3)
        return face_coords

    # Case 3: list of files/filenames returned by generator -> cannot sample directly
    # Return None and let caller handle file loading if needed
    return None



def sample_points_from_mesh(vertices: torch.Tensor, faces: torch.Tensor, n_points: int):
    """
    vertices: (V,3) CPU or GPU tensor
    faces: (F,3) long indices (no padding)
    returns: (n_points,3) float32 tensor on CPU
    """
    device = vertices.device
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # triangle areas
    tri_cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    tri_areas = 0.5 * torch.norm(tri_cross, dim=1)
    tri_areas = tri_areas.clamp(min=1e-12)  # avoid zeros

    # sample triangles proportional to area
    probs = tri_areas / tri_areas.sum()
    # sample indices (with replacement)
    idx = torch.multinomial(probs, n_points, replacement=True)

    # barycentric sampling inside triangles
    r1 = torch.sqrt(torch.rand(n_points, device=device))
    r2 = torch.rand(n_points, device=device)
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2

    tri_v0 = v0[idx]
    tri_v1 = v1[idx]
    tri_v2 = v2[idx]

    samples = (tri_v0 * a.unsqueeze(1) + tri_v1 * b.unsqueeze(1) + tri_v2 * c.unsqueeze(1))
    return samples.detach().cpu()

# ---------------------------------------
# Chunked pairwise nearest distances utils
# ---------------------------------------
def pairwise_distances_chunked(x: torch.Tensor, y: torch.Tensor, device='cpu', chunk_size=4096):
    """
    returns dists: (x_len, y_len) possibly huge - but we will compute nearest distances one side at a time.
    We'll instead compute one-way nearest distances: for each point in x -> min dist to y
    """
    x = x.to(device)
    y = y.to(device)
    n_x = x.shape[0]
    mins = []
    with torch.no_grad():
        for i in range(0, n_x, chunk_size):
            x_chunk = x[i:i+chunk_size]  # (chunk, 3)
            # compute squared dists to all y
            d = torch.cdist(x_chunk, y, p=2)  # (chunk, y_len)
            m, _ = torch.min(d, dim=1)
            mins.append(m.cpu())
    return torch.cat(mins).numpy()  # shape (n_x,)

def chamfer_distance_from_pointclouds(pc1: torch.Tensor, pc2: torch.Tensor, device='cuda', chunk_size=4096):
    """
    pc1, pc2: numpy arrays or torch tensors (N,3) ; return scalar chamfer = mean(min d^2) both directions
    We'll compute L2 distances (not squared) and return mean of distances (consistent with many works).
    """
    if not isinstance(pc1, torch.Tensor):
        pc1 = torch.from_numpy(pc1)
    if not isinstance(pc2, torch.Tensor):
        pc2 = torch.from_numpy(pc2)
    # move smaller arrays to device for cdist but chunk from the other side when large
    pc1_device = pc1.to(device)
    pc2_device = pc2.to(device)
    # compute one-way distances
    d1 = pairwise_distances_chunked(pc1_device, pc2_device, device=device, chunk_size=chunk_size)
    d2 = pairwise_distances_chunked(pc2_device, pc1_device, device=device, chunk_size=chunk_size)
    return float(d1.mean() + d2.mean())

def f_score(pc_pred, pc_gt, tau=0.01, device='cuda', chunk_size=4096):
    """
    Compute F-score at distance threshold tau.
    Returns precision, recall, f.
    """
    if not isinstance(pc_pred, torch.Tensor):
        pc_pred = torch.from_numpy(pc_pred)
    if not isinstance(pc_gt, torch.Tensor):
        pc_gt = torch.from_numpy(pc_gt)
    pc_pred_d = pc_pred.to(device)
    pc_gt_d = pc_gt.to(device)
    # one-way min dists
    d_pred_to_gt = pairwise_distances_chunked(pc_pred_d, pc_gt_d, device=device, chunk_size=chunk_size)
    d_gt_to_pred = pairwise_distances_chunked(pc_gt_d, pc_pred_d, device=device, chunk_size=chunk_size)
    prec = float((d_pred_to_gt <= tau).mean())
    rec = float((d_gt_to_pred <= tau).mean())
    if prec + rec == 0:
        f = 0.0
    else:
        f = 2 * prec * rec / (prec + rec)
    return prec, rec, f

# -------------------------
# Renderer helper (pyrender/trimesh fallback)
# -------------------------
def default_render_mesh_to_images(verts_np, faces_np, views=8, image_size=299, bgcolor=(255,255,255)):
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    mesh.apply_translation(-mesh.centroid)
    scale = 1.0 / np.linalg.norm(mesh.extents)
    mesh.apply_scale(scale)

    scene = pyrender.Scene(bg_color=bgcolor, ambient_light=(0.7, 0.7, 0.7, 1.0))
    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_node)

    # Create camera and add it
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    # Position camera at some initial pose
    cam_distance = 2.0
    pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -cam_distance],
        [0.0, 0.0, 1.0, cam_distance],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
    cam_node = scene.add(cam, pose=pose)

    images = []
    renderer = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size)
    for i in range(views):
        theta = 2.0 * math.pi * i / views
        # Rotate camera around Z axis
        rot = trimesh.transformations.rotation_matrix(theta, [0, 0, 1])
        new_pose = rot @ pose
        scene.set_pose(cam_node, pose=new_pose)

        color, _ = renderer.render(scene)
        images.append(Image.fromarray(color))

    renderer.delete()
    return images

# -------------------------
# Inception feature extractor (pool features, 2048-d)
# -------------------------
class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        # Use pretrained InceptionV3 and return pool features by replacing fc with Identity
        self.device = device
        self.model = models.inception_v3(pretrained=True, aux_logits=False)
        self.model.fc = nn.Identity()  # returns 2048-d features (pool output)
        self.model.eval().to(device)
        # preprocessing transform
        self.transform = T.Compose([
            T.Resize((299,299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],  # Inception normalization
                        std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def get_activations(self, pil_images, batch_size=64):
        """
        pil_images: list of PIL.Image
        returns: numpy array (N, 2048)
        """
        acts = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i+batch_size]
            tensors = torch.stack([self.transform(img) for img in batch]).to(self.device)
            with torch.no_grad():
                out_tensor = self.model(tensors)
            out_cpu = out_tensor.detach().cpu().numpy()
            acts.append(out_cpu)
        return np.concatenate(acts, axis=0)

# -------------------------
# FID computation
# -------------------------
def calculate_fid_from_activations(act1, act2, eps=1e-6):
    # act1, act2: numpy arrays (N, dim)
    mu1 = np.mean(act1, axis=0)
    mu2 = np.mean(act2, axis=0)
    cov1 = np.cov(act1, rowvar=False)
    cov2 = np.cov(act2, rowvar=False)
    diff = mu1 - mu2
    # sqrt of product
    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    if not np.isfinite(covmean).all():
        # numerical fallback
        offset = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))
    # if imaginary parts appear
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(cov1 + cov2 - 2.0 * covmean)
    return float(np.real(fid))

# -------------------------
# KID computation (unbiased polynomial kernel MMD)
# -------------------------
def polynomial_kernel(x, y, degree=3, gamma=None, coef0=1):
    # x,y are numpy arrays (n,d) and (m,d)
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * x.dot(y.T) + coef0) ** degree

def kid_unbiased(x, y, degree=3, gamma=None, coef0=1):
    """
    unbiased estimator of squared MMD with polynomial kernel (KID).
    Returns mean KID and var estimate.
    """
    x = np.asarray(x); y = np.asarray(y)
    nx = x.shape[0]; ny = y.shape[0]
    Kxx = polynomial_kernel(x, x, degree, gamma, coef0)
    Kyy = polynomial_kernel(y, y, degree, gamma, coef0)
    Kxy = polynomial_kernel(x, y, degree, gamma, coef0)

    # Remove diagonal
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    term_x = Kxx.sum() / (nx * (nx - 1))
    term_y = Kyy.sum() / (ny * (ny - 1))
    term_xy = 2.0 * Kxy.sum() / (nx * ny)
    m = term_x + term_y - term_xy
    return float(m)

# -------------------------
# 1-NNA (nearest neighbor test)
# -------------------------
def one_nn_accuracy_from_pointclouds(gen_pcs, gt_pcs):
    """
    gen_pcs, gt_pcs: lists of numpy (N_points,3)
    We'll compute a 1-NN classifier using KDTree over combined set and leave-one-out accuracy.
    Implementation:
      - Build KDTree on all pointcloud centroids or simple global features (we use centroid+PCD signature)
    Simpler proxy: use flattened PCA features or avg furthest distances; but here we compute pairwise Chamfer distances
    between pointcloud centroids (cheap). For strict 1-NNA you'd need to vectorize better.
    For reasonable speed, we'll compute a small signature: [centroid (3), std (3), median distance (1)] -> 7D
    """
    def signature(pc):
        pc = np.asarray(pc)
        centroid = pc.mean(axis=0)
        std = pc.std(axis=0)
        dists = np.linalg.norm(pc - centroid[None, :], axis=1)
        med = np.median(dists)
        return np.concatenate([centroid, std, [med]])

    X = np.vstack([signature(pc) for pc in gen_pcs])
    Y = np.vstack([signature(pc) for pc in gt_pcs])
    labels = np.concatenate([np.zeros(X.shape[0]), np.ones(Y.shape[0])])
    data = np.vstack([X, Y])
    # KDTree
    tree = KDTree(data)
    # find 1-NN (excluding self)
    neigh_ind = tree.query(data, k=2, return_distance=False)[:,1]  # 0 is self
    neigh_labels = labels[neigh_ind]
    # accuracy = fraction where neighbor label equals own label
    acc = (neigh_labels == labels).mean()
    # For 1-NN test p-value you'd do permutation; here we return accuracy
    return float(acc)

# -------------------------
# Top-level helpers to compute FID/KID/1-NNA for whole sets
# -------------------------
def compute_image_fid_kid(gen_meshes, gt_meshes, render_fn,
                          views=8, image_size=299, device='cuda', tmpdir='tmp_renders'):
    """
    gen_meshes: list of (verts_np, faces_np)
    gt_meshes: same
    render_fn: function(verts_np, faces_np) -> list[PIL.Image]
    returns: fid_value, kid_value
    """
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(exist_ok=True, parents=True)

    # render images for gen and gt; store lists
    gen_images = []
    gt_images = []

    print(f"Rendering {len(gen_meshes)} gen meshes and {len(gt_meshes)} gt meshes with {views} views each...")
    t0 = time.time()
    for verts, faces in gen_meshes:
        imgs = render_fn(verts, faces)
        gen_images.extend(imgs)
    for verts, faces in gt_meshes:
        imgs = render_fn(verts, faces)
        gt_images.extend(imgs)
    t1 = time.time()
    print("Rendered images in", t1 - t0, "s. Total images:", len(gen_images), len(gt_images))

    # get inception activations
    inc = InceptionV3FeatureExtractor(device=device)
    act_gen = inc.get_activations(gen_images, batch_size=32)
    act_gt = inc.get_activations(gt_images, batch_size=32)

    print(type(act_gt), act_gt)
    print(type(act_gen), act_gen)

    fid_val = calculate_fid_from_activations(act_gen, act_gt)
    kid_val = kid_unbiased(act_gen, act_gt)
    return fid_val, kid_val


# -------------------------
# High-level per-category eval
# -------------------------
def evaluate_categories(transformer,
                        dataset,
                        mesh_render,
                        out_folder: str,
                        categories=None,
                        samples_per_category=5,
                        points_per_mesh=10000,
                        taus=(0.01, 0.02, 0.05),
                        device='cuda'):
    """
    transformer: your transformer object with .generate(texts=[...])
    dataset: MeshDataset instance with .data list of dicts, each dict has 'texts','vertices','faces'
    mesh_render: only used if you want to save meshes; optional (pass None to skip saving)
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    texts = sorted(list({item['texts'] for item in dataset.data}))
    print(texts)
    if categories is None:
        categories = texts  # all
    # categories_df = {'chair':6000, 'bench':1200, 'lamp':1000, 'table':8000}
    categories_df = {'bench':2}
    results = []
    categories = list(categories_df.keys())
    print("categories:", categories)
    # Pre-build dict: category -> list of gt mesh entries
    gt_by_cat = defaultdict(list)
    for item in dataset.data:
        gt_by_cat[item['texts']].append(item)

    for cat in categories:
        print(f"\n=== Eval category: {cat} ===")
        gts = gt_by_cat[cat]
        if len(gts) == 0:
            print("No GTs found for category", cat)
            continue

        # build GT pointclouds list (sample each GT once)
        gt_pointclouds = []
        gt_meshes = []
        for gt in gts:
            verts = torch.as_tensor(gt['vertices'], dtype=torch.float32)
            faces = torch.as_tensor(gt['faces'], dtype=torch.long)
            pc = sample_points_from_mesh(verts, faces, points_per_mesh)
            gt_pointclouds.append(pc.numpy())
            verts_np = verts.cpu().numpy()
            faces_np = faces.cpu().numpy()
            gt_meshes.append((verts_np, faces_np))

        # generate samples (G per category)
        # inside evaluate loop where you generate samples_per_category
        gen_pointclouds = []
        gen_meshes = []
        samples_per_category = categories_df[cat]
        # samples_per_category = 1
        for g_idx in range(samples_per_category):
            t0 = time.time()
            out = transformer.generate(texts=[cat], temperature=0.0)
            t1 = time.time()
            print(f" generated sample {g_idx + 1}/{samples_per_category} (in {t1 - t0:.3f}s)")

            # Attempt to extract face_coords (F,3,3) numpy array
            face_coords_np = extract_face_coords_from_generate_output(out)

            if face_coords_np is not None:
                # sample points directly from triangular faces
                pc_g = sample_points_from_face_coords(face_coords_np, n_points=points_per_mesh, device='cpu')
                gen_pointclouds.append(pc_g)

                verts_np, faces_np = faces_coords_to_vertices_faces(face_coords_np)
                gen_meshes.append((verts_np, faces_np))
                # optionally reconstruct indexed mesh for saving
                if mesh_render is not None:
                    mesh_fname = out_folder / f"{cat.replace(' ', '_')}_gen_{g_idx}.obj"
                    mesh_render.save_single_mesh(verts_np, faces_np, mesh_fname)
            else:
                # fallback: if generator returned files (obj paths), read them into verts/faces
                # or if generator returned dict that extract didn't catch, adapt here
                raise RuntimeError("Unable to extract face_coords from generator output; adapt extract function.")

                # out expected: structure containing vertices+faces or face coords; adapt accordingly.
                # I assume out is a dict/list with 'vertices' and 'faces'
                # If your generate returns something else, modify the extraction below.
                gen = out  # adjust if out is list
                # try to normalize retrieval:
                if isinstance(out, list):
                    gen = out[0]
                verts_g = torch.as_tensor(gen['vertices'], dtype=torch.float32)
                faces_g = torch.as_tensor(gen['faces'], dtype=torch.long)
                pc_g = sample_points_from_mesh(verts_g, faces_g, points_per_mesh)
                gen_pointclouds.append(pc_g.numpy())
                gen_meshes.append((verts_g.numpy(), faces_g.numpy()))

                if mesh_render is not None:
                    # optional: store generated mesh to file
                    mesh_filename = out_folder / f"{cat.replace(' ','_')}_gen_{g_idx}.obj"
                    mesh_render.save_single_mesh(verts_g.numpy(), faces_g.numpy(), mesh_filename)

        # chamfer_matrix computation (unchanged)
        chamfer_matrix = np.zeros((len(gen_pointclouds), len(gt_pointclouds)), dtype=float)

        for i, pg in enumerate(gen_pointclouds):
            for j, rg in enumerate(gt_pointclouds):
                # compute chamfer distance (one-way both directions inside function)
                c = chamfer_distance_from_pointclouds(torch.from_numpy(pg), torch.from_numpy(rg), device=device)
                chamfer_matrix[i, j] = c

        # per-generated minimum to any GT (useful if generated matches any GT)
        per_gen_min = chamfer_matrix.min(axis=1)  # shape (G,)
        per_gt_min = chamfer_matrix.min(axis=0)  # shape (R,)

        # compute F-scores for each generated against its best GT
        f_at_taus_per_gen = []
        for i, pg in enumerate(gen_pointclouds):
            jbest = int(np.argmin(chamfer_matrix[i]))  # index of best GT
            f_per_tau = []
            for tau in taus:
                _, _, f = f_score(torch.from_numpy(pg), torch.from_numpy(gt_pointclouds[jbest]), tau=tau, device=device)
                f_per_tau.append(f)
            f_at_taus_per_gen.append(f_per_tau)

        # MMD: average over gt meshes of min chamfer to any generated
        MMD = float(per_gt_min.mean())

        # Coverage: fraction of GTs with min chamfer < threshold (use tau as coverage threshold)
        covs = {}
        for tau in taus:
            covs[tau] = float((per_gt_min <= tau).sum() / len(per_gt_min))

        # Extra metrics: 1-NNA (signature-based) and FID/KID on rendered images
        # NOTE: compute_image_fid_kid may take a long time (renders many images). We cap meshes used to a reasonable number.
        extra_metrics = {}
        try:
            # 1-NNA: uses pointcloud signatures and KDTree (fast)
            nna_val = one_nn_accuracy_from_pointclouds(gen_pointclouds, gt_pointclouds)
            extra_metrics['1-NNA'] = float(nna_val)
        except Exception as e:
            warnings.warn(f"1-NNA computation failed: {e}")
            extra_metrics['1-NNA'] = None

        # FID/KID: render images and compute Inception activations; expensive — cap to first N meshes
        try:
            # Only compute FID/KID if we have meshes (verts+faces) for gen and gt
            if len(gen_meshes) > 0 and len(gt_meshes) > 0:
                fid_cap = 200  # reduce for speed; increase to match paper if you have time
                gen_for_fid = gen_meshes[:min(len(gen_meshes), fid_cap)]
                gt_for_fid = gt_meshes[:min(len(gt_meshes), fid_cap)]
                # compute_image_fid_kid should be available (uses default renderer or your provided one)
                fid_val, kid_val = compute_image_fid_kid(gen_for_fid, gt_for_fid,
                                                         render_fn=default_render_mesh_to_images,
                                                         views=8, image_size=299, device=device,
                                                         tmpdir=str(out_folder / 'tmp_renders'))
                print(fid_val, kid_val)
                extra_metrics['FID'] = float(fid_val)
                extra_metrics['KID'] = float(kid_val)
            else:
                extra_metrics['FID'] = None
                extra_metrics['KID'] = None
        except Exception as e:
            warnings.warn(f"FID/KID computation failed: {e}")
            extra_metrics['FID'] = None
            extra_metrics['KID'] = None
        print(extra_metrics.get('FID'))
        print(extra_metrics.get('KID'))
        # aggregate stats
        results.append({
            'category': cat,
            'n_gt': len(gt_pointclouds),
            'n_gen': len(gen_pointclouds),
            'MMD': MMD,
            'COV': {tau: covs[tau] for tau in taus},
            'chamfer_mean_gen': float(per_gen_min.mean()),
            'chamfer_min_gen': float(per_gen_min.min()),
            'f_at_taus_mean': {tau: float(np.mean([f[i] for f in f_at_taus_per_gen])) for i, tau in enumerate(taus)},
            '1-NNA': extra_metrics.get('1-NNA', None),
            'FID': extra_metrics.get('FID'),
            'KID': extra_metrics.get('KID')
        })

        # quick printing
        print(f" MMD (chamfer) = {MMD:.6f}")
        for tau in taus:
            print(f" COV@{tau} = {covs[tau]:.3f}")
        print(f" gen mean chamfer (min over GTs) = {per_gen_min.mean():.6f}")
        print(" F@taus (mean over generated):",
              {taus[i]: np.mean([f[i] for f in f_at_taus_per_gen]) for i in range(len(taus))})
        print(" 1-NNA:", extra_metrics.get('1-NNA'))
        print(" FID:", extra_metrics.get('FID'), "KID:", extra_metrics.get('KID'))

    # Save CSV / JSON (extended header)
    import csv, json
    csv_path = out_folder / "eval_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['category', 'n_gt', 'n_gen', 'MMD', 'chamfer_mean_gen', 'chamfer_min_gen'] + [f"COV@{t}" for t in
                                                                                                taus] + [f"F@{t}"
                                                                                                         for t in
                                                                                                         taus] + [
                     '1-NNA', 'FID', 'KID']
        writer.writerow(header)
        for r in results:
            row = [
                      r['category'], r['n_gt'], r['n_gen'], r['MMD'], r['chamfer_mean_gen'], r['chamfer_min_gen']
                  ] + [r['COV'][t] for t in taus] + [r['f_at_taus_mean'][t] for t in taus] + [r.get('1-NNA'),
                                                                                              r.get('FID'),
                                                                                              r.get('KID')]
            writer.writerow(row)
    print("Saved CSV to", csv_path)
    return results

# -------------------------
# Example usage (adapt to your code)
# -------------------------
if __name__ == "__main__":
    # assume these already exist in your namespace / project:
    # transformer, dataset, mesh_render
    # e.g. dataset = MeshDataset.load('Data/meshData_org.npz')
    out = evaluate_categories(
        transformer=transformer,
        dataset=dataset,
        mesh_render=None,                # or pass your renderer
        out_folder="eval_out",
        samples_per_category=5,
        points_per_mesh=2048,            # reduce to 5k if slow; 10k is common
        taus=(0.01, 0.02, 0.05),
        device='cuda'
    )