from __future__ import annotations
import torch
from torch import is_tensor
from torch.nn.utils.rnn import pad_sequence
import trimesh
import numpy as np
from collections import OrderedDict
from math import ceil

from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from beartype.typing import Tuple
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from torchtyping import TensorType

from data_scripts.typing_mod import typecheck, Float


def get_mesh_data(file_path):
    mesh = trimesh.load(file_path, force='mesh')
    vertices = mesh.vertices
    faces = mesh.faces

    centered_vertices = vertices - np.mean(vertices, axis=0)
    max_abs = np.max(np.abs(centered_vertices))
    vertices = centered_vertices / (max_abs / 0.95)

    min_y = np.min(vertices[:, 1])
    vertices[:, 1] += -0.95 - min_y

    def sort_vertices(vertex):
        return vertex[1], vertex[2], vertex[0]

    seen = OrderedDict()
    for point in vertices:
        key = tuple(point)
        if key not in seen:
            seen[key] = point

    unique_vertices = list(seen.values())
    sorted_vertices = sorted(unique_vertices, key=sort_vertices)

    vertices_as_tuples = [tuple(v) for v in vertices]
    sorted_vertices_as_tuples = [tuple(v) for v in sorted_vertices]

    vertex_map = {old_index: new_index for old_index, vertex_tuple in enumerate(vertices_as_tuples)
                  for new_index, sorted_vertex_tuple in enumerate(sorted_vertices_as_tuples)
                  if vertex_tuple == sorted_vertex_tuple}
    reindexed_faces = [[vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]] for face in faces]
    sorted_faces = [sorted(sub_arr) for sub_arr in reindexed_faces]
    return np.array(sorted_vertices), np.array(sorted_faces)


def augment_mesh(vertices, scale_factor):
    jitter_factor = 0.1
    offsets = np.random.uniform(-jitter_factor, jitter_factor, size=vertices.shape)
    vertices = vertices + offsets
    vertices = vertices * scale_factor

    min_y = np.min(vertices[:, 1])
    vertices[:, 1] += -0.95 - min_y
    return vertices


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def maybe_del(d: dict, *keys):
    for key in keys:
        if key not in d:
            continue

        del d[key]


def first(it):
    return it[0]


def identity(t, *args, **kwargs):
    return t


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def is_empty(x):
    return len(x) == 0


def is_tensor_empty(t: Tensor):
    return t.numel() == 0


def set_module_requires_grad_(
        module: Module,
        requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad


def l1norm(t):
    return F.normalize(t, dim=-1, p=1)


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)]

    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return first(tensors)

    return torch.cat(tensors, dim=dim)


def pad_at_dim(t, padding, dim=-1, value=0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value=value)


def pad_to_length(t, length, dim=-1, value=0, right=True):
    curr_length = t.shape[dim]
    remainder = length - curr_length

    if remainder <= 0:
        return t

    padding = (0, remainder) if right else (remainder, 0)
    return pad_at_dim(t, padding, dim=dim, value=value)


def masked_mean(tensor, mask, dim=-1, eps=1e-5):
    if not exists(mask):
        return tensor.mean(dim=dim)

    mask = rearrange(mask, '... -> ... 1')
    tensor = tensor.masked_fill(~mask, 0.)

    total_el = mask.sum(dim=dim)
    num = tensor.sum(dim=dim)
    den = total_el.float().clamp(min=eps)
    mean = num / den
    mean = mean.masked_fill(total_el == 0, 0.)
    return mean


# continuous embed

def ContinuousEmbed(dim_cont):
    return nn.Sequential(
        Rearrange('... -> ... 1'),
        nn.Linear(1, dim_cont),
        nn.SiLU(),
        nn.Linear(dim_cont, dim_cont),
        nn.LayerNorm(dim_cont)
    )


# additional encoder features
# 1. angle (3), 2. area (1), 3. normals (3)

def derive_angle(x, y, eps=1e-5):
    z = einsum('... d, ... d -> ...', l2norm(x), l2norm(y))
    return z.clip(-1 + eps, 1 - eps).arccos()


@torch.no_grad()
@typecheck
def get_derived_face_features(
        face_coords: Float['b nf nvf 3']  # 3 or 4 vertices with 3 coordinates
):
    is_quad = face_coords.shape[-2] == 4

    # shift face coordinates depending on triangles or quads

    shifted_face_coords = torch.roll(face_coords, 1, dims=(2,))

    angles = derive_angle(face_coords, shifted_face_coords)

    if is_quad:
        # @sbriseid says quads need to be shifted by 2
        shifted_face_coords = torch.roll(shifted_face_coords, 1, dims=(2,))

    edge1, edge2, *_ = (face_coords - shifted_face_coords).unbind(dim=2)

    cross_product = torch.cross(edge1, edge2, dim=-1)

    normals = l2norm(cross_product)
    area = cross_product.norm(dim=-1, keepdim=True) * 0.5

    return dict(
        angles=angles,
        area=area,
        normals=normals
    )


@typecheck
def discretize(
        t: Tensor,
        *,
        continuous_range: Tuple[float, float],
        num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min=0, max=num_discrete - 1)


@typecheck
def undiscretize(
        t: Tensor,
        *,
        continuous_range=Tuple[float, float],
        num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = t.float()

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo


@typecheck
def gaussian_blur_1d(
        t: Tensor,
        *,
        sigma: float = 1.
) -> Tensor:
    _, _, channels, device, dtype = *t.shape, t.device, t.dtype

    width = int(ceil(sigma * 5))
    width += (width + 1) % 2
    half_width = width // 2

    distance = torch.arange(-half_width, half_width + 1, dtype=dtype, device=device)

    gaussian = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
    gaussian = l1norm(gaussian)

    kernel = repeat(gaussian, 'n -> c 1 n', c=channels)

    t = rearrange(t, 'b n c -> b c n')
    out = F.conv1d(t, kernel, padding=half_width, groups=channels)
    return rearrange(out, 'b c n -> b n c')


@typecheck
def scatter_mean(
        tgt: Tensor,
        indices: Tensor,
        src=Tensor,
        *,
        dim: int = -1,
        eps: float = 1e-5
):
    """
    todo: update to pytorch 2.1 and try https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_
    """
    num = tgt.scatter_add(dim, indices, src)
    den = torch.zeros_like(tgt).scatter_add(dim, indices, torch.ones_like(src))
    return num / den.clamp(min=eps)


def derive_face_edges_from_faces(
        faces: TensorType['b', 'nf', 3, int],
        pad_id=-1,
        neighbor_if_share_one_vertex=False,
        include_self=True
) -> TensorType['b', 'e', 2, int]:
    is_one_face, device = faces.ndim == 2, faces.device

    if is_one_face:
        faces = rearrange(faces, 'nf c -> 1 nf c')

    max_num_faces = faces.shape[1]
    face_edges_vertices_threshold = 1 if neighbor_if_share_one_vertex else 2

    all_edges = torch.stack(torch.meshgrid(
        torch.arange(max_num_faces, device=device),
        torch.arange(max_num_faces, device=device),
        indexing='ij'), dim=-1)

    face_masks = reduce(faces != pad_id, 'b nf c -> b nf', 'all')
    face_edges_masks = rearrange(face_masks, 'b i -> b i 1') & rearrange(face_masks, 'b j -> b 1 j')

    face_edges = []

    for face, face_edge_mask in zip(faces, face_edges_masks):

        shared_vertices = rearrange(face, 'i c -> i 1 c 1') == rearrange(face, 'j c -> 1 j 1 c')
        num_shared_vertices = shared_vertices.any(dim=-1).sum(dim=-1)

        is_neighbor_face = (num_shared_vertices >= face_edges_vertices_threshold) & face_edge_mask

        if not include_self:
            is_neighbor_face &= num_shared_vertices != 3

        face_edge = all_edges[is_neighbor_face]
        face_edges.append(face_edge)

    face_edges = pad_sequence(face_edges, padding_value=pad_id, batch_first=True)

    if is_one_face:
        face_edges = rearrange(face_edges, '1 e ij -> e ij')

    return face_edges


# custom collater

def custom_collate(data, pad_id=-1):
    # is_dict = isinstance(first(data), dict)
    #
    # if is_dict:
    #     keys = first(data).keys()
    #     data = [d.values() for d in data]
    #
    # output = []
    #
    # for datum in zip(*data):
    #     if is_tensor(first(datum)):
    #         datum = pad_sequence(datum, batch_first=True, padding_value=pad_id)
    #     else:
    #         datum = list(datum)
    #         output.append(datum)
    #
    #     output.append(datum)
    #
    # output = tuple(output)
    #
    # if is_dict:
    #     output = dict(zip(keys, output))
    #
    # return output

    """
    Robust collate function:
      - Pads tensor fields with pad_sequence if they have variable length along dim=0.
      - Stacks tensor fields if they already have consistent shapes.
      - Keeps non-tensor fields as Python lists (e.g., strings, uids).
    Expects `data` to be a list of dicts or a list of tuples/lists consistent with Dataset.__getitem__.
    """
    is_dict = isinstance(first(data), dict)

    if is_dict:
        keys = list(first(data).keys())
        # turn list of dicts -> list of lists (values)
        data = [list(d.values()) for d in data]

    # transpose: fields are grouped across the batch
    transposed = list(zip(*data))
    output_fields = []

    for field_values in transposed:
        first_val = field_values[0]
        if is_tensor(first_val):
            # try padding variable-length tensors (common case for sequences)
            try:
                padded = pad_sequence(field_values, batch_first=True, padding_value=pad_id)
            except Exception:
                # if pad_sequence fails (all same shape), stack
                padded = torch.stack(field_values, dim=0)
            output_fields.append(padded)
        else:
            # keep raw python objects in a list
            output_fields.append(list(field_values))

    if is_dict:
        return dict(zip(keys, output_fields))
    return tuple(output_fields)

