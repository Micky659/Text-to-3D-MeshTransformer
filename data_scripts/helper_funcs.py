from __future__ import annotations
import torch
from torch import is_tensor
from torch.nn.utils.rnn import pad_sequence
import trimesh
import numpy as np
from collections import OrderedDict
from math import ceil
from torch import nn, Tensor, einsum
from torch.nn import Module
import torch.nn.functional as F
from beartype.typing import Tuple
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torchtyping import TensorType
from data_scripts.typing_mod import typecheck, Float


def get_mesh_data(file_path):
    m = trimesh.load(file_path, force='mesh')
    v = m.vertices
    f = m.faces

    cv = v - np.mean(v, axis=0)
    ma = np.max(np.abs(cv))
    v = cv / (ma / 0.95)

    my = np.min(v[:, 1])
    v[:, 1] += -0.95 - my

    def sv(vertex):
        return vertex[1], vertex[2], vertex[0]

    seen = OrderedDict()
    for p in v:
        k = tuple(p)
        if k not in seen:
            seen[k] = p

    uv = list(seen.values())
    sv_list = sorted(uv, key=sv)

    vat = [tuple(x) for x in v]
    svat = [tuple(x) for x in sv_list]

    vm = {oi: ni for oi, vt in enumerate(vat) for ni, svt in enumerate(svat) if vt == svt}
    rf = [[vm[face[0]], vm[face[1]], vm[face[2]]] for face in f]
    sf = [sorted(sa) for sa in rf]
    return np.array(sv_list), np.array(sf)


def augment_mesh(vertices, scale_factor):
    jf = 0.1
    offs = np.random.uniform(-jf, jf, size=vertices.shape)
    v = vertices + offs
    v = v * scale_factor

    my = np.min(v[:, 1])
    v[:, 1] += -0.95 - my
    return v


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


def set_module_requires_grad_(module: Module, requires_grad: bool):
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
    nd = t.ndim
    rd = (nd - dim - 1) if dim >= 0 else (-dim - 1)
    z = (0, 0) * rd
    return F.pad(t, (*z, *padding), value=value)


def pad_to_length(t, length, dim=-1, value=0, right=True):
    cl = t.shape[dim]
    rem = length - cl

    if rem <= 0:
        return t

    pad = (0, rem) if right else (rem, 0)
    return pad_at_dim(t, pad, dim=dim, value=value)


def masked_mean(tensor, mask, dim=-1, eps=1e-5):
    if not exists(mask):
        return tensor.mean(dim=dim)

    mask = rearrange(mask, '... -> ... 1')
    tensor = tensor.masked_fill(~mask, 0.)

    te = mask.sum(dim=dim)
    num = tensor.sum(dim=dim)
    den = te.float().clamp(min=eps)
    mean = num / den
    mean = mean.masked_fill(te == 0, 0.)
    return mean


def ContinuousEmbed(dim_cont):
    return nn.Sequential(
        Rearrange('... -> ... 1'),
        nn.Linear(1, dim_cont),
        nn.SiLU(),
        nn.Linear(dim_cont, dim_cont),
        nn.LayerNorm(dim_cont)
    )


def derive_angle(x, y, eps=1e-5):
    z = einsum('... d, ... d -> ...', l2norm(x), l2norm(y))
    return z.clip(-1 + eps, 1 - eps).arccos()


@torch.no_grad()
@typecheck
def get_derived_face_features(face_coords: Float['b nf nvf 3']):
    iq = face_coords.shape[-2] == 4

    sfc = torch.roll(face_coords, 1, dims=(2,))
    ang = derive_angle(face_coords, sfc)

    if iq:
        sfc = torch.roll(sfc, 1, dims=(2,))

    e1, e2, *_ = (face_coords - sfc).unbind(dim=2)

    cp = torch.cross(e1, e2, dim=-1)

    nrm = l2norm(cp)
    ar = cp.norm(dim=-1, keepdim=True) * 0.5

    return dict(angles=ang, area=ar, normals=nrm)


@typecheck
def discretize(t: Tensor, *, continuous_range: Tuple[float, float], num_discrete: int = 128) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min=0, max=num_discrete - 1)


@typecheck
def undiscretize(t: Tensor, *, continuous_range=Tuple[float, float], num_discrete: int = 128) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = t.float()

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo


@typecheck
def gaussian_blur_1d(t: Tensor, *, sigma: float = 1.) -> Tensor:
    _, _, ch, dev, dt = *t.shape, t.device, t.dtype

    w = int(ceil(sigma * 5))
    w += (w + 1) % 2
    hw = w // 2

    dist = torch.arange(-hw, hw + 1, dtype=dt, device=dev)

    gauss = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    gauss = l1norm(gauss)

    kern = repeat(gauss, 'n -> c 1 n', c=ch)

    t = rearrange(t, 'b n c -> b c n')
    out = F.conv1d(t, kern, padding=hw, groups=ch)
    return rearrange(out, 'b c n -> b n c')


@typecheck
def scatter_mean(tgt: Tensor, indices: Tensor, src=Tensor, *, dim: int = -1, eps: float = 1e-5):
    num = tgt.scatter_add(dim, indices, src)
    den = torch.zeros_like(tgt).scatter_add(dim, indices, torch.ones_like(src))
    return num / den.clamp(min=eps)


def derive_face_edges_from_faces(
        faces: TensorType['b', 'nf', 3, int],
        pad_id=-1,
        neighbor_if_share_one_vertex=False,
        include_self=True
) -> TensorType['b', 'e', 2, int]:
    iof, dev = faces.ndim == 2, faces.device

    if iof:
        faces = rearrange(faces, 'nf c -> 1 nf c')

    mnf = faces.shape[1]
    fevt = 1 if neighbor_if_share_one_vertex else 2

    ae = torch.stack(torch.meshgrid(
        torch.arange(mnf, device=dev),
        torch.arange(mnf, device=dev),
        indexing='ij'), dim=-1)

    fm = reduce(faces != pad_id, 'b nf c -> b nf', 'all')
    fem = rearrange(fm, 'b i -> b i 1') & rearrange(fm, 'b j -> b 1 j')

    fe_list = []

    for f, fem_item in zip(faces, fem):
        sv = rearrange(f, 'i c -> i 1 c 1') == rearrange(f, 'j c -> 1 j 1 c')
        nsv = sv.any(dim=-1).sum(dim=-1)

        inf = (nsv >= fevt) & fem_item

        if not include_self:
            inf &= nsv != 3

        fe_item = ae[inf]
        fe_list.append(fe_item)

    fe_seq = pad_sequence(fe_list, padding_value=pad_id, batch_first=True)

    if iof:
        fe_seq = rearrange(fe_seq, '1 e ij -> e ij')

    return fe_seq


def custom_collate(data, pad_id=-1):
    isd = isinstance(first(data), dict)

    if isd:
        keys = list(first(data).keys())
        data = [list(d.values()) for d in data]

    trans = list(zip(*data))
    out_flds = []

    for fv in trans:
        fval = fv[0]
        if is_tensor(fval):
            try:
                pad = pad_sequence(fv, batch_first=True, padding_value=pad_id)
            except Exception:
                pad = torch.stack(fv, dim=0)
            out_flds.append(pad)
        else:
            out_flds.append(list(fv))

    if isd:
        return dict(zip(keys, out_flds))
    return tuple(out_flds)