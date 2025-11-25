from __future__ import annotations
from pathlib import Path
from functools import partial
from math import ceil, pi
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
from pytorch_custom_utils import save_load
from beartype.typing import Tuple
from einops import rearrange, repeat, reduce, pack
from einx import get_at
from x_transformers.x_transformers import RMSNorm, FeedForward
from local_attention import LocalMHA
from vector_quantize_pytorch import ResidualVQ, ResidualLFQ, ResidualSimVQ
from taylor_series_linear_attention import TaylorSeriesLinearAttn
from torch_geometric.nn.conv import SAGEConv
from data_scripts.helper_funcs import *
from model_scripts.resnet import ResnetBlock
from data_scripts.typing_mod import Float, Int, Bool


@save_load()
class MeshAutoencoder(Module):
    @typecheck
    def __init__(
            self,
            num_discrete_coors=128,
            coor_continuous_range: Tuple[float, float] = (-1., 1.),
            dim_coor_embed=64,
            num_discrete_area=128,
            dim_area_embed=16,
            num_discrete_normals=128,
            dim_normal_embed=64,
            num_discrete_angle=128,
            dim_angle_embed=16,
            encoder_dims_through_depth: Tuple[int, ...] = (64, 128, 256, 256, 576),
            init_decoder_conv_kernel=7,
            decoder_dims_through_depth: Tuple[int, ...] = (128, 128, 128, 128, 192, 192, 192, 192,
                                                           256, 256, 256, 256, 256, 256, 384, 384, 384),
            dim_codebook=192,
            num_quantizers=2,
            codebook_size=16384,
            use_residual_lfq=True,
            use_residual_simvq=False,
            rq_kwargs: dict = dict(quantize_dropout=True, quantize_dropout_cutoff_index=1,
                                   quantize_dropout_multiple_of=1),
            rvq_kwargs: dict = dict(kmeans_init=True, threshold_ema_dead_code=2),
            rlfq_kwargs: dict = dict(frac_per_sample_entropy=1., soft_clamp_input_value=10.),
            rvq_stochastic_sample_codes=True,
            sageconv_kwargs: dict = dict(normalize=True, project=True),
            commit_loss_weight=0.1,
            bin_smooth_blur_sigma=0.4,
            attn_encoder_depth=0,
            attn_decoder_depth=0,
            local_attn_kwargs: dict = dict(dim_head=32, heads=8),
            local_attn_window_size=64,
            linear_attn_kwargs: dict = dict(dim_head=8, heads=16),
            use_linear_attn=True,
            pad_id=-1,
            flash_attn=True,
            attn_dropout=0.,
            ff_dropout=0.,
            resnet_dropout=0,
            checkpoint_quantizer=False,
            quads=False
    ):
        super().__init__()

        self.num_vertices_per_face = 3 if not quads else 4
        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.checkpoint_quantizer = checkpoint_quantizer
        self.pad_id = pad_id
        self.commit_loss_weight = commit_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

        self.discretize_face_coords = partial(discretize, num_discrete=num_discrete_coors,
                                              continuous_range=coor_continuous_range)
        self.discretize_angle = partial(discretize, num_discrete=num_discrete_angle, continuous_range=(0., pi))
        lo, hi = coor_continuous_range
        self.discretize_area = partial(discretize, num_discrete=num_discrete_area,
                                       continuous_range=(0., (hi - lo) ** 2))
        self.discretize_normals = partial(discretize, num_discrete=num_discrete_normals,
                                          continuous_range=coor_continuous_range)

        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed)
        self.angle_embed = nn.Embedding(num_discrete_angle, dim_angle_embed)
        self.area_embed = nn.Embedding(num_discrete_area, dim_area_embed)
        self.normal_embed = nn.Embedding(num_discrete_normals, dim_normal_embed)

        attn_cfg = dict(causal=False, prenorm=True, dropout=attn_dropout, window_size=local_attn_window_size)

        init_dim = dim_coor_embed * (
                    3 * self.num_vertices_per_face) + dim_angle_embed * self.num_vertices_per_face + dim_normal_embed * 3 + dim_area_embed
        self.project_in = nn.Linear(init_dim, dim_codebook)

        init_enc_dim, *enc_dims = encoder_dims_through_depth
        self.init_sage_conv = SAGEConv(dim_codebook, init_enc_dim, **sageconv_kwargs)
        self.init_encoder_act_and_norm = nn.Sequential(nn.SiLU(), nn.LayerNorm(init_enc_dim))

        self.encoders = ModuleList([])
        curr = init_enc_dim
        for d in enc_dims:
            self.encoders.append(SAGEConv(curr, d, **sageconv_kwargs))
            curr = d

        self.encoder_attn_blocks = ModuleList([])
        for _ in range(attn_encoder_depth):
            self.encoder_attn_blocks.append(nn.ModuleList([
                TaylorSeriesLinearAttn(curr, prenorm=True, **linear_attn_kwargs) if use_linear_attn else None,
                LocalMHA(dim=curr, **attn_cfg, **local_attn_kwargs),
                nn.Sequential(RMSNorm(curr), FeedForward(curr, glu=True, dropout=ff_dropout))
            ]))

        self.project_dim_codebook = nn.Linear(curr, dim_codebook * self.num_vertices_per_face)

        if use_residual_simvq:
            self.quantizer = ResidualSimVQ(dim=dim_codebook, num_quantizers=num_quantizers,
                                           codebook_size=codebook_size, commitment_weight=1., **rq_kwargs)
        elif use_residual_lfq:
            self.quantizer = ResidualLFQ(dim=dim_codebook, num_quantizers=num_quantizers,
                                         codebook_size=codebook_size, commitment_loss_weight=1., **rlfq_kwargs,
                                         **rq_kwargs)
        else:
            self.quantizer = ResidualVQ(dim=dim_codebook, num_quantizers=num_quantizers, codebook_size=codebook_size,
                                        shared_codebook=True, commitment_weight=1.,
                                        stochastic_sample_codes=rvq_stochastic_sample_codes, **rvq_kwargs, **rq_kwargs)

        dec_input_dim = dim_codebook * 3
        self.decoder_attn_blocks = ModuleList([])
        for _ in range(attn_decoder_depth):
            self.decoder_attn_blocks.append(nn.ModuleList([
                TaylorSeriesLinearAttn(dec_input_dim, prenorm=True, **linear_attn_kwargs) if use_linear_attn else None,
                LocalMHA(dim=dec_input_dim, **attn_cfg, **local_attn_kwargs),
                nn.Sequential(RMSNorm(dec_input_dim), FeedForward(dec_input_dim, glu=True, dropout=ff_dropout))
            ]))

        init_dec_dim, *dec_dims = decoder_dims_through_depth
        self.init_decoder_conv = nn.Sequential(
            nn.Conv1d(dim_codebook * self.num_vertices_per_face, init_dec_dim, kernel_size=init_decoder_conv_kernel,
                      padding=init_decoder_conv_kernel // 2),
            nn.SiLU(), Rearrange('b c n -> b n c'), nn.LayerNorm(init_dec_dim), Rearrange('b n c -> b c n'))

        self.decoders = ModuleList([])
        curr = init_dec_dim
        for d in dec_dims:
            self.decoders.append(ResnetBlock(curr, d, dropout=resnet_dropout))
            curr = d

        total_coords = self.num_vertices_per_face * 3
        self.to_coor_logits = nn.Sequential(nn.Linear(curr, num_discrete_coors * total_coords),
                                            Rearrange('... (v c) -> ... v c', v=total_coords))

    @property
    def device(self):
        return next(self.parameters()).device

    @typecheck
    def encode(self, *, vertices: Float['b nv 3'], faces: Int['b nf nvf'], face_edges: Int['b e 2'],
               face_mask: Bool['b nf'], face_edges_mask: Bool['b e'], return_face_coordinates=False):

        face_no_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)
        fc = get_at('b [nv] c, b nf mv -> b nf mv c', vertices, face_no_pad)

        feat = get_derived_face_features(fc)
        ang = self.angle_embed(self.discretize_angle(feat['angles']))
        ar = self.area_embed(self.discretize_area(feat['area']))
        nrm = self.normal_embed(self.discretize_normals(feat['normals']))

        dfc = self.discretize_face_coords(fc)
        dfc = rearrange(dfc, 'b nf nv c -> b nf (nv c)')
        fce = self.coor_embed(dfc)
        fce = rearrange(fce, 'b nf c d -> b nf (c d)')

        emb, _ = pack([fce, ang, ar, nrm], 'b nf *')
        emb = self.project_in(emb)

        fo = reduce(face_mask.long(), 'b nf -> b', 'sum')
        fo = F.pad(fo.cumsum(dim=0), (1, -1), value=0)
        fo = rearrange(fo, 'b -> b 1 1')

        fe = face_edges + fo
        fe = fe[face_edges_mask]
        fe = rearrange(fe, 'be ij -> ij be')

        orig_shape = emb.shape[:2]
        emb = emb[face_mask]

        emb = self.init_sage_conv(emb, fe)
        emb = self.init_encoder_act_and_norm(emb)

        for conv in self.encoders:
            emb = conv(emb, fe)

        shape = (*orig_shape, emb.shape[-1])
        emb = emb.new_zeros(shape).masked_scatter(rearrange(face_mask, '... -> ... 1'), emb)

        for lin_attn, attn, ff in self.encoder_attn_blocks:
            if exists(lin_attn):
                emb = lin_attn(emb, mask=face_mask) + emb
            emb = attn(emb, mask=face_mask) + emb
            emb = ff(emb) + emb

        return (emb, dfc) if return_face_coordinates else emb

    @typecheck
    def quantize(self, *, faces: Int['b nf nvf'], face_mask: Bool['b n'], face_embed: Float['b nf d'],
                 pad_id=None, rvq_sample_codebook_temp=1.):

        pad_id = default(pad_id, self.pad_id)
        b, dev = faces.shape[0], faces.device

        mx = faces.amax()
        nv = int(mx.item() + 1)

        fe = self.project_dim_codebook(face_embed)
        fe = rearrange(fe, 'b nf (nvf d) -> b nf nvf d', nvf=self.num_vertices_per_face)

        vd = fe.shape[-1]
        verts = torch.zeros((b, nv, vd), device=dev)

        pad_v_id = nv
        verts = pad_at_dim(verts, (0, 1), dim=-2, value=0.)
        faces = faces.masked_fill(~rearrange(face_mask, 'b n -> b n 1'), pad_v_id)

        f_dim = repeat(faces, 'b nf nvf -> b (nf nvf) d', d=vd)
        fe = rearrange(fe, 'b ... d -> b (...) d')
        avg_v = scatter_mean(verts, f_dim, fe, dim=-2)

        msk = torch.ones((b, nv + 1), device=dev, dtype=torch.bool)
        msk[:, -1] = False

        qkw = dict()
        if not isinstance(self.quantizer, ResidualSimVQ):
            qkw.update(mask=msk)
        if isinstance(self.quantizer, ResidualVQ):
            qkw.update(sample_codebook_temp=rvq_sample_codebook_temp)

        def q_wrap(inp):
            unq, kw = inp
            return self.quantizer(unq, **kw)

        if self.checkpoint_quantizer:
            q_wrap = partial(checkpoint, q_wrap, use_reentrant=False)

        quant, codes, cl = q_wrap((avg_v, qkw))

        if isinstance(self.quantizer, ResidualSimVQ):
            quant = quant * msk.unsqueeze(-1)
            codes = codes * msk.unsqueeze(-1).long()

        fe_out = get_at('b [n] d, b nf nvf -> b nf (nvf d)', quant, faces)
        c_out = get_at('b [n] q, b nf nvf -> b (nf nvf) q', codes, faces)

        fm = repeat(face_mask, 'b nf -> b (nf nvf) 1', nvf=self.num_vertices_per_face)
        c_out = c_out.masked_fill(~fm, self.pad_id)

        return fe_out, c_out, cl

    @typecheck
    def decode(self, quantized: Float['b n d'], face_mask: Bool['b n']):
        cfm = rearrange(face_mask, 'b n -> b 1 n')
        x = quantized

        for lin_attn, attn, ff in self.decoder_attn_blocks:
            if exists(lin_attn):
                x = lin_attn(x, mask=face_mask) + x
            x = attn(x, mask=face_mask) + x
            x = ff(x) + x

        x = rearrange(x, 'b n d -> b d n')
        x = x.masked_fill(~cfm, 0.)
        x = self.init_decoder_conv(x)

        for rb in self.decoders:
            x = rb(x, mask=cfm)

        return rearrange(x, 'b d n -> b n d')

    @typecheck
    @torch.no_grad()
    def decode_from_codes_to_faces(self, codes: Tensor, face_mask: Bool['b n'] | None = None,
                                   return_discrete_codes=False):
        dev = next(self.parameters()).device
        if not isinstance(codes, torch.Tensor):
            codes = torch.as_tensor(codes)
        codes = codes.to(dev)

        codes = rearrange(codes, 'b ... -> b (...)')

        if not exists(face_mask):
            face_mask = reduce(codes != self.pad_id, 'b (nf nvf q) -> b nf', 'all',
                               nvf=self.num_vertices_per_face, q=self.num_quantizers)

        codes = rearrange(codes, 'b (n q) -> b n q', q=self.num_quantizers)

        quant = self.quantizer.get_output_from_indices(codes)
        quant = rearrange(quant, 'b (nf nvf) d -> b nf (nvf d)', nvf=self.num_vertices_per_face)

        dec = self.decode(quant, face_mask=face_mask)
        dec = dec.masked_fill(~face_mask[..., None], 0.)
        pfc = self.to_coor_logits(dec)
        pfc = pfc.argmax(dim=-1)
        pfc = rearrange(pfc, '... (v c) -> ... v c', v=self.num_vertices_per_face)

        cont = undiscretize(pfc, num_discrete=self.num_discrete_coors, continuous_range=self.coor_continuous_range)
        cont = cont.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1 1'), float('nan'))

        return (cont, pfc, face_mask) if return_discrete_codes else (cont, face_mask)

    @torch.no_grad()
    def tokenize(self, vertices, faces, face_edges=None, **kwargs):
        assert 'return_codes' not in kwargs

        inp = [vertices, faces, face_edges]
        inp = [*filter(exists, inp)]
        nd = {i.ndim for i in inp}

        assert len(nd) == 1
        bl = first(list(nd)) == 2

        if bl:
            inp = [rearrange(i, '... -> 1 ...') for i in inp]

        ikw = dict(zip(['vertices', 'faces', 'face_edges'], inp))
        self.eval()
        codes = self.forward(**ikw, return_codes=True, **kwargs)

        if bl:
            codes = rearrange(codes, '1 ... -> ...')

        return codes

    @typecheck
    def forward(self, *, vertices: Float['b nv 3'], faces: Int['b nf nvf'], face_edges: Int['b e 2'] | None = None,
                return_codes=False, return_loss_breakdown=False, return_recon_faces=False,
                only_return_recon_faces=False, rvq_sample_codebook_temp=1.):

        if not exists(face_edges):
            face_edges = derive_face_edges_from_faces(faces, pad_id=self.pad_id)

        fm = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')
        fem = reduce(face_edges != self.pad_id, 'b e ij -> b e', 'all')

        enc, fc = self.encode(vertices=vertices, faces=faces, face_edges=face_edges,
                              face_edges_mask=fem, face_mask=fm, return_face_coordinates=True)

        quant, codes, cl = self.quantize(face_embed=enc, faces=faces, face_mask=fm,
                                         rvq_sample_codebook_temp=rvq_sample_codebook_temp)

        if return_codes:
            codes = codes.masked_fill(~repeat(fm, 'b nf -> b (nf nvf) 1', nvf=self.num_vertices_per_face), self.pad_id)
            return codes

        dec = self.decode(quant, face_mask=fm)
        pfc = self.to_coor_logits(dec)

        if return_recon_faces or only_return_recon_faces:
            rf = undiscretize(pfc.argmax(dim=-1), num_discrete=self.num_discrete_coors,
                              continuous_range=self.coor_continuous_range)
            rf = rearrange(rf, 'b nf (nvf c) -> b nf nvf c', nvf=self.num_vertices_per_face)
            fm_exp = rearrange(fm, 'b nf -> b nf 1 1')
            rf = rf.masked_fill(~fm_exp, float('nan'))

        if only_return_recon_faces:
            return rf

        pfc = rearrange(pfc, 'b ... c -> b c (...)')
        fc = rearrange(fc, 'b ... -> b 1 (...)')

        with autocast(enabled=False):
            plp = pfc.log_softmax(dim=1)
            toh = torch.zeros_like(plp).scatter(1, fc, 1.)

            if self.bin_smooth_blur_sigma >= 0.:
                toh = gaussian_blur_1d(toh, sigma=self.bin_smooth_blur_sigma)

            rl = (-toh * plp).sum(dim=1)
            fm_exp = repeat(fm, 'b nf -> b (nf r)', r=self.num_vertices_per_face * 3)
            rl = rl[fm_exp].mean()

        tl = rl + cl.sum() * self.commit_loss_weight
        lb = (rl, cl)

        if not return_loss_breakdown:
            return (rf, tl) if return_recon_faces else tl

        return (rf, tl, lb) if return_recon_faces else (tl, lb)