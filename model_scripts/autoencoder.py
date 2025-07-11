from __future__ import annotations

from pathlib import Path
from functools import partial
from math import ceil, pi, sqrt

from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast

from pytorch_custom_utils import save_load

from beartype.typing import Tuple, Callable, List, Dict, Any

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from einops import rearrange, repeat, reduce, pack, unpack

from einx import get_at

from x_transformers.x_transformers import RMSNorm, FeedForward, LayerIntermediates

from local_attention import LocalMHA

from vector_quantize_pytorch import (
    ResidualVQ,
    ResidualLFQ
)

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from torch_geometric.nn.conv import SAGEConv

from data_scripts.helper_funcs import *
from resnet import ResnetBlock
from data_scripts.typing import Float, Int, Bool


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
            encoder_dims_through_depth: Tuple[int, ...] = (
                    64, 128, 256, 256, 576
            ),
            init_decoder_conv_kernel=7,
            decoder_dims_through_depth: Tuple[int, ...] = (
                    128, 128, 128, 128,
                    192, 192, 192, 192,
                    256, 256, 256, 256, 256, 256,
                    384, 384, 384
            ),
            dim_codebook=192,
            num_quantizers=2,  # or 'D' in the paper
            codebook_size=16384,  # they use 16k, shared codebook between layers
            use_residual_lfq=True,  # whether to use the latest lookup-free quantization
            rq_kwargs: dict = dict(
                quantize_dropout=True,
                quantize_dropout_cutoff_index=1,
                quantize_dropout_multiple_of=1,
            ),
            rvq_kwargs: dict = dict(
                kmeans_init=True,
                threshold_ema_dead_code=2,
            ),
            rlfq_kwargs: dict = dict(
                frac_per_sample_entropy=1.,
                soft_clamp_input_value=10.
            ),
            rvq_stochastic_sample_codes=True,
            sageconv_kwargs: dict = dict(
                normalize=True,
                project=True
            ),
            commit_loss_weight=0.1,
            bin_smooth_blur_sigma=0.4,  # they blur the one hot discretized coordinate positions
            attn_encoder_depth=0,
            attn_decoder_depth=0,
            local_attn_kwargs: dict = dict(
                dim_head=32,
                heads=8
            ),
            local_attn_window_size=64,
            linear_attn_kwargs: dict = dict(
                dim_head=8,
                heads=16
            ),
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
        total_coordinates_per_face = self.num_vertices_per_face * 3

        # main face coordinate embedding

        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range

        self.discretize_face_coords = partial(discretize, num_discrete=num_discrete_coors,
                                              continuous_range=coor_continuous_range)
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed)

        # derived feature embedding

        self.discretize_angle = partial(discretize, num_discrete=num_discrete_angle, continuous_range=(0., pi))
        self.angle_embed = nn.Embedding(num_discrete_angle, dim_angle_embed)

        lo, hi = coor_continuous_range
        self.discretize_area = partial(discretize, num_discrete=num_discrete_area,
                                       continuous_range=(0., (hi - lo) ** 2))
        self.area_embed = nn.Embedding(num_discrete_area, dim_area_embed)

        self.discretize_normals = partial(discretize, num_discrete=num_discrete_normals,
                                          continuous_range=coor_continuous_range)
        self.normal_embed = nn.Embedding(num_discrete_normals, dim_normal_embed)

        # attention related

        attn_kwargs = dict(
            causal=False,
            prenorm=True,
            dropout=attn_dropout,
            window_size=local_attn_window_size,
        )

        # initial dimension

        init_dim = dim_coor_embed * (
                    3 * self.num_vertices_per_face) + dim_angle_embed * self.num_vertices_per_face + dim_normal_embed * 3 + dim_area_embed

        # project into model dimension

        self.project_in = nn.Linear(init_dim, dim_codebook)

        # initial sage conv
        init_encoder_dim, *encoder_dims_through_depth = encoder_dims_through_depth
        curr_dim = init_encoder_dim

        self.init_sage_conv = SAGEConv(dim_codebook, init_encoder_dim, **sageconv_kwargs)

        self.init_encoder_act_and_norm = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(init_encoder_dim)
        )

        self.encoders = ModuleList([])

        for dim_layer in encoder_dims_through_depth:
            sage_conv = SAGEConv(
                curr_dim,
                dim_layer,
                **sageconv_kwargs
            )

            self.encoders.append(sage_conv)
            curr_dim = dim_layer

        self.encoder_attn_blocks = ModuleList([])

        for _ in range(attn_encoder_depth):
            self.encoder_attn_blocks.append(nn.ModuleList([
                TaylorSeriesLinearAttn(curr_dim, prenorm=True, **linear_attn_kwargs) if use_linear_attn else None,
                LocalMHA(dim=curr_dim, **attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(curr_dim), FeedForward(curr_dim, glu=True, dropout=ff_dropout))
            ]))

        # residual quantization

        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        self.project_dim_codebook = nn.Linear(curr_dim, dim_codebook * self.num_vertices_per_face)

        if use_residual_lfq:
            self.quantizer = ResidualLFQ(
                dim=dim_codebook,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                commitment_loss_weight=1.,
                **rlfq_kwargs,
                **rq_kwargs
            )
        else:
            self.quantizer = ResidualVQ(
                dim=dim_codebook,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                shared_codebook=True,
                commitment_weight=1.,
                stochastic_sample_codes=rvq_stochastic_sample_codes,
                **rvq_kwargs,
                **rq_kwargs
            )

        self.checkpoint_quantizer = checkpoint_quantizer  # whether to memory checkpoint the quantizer

        self.pad_id = pad_id  # for variable lengthed faces, padding quantized ids will be set to this value

        # decoder

        decoder_input_dim = dim_codebook * 3

        self.decoder_attn_blocks = ModuleList([])

        for _ in range(attn_decoder_depth):
            self.decoder_attn_blocks.append(nn.ModuleList([
                TaylorSeriesLinearAttn(decoder_input_dim, prenorm=True,
                                       **linear_attn_kwargs) if use_linear_attn else None,
                LocalMHA(dim=decoder_input_dim, **attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(decoder_input_dim), FeedForward(decoder_input_dim, glu=True, dropout=ff_dropout))
            ]))

        init_decoder_dim, *decoder_dims_through_depth = decoder_dims_through_depth
        curr_dim = init_decoder_dim

        assert is_odd(init_decoder_conv_kernel)

        self.init_decoder_conv = nn.Sequential(
            nn.Conv1d(dim_codebook * self.num_vertices_per_face, init_decoder_dim, kernel_size=init_decoder_conv_kernel,
                      padding=init_decoder_conv_kernel // 2),
            nn.SiLU(),
            Rearrange('b c n -> b n c'),
            nn.LayerNorm(init_decoder_dim),
            Rearrange('b n c -> b c n')
        )

        self.decoders = ModuleList([])

        for dim_layer in decoder_dims_through_depth:
            resnet_block = ResnetBlock(curr_dim, dim_layer, dropout=resnet_dropout)

            self.decoders.append(resnet_block)
            curr_dim = dim_layer

        self.to_coor_logits = nn.Sequential(
            nn.Linear(curr_dim, num_discrete_coors * total_coordinates_per_face),
            Rearrange('... (v c) -> ... v c', v=total_coordinates_per_face)
        )

        # loss related

        self.commit_loss_weight = commit_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: str | None,
            cache_dir: str | Path | None,
            force_download: bool,
            proxies: Dict | None,
            resume_download: bool,
            local_files_only: bool,
            token: str | bool | None,
            map_location: str = "cpu",
            strict: bool = False,
            **model_kwargs,
    ):
        model_filename = "mesh-autoencoder.bin"
        model_file = Path(model_id) / model_filename
        if not model_file.exists():
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=model_filename,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        model = cls.init_and_load(model_file, strict=strict)
        model.to(map_location)
        return model

    @typecheck
    def encode(
            self,
            *,
            vertices: Float['b nv 3'],
            faces: Int['b nf nvf'],
            face_edges: Int['b e 2'],
            face_mask: Bool['b nf'],
            face_edges_mask: Bool['b e'],
            return_face_coordinates=False
    ):
        """
        einops:
        b - batch
        nf - number of faces
        nv - number of vertices (3)
        nvf - number of vertices per face (3 or 4) - triangles vs quads
        c - coordinates (3)
        d - embed dim
        """

        _, num_faces, num_vertices_per_face = faces.shape

        assert self.num_vertices_per_face == num_vertices_per_face

        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)

        # continuous face coords

        face_coords = get_at('b [nv] c, b nf mv -> b nf mv c', vertices, face_without_pad)

        # compute derived features and embed

        derived_features = get_derived_face_features(face_coords)

        discrete_angle = self.discretize_angle(derived_features['angles'])
        angle_embed = self.angle_embed(discrete_angle)

        discrete_area = self.discretize_area(derived_features['area'])
        area_embed = self.area_embed(discrete_area)

        discrete_normal = self.discretize_normals(derived_features['normals'])
        normal_embed = self.normal_embed(discrete_normal)

        # discretize vertices for face coordinate embedding

        discrete_face_coords = self.discretize_face_coords(face_coords)
        discrete_face_coords = rearrange(discrete_face_coords,
                                         'b nf nv c -> b nf (nv c)')  # 9 or 12 coordinates per face

        face_coor_embed = self.coor_embed(discrete_face_coords)
        face_coor_embed = rearrange(face_coor_embed, 'b nf c d -> b nf (c d)')

        # combine all features and project into model dimension

        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed], 'b nf *')
        face_embed = self.project_in(face_embed)

        # handle variable lengths by using masked_select and masked_scatter

        # first handle edges
        # needs to be offset by number of faces for each batch

        face_index_offsets = reduce(face_mask.long(), 'b nf -> b', 'sum')
        face_index_offsets = F.pad(face_index_offsets.cumsum(dim=0), (1, -1), value=0)
        face_index_offsets = rearrange(face_index_offsets, 'b -> b 1 1')

        face_edges = face_edges + face_index_offsets
        face_edges = face_edges[face_edges_mask]
        face_edges = rearrange(face_edges, 'be ij -> ij be')

        # next prepare the face_mask for using masked_select and masked_scatter

        orig_face_embed_shape = face_embed.shape[:2]

        face_embed = face_embed[face_mask]

        # initial sage conv followed by activation and norm

        face_embed = self.init_sage_conv(face_embed, face_edges)

        face_embed = self.init_encoder_act_and_norm(face_embed)

        for conv in self.encoders:
            face_embed = conv(face_embed, face_edges)

        shape = (*orig_face_embed_shape, face_embed.shape[-1])

        face_embed = face_embed.new_zeros(shape).masked_scatter(rearrange(face_mask, '... -> ... 1'), face_embed)

        for linear_attn, attn, ff in self.encoder_attn_blocks:
            if exists(linear_attn):
                face_embed = linear_attn(face_embed, mask=face_mask) + face_embed

            face_embed = attn(face_embed, mask=face_mask) + face_embed
            face_embed = ff(face_embed) + face_embed

        if not return_face_coordinates:
            return face_embed

        return face_embed, discrete_face_coords

    @typecheck
    def quantize(
            self,
            *,
            faces: Int['b nf nvf'],
            face_mask: Bool['b n'],
            face_embed: Float['b nf d'],
            pad_id=None,
            rvq_sample_codebook_temp=1.
    ):
        pad_id = default(pad_id, self.pad_id)
        batch, device = faces.shape[0], faces.device

        max_vertex_index = faces.amax()
        num_vertices = int(max_vertex_index.item() + 1)

        face_embed = self.project_dim_codebook(face_embed)
        face_embed = rearrange(face_embed, 'b nf (nvf d) -> b nf nvf d', nvf=self.num_vertices_per_face)

        vertex_dim = face_embed.shape[-1]
        vertices = torch.zeros((batch, num_vertices, vertex_dim), device=device)

        # create pad vertex, due to variable lengthed faces

        pad_vertex_id = num_vertices
        vertices = pad_at_dim(vertices, (0, 1), dim=-2, value=0.)

        faces = faces.masked_fill(~rearrange(face_mask, 'b n -> b n 1'), pad_vertex_id)

        # prepare for scatter mean

        faces_with_dim = repeat(faces, 'b nf nvf -> b (nf nvf) d', d=vertex_dim)

        face_embed = rearrange(face_embed, 'b ... d -> b (...) d')

        # scatter mean

        averaged_vertices = scatter_mean(vertices, faces_with_dim, face_embed, dim=-2)

        # mask out null vertex token

        mask = torch.ones((batch, num_vertices + 1), device=device, dtype=torch.bool)
        mask[:, -1] = False

        # rvq specific kwargs

        quantize_kwargs = dict(mask=mask)

        if isinstance(self.quantizer, ResidualVQ):
            quantize_kwargs.update(sample_codebook_temp=rvq_sample_codebook_temp)

        # a quantize function that makes it memory checkpointable

        def quantize_wrapper_fn(inp):
            unquantized, quantize_kwargs = inp
            return self.quantizer(unquantized, **quantize_kwargs)

        # maybe checkpoint the quantize fn

        if self.checkpoint_quantizer:
            quantize_wrapper_fn = partial(checkpoint, quantize_wrapper_fn, use_reentrant=False)

        # residual VQ

        quantized, codes, commit_loss = quantize_wrapper_fn((averaged_vertices, quantize_kwargs))

        # gather quantized vertexes back to faces for decoding
        # now the faces have quantized vertices

        face_embed_output = get_at('b [n] d, b nf nvf -> b nf (nvf d)', quantized, faces)

        # vertex codes also need to be gathered to be organized by face sequence
        # for autoregressive learning

        codes_output = get_at('b [n] q, b nf nvf -> b (nf nvf) q', codes, faces)

        # make sure codes being outputted have this padding

        face_mask = repeat(face_mask, 'b nf -> b (nf nvf) 1', nvf=self.num_vertices_per_face)
        codes_output = codes_output.masked_fill(~face_mask, self.pad_id)

        # output quantized, codes, as well as commitment loss

        return face_embed_output, codes_output, commit_loss

    @typecheck
    def decode(
            self,
            quantized: Float['b n d'],
            face_mask: Bool['b n']
    ):
        conv_face_mask = rearrange(face_mask, 'b n -> b 1 n')

        x = quantized

        for linear_attn, attn, ff in self.decoder_attn_blocks:
            if exists(linear_attn):
                x = linear_attn(x, mask=face_mask) + x

            x = attn(x, mask=face_mask) + x
            x = ff(x) + x

        x = rearrange(x, 'b n d -> b d n')
        x = x.masked_fill(~conv_face_mask, 0.)
        x = self.init_decoder_conv(x)

        for resnet_block in self.decoders:
            x = resnet_block(x, mask=conv_face_mask)

        return rearrange(x, 'b d n -> b n d')

    @typecheck
    @torch.no_grad()
    def decode_from_codes_to_faces(
            self,
            codes: Tensor,
            face_mask: Bool['b n'] | None = None,
            return_discrete_codes=False
    ):
        codes = rearrange(codes, 'b ... -> b (...)')

        if not exists(face_mask):
            face_mask = reduce(codes != self.pad_id, 'b (nf nvf q) -> b nf', 'all', nvf=self.num_vertices_per_face,
                               q=self.num_quantizers)

        # handle different code shapes

        codes = rearrange(codes, 'b (n q) -> b n q', q=self.num_quantizers)

        # decode

        quantized = self.quantizer.get_output_from_indices(codes)
        quantized = rearrange(quantized, 'b (nf nvf) d -> b nf (nvf d)', nvf=self.num_vertices_per_face)

        decoded = self.decode(
            quantized,
            face_mask=face_mask
        )

        decoded = decoded.masked_fill(~face_mask[..., None], 0.)
        pred_face_coords = self.to_coor_logits(decoded)

        pred_face_coords = pred_face_coords.argmax(dim=-1)

        pred_face_coords = rearrange(pred_face_coords, '... (v c) -> ... v c', v=self.num_vertices_per_face)

        # back to continuous space

        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete=self.num_discrete_coors,
            continuous_range=self.coor_continuous_range
        )

        # mask out with nan

        continuous_coors = continuous_coors.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1 1'), float('nan'))

        if not return_discrete_codes:
            return continuous_coors, face_mask

        return continuous_coors, pred_face_coords, face_mask

    @torch.no_grad()
    def tokenize(self, vertices, faces, face_edges=None, **kwargs):
        assert 'return_codes' not in kwargs

        inputs = [vertices, faces, face_edges]
        inputs = [*filter(exists, inputs)]
        ndims = {i.ndim for i in inputs}

        assert len(ndims) == 1
        batch_less = first(list(ndims)) == 2

        if batch_less:
            inputs = [rearrange(i, '... -> 1 ...') for i in inputs]

        input_kwargs = dict(zip(['vertices', 'faces', 'face_edges'], inputs))

        self.eval()

        codes = self.forward(
            **input_kwargs,
            return_codes=True,
            **kwargs
        )

        if batch_less:
            codes = rearrange(codes, '1 ... -> ...')

        return codes

    @typecheck
    def forward(
            self,
            *,
            vertices: Float['b nv 3'],
            faces: Int['b nf nvf'],
            face_edges: Int['b e 2'] | None = None,
            return_codes=False,
            return_loss_breakdown=False,
            return_recon_faces=False,
            only_return_recon_faces=False,
            rvq_sample_codebook_temp=1.
    ):
        if not exists(face_edges):
            face_edges = derive_face_edges_from_faces(faces, pad_id=self.pad_id)

        device = faces.device

        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')
        face_edges_mask = reduce(face_edges != self.pad_id, 'b e ij -> b e', 'all')

        encoded, face_coordinates = self.encode(
            vertices=vertices,
            faces=faces,
            face_edges=face_edges,
            face_edges_mask=face_edges_mask,
            face_mask=face_mask,
            return_face_coordinates=True
        )

        quantized, codes, commit_loss = self.quantize(
            face_embed=encoded,
            faces=faces,
            face_mask=face_mask,
            rvq_sample_codebook_temp=rvq_sample_codebook_temp
        )

        if return_codes:
            assert not return_recon_faces, 'cannot return reconstructed faces when just returning raw codes'

            codes = codes.masked_fill(~repeat(face_mask, 'b nf -> b (nf nvf) 1', nvf=self.num_vertices_per_face),
                                      self.pad_id)
            return codes

        decode = self.decode(
            quantized,
            face_mask=face_mask
        )

        pred_face_coords = self.to_coor_logits(decode)

        # compute reconstructed faces if needed

        if return_recon_faces or only_return_recon_faces:
            recon_faces = undiscretize(
                pred_face_coords.argmax(dim=-1),
                num_discrete=self.num_discrete_coors,
                continuous_range=self.coor_continuous_range,
            )

            recon_faces = rearrange(recon_faces, 'b nf (nvf c) -> b nf nvf c', nvf=self.num_vertices_per_face)
            face_mask = rearrange(face_mask, 'b nf -> b nf 1 1')
            recon_faces = recon_faces.masked_fill(~face_mask, float('nan'))
            face_mask = rearrange(face_mask, 'b nf 1 1 -> b nf')

        if only_return_recon_faces:
            return recon_faces

        # prepare for recon loss

        pred_face_coords = rearrange(pred_face_coords, 'b ... c -> b c (...)')
        face_coordinates = rearrange(face_coordinates, 'b ... -> b 1 (...)')

        # reconstruction loss on discretized coordinates on each face
        # they also smooth (blur) the one hot positions, localized label smoothing basically

        with autocast(enabled=False):
            pred_log_prob = pred_face_coords.log_softmax(dim=1)

            target_one_hot = torch.zeros_like(pred_log_prob).scatter(1, face_coordinates, 1.)

            if self.bin_smooth_blur_sigma >= 0.:
                target_one_hot = gaussian_blur_1d(target_one_hot, sigma=self.bin_smooth_blur_sigma)

            # cross entropy with localized smoothing

            recon_losses = (-target_one_hot * pred_log_prob).sum(dim=1)

            face_mask = repeat(face_mask, 'b nf -> b (nf r)', r=self.num_vertices_per_face * 3)
            recon_loss = recon_losses[face_mask].mean()

        # calculate total loss

        total_loss = recon_loss + \
                     commit_loss.sum() * self.commit_loss_weight

        # calculate loss breakdown if needed

        loss_breakdown = (recon_loss, commit_loss)

        # some return logic

        if not return_loss_breakdown:
            if not return_recon_faces:
                return total_loss

            return recon_faces, total_loss

        if not return_recon_faces:
            return total_loss, loss_breakdown

        return recon_faces, total_loss, loss_breakdown