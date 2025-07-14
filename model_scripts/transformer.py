from __future__ import annotations

from pathlib import Path

from pytorch_custom_utils import save_load

from beartype.typing import Tuple, Callable, List, Dict, Any

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from einops import rearrange, repeat, reduce, pack, unpack

from x_transformers import Decoder
from x_transformers.x_transformers import RMSNorm, FeedForward, LayerIntermediates

from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    top_k,
    top_p,
)

from classifier_free_guidance_pytorch import (
    classifier_free_guidance,
    TextEmbeddingReturner
)

from gateloop_transformer import SimpleGateLoopLayer

from tqdm import tqdm

from data_scripts.helper_funcs import *
from model_scripts.resnet import FiLM
from data_scripts.typing_mod import Int
from model_scripts.autoencoder import MeshAutoencoder


class GateLoopBlock(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        use_heinsen = True
    ):
        super().__init__()
        self.gateloops = ModuleList([])

        for _ in range(depth):
            gateloop = SimpleGateLoopLayer(dim = dim, use_heinsen = use_heinsen)
            self.gateloops.append(gateloop)

    def forward(
        self,
        x,
        cache = None
    ):
        received_cache = exists(cache)

        if is_tensor_empty(x):
            return x, None

        if received_cache:
            prev, x = x[:, :-1], x[:, -1:]

        cache = default(cache, [])
        cache = iter(cache)

        new_caches = []
        for gateloop in self.gateloops:
            layer_cache = next(cache, None)
            out, new_cache = gateloop(x, cache = layer_cache, return_cache = True)
            new_caches.append(new_cache)
            x = x + out

        if received_cache:
            x = torch.cat((prev, x), dim = -2)

        return x, new_caches


@save_load()
class MeshTransformer(Module, PyTorchModelHubMixin):
    @typecheck
    def __init__(
            self,
            autoencoder: MeshAutoencoder,
            *,
            dim: int | Tuple[int, int] = 512,
            max_seq_len=8192,
            flash_attn=True,
            attn_depth=12,
            attn_dim_head=64,
            attn_heads=16,
            attn_kwargs: dict = dict(
                ff_glu=True,
                attn_num_mem_kv=4
            ),
            cross_attn_num_mem_kv=4,  # needed for preventing nan when dropping out text condition
            dropout=0.,
            coarse_pre_gateloop_depth=2,
            coarse_post_gateloop_depth=0,
            coarse_adaptive_rmsnorm=False,
            fine_pre_gateloop_depth=2,
            gateloop_use_heinsen=False,
            fine_attn_depth=2,
            fine_attn_dim_head=32,
            fine_attn_heads=8,
            fine_cross_attend_text=False,  # additional conditioning - fine transformer cross attention to text tokens
            pad_id=-1,
            num_sos_tokens=None,
            condition_on_text=False,
            text_cond_with_film=False,
            text_condition_model_types=('t5',),
            text_condition_model_kwargs=(dict(),),
            text_condition_cond_drop_prob=0.25,
            quads=False,
    ):
        super().__init__()
        self.num_vertices_per_face = 3 if not quads else 4

        assert autoencoder.num_vertices_per_face == self.num_vertices_per_face, 'autoencoder and transformer must both support the same type of mesh (either all triangles, or all quads)'

        dim, dim_fine = (dim, dim) if isinstance(dim, int) else dim

        self.autoencoder = autoencoder
        set_module_requires_grad_(autoencoder, False)

        self.codebook_size = autoencoder.codebook_size
        self.num_quantizers = autoencoder.num_quantizers

        self.eos_token_id = self.codebook_size

        # the fine transformer sos token
        # as well as a projection of pooled text embeddings to condition it

        num_sos_tokens = default(num_sos_tokens, 1 if not condition_on_text else 4)
        assert num_sos_tokens > 0

        self.num_sos_tokens = num_sos_tokens
        self.sos_token = nn.Parameter(torch.randn(num_sos_tokens, dim))

        # they use axial positional embeddings

        assert divisible_by(max_seq_len,
                            self.num_vertices_per_face * self.num_quantizers), f'max_seq_len ({max_seq_len}) must be divisible by (3 x {self.num_quantizers}) = {3 * self.num_quantizers}'  # 3 or 4 vertices per face, with D codes per vertex

        self.token_embed = nn.Embedding(self.codebook_size + 1, dim)

        self.quantize_level_embed = nn.Parameter(torch.randn(self.num_quantizers, dim))
        self.vertex_embed = nn.Parameter(torch.randn(self.num_vertices_per_face, dim))

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len

        # text condition

        self.condition_on_text = condition_on_text
        self.conditioner = None

        cross_attn_dim_context = None
        dim_text = None

        if condition_on_text:
            self.conditioner = TextEmbeddingReturner(
                model_types=text_condition_model_types,
                model_kwargs=text_condition_model_kwargs,
                cond_drop_prob=text_condition_cond_drop_prob,
                text_embed_pad_value=-1.
            )

            dim_text = self.conditioner.dim_latent
            cross_attn_dim_context = dim_text

            self.text_coarse_film_cond = FiLM(dim_text, dim) if text_cond_with_film else identity
            self.text_fine_film_cond = FiLM(dim_text, dim_fine) if text_cond_with_film else identity

        # for summarizing the vertices of each face

        self.to_face_tokens = nn.Sequential(
            nn.Linear(self.num_quantizers * self.num_vertices_per_face * dim, dim),
            nn.LayerNorm(dim)
        )

        self.coarse_gateloop_block = GateLoopBlock(dim, depth=coarse_pre_gateloop_depth,
                                                   use_heinsen=gateloop_use_heinsen) if coarse_pre_gateloop_depth > 0 else None

        self.coarse_post_gateloop_block = GateLoopBlock(dim, depth=coarse_post_gateloop_depth,
                                                        use_heinsen=gateloop_use_heinsen) if coarse_post_gateloop_depth > 0 else None

        # main autoregressive attention network
        # attending to a face token

        self.coarse_adaptive_rmsnorm = coarse_adaptive_rmsnorm

        self.decoder = Decoder(
            dim=dim,
            depth=attn_depth,
            heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_flash=flash_attn,
            attn_dropout=dropout,
            ff_dropout=dropout,
            use_adaptive_rmsnorm=coarse_adaptive_rmsnorm,
            dim_condition=dim_text,
            cross_attend=condition_on_text,
            cross_attn_dim_context=cross_attn_dim_context,
            cross_attn_num_mem_kv=cross_attn_num_mem_kv,
            **attn_kwargs
        )

        # projection from coarse to fine, if needed

        self.maybe_project_coarse_to_fine = nn.Linear(dim, dim_fine) if dim != dim_fine else nn.Identity()

        # address a weakness in attention

        self.fine_gateloop_block = GateLoopBlock(dim, depth=fine_pre_gateloop_depth,
                                                 use_heinsen=gateloop_use_heinsen) if fine_pre_gateloop_depth > 0 else None

        # decoding the vertices, 2-stage hierarchy

        self.fine_cross_attend_text = condition_on_text and fine_cross_attend_text

        self.fine_decoder = Decoder(
            dim=dim_fine,
            depth=fine_attn_depth,
            heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_flash=flash_attn,
            attn_dropout=dropout,
            ff_dropout=dropout,
            cross_attend=self.fine_cross_attend_text,
            cross_attn_dim_context=cross_attn_dim_context,
            cross_attn_num_mem_kv=cross_attn_num_mem_kv,
            **attn_kwargs
        )

        # to logits

        self.to_logits = nn.Linear(dim_fine, self.codebook_size + 1)

        # padding id
        # force the autoencoder to use the same pad_id given in transformer

        self.pad_id = pad_id
        autoencoder.pad_id = pad_id

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
        model_filename = "mesh-transformer.bin"
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

    @property
    def device(self):
        return next(self.parameters()).device

    @typecheck
    @torch.no_grad()
    def embed_texts(self, texts: str | List[str]):
        single_text = not isinstance(texts, list)
        if single_text:
            texts = [texts]

        assert exists(self.conditioner)
        text_embeds = self.conditioner.embed_texts(texts).detach()

        if single_text:
            text_embeds = text_embeds[0]

        return text_embeds

    @eval_decorator
    @torch.no_grad()
    @typecheck
    def generate(
            self,
            prompt: Tensor | None = None,
            batch_size: int | None = None,
            filter_logits_fn: Callable = top_k,
            filter_kwargs: dict = dict(),
            temperature=1.,
            return_codes=False,
            texts: List[str] | None = None,
            text_embeds: Tensor | None = None,
            cond_scale=1.,
            cache_kv=True,
            max_seq_len=None,
            face_coords_to_file: Callable[[Tensor], Any] | None = None
    ):
        max_seq_len = default(max_seq_len, self.max_seq_len)

        if exists(prompt):
            assert not exists(batch_size)

            prompt = rearrange(prompt, 'b ... -> b (...)')
            assert prompt.shape[-1] <= self.max_seq_len

            batch_size = prompt.shape[0]

        if self.condition_on_text:
            assert exists(texts) ^ exists(
                text_embeds), '`text` or `text_embeds` must be passed in if `condition_on_text` is set to True'
            if exists(texts):
                text_embeds = self.embed_texts(texts)

            batch_size = default(batch_size, text_embeds.shape[0])

        batch_size = default(batch_size, 1)

        codes = default(prompt, torch.empty((batch_size, 0), dtype=torch.long, device=self.device))

        curr_length = codes.shape[-1]

        cache = (None, None)

        for i in tqdm(range(curr_length, max_seq_len)):

            # example below for triangles, extrapolate for quads
            # v1([q1] [q2] [q1] [q2] [q1] [q2]) v2([eos| q1] [q2] [q1] [q2] [q1] [q2]) -> 0 1 2 3 4 5 6 7 8 9 10 11 12 -> v1(F F F F F F) v2(T F F F F F) v3(T F F F F F)

            can_eos = i != 0 and divisible_by(i,
                                              self.num_quantizers * self.num_vertices_per_face)  # only allow for eos to be decoded at the end of each face, defined as 3 or 4 vertices with D residual VQ codes

            output = self.forward_on_codes(
                codes,
                text_embeds=text_embeds,
                return_loss=False,
                return_cache=cache_kv,
                append_eos=False,
                cond_scale=cond_scale,
                cfg_routed_kwargs=dict(
                    cache=cache
                )
            )

            if cache_kv:
                logits, cache = output

                if cond_scale == 1.:
                    cache = (cache, None)
            else:
                logits = output

            logits = logits[:, -1]

            if not can_eos:
                logits[:, -1] = -torch.finfo(logits.dtype).max

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            if temperature == 0.:
                sample = filtered_logits.argmax(dim=-1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            codes, _ = pack([codes, sample], 'b *')

            # check for all rows to have [eos] to terminate

            is_eos_codes = (codes == self.eos_token_id)

            if is_eos_codes.any(dim=-1).all():
                break

        # mask out to padding anything after the first eos

        mask = is_eos_codes.float().cumsum(dim=-1) >= 1
        codes = codes.masked_fill(mask, self.pad_id)

        # remove a potential extra token from eos, if breaked early

        code_len = codes.shape[-1]
        round_down_code_len = code_len // self.num_vertices_per_face * self.num_vertices_per_face
        codes = codes[:, :round_down_code_len]

        # early return of raw residual quantizer codes

        if return_codes:
            codes = rearrange(codes, 'b (n q) -> b n q', q=self.num_quantizers)
            return codes

        self.autoencoder.eval()
        face_coords, face_mask = self.autoencoder.decode_from_codes_to_faces(codes)

        if not exists(face_coords_to_file):
            return face_coords, face_mask

        files = [face_coords_to_file(coords[mask]) for coords, mask in zip(face_coords, face_mask)]
        return files

    def forward(
            self,
            *,
            vertices: Int['b nv 3'],
            faces: Int['b nf nvf'],
            face_edges: Int['b e 2'] | None = None,
            codes: Tensor | None = None,
            cache: LayerIntermediates | None = None,
            **kwargs
    ):
        if not exists(codes):
            codes = self.autoencoder.tokenize(
                vertices=vertices,
                faces=faces,
                face_edges=face_edges
            )

        return self.forward_on_codes(codes, cache=cache, **kwargs)

    @classifier_free_guidance
    def forward_on_codes(
            self,
            codes=None,
            return_loss=True,
            return_cache=False,
            append_eos=True,
            cache=None,
            texts: List[str] | None = None,
            text_embeds: Tensor | None = None,
            cond_drop_prob=None
    ):
        # handle text conditions

        attn_context_kwargs = dict()

        if self.condition_on_text:
            assert exists(texts) ^ exists(
                text_embeds), '`text` or `text_embeds` must be passed in if `condition_on_text` is set to True'

            if exists(texts):
                text_embeds = self.conditioner.embed_texts(texts)

            if exists(codes):
                assert text_embeds.shape[0] == codes.shape[
                    0], 'batch size of texts or text embeddings is not equal to the batch size of the mesh codes'

            _, maybe_dropped_text_embeds = self.conditioner(
                text_embeds=text_embeds,
                cond_drop_prob=cond_drop_prob
            )

            text_embed, text_mask = maybe_dropped_text_embeds

            pooled_text_embed = masked_mean(text_embed, text_mask, dim=1)

            attn_context_kwargs = dict(
                context=text_embed,
                context_mask=text_mask
            )

            if self.coarse_adaptive_rmsnorm:
                attn_context_kwargs.update(
                    condition=pooled_text_embed
                )

        # take care of codes that may be flattened

        if codes.ndim > 2:
            codes = rearrange(codes, 'b ... -> b (...)')

        # get some variable

        batch, seq_len, device = *codes.shape, codes.device

        assert seq_len <= self.max_seq_len, f'received codes of length {seq_len} but needs to be less than or equal to set max_seq_len {self.max_seq_len}'

        # auto append eos token

        if append_eos:
            assert exists(codes)

            code_lens = ((codes == self.pad_id).cumsum(dim=-1) == 0).sum(dim=-1)

            codes = F.pad(codes, (0, 1), value=0)

            batch_arange = torch.arange(batch, device=device)

            batch_arange = rearrange(batch_arange, '... -> ... 1')
            code_lens = rearrange(code_lens, '... -> ... 1')

            codes[batch_arange, code_lens] = self.eos_token_id

        # if returning loss, save the labels for cross entropy

        if return_loss:
            assert seq_len > 0
            codes, labels = codes[:, :-1], codes

        # token embed (each residual VQ id)

        codes = codes.masked_fill(codes == self.pad_id, 0)
        codes = self.token_embed(codes)

        # codebook embed + absolute positions

        seq_arange = torch.arange(codes.shape[-2], device=device)

        codes = codes + self.abs_pos_emb(seq_arange)

        # embedding for quantizer level

        code_len = codes.shape[1]

        level_embed = repeat(self.quantize_level_embed, 'q d -> (r q) d', r=ceil(code_len / self.num_quantizers))
        codes = codes + level_embed[:code_len]

        # embedding for each vertex

        vertex_embed = repeat(self.vertex_embed, 'nv d -> (r nv q) d',
                              r=ceil(code_len / (self.num_vertices_per_face * self.num_quantizers)),
                              q=self.num_quantizers)
        codes = codes + vertex_embed[:code_len]

        # create a token per face, by summarizing the 3 or 4 vertices
        # this is similar in design to the RQ transformer from Lee et al. https://arxiv.org/abs/2203.01941

        num_tokens_per_face = self.num_quantizers * self.num_vertices_per_face

        curr_vertex_pos = code_len % num_tokens_per_face  # the current intra-face vertex-code position id, needed for caching at the fine decoder stage

        code_len_is_multiple_of_face = divisible_by(code_len, num_tokens_per_face)

        next_multiple_code_len = ceil(code_len / num_tokens_per_face) * num_tokens_per_face

        codes = pad_to_length(codes, next_multiple_code_len, dim=-2)

        # grouped codes will be used for the second stage

        grouped_codes = rearrange(codes, 'b (nf n) d -> b nf n d', n=num_tokens_per_face)

        # create the coarse tokens for the first attention network

        face_codes = grouped_codes if code_len_is_multiple_of_face else grouped_codes[:, :-1]
        face_codes = rearrange(face_codes, 'b nf n d -> b nf (n d)')
        face_codes = self.to_face_tokens(face_codes)

        face_codes_len = face_codes.shape[-2]

        # cache logic

        (
            cached_attended_face_codes,
            coarse_cache,
            fine_cache,
            coarse_gateloop_cache,
            coarse_post_gateloop_cache,
            fine_gateloop_cache
        ) = cache if exists(cache) else ((None,) * 6)

        if exists(cache):
            cached_face_codes_len = cached_attended_face_codes.shape[-2]
            cached_face_codes_len_without_sos = cached_face_codes_len - 1

            need_call_first_transformer = face_codes_len > cached_face_codes_len_without_sos
        else:
            # auto prepend sos token

            sos = repeat(self.sos_token, 'n d -> b n d', b=batch)
            face_codes, packed_sos_shape = pack([sos, face_codes], 'b * d')

            # if no kv cache, always call first transformer

            need_call_first_transformer = True

        should_cache_fine = not divisible_by(curr_vertex_pos + 1, num_tokens_per_face)

        # condition face codes with text if needed

        if self.condition_on_text:
            face_codes = self.text_coarse_film_cond(face_codes, pooled_text_embed)

        # attention on face codes (coarse)

        if need_call_first_transformer:
            if exists(self.coarse_gateloop_block):
                face_codes, coarse_gateloop_cache = self.coarse_gateloop_block(face_codes, cache=coarse_gateloop_cache)

            attended_face_codes, coarse_cache = self.decoder(
                face_codes,
                cache=coarse_cache,
                return_hiddens=True,
                **attn_context_kwargs
            )

            if exists(self.coarse_post_gateloop_block):
                face_codes, coarse_post_gateloop_cache = self.coarse_post_gateloop_block(face_codes,
                                                                                         cache=coarse_post_gateloop_cache)

        else:
            attended_face_codes = None

        attended_face_codes = safe_cat((cached_attended_face_codes, attended_face_codes), dim=-2)

        # if calling without kv cache, pool the sos tokens, if greater than 1 sos token

        if not exists(cache):
            sos_tokens, attended_face_codes = unpack(attended_face_codes, packed_sos_shape, 'b * d')
            last_sos_token = sos_tokens[:, -1:]
            attended_face_codes = torch.cat((last_sos_token, attended_face_codes), dim=1)

        # maybe project from coarse to fine dimension for hierarchical transformers

        attended_face_codes = self.maybe_project_coarse_to_fine(attended_face_codes)

        grouped_codes = pad_to_length(grouped_codes, attended_face_codes.shape[-2], dim=1)
        fine_vertex_codes, _ = pack([attended_face_codes, grouped_codes], 'b n * d')

        fine_vertex_codes = fine_vertex_codes[..., :-1, :]

        # gateloop layers

        if exists(self.fine_gateloop_block):
            fine_vertex_codes = rearrange(fine_vertex_codes, 'b nf n d -> b (nf n) d')
            orig_length = fine_vertex_codes.shape[-2]
            fine_vertex_codes = fine_vertex_codes[:, :(code_len + 1)]

            fine_vertex_codes, fine_gateloop_cache = self.fine_gateloop_block(fine_vertex_codes,
                                                                              cache=fine_gateloop_cache)

            fine_vertex_codes = pad_to_length(fine_vertex_codes, orig_length, dim=-2)
            fine_vertex_codes = rearrange(fine_vertex_codes, 'b (nf n) d -> b nf n d', n=num_tokens_per_face)

        # fine attention - 2nd stage

        if exists(cache):
            fine_vertex_codes = fine_vertex_codes[:, -1:]

            if exists(fine_cache):
                for attn_intermediate in fine_cache.attn_intermediates:
                    ck, cv = attn_intermediate.cached_kv
                    ck, cv = [rearrange(t, '(b nf) ... -> b nf ...', b=batch) for t in (ck, cv)]

                    # when operating on the cached key / values, treat self attention and cross attention differently

                    layer_type = attn_intermediate.layer_type

                    if layer_type == 'a':
                        ck, cv = [t[:, -1, :, :curr_vertex_pos] for t in (ck, cv)]
                    elif layer_type == 'c':
                        ck, cv = [t[:, -1, ...] for t in (ck, cv)]

                    attn_intermediate.cached_kv = (ck, cv)

        num_faces = fine_vertex_codes.shape[1]
        one_face = num_faces == 1

        fine_vertex_codes = rearrange(fine_vertex_codes, 'b nf n d -> (b nf) n d')

        if one_face:
            fine_vertex_codes = fine_vertex_codes[:, :(curr_vertex_pos + 1)]

        # handle maybe cross attention conditioning of fine transformer with text

        fine_attn_context_kwargs = dict()

        # optional text cross attention conditioning for fine transformer

        if self.fine_cross_attend_text:
            repeat_batch = fine_vertex_codes.shape[0] // text_embed.shape[0]

            text_embed = repeat(text_embed, 'b ... -> (b r) ...', r=repeat_batch)
            text_mask = repeat(text_mask, 'b ... -> (b r) ...', r=repeat_batch)

            fine_attn_context_kwargs = dict(
                context=text_embed,
                context_mask=text_mask
            )

        # also film condition the fine vertex codes

        if self.condition_on_text:
            repeat_batch = fine_vertex_codes.shape[0] // pooled_text_embed.shape[0]

            pooled_text_embed = repeat(pooled_text_embed, 'b ... -> (b r) ...', r=repeat_batch)
            fine_vertex_codes = self.text_fine_film_cond(fine_vertex_codes, pooled_text_embed)

        # fine transformer

        attended_vertex_codes, fine_cache = self.fine_decoder(
            fine_vertex_codes,
            cache=fine_cache,
            **fine_attn_context_kwargs,
            return_hiddens=True
        )

        if not should_cache_fine:
            fine_cache = None

        if not one_face:
            # reconstitute original sequence

            embed = rearrange(attended_vertex_codes, '(b nf) n d -> b (nf n) d', b=batch)
            embed = embed[:, :(code_len + 1)]
        else:
            embed = attended_vertex_codes

        # logits

        logits = self.to_logits(embed)

        if not return_loss:
            if not return_cache:
                return logits

            next_cache = (
                attended_face_codes,
                coarse_cache,
                fine_cache,
                coarse_gateloop_cache,
                coarse_post_gateloop_cache,
                fine_gateloop_cache
            )

            return logits, next_cache

        # loss

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index=self.pad_id
        )

        return ce_loss