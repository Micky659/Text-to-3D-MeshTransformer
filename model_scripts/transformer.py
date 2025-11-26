from __future__ import annotations
from pathlib import Path
from pytorch_custom_utils import save_load
from beartype.typing import Tuple, Callable, List, Any
from huggingface_hub import PyTorchModelHubMixin
from einops import rearrange, repeat, reduce, pack, unpack
from x_transformers import Decoder
from x_transformers.x_transformers import RMSNorm, FeedForward
from x_transformers.autoregressive_wrapper import top_k
from classifier_free_guidance_pytorch import classifier_free_guidance, TextEmbeddingReturner
from gateloop_transformer import SimpleGateLoopLayer
from tqdm import tqdm
from data_scripts.helper_funcs import *
from model_scripts.resnet import FiLM
from data_scripts.typing_mod import Int
from model_scripts.autoencoder import MeshAutoencoder


class GateLoopBlock(Module):
    def __init__(self, dim, *, depth, use_heinsen=True):
        super().__init__()
        self.gateloops = nn.ModuleList([SimpleGateLoopLayer(dim=dim, use_heinsen=use_heinsen) for _ in range(depth)])

    def forward(self, x, cache=None):
        rcv = exists(cache)

        if is_tensor_empty(x):
            return x, None

        if rcv:
            prev, x = x[:, :-1], x[:, -1:]

        cache = default(cache, [])
        cache = iter(cache)

        nc = []
        for gl in self.gateloops:
            lc = next(cache, None)
            out, new_c = gl(x, cache=lc, return_cache=True)
            nc.append(new_c)
            x = x + out

        if rcv:
            x = torch.cat((prev, x), dim=-2)

        return x, nc


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
            attn_kwargs: dict = dict(ff_glu=True, attn_num_mem_kv=4),
            cross_attn_num_mem_kv=4,
            dropout=0.,
            coarse_pre_gateloop_depth=2,
            coarse_post_gateloop_depth=0,
            coarse_adaptive_rmsnorm=False,
            fine_pre_gateloop_depth=2,
            gateloop_use_heinsen=False,
            fine_attn_depth=2,
            fine_attn_dim_head=32,
            fine_attn_heads=8,
            fine_cross_attend_text=False,
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

        assert autoencoder.num_vertices_per_face == self.num_vertices_per_face

        d, df = (dim, dim) if isinstance(dim, int) else dim

        self.autoencoder = autoencoder
        set_module_requires_grad_(autoencoder, False)

        self.codebook_size = autoencoder.codebook_size
        self.num_quantizers = autoencoder.num_quantizers
        self.eos_token_id = self.codebook_size

        num_sos_tokens = default(num_sos_tokens, 1 if not condition_on_text else 4)
        self.num_sos_tokens = num_sos_tokens
        self.sos_token = nn.Parameter(torch.randn(num_sos_tokens, d))

        assert divisible_by(max_seq_len, self.num_vertices_per_face * self.num_quantizers)

        self.token_embed = nn.Embedding(self.codebook_size + 1, d)
        self.quantize_level_embed = nn.Parameter(torch.randn(self.num_quantizers, d))
        self.vertex_embed = nn.Parameter(torch.randn(self.num_vertices_per_face, d))
        self.abs_pos_emb = nn.Embedding(max_seq_len, d)
        self.max_seq_len = max_seq_len

        self.condition_on_text = condition_on_text
        self.conditioner = None

        ca_ctx = None
        dt = None

        if condition_on_text:
            self.conditioner = TextEmbeddingReturner(
                model_types=text_condition_model_types,
                model_kwargs=text_condition_model_kwargs,
                cond_drop_prob=text_condition_cond_drop_prob,
                text_embed_pad_value=-1.)

            dt = self.conditioner.dim_latent
            ca_ctx = dt

            self.text_coarse_film_cond = FiLM(dt, d) if text_cond_with_film else identity
            self.text_fine_film_cond = FiLM(dt, df) if text_cond_with_film else identity

        self.to_face_tokens = nn.Sequential(nn.Linear(self.num_quantizers * self.num_vertices_per_face * d, d),
                                            nn.LayerNorm(d))

        self.coarse_gateloop_block = GateLoopBlock(d, depth=coarse_pre_gateloop_depth,
                                                   use_heinsen=gateloop_use_heinsen) if coarse_pre_gateloop_depth > 0 else None

        self.coarse_post_gateloop_block = GateLoopBlock(d, depth=coarse_post_gateloop_depth,
                                                        use_heinsen=gateloop_use_heinsen) if coarse_post_gateloop_depth > 0 else None

        self.coarse_adaptive_rmsnorm = coarse_adaptive_rmsnorm

        self.decoder = Decoder(dim=d, depth=attn_depth, heads=attn_heads, attn_dim_head=attn_dim_head,
                               attn_flash=flash_attn, attn_dropout=dropout, ff_dropout=dropout,
                               use_adaptive_rmsnorm=coarse_adaptive_rmsnorm, dim_condition=dt,
                               cross_attend=condition_on_text, cross_attn_dim_context=ca_ctx,
                               cross_attn_num_mem_kv=cross_attn_num_mem_kv, **attn_kwargs)

        self.maybe_project_coarse_to_fine = nn.Linear(d, df) if d != df else nn.Identity()

        self.fine_gateloop_block = GateLoopBlock(d, depth=fine_pre_gateloop_depth,
                                                 use_heinsen=gateloop_use_heinsen) if fine_pre_gateloop_depth > 0 else None

        self.fine_cross_attend_text = condition_on_text and fine_cross_attend_text

        self.fine_decoder = Decoder(dim=df, depth=fine_attn_depth, heads=attn_heads, attn_dim_head=attn_dim_head,
                                    attn_flash=flash_attn, attn_dropout=dropout, ff_dropout=dropout,
                                    cross_attend=self.fine_cross_attend_text, cross_attn_dim_context=ca_ctx,
                                    cross_attn_num_mem_kv=cross_attn_num_mem_kv, **attn_kwargs)

        self.to_logits = nn.Linear(df, self.codebook_size + 1)
        self.pad_id = pad_id
        autoencoder.pad_id = pad_id

    @property
    def device(self):
        return next(self.parameters()).device

    @typecheck
    @torch.no_grad()
    def embed_texts(self, texts: str | List[str]):
        s = not isinstance(texts, list)
        if s:
            texts = [texts]

        assert exists(self.conditioner)
        te = self.conditioner.embed_texts(texts).detach()

        if s:
            te = te[0]

        return te

    @torch.no_grad()
    @typecheck
    def generate(self, prompt: Tensor | None = None, batch_size: int | None = None, filter_logits_fn: Callable = top_k,
                 filter_kwargs: dict = dict(), temperature=1., return_codes=False, texts: List[str] | None = None,
                 text_embeds: Tensor | None = None, cond_scale=1., cache_kv=True, max_seq_len=None,
                 face_coords_to_file: Callable[[Tensor], Any] | None = None):

        msl = default(max_seq_len, self.max_seq_len)

        if exists(prompt):
            assert not exists(batch_size)
            prompt = rearrange(prompt, 'b ... -> b (...)')
            assert prompt.shape[-1] <= self.max_seq_len
            batch_size = prompt.shape[0]

        if self.condition_on_text:
            assert exists(texts) ^ exists(text_embeds)
            if exists(texts):
                text_embeds = self.embed_texts(texts)
            batch_size = default(batch_size, text_embeds.shape[0])

        batch_size = default(batch_size, 1)
        codes = default(prompt, torch.empty((batch_size, 0), dtype=torch.long, device=self.device))
        cl = codes.shape[-1]
        cache = (None, None)

        for i in tqdm(range(cl, msl)):
            can_eos = i != 0 and divisible_by(i, self.num_quantizers * self.num_vertices_per_face)

            out = self.forward_on_codes(codes, text_embeds=text_embeds, return_loss=False, return_cache=cache_kv,
                                        append_eos=False, cond_scale=cond_scale, cfg_routed_kwargs=dict(cache=cache))

            if cache_kv:
                logits, cache = out
                if cond_scale == 1.:
                    cache = (cache, None)
            else:
                logits = out

            logits = logits[:, -1]

            if not can_eos:
                logits[:, -1] = -torch.finfo(logits.dtype).max

            fl = filter_logits_fn(logits, **filter_kwargs)

            if temperature == 0.:
                samp = fl.argmax(dim=-1)
            else:
                probs = F.softmax(fl / temperature, dim=-1)
                samp = torch.multinomial(probs, 1)

            codes, _ = pack([codes, samp], 'b *')

            is_eos = (codes == self.eos_token_id)
            if is_eos.any(dim=-1).all():
                break

        msk = is_eos.float().cumsum(dim=-1) >= 1
        codes = codes.masked_fill(msk, self.pad_id)

        cl = codes.shape[-1]
        rd_cl = cl // self.num_vertices_per_face * self.num_vertices_per_face
        codes = codes[:, :rd_cl]

        if return_codes:
            codes = rearrange(codes, 'b (n q) -> b n q', q=self.num_quantizers)
            return codes

        self.autoencoder.eval()
        fc, fm = self.autoencoder.decode_from_codes_to_faces(codes)

        if not exists(face_coords_to_file):
            return fc, fm

        files = [face_coords_to_file(coords[mask]) for coords, mask in zip(fc, fm)]
        return files

    def forward(self, *, vertices: Int['b nv 3'], faces: Int['b nf nvf'], face_edges: Int['b e 2'] | None = None,
                codes: Tensor | None = None, cache=None, **kwargs):

        if not exists(codes):
            codes = self.autoencoder.tokenize(vertices=vertices, faces=faces, face_edges=face_edges)

        return self.forward_on_codes(codes, cache=cache, **kwargs)

    @classifier_free_guidance
    def forward_on_codes(self, codes=None, return_loss=True, return_cache=False, append_eos=True, cache=None,
                         texts: List[str] | None = None, text_embeds: Tensor | None = None, cond_drop_prob=None):

        akw = dict()

        if self.condition_on_text:
            assert exists(texts) ^ exists(text_embeds)

            if exists(texts):
                text_embeds = self.conditioner.embed_texts(texts)

            if exists(codes):
                assert text_embeds.shape[0] == codes.shape[0]

            _, mdte = self.conditioner(text_embeds=text_embeds, cond_drop_prob=cond_drop_prob)
            te, tm = mdte

            pte = masked_mean(te, tm, dim=1)

            akw = dict(context=te, context_mask=tm)

            if self.coarse_adaptive_rmsnorm:
                akw.update(condition=pte)

        if codes.ndim > 2:
            codes = rearrange(codes, 'b ... -> b (...)')

        b, sl, dev = *codes.shape, codes.device

        assert sl <= self.max_seq_len

        if append_eos:
            assert exists(codes)
            cl = ((codes == self.pad_id).cumsum(dim=-1) == 0).sum(dim=-1)
            codes = F.pad(codes, (0, 1), value=0)
            ba = torch.arange(b, device=dev)
            ba = rearrange(ba, '... -> ... 1')
            cl = rearrange(cl, '... -> ... 1')
            codes[ba, cl] = self.eos_token_id

        if return_loss:
            assert sl > 0
            codes, labels = codes[:, :-1], codes

        codes = codes.masked_fill(codes == self.pad_id, 0)
        codes = self.token_embed(codes)

        sa = torch.arange(codes.shape[-2], device=dev)
        codes = codes + self.abs_pos_emb(sa)

        cl = codes.shape[1]

        le = repeat(self.quantize_level_embed, 'q d -> (r q) d', r=ceil(cl / self.num_quantizers))
        codes = codes + le[:cl]

        ve = repeat(self.vertex_embed, 'nv d -> (r nv q) d',
                    r=ceil(cl / (self.num_vertices_per_face * self.num_quantizers)), q=self.num_quantizers)
        codes = codes + ve[:cl]

        ntpf = self.num_quantizers * self.num_vertices_per_face
        cvp = cl % ntpf
        cl_mult = divisible_by(cl, ntpf)
        nmc = ceil(cl / ntpf) * ntpf

        codes = pad_to_length(codes, nmc, dim=-2)
        gc = rearrange(codes, 'b (nf n) d -> b nf n d', n=ntpf)

        fc = gc if cl_mult else gc[:, :-1]
        fc = rearrange(fc, 'b nf n d -> b nf (n d)')
        fc = self.to_face_tokens(fc)

        fcl = fc.shape[-2]

        (cafc, cc, finec, cglc, cpglc, fglc) = cache if exists(cache) else ((None,) * 6)

        if exists(cache):
            cfcl = cafc.shape[-2]
            cfcl_nos = cfcl - 1
            ncft = fcl > cfcl_nos
        else:
            sos = repeat(self.sos_token, 'n d -> b n d', b=b)
            fc, pss = pack([sos, fc], 'b * d')
            ncft = True

        scf = not divisible_by(cvp + 1, ntpf)

        if self.condition_on_text:
            fc = self.text_coarse_film_cond(fc, pte)

        if ncft:
            if exists(self.coarse_gateloop_block):
                fc, cglc = self.coarse_gateloop_block(fc, cache=cglc)

            afc, cc = self.decoder(fc, cache=cc, return_hiddens=True, **akw)

            if exists(self.coarse_post_gateloop_block):
                fc, cpglc = self.coarse_post_gateloop_block(fc, cache=cpglc)
        else:
            afc = None

        afc = safe_cat((cafc, afc), dim=-2)

        if not exists(cache):
            st, afc = unpack(afc, pss, 'b * d')
            lst = st[:, -1:]
            afc = torch.cat((lst, afc), dim=1)

        afc = self.maybe_project_coarse_to_fine(afc)

        gc = pad_to_length(gc, afc.shape[-2], dim=1)
        fvc, _ = pack([afc, gc], 'b n * d')
        fvc = fvc[..., :-1, :]

        if exists(self.fine_gateloop_block):
            fvc = rearrange(fvc, 'b nf n d -> b (nf n) d')
            ol = fvc.shape[-2]
            fvc = fvc[:, :(cl + 1)]
            fvc, fglc = self.fine_gateloop_block(fvc, cache=fglc)
            fvc = pad_to_length(fvc, ol, dim=-2)
            fvc = rearrange(fvc, 'b (nf n) d -> b nf n d', n=ntpf)

        if exists(cache):
            fvc = fvc[:, -1:]

            if exists(finec):
                for ai in finec.attn_intermediates:
                    ck, cv = ai.cached_kv
                    ck, cv = [rearrange(t, '(b nf) ... -> b nf ...', b=b) for t in (ck, cv)]

                    lt = ai.layer_type

                    if lt == 'a':
                        ck, cv = [t[:, -1, :, :cvp] for t in (ck, cv)]
                    elif lt == 'c':
                        ck, cv = [t[:, -1, ...] for t in (ck, cv)]

                    ai.cached_kv = (ck, cv)

        nf = fvc.shape[1]
        of = nf == 1

        fvc = rearrange(fvc, 'b nf n d -> (b nf) n d')

        if of:
            fvc = fvc[:, :(cvp + 1)]

        fakw = dict()

        if self.fine_cross_attend_text:
            rb = fvc.shape[0] // te.shape[0]
            te = repeat(te, 'b ... -> (b r) ...', r=rb)
            tm = repeat(tm, 'b ... -> (b r) ...', r=rb)
            fakw = dict(context=te, context_mask=tm)

        if self.condition_on_text:
            rb = fvc.shape[0] // pte.shape[0]
            pte = repeat(pte, 'b ... -> (b r) ...', r=rb)
            fvc = self.text_fine_film_cond(fvc, pte)

        avc, finec = self.fine_decoder(fvc, cache=finec, **fakw, return_hiddens=True)

        if not scf:
            finec = None

        if not of:
            emb = rearrange(avc, '(b nf) n d -> b (nf n) d', b=b)
            emb = emb[:, :(cl + 1)]
        else:
            emb = avc

        logits = self.to_logits(emb)

        if not return_loss:
            if not return_cache:
                return logits

            nc = (afc, cc, finec, cglc, cpglc, fglc)
            return logits, nc

        cel = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index=self.pad_id)
        return cel