from __future__ import annotations
from pathlib import Path
from functools import partial
from contextlib import nullcontext
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_custom_utils import get_adam_optimizer, OptimizerWithWarmupSchedule
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from beartype.typing import Tuple, Type, List
from data_scripts.typing_mod import typecheck, beartype_isinstance
from ema_pytorch import EMA
from data_scripts.helper_funcs import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_scripts.autoencoder import MeshAutoencoder


DDPK = DistributedDataParallelKwargs(find_unused_parameters=True)


class MeshAutoencoderTrainer(Module):
    @typecheck
    def __init__(
            self,
            model: MeshAutoencoder,
            dataset: Dataset,
            num_train_steps: int,
            batch_size: int,
            grad_accum_every: int,
            val_dataset: Dataset | None = None,
            val_every: int = 100,
            val_num_batches: int = 5,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.,
            max_grad_norm: float | None = None,
            ema_kwargs: dict = dict(use_foreach=True),
            scheduler: Type[_LRScheduler] | None = None,
            scheduler_kwargs: dict = dict(),
            accelerator_kwargs: dict = dict(),
            optimizer_kwargs: dict = dict(),
            checkpoint_every=1000,
            checkpoint_every_epoch: Type[int] | None = None,
            checkpoint_folder='./checkpoints',
            data_kwargs: Tuple[str, ...] = ('vertices', 'faces', 'face_edges'),
            warmup_steps=1000,
            use_wandb_tracking=False
    ):
        super().__init__()

        self.use_wandb = use_wandb_tracking

        if use_wandb_tracking:
            accelerator_kwargs['log_with'] = 'wandb'

        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DDPK]

        self.acc = Accelerator(**accelerator_kwargs)
        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, **ema_kwargs)

        self.opt = OptimizerWithWarmupSchedule(
            accelerator=self.acc,
            optimizer=get_adam_optimizer(model.parameters(), lr=learning_rate, wd=weight_decay, **optimizer_kwargs),
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm)

        self.dl = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=True,
                            collate_fn=partial(custom_collate, pad_id=model.pad_id))

        self.val_flag = exists(val_dataset)

        if self.val_flag:
            assert len(val_dataset) > 0
            self.val_every = val_every
            self.val_nb = val_num_batches
            self.vdl = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, drop_last=True,
                                 collate_fn=partial(custom_collate, pad_id=model.pad_id))

        if hasattr(dataset, 'data_kwargs') and exists(dataset.data_kwargs):
            assert beartype_isinstance(dataset.data_kwargs, List[str])
            self.dk = dataset.data_kwargs
        else:
            self.dk = data_kwargs

        (self.model, self.dl) = self.acc.prepare(self.model, self.dl)

        self.gae = grad_accum_every
        self.nts = num_train_steps
        self.register_buffer('step', torch.tensor(0))

        self.cke = checkpoint_every_epoch
        self.ck = checkpoint_every
        self.ckf = Path(checkpoint_folder)
        self.ckf.mkdir(exist_ok=True, parents=True)

    @property
    def ema_tokenizer(self):
        return self.ema_model.ema_model

    def tokenize(self, *args, **kwargs):
        return self.ema_tokenizer.tokenize(*args, **kwargs)

    def log(self, **kw):
        self.acc.log(kw, step=self.step.item())

    @property
    def device(self):
        return self.unwrapped_model.device

    @property
    def is_main(self):
        return self.acc.is_main_process

    @property
    def unwrapped_model(self):
        return self.acc.unwrap_model(self.model)

    @property
    def is_local_main(self):
        return self.acc.is_local_main_process

    def wait(self):
        return self.acc.wait_for_everyone()

    def print(self, msg):
        return self.acc.print(msg)

    def save(self, path, overwrite=True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(model=self.unwrapped_model.state_dict(), ema_model=self.ema_model.state_dict(),
                  optimizer=self.opt.state_dict(), step=self.step.item(), config=self.unwrapped_model._config)

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))
        self.model.load_state_dict(pkg['model'])
        self.ema_model.load_state_dict(pkg['ema_model'])
        self.opt.load_state_dict(pkg['optimizer'])
        self.step.copy_(pkg['step'])

    def next_data_to_forward_kwargs(self, dli) -> dict:
        d = next(dli)

        if isinstance(d, tuple):
            fk = dict(zip(self.dk, d))
        elif isinstance(d, dict):
            fk = d

        maybe_del(fk, 'texts', 'text_embeds')
        return fk

    def forward(self):
        s = self.step.item()
        dli = cycle(self.dl)

        if self.is_main and self.val_flag:
            vdli = cycle(self.vdl)

        while s < self.nts:
            for i in range(self.gae):
                isl = i == (self.gae - 1)
                mns = partial(self.acc.no_sync, self.model) if not isl else nullcontext

                fk = self.next_data_to_forward_kwargs(dli)

                with self.acc.autocast(), mns():
                    tl, (rl, cl) = self.model(**fk, return_loss_breakdown=True)
                    self.acc.backward(tl / self.gae)

            self.print(f'recon loss: {rl.item():.3f} | commit loss: {cl.sum().item():.3f}')
            self.log(total_loss=tl.item(), commit_loss=cl.sum().item(), recon_loss=rl.item())

            self.opt.step()
            self.opt.zero_grad()

            s += 1
            self.step.add_(1)
            self.wait()

            if self.is_main:
                self.ema_model.update()

            self.wait()

            if self.is_main and self.val_flag and divisible_by(s, self.val_every):
                tvrl = 0.
                self.ema_model.eval()
                nvb = self.val_nb * self.gae

                for _ in range(nvb):
                    with self.acc.autocast(), torch.no_grad():
                        fk = self.next_data_to_forward_kwargs(vdli)
                        vl, (vrl, vcl) = self.ema_model(**fk, return_loss_breakdown=True)
                        tvrl += (vrl / nvb)

                self.print(f'valid recon loss: {tvrl:.3f}')
                self.log(val_loss=tvrl)

            self.wait()

            if self.is_main and divisible_by(s, self.ck):
                cn = s // self.ck
                self.save(self.ckf / f'mesh-autoencoder.ckpt.{cn}.pt')

            self.wait()

        self.print('training complete')

    def train(self, num_epochs, stop_at_loss=None, diplay_graph=False, start_epoch=0):
        el, erl, ecl = [], [], []
        self.model.train()

        for li in range(num_epochs):
            e = start_epoch + li
            tel, terl, tecl = 0.0, 0.0, 0.0

            pb = tqdm(enumerate(self.dl), desc=f'Epoch {e + 1}/{start_epoch + num_epochs}', total=len(self.dl))
            for bi, b in pb:
                isl = (bi + 1) % self.gae == 0
                mns = partial(self.acc.no_sync, self.model) if not isl else nullcontext

                if isinstance(b, tuple):
                    fk = dict(zip(self.dk, b))
                elif isinstance(b, dict):
                    fk = b
                maybe_del(fk, 'texts', 'text_embeds')

                with self.acc.autocast(), mns():
                    tl, (rl, cl) = self.model(**fk, return_loss_breakdown=True)
                    self.acc.backward(tl / self.gae)

                cl_val = tl.item()
                tel += cl_val
                terl += rl.item()
                tecl += cl.sum().item()

                pb.set_postfix(loss=cl_val, recon_loss=round(rl.item(), 3), commit_loss=round(cl.sum().item(), 4))

                if isl or (bi + 1 == len(self.dl)):
                    self.opt.step()
                    self.opt.zero_grad()

            arl = terl / len(self.dl)
            acl = tecl / len(self.dl)
            ael = tel / len(self.dl)

            el.append(ael)
            erl.append(arl)
            ecl.append(acl)

            eo = f'Epoch {e + 1} average loss: {ael} recon loss: {arl:.4f}: commit_loss {acl:.4f}'

            if len(el) >= 4 and ael > 0:
                ali = sum(el[-4:-1]) / 3 - ael
                eo += f'          avg loss speed: {ali}'
                if ali > 0 and ali < 0.2:
                    eu03 = max(0, abs(ael - 0.3) / ali)
                    if eu03 > 0:
                        eo += f' epochs left: {eu03:.2f}'

            self.wait()
            self.print(eo)

            if self.is_main and self.cke is not None and (self.cke == 1 or (e != 0 and e % self.cke == 0)):
                self.save(self.ckf / f'mesh-autoencoder.ckpt.epoch_{e}_avg_loss_{ael:.5f}_recon_{arl:.4f}_commit_{acl:.4f}.pt')

            if stop_at_loss is not None and ael < stop_at_loss:
                self.print(f'Stopping training at epoch {e} with average loss {ael}')
                if self.is_main and self.cke is not None:
                    self.save(self.ckf / f'mesh-autoencoder.ckpt.stop_at_loss_avg_loss_{ael:.3f}.pt')
                break

        self.print('Training complete')
        if diplay_graph:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(el) + 1), el, marker='o', label='Total Loss')
            plt.plot(range(1, len(el) + 1), erl, marker='o', label='Recon Loss')
            plt.plot(range(1, len(el) + 1), ecl, marker='o', label='Commit Loss')
            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.grid(True)
            plt.show()
        return el[-1]