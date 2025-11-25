from __future__ import annotations
from pathlib import Path
from functools import partial
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_custom_utils import get_adam_optimizer, OptimizerWithWarmupSchedule
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from beartype.typing import Tuple, Type, List
from data_scripts.typing_mod import typecheck, beartype_isinstance
from data_scripts.helper_funcs import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_scripts.transformer import MeshTransformer


DDPK = DistributedDataParallelKwargs(find_unused_parameters=True)


class MeshTransformerTrainer(Module):
    @typecheck
    def __init__(
            self,
            model: MeshTransformer,
            dataset: Dataset,
            num_train_steps: int,
            batch_size: int,
            grad_accum_every: int,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.,
            max_grad_norm: float | None = 0.5,
            val_dataset: Dataset | None = None,
            val_every=1,
            val_num_batches=5,
            scheduler: Type[_LRScheduler] | None = None,
            scheduler_kwargs: dict = dict(),
            ema_kwargs: dict = dict(),
            accelerator_kwargs: dict = dict(),
            optimizer_kwargs: dict = dict(),
            checkpoint_every=1000,
            checkpoint_every_epoch: Type[int] | None = None,
            checkpoint_folder='./checkpoints',
            data_kwargs: Tuple[str, ...] = ('vertices', 'faces', 'face_edges', 'text'),
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

        opt = get_adam_optimizer(model.parameters(), lr=learning_rate, wd=weight_decay,
                                 filter_by_requires_grad=True, **optimizer_kwargs)

        self.opt = OptimizerWithWarmupSchedule(
            accelerator=self.acc,
            optimizer=opt,
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

    def next_data_to_forward_kwargs(self, dli) -> dict:
        d = next(dli)

        if isinstance(d, tuple):
            fk = dict(zip(self.dk, d))
        elif isinstance(d, dict):
            fk = d

        return fk

    def save(self, path, overwrite=True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(model=self.unwrapped_model.state_dict(), optimizer=self.opt.state_dict(), step=self.step.item())
        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))
        self.model.load_state_dict(pkg['model'])
        self.opt.load_state_dict(pkg['optimizer'])
        self.step.copy_(pkg['step'])

    def forward(self):
        s = self.step.item()
        dli = cycle(self.dl)

        if self.val_flag:
            vdli = cycle(self.vdl)

        while s < self.nts:
            for i in range(self.gae):
                isl = i == (self.gae - 1)
                mns = partial(self.acc.no_sync, self.model) if not isl else nullcontext

                fk = self.next_data_to_forward_kwargs(dli)

                with self.acc.autocast(), mns():
                    l = self.model(**fk)
                    self.acc.backward(l / self.gae)

            self.print(f'loss: {l.item():.3f}')
            self.log(loss=l.item())

            self.opt.step()
            self.opt.zero_grad()

            s += 1
            self.step.add_(1)
            self.wait()

            if self.is_main and self.val_flag and divisible_by(s, self.val_every):
                tvl = 0.
                self.unwrapped_model.eval()
                nvb = self.val_nb * self.gae

                for _ in range(nvb):
                    with self.acc.autocast(), torch.no_grad():
                        fk = self.next_data_to_forward_kwargs(vdli)
                        vl = self.unwrapped_model(**fk)
                        tvl += (vl / nvb)

                self.print(f'valid recon loss: {tvl:.3f}')
                self.log(val_loss=tvl)

            self.wait()

            if self.is_main and divisible_by(s, self.ck):
                cn = s // self.ck
                self.save(self.ckf / f'mesh-transformer.ckpt.{cn}.pt')

            self.wait()

        self.print('training complete')

    def train(self, num_epochs, stop_at_loss=None, diplay_graph=False, start_epoch=0):
        el = []
        es = len(self.dl)
        self.model.train()

        for e in range(num_epochs):
            ne = start_epoch + e
            tel = 0.0

            pb = tqdm(enumerate(self.dl), desc=f'Epoch {ne + 1}/{start_epoch + num_epochs}', total=len(self.dl))
            for bi, b in pb:
                isl = (bi + 1) % self.gae == 0
                mns = partial(self.acc.no_sync, self.model) if not isl else nullcontext

                with self.acc.autocast(), mns():
                    tl = self.model(**b)
                    self.acc.backward(tl / self.gae)

                cl = tl.item()
                tel += cl
                pb.set_postfix(loss=cl)

                if isl or (bi + 1 == len(self.dl)):
                    self.opt.step()
                    self.opt.zero_grad()

            ael = tel / es
            eo = f'Epoch {ne + 1} average loss: {ael}'

            el.append(ael)

            if len(el) >= 4 and ael > 0:
                ali = sum(el[-4:-1]) / 3 - ael
                eo += f'          avg loss speed: {ali}'
                if ali > 0 and ali < 0.2:
                    eu03 = max(0, abs(ael - 0.3) / ali)
                    if eu03 > 0:
                        eo += f' epochs left: {eu03:.2f}'

            self.wait()
            self.print(eo)

            if self.is_main and self.cke is not None and (self.cke == 1 or (ne != 0 and ne % self.cke == 0)):
                self.save(self.ckf / f'mesh-transformer.ckpt.epoch_{ne}_avg_loss_{ael:.3f}.pt')

            if stop_at_loss is not None and ael < stop_at_loss:
                self.print(f'Stopping training at epoch {ne} with average loss {ael}')
                if self.is_main and self.cke is not None:
                    self.save(self.ckf / f'mesh-transformer.ckpt.stop_at_loss_avg_loss_{ael:.3f}.pt')
                break

        self.print('Training complete')
        if diplay_graph:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(el) + 1), el, marker='o', label='Total Loss')
            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.grid(True)
            plt.show()
        return el[-1]