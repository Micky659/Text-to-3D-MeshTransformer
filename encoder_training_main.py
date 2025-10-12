from pathlib import Path
import re

import torch

from data_scripts.Mesh_dataset import MeshDataset
from model_scripts.autoencoder import MeshAutoencoder
from training_scripts.autoencoder_trainer import MeshAutoencoderTrainer


script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

working_dir = project_root / "Data"
working_dir.mkdir(exist_ok=True, parents=True)

# torch.cuda.set_per_process_memory_fraction(0.95)

def main():
    dataset_path = "Data/meshData_org.npz"
    dataset = MeshDataset.load(dataset_path)
    print(dataset_path.split("/")[1].split(".")[0])

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
        attn_encoder_depth=2,

        use_residual_lfq=True # If true uses residual lfq otherwise residual rq
    ).to("cuda")
    total_params = sum(p.numel() for p in autoencoder.parameters())
    total_params = f"{total_params / 1000000:.1f}M"
    print(f"Total parameters: {total_params}")


    batch_size=16# The batch size should be max 64.
    grad_accum_every = 4
    # So set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  16 * 4 = 64
    learning_rate = 1e-4 # Start with 1e-3 then at staggnation around 0.35, you can lower it to 1e-4.

    autoencoder.commit_loss_weight = 0.4 # Set dependant on the dataset size, on smaller datasets, 0.1 is fine, otherwise try from 0.25 to 0.4.
    autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder ,warmup_steps = 10, dataset = dataset, num_train_steps=100,
                                                 batch_size=batch_size,
                                                 grad_accum_every = grad_accum_every,
                                                 learning_rate = learning_rate,
                                                 checkpoint_every_epoch=1,
                                                 checkpoint_folder = './checkpoints')

    # find last epoch checkpoint
    trainer_ckpt_dir = Path("./checkpoints")
    ckpts = sorted(trainer_ckpt_dir.glob("mesh-autoencoder.ckpt.epoch_*.pt"))
    if not ckpts:
        raise RuntimeError("No checkpoint to resume from.")
    def _epoch_from_name(p):
        m = re.search(r'epoch_(\d+)_', p.name)
        return int(m.group(1)) if m else -1
    last_ckpt = max(ckpts, key=_epoch_from_name)

    saved_model = Path("Model/mesh-encoder_16k_4_2.pt")

    resume_epoch = _epoch_from_name(last_ckpt) + 1

    # load state
    print(last_ckpt)
    autoencoder_trainer.load(last_ckpt)

    # change lr AFTER load
    for pg in autoencoder_trainer.optimizer.optimizer.param_groups:
        print(pg["lr"])
        pg["lr"] = learning_rate

    # resume training for additional epochs
    autoencoder_trainer.train(
        num_epochs=50,                # how many *more*
        stop_at_loss=0.2,
        diplay_graph=True,
        start_epoch=resume_epoch       # <-- requires patch above
    )


    # --- Optional: precompute BPT codes for transformer training later.
    # Set this to True if you want to generate BPT tokenization for every mesh and store it into dataset['codes'].
    # This does not affect autoencoder training, but helps when you later train a transformer on BPT tokens.
    # use_bpt = False
    # if use_bpt:
    #     print("[encoder_training_main] Generating BPT codes for dataset (this may be slow)...")
    #     # defaults match Tencent BPT settings
    #     dataset.generate_codes_bpt(batch_size=32, pad_id=-1, block_size=8, offset_size=16, use_special_block=True)
    #     # save dataset to file for reuse (overwrites current dataset file)
    #     out_path = Path(dataset_path).with_suffix('.bpt.npz')
    #     np.savez_compressed(out_path, dataset.data, allow_pickle=True)
    #     print(f"[encoder_training_main] Saved BPT-augmented dataset to {out_path}")



    # loss = autoencoder_trainer.train(100,stop_at_loss = 0.2, diplay_graph= True)

    autoencoder_trainer.save(f'{"Model"}/mesh-encoder_16k_4_2.pt')


if __name__ == '__main__':
    # Optional but helps with certain Windows launchers
    import multiprocessing
    multiprocessing.freeze_support()
    main()


