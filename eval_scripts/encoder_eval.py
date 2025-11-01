import torch
import random
from tqdm import tqdm
from pathlib import Path
import re
from data_scripts.Mesh_dataset import MeshDataset
from model_scripts.autoencoder import MeshAutoencoder
from training_scripts.autoencoder_trainer import MeshAutoencoderTrainer
from data_scripts import mesh_render


script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

working_dir = project_root / "Data"
working_dir.mkdir(exist_ok=True, parents=True)
# dataset_path =  working_dir / "meshData_org_augmented.npz"
dataset_path =  working_dir / "meshData_augmented_2.npz"
dataset = MeshDataset.load(dataset_path)

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
    use_residual_lfq=False
).to("cuda")
total_params = sum(p.numel() for p in autoencoder.parameters())
total_params = f"{total_params / 1000000:.1f}M"
print(f"Total parameters: {total_params}")


batch_size=16 # The batch size should be max 64.
grad_accum_every = 4
# So set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  16 * 4 = 64
learning_rate = 1e-3 # Start with 1e-3 then at staggnation around 0.35, you can lower it to 1e-4.

autoencoder.commit_loss_weight = 0.4 # Set dependant on the dataset size, on smaller datasets, 0.1 is fine, otherwise try from 0.25 to 0.4.
autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder ,warmup_steps = 10, dataset = dataset, num_train_steps=100,
                                             batch_size=batch_size,
                                             grad_accum_every = grad_accum_every,
                                             learning_rate = learning_rate,
                                             checkpoint_every_epoch=1,
                                             checkpoint_folder = './checkpoints')

# find last epoch checkpoint
trainer_ckpt_dir = Path(project_root / "checkpoints")
ckpts = sorted(trainer_ckpt_dir.glob("mesh-autoencoder.ckpt.epoch_*.pt"))
if not ckpts:
    raise RuntimeError("No checkpoint to resume from.")
def _epoch_from_name(p):
    m = re.search(r'epoch_(\d+)_', p.name)
    return int(m.group(1)) if m else -1
last_ckpt = max(ckpts, key=_epoch_from_name)
print(last_ckpt)
# last_ckpt = "checkpoints/mesh-autoencoder.ckpt.epoch_96_avg_loss_0.43177_recon_0.3578_commit_0.3698.pt"
saved_model = Path(project_root / "checkpoints_chairs_rq" / 'mesh-autoencoder.ckpt.epoch_50_avg_loss_1.26883_recon_1.2484_commit_0.2045.pt')
print(saved_model, Path(f'{project_root}/{"Model"}/mesh-encoder_16k_4_2.pt'), Path(f'{project_root}/{"Model"}/mesh-encoder_16k_4_2.pt').exists())
# load state
autoencoder_trainer.load(saved_model)

# # change lr AFTER load
# for pg in autoencoder_trainer.optimizer.optimizer.param_groups:
#     pg["lr"] = learning_rate

autoencoder_trainer.load(saved_model)
min_mse, max_mse = float('inf'), float('-inf')
min_coords, min_orgs, max_coords, max_orgs = None, None, None, None
random_samples, random_samples_pred, all_random_samples = [], [], []
total_mse, sample_size = 0.0, 500

random.shuffle(dataset.data)
from collections import defaultdict
uni = defaultdict(int)

for item in tqdm(dataset.data[:sample_size]):
    codes = autoencoder.tokenize(vertices=item['vertices'], faces=item['faces'], face_edges=item['face_edges'])

    codes = codes.flatten().unsqueeze(0)
    codes = codes[:, :codes.shape[-1] // autoencoder.num_quantizers * autoencoder.num_quantizers]

    coords, mask = autoencoder.decode_from_codes_to_faces(codes)
    orgs = item['vertices'][item['faces']].unsqueeze(0)

    mse = torch.mean((orgs.view(-1, 3).cpu() - coords.view(-1, 3).cpu()) ** 2)
    uni[item['texts']] += mse
    total_mse += mse

    if mse < min_mse: min_mse, min_coords, min_orgs = mse, coords, orgs
    if mse > max_mse: max_mse, max_coords, max_orgs = mse, coords, orgs

    if len(random_samples) <= 30:
        random_samples.append(coords)
        random_samples_pred.append(orgs)
    else:
        all_random_samples.extend([random_samples_pred, random_samples])
        random_samples, random_samples_pred = [], []
print(uni)
print(f'MSE AVG: {total_mse / sample_size:.10f}, Min: {min_mse:.10f}, Max: {max_mse:.10f}')
mesh_render.combind_mesh_with_rows(project_root/'Data'/'mse_rows.obj', all_random_samples)
