from data_scripts.Mesh_dataset import MeshDataset
from model_scripts.autoencoder import MeshAutoencoder
from model_scripts.transformer import MeshTransformer
import torch
import re
from pathlib import Path
from training_scripts.transformer_trainer import MeshTransformerTrainer


script_dir = Path(__file__).resolve().parent
DATASET_PATH = "Data/meshData_org.npz"

dataset = MeshDataset.load(DATASET_PATH)
# ds.data = ds.data[:400]

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
        attn_encoder_depth=2
    ).to("cuda")


pkg = torch.load(str("Model/mesh-encoder_16k_4_2.pt"))
autoencoder.load_state_dict(pkg['model'])

import gc
gc.collect()
torch.cuda.empty_cache()
max_seq = max(len(d["faces"]) for d in dataset if "faces" in d)  * (autoencoder.num_vertices_per_face * autoencoder.num_quantizers)
print("Max token sequence:" , max_seq)

# GPT2-Small model
transformer = MeshTransformer(
    autoencoder,
    dim = 768,
    coarse_pre_gateloop_depth = 3,
    fine_pre_gateloop_depth= 3,
    attn_depth = 12,
    attn_heads = 12,
    max_seq_len = max_seq,
    condition_on_text = True,
    gateloop_use_heinsen = False,
    dropout  = 0.0,
    text_condition_model_types = "bge",
    text_condition_cond_drop_prob = 0.0
)

total_params = sum(p.numel() for p in transformer.decoder.parameters())
total_params = f"{total_params / 1000000:.1f}M"
print(f"Decoder total parameters: {total_params}")

# filter dataset to categories_to_ft (list of strings)
def filter_dataset_by_categories(dataset, categories_to_ft):
    new_data = [d for d in dataset.data if d['texts'] in categories_to_ft]
    return MeshDataset(new_data)

# Example:
categories_to_ft = ['ashcan trash can garbage can wastebin ash bin ash-bin ashbin dustbin trash barrel trash bin', 'bag traveling bag travelling bag grip suitcase', 'bed', 'bench', 'bookshelf', 'bus autobus coach charabanc double-decker jitney motorbus motorcoach omnibus passenger vehi', 'cabinet', 'car auto automobile machine motorcar', 'chair', 'lamp', 'mailbox letter box', 'motorcycle bike', 'pot flowerpot', 'sofa couch lounge', 'table']
ft_dataset = filter_dataset_by_categories(dataset, categories_to_ft)

trainer_ckpt_dir = Path("./checkpoints")
ckpts = sorted(trainer_ckpt_dir.glob("mesh-transformer.ckpt.epoch_*.pt"))
if not ckpts:
    raise RuntimeError("No checkpoint to resume from.")

def _epoch_from_name(p):
    m = re.search(r'epoch_(\d+)_', p.name)
    return int(m.group(1)) if m else -1

last_ckpt = max(ckpts, key=_epoch_from_name)
resume_epoch = _epoch_from_name(last_ckpt) + 1

pkg = torch.load(str(last_ckpt))
transformer.load_state_dict(pkg['model'])


labels = list(set(item["texts"] for item in ft_dataset.data))
ft_dataset.embed_texts(transformer, batch_size = 25)
ft_dataset.generate_codes(autoencoder, batch_size = 50)
print(ft_dataset.data[0].keys())
print(labels)

# pkg = torch.load(str(f'{working_dir}\mesh-transformer_{project_name}.pt'))
# transformer.load_state_dict(pkg['model'])


batch_size = 4 # Max 64
grad_accum_every = 16

# Set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  4 * 16 = 64
learning_rate = 1e-4 # Start training with the learning rate at 1e-2 then lower it to 1e-3 at stagnation or at 0.5 loss.

trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,num_train_steps=100, dataset = ft_dataset,
                                 grad_accum_every=grad_accum_every,
                                 learning_rate = learning_rate,
                                 batch_size=batch_size,
                                 checkpoint_every_epoch = 10,
                                 # FP16 training, it doesn't speed up very much but can increase the batch size which will in turn speed up the training.
                                 # However it might cause nan after a while.
                                 # accelerator_kwargs = {"mixed_precision" : "fp16"}, optimizer_kwargs = { "eps": 1e-7}
                                 )

# trainer.load(last_ckpt)
#
# for pg in trainer.optimizer.optimizer.param_groups:
#     print(pg["lr"])
#     pg["lr"] = learning_rate
#     print(pg["lr"])

loss = trainer.train(50, stop_at_loss = 0.001, diplay_graph=True, start_epoch=resume_epoch)

trainer.save(f'{"Model"}/mesh-transformer_16k_4_2_finetuned.pt')

