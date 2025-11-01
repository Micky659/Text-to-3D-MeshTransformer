from pathlib import Path
import torch
from datetime import datetime

from data_scripts.Mesh_dataset import MeshDataset
from model_scripts.autoencoder import MeshAutoencoder
from training_scripts.autoencoder_trainer import MeshAutoencoderTrainer

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

working_dir = project_root / "Data"
working_dir.mkdir(exist_ok=True, parents=True)


def train_quantizer_variant(
        variant_name: str,
        use_residual_lfq: bool,
        use_residual_simvq: bool,
        dataset: MeshDataset,
        base_checkpoint_folder: str = './checkpoints',
        base_model_folder: str = './Model',
        num_epochs: int = 100,
        batch_size: int = 16,
        grad_accum_every: int = 4,
        learning_rate: float = 1e-4,
        commit_loss_weight: float = 0.1,
        checkpoint_every_epoch: int = 1,
        stop_at_loss: float = 0.35
):
    """
    Train a single quantizer variant and save checkpoints/final model.

    Args:
        variant_name: Name of the variant (e.g., 'RVQ', 'ResidualLFQ', 'ResidualSimVQ')
        use_residual_lfq: Whether to use ResidualLFQ
        use_residual_simvq: Whether to use ResidualSimVQ
        dataset: Training dataset
        base_checkpoint_folder: Base folder for checkpoints
        base_model_folder: Base folder for final models
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        grad_accum_every: Gradient accumulation steps
        learning_rate: Learning rate
        commit_loss_weight: Weight for commitment loss
        checkpoint_every_epoch: Save checkpoint every N epochs
        stop_at_loss: Stop training if loss reaches this value
    """
    print(f"\n{'=' * 80}")
    print(f"Training {variant_name} variant")
    print(f"{'=' * 80}\n")

    # Create variant-specific folders
    checkpoint_folder = Path(base_checkpoint_folder) / variant_name.lower()
    model_folder = Path(base_model_folder) / variant_name.lower()

    checkpoint_folder.mkdir(exist_ok=True, parents=True)
    model_folder.mkdir(exist_ok=True, parents=True)

    # Initialize autoencoder with specific quantizer configuration
    autoencoder = MeshAutoencoder(
        decoder_dims_through_depth=(128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
        codebook_size=16384,
        dim_codebook=192,
        dim_area_embed=16,
        dim_coor_embed=16,
        dim_normal_embed=16,
        dim_angle_embed=8,
        attn_decoder_depth=4,
        attn_encoder_depth=2,
        use_residual_lfq=use_residual_lfq,
        use_residual_simvq=use_residual_simvq
    ).to("cuda")

    # Calculate and display model parameters
    total_params = sum(p.numel() for p in autoencoder.parameters())
    total_params_str = f"{total_params / 1000000:.1f}M"
    print(f"Total parameters: {total_params_str}")
    print(f"Quantizer type: {variant_name}")
    print(f"Effective batch size: {batch_size * grad_accum_every}")
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoint folder: {checkpoint_folder}")
    print(f"Model folder: {model_folder}\n")

    # Set commitment loss weight
    autoencoder.commit_loss_weight = commit_loss_weight

    # Initialize trainer
    autoencoder_trainer = MeshAutoencoderTrainer(
        model=autoencoder,
        warmup_steps=10,
        dataset=dataset,
        num_train_steps=num_epochs,
        batch_size=batch_size,
        grad_accum_every=grad_accum_every,
        learning_rate=learning_rate,
        checkpoint_every_epoch=checkpoint_every_epoch,
        checkpoint_folder=str(checkpoint_folder)
    )

    # Train the model
    try:
        loss = autoencoder_trainer.train(
            num_epochs,
            stop_at_loss=stop_at_loss,
            diplay_graph=False
        )

        print(f"\n{variant_name} training completed!")
        print(f"Final loss: {loss:.6f}")

    except Exception as e:
        print(f"\nError during {variant_name} training: {str(e)}")
        print(f"Saving checkpoint at current state...")

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = model_folder / f'mesh_encoder_{variant_name.lower()}_chairs_final_{timestamp}.pt'
    autoencoder_trainer.save(str(final_model_path))
    print(f"Final model saved to: {final_model_path}\n")

    # Clear GPU memory
    del autoencoder
    del autoencoder_trainer
    torch.cuda.empty_cache()

    return final_model_path


def main():
    """Main training function that trains all three quantizer variants."""

    # Load dataset
    dataset_path = "Data/meshData_org_onlychair.npz"
    print(f"Loading dataset from: {dataset_path}")
    dataset = MeshDataset.load(dataset_path)
    dataset_name = dataset_path.split("/")[1].split(".")[0]
    print(f"Dataset loaded: {dataset_name}")
    print(f"Dataset size: {len(dataset)} samples\n")

    # Training configuration
    training_config = {
        'num_epochs': 100,
        'batch_size': 16,
        'grad_accum_every': 4,
        'learning_rate': 1e-4,
        'commit_loss_weight': 0.1,
        'checkpoint_every_epoch': 1,
        'stop_at_loss': 0.35,
        'base_checkpoint_folder': './checkpoints_chairs',
        'base_model_folder': './Model'
    }

    # Define quantizer variants to train
    quantizer_variants = [
        {
            'name': 'RVQ',
            'use_residual_lfq': False,
            'use_residual_simvq': False,
            'description': 'Residual Vector Quantization (original MeshGPT)'
        },
        {
            'name': 'ResidualLFQ',
            'use_residual_lfq': True,
            'use_residual_simvq': False,
            'description': 'Residual Lookup-Free Quantization'
        },
        {
            'name': 'ResidualSimVQ',
            'use_residual_lfq': False,
            'use_residual_simvq': True,
            'description': 'Residual Similarity-based Vector Quantization'
        }
    ]

    # Train all variants
    results = {}

    print(f"\n{'=' * 80}")
    print(f"Starting training for all quantizer variants")
    print(f"Total variants: {len(quantizer_variants)}")
    print(f"Epochs per variant: {training_config['num_epochs']}")
    print(f"{'=' * 80}\n")

    for variant in quantizer_variants:
        print(f"\n{variant['name']}: {variant['description']}")

        try:
            model_path = train_quantizer_variant(
                variant_name=variant['name'],
                use_residual_lfq=variant['use_residual_lfq'],
                use_residual_simvq=variant['use_residual_simvq'],
                dataset=dataset,
                **training_config
            )
            results[variant['name']] = {
                'status': 'completed',
                'model_path': model_path
            }
        except Exception as e:
            print(f"\nFailed to train {variant['name']}: {str(e)}")
            results[variant['name']] = {
                'status': 'failed',
                'error': str(e)
            }

    # Print summary
    print(f"\n{'=' * 80}")
    print("Training Summary")
    print(f"{'=' * 80}\n")

    for variant_name, result in results.items():
        print(f"{variant_name}:")
        print(f"  Status: {result['status']}")
        if result['status'] == 'completed':
            print(f"  Model path: {result['model_path']}")
        else:
            print(f"  Error: {result['error']}")
        print()

    print("All training completed!")


if __name__ == '__main__':
    # Optional but helps with certain Windows launchers
    import multiprocessing

    multiprocessing.freeze_support()

    # Set CUDA memory fraction if needed
    # torch.cuda.set_per_process_memory_fraction(0.95)

    main()