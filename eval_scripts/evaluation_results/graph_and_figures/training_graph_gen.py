from pathlib import Path

import matplotlib.pyplot as plt
import re
from typing import List, Dict, Tuple


def parse_encoder_checkpoint(path: str) -> Dict:
    """Parse encoder checkpoint path to extract metrics."""
    pattern = r'epoch_(\d+)_avg_loss_([\d.]+)_recon_([\d.]+)_commit_([\d.]+).pt'
    match = re.search(pattern, path)
    if match:
        return {
            'epoch': int(match.group(1)),
            'avg_loss': float(match.group(2)),
            'recon': float(match.group(3)),
            'commit': float(match.group(4))
        }
    return None


def parse_transformer_checkpoint(path: str) -> Dict:
    """Parse transformer checkpoint path to extract metrics."""
    pattern = r'epoch_(\d+)_avg_loss_([\d.]+).pt'
    match = re.search(pattern, path)
    if match:
        return {
            'epoch': int(match.group(1)),
            'avg_loss': float(match.group(2))
        }
    return None


# ============================================================================
# CONFIGURABLE INPUT VARIABLES
# ============================================================================

# Encoder checkpoint paths
# ENCODER_CHECKPOINTS = [
#     'mesh-autoencoder.ckpt.epoch_0_avg_loss_4.41626_recon_4.1300_commit_0.7156.pt',
#     'mesh-autoencoder.ckpt.epoch_1_avg_loss_3.97560_recon_3.6960_commit_0.6989.pt',
#     'mesh-autoencoder.ckpt.epoch_2_avg_loss_3.83200_recon_3.5468_commit_0.7129.pt',
#     'mesh-autoencoder.ckpt.epoch_3_avg_loss_3.73473_recon_3.4497_commit_0.7125.pt'
# ]

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
print(project_root)

trainer_ckpt_dir = Path(project_root / "checkpoints")

# if not trainer_ckpt_dir.exists():
#     raise FileNotFoundError(f"Directory not found: {trainer_ckpt_dir}")
#
# # List all files in the folder
# all_files = sorted(trainer_ckpt_dir.iterdir())
# print("Files found:")
# for f in all_files:
#     print(f.name)

# ENCODER_CHECKPOINTS = [str(f) for f in sorted(trainer_ckpt_dir.glob("mesh-autoencoder.ckpt.epoch_*.pt"))]
ENCODER_CHECKPOINTS = [
    str(f) for f in sorted(trainer_ckpt_dir.glob("mesh-autoencoder.ckpt.epoch_*.pt"))
    if (m := re.search(r"epoch_(\d+)_", f.name)) and int(m.group(1)) % 10 == 0
]

print(ENCODER_CHECKPOINTS)

# Transformer checkpoint paths
TRANSFORMER_CHECKPOINTS = [
    "mesh-transformer.ckpt.epoch_0_avg_loss_3.429.pt",
   "mesh-transformer.ckpt.epoch_10_avg_loss_1.426.pt",
   "mesh-transformer.ckpt.epoch_20_avg_loss_1.029.pt",
   "mesh-transformer.ckpt.epoch_30_avg_loss_0.741.pt",
   "mesh-transformer.ckpt.epoch_40_avg_loss_0.516.pt",
   "mesh-transformer.ckpt.epoch_50_avg_loss_0.365.pt",
   "mesh-transformer.ckpt.epoch_60_avg_loss_0.264.pt",
   "mesh-transformer.ckpt.epoch_70_avg_loss_0.196.pt",
   "mesh-transformer.ckpt.epoch_80_avg_loss_0.149.pt",
   "mesh-transformer.ckpt.epoch_90_avg_loss_0.118.pt",
    "mesh-transformer.ckpt.epoch_100_avg_loss_0.007.pt",
    "mesh-transformer.ckpt.epoch_110_avg_loss_0.006.pt",
    "mesh-transformer.ckpt.epoch_120_avg_loss_0.005.pt",
    "mesh-transformer.ckpt.epoch_130_avg_loss_0.005.pt",
    "mesh-transformer.ckpt.epoch_140_avg_loss_0.005.pt",
    "mesh-transformer.ckpt.epoch_150_avg_loss_0.005.pt"
]

residuallfq_ckpt_dir = Path(project_root / "checkpoints_chairs" / "residuallfq")
residualsimvq_ckpt_dir = Path(project_root / "checkpoints_chairs" / "residualsimvq")
rvq_ckpt_dir = Path(project_root / "checkpoints_chairs" / "rvq")


# Quantization variation checkpoint paths
QUANT_VARIATIONS = {
    'residuallfq': [
    str(f) for f in sorted(residuallfq_ckpt_dir.glob("mesh-autoencoder.ckpt.epoch_*.pt"))
    if (m := re.search(r"epoch_(\d+)_", f.name)) and int(m.group(1)) % 10 == 0
],
    'residualsimvq': [
    str(f) for f in sorted(residualsimvq_ckpt_dir.glob("mesh-autoencoder.ckpt.epoch_*.pt"))
    if (m := re.search(r"epoch_(\d+)_", f.name)) and int(m.group(1)) % 10 == 0
],
    'rvq': [
    str(f) for f in sorted(rvq_ckpt_dir.glob("mesh-autoencoder.ckpt.epoch_*.pt"))
    if (m := re.search(r"epoch_(\d+)_", f.name)) and int(m.group(1)) % 10 == 0
]
}


# ============================================================================
# MAIN VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_encoder_training(checkpoint_paths: List[str]):
    """Create visualization for encoder training with all 3 error metrics."""
    # Parse checkpoints
    data = [parse_encoder_checkpoint(path) for path in checkpoint_paths]
    data = [d for d in data if d is not None]  # Filter out None values
    data.sort(key=lambda x: x['epoch'])
    print(data)

    epochs = [d['epoch'] for d in data]
    avg_losses = [d['avg_loss'] for d in data]
    recon_losses = [d['recon'] for d in data]
    commit_losses = [d['commit'] for d in data]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot all three metrics
    ax.plot(epochs, avg_losses, marker='o', linewidth=2, markersize=8,
            label='Average Loss', color='#2563eb')
    ax.plot(epochs, recon_losses, marker='o', linewidth=2, markersize=8,
            label='Reconstruction Loss', color='#dc2626')
    ax.plot(epochs, commit_losses, marker='o', linewidth=2, markersize=8,
            label='Commit Loss', color='#16a34a')

    # Styling
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Mesh Autoencoder Training Progress', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(epochs)

    plt.tight_layout()
    plt.savefig('encoder_training.png', dpi=300, bbox_inches='tight')
    print("✓ Encoder training visualization saved as 'encoder_training.png'")
    plt.show()


def visualize_transformer_training(checkpoint_paths: List[str]):
    """Create visualization for transformer training."""
    # Parse checkpoints
    data = [parse_transformer_checkpoint(path) for path in checkpoint_paths]
    data = [d for d in data if d is not None]

    epochs = [d['epoch'] for d in data]
    avg_losses = [d['avg_loss'] for d in data]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot transformer loss
    ax.plot(epochs, avg_losses, marker='o', linewidth=2.5, markersize=10,
            label='Average Loss', color='#7c3aed', markerfacecolor='#a78bfa')

    # Styling
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Mesh Transformer Training Progress', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(epochs)

    plt.tight_layout()
    plt.savefig('transformer_training.png', dpi=300, bbox_inches='tight')
    print("✓ Transformer training visualization saved as 'transformer_training.png'")
    plt.show()


def visualize_quantization_variations(quant_variations: Dict[str, List[str]]):
    """Create 3 separate graphs comparing quantization variations for each error metric."""

    # Parse all quantization data
    parsed_data = {}
    for quant_type, paths in quant_variations.items():
        parsed_data[quant_type] = [parse_encoder_checkpoint(path) for path in paths]
        parsed_data[quant_type] = [d for d in parsed_data[quant_type] if d is not None]

    # Define colors for each quantization type
    colors = {
        'recon': '#ef4444',
        'avg_loss': '#3b82f6',
        'commit': '#10b981'
    }

    # Define markers
    markers = {
        'residuallfq': 'o',
        'residualsimvq': 's',
        'rvq': '^'
    }

    lable = {
        'residuallfq': 'Residual LFQ',
        'residualsimvq': 'residual SimVQ',
        'rvq': 'Residual VQ'
    }

    # Create 3 subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ('avg_loss', 'Average Loss', axes[0]),
        ('recon', 'Reconstruction Loss', axes[1]),
        ('commit', 'Commit Loss', axes[2])
    ]

    for metric_key, metric_name, ax in metrics:
        for quant_type, data in parsed_data.items():
            epochs = [d['epoch'] for d in data]
            values = [d[metric_key] for d in data]

            ax.plot(epochs, values,
                    marker=markers[quant_type],
                    linewidth=2,
                    markersize=8,
                    label=lable[quant_type],
                    color=colors[metric_key])

        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold', pad=15)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Set integer epochs on x-axis
        all_epochs = []
        for data in parsed_data.values():
            all_epochs.extend([d['epoch'] for d in data])
        ax.set_xticks(sorted(set(all_epochs)))

    plt.suptitle('Quantization Variation Evaluation', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('quantization_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Quantization comparison visualization saved as 'quantization_comparison.png'")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Mesh Training Visualization Suite")
    print("=" * 70)
    print()

    # Generate encoder training visualization
    print("1. Generating Encoder Training Visualization...")
    visualize_encoder_training(ENCODER_CHECKPOINTS)
    print()

    # Generate transformer training visualization
    print("2. Generating Transformer Training Visualization...")
    visualize_transformer_training(TRANSFORMER_CHECKPOINTS)
    print()

    # Generate quantization comparison visualization
    print("3. Generating Quantization Variation Comparison...")
    visualize_quantization_variations(QUANT_VARIATIONS)
    print()

    print("=" * 70)
    print("All visualizations completed successfully!")
    print("=" * 70)