"""
Train DDPM on CIFAR-10 images, then sample.
"""

from pathlib import Path

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


def find_latest_milestone(results_folder: str):
    results_path = Path(results_folder)
    checkpoints = []

    for ckpt in results_path.glob('model-*.pt'):
        try:
            milestone = int(ckpt.stem.split('-')[-1])
            checkpoints.append(milestone)
        except ValueError:
            continue

    return max(checkpoints) if checkpoints else None


if __name__ == '__main__':
    # GPU optimization: let cuDNN auto-tune convolution algorithms for fixed input size
    torch.backends.cudnn.benchmark = True

    # Step 1: CIFAR-10 image folder (50k 32x32 RGB PNGs)
    cifar_folder = './cifar-10/train'

    # Step 2: Build model and diffusion
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True  # uses PyTorch SDPA — fast on RTX 4070 SUPER (compute 8.9)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000
    )

    # Step 3: Train with FID evaluation
    results_folder = './results_cifar'

    trainer = Trainer(
        diffusion,
        folder=cifar_folder,
        train_batch_size=64,
        train_num_steps=200000,
        save_and_sample_every=4000,
        calculate_fid=True,
        num_fid_samples=10000,
        amp=True,
        mixed_precision_type='fp16',
        num_samples=4,
        results_folder=results_folder,
        save_samples=False
    )

    # Resume from latest checkpoint if one is available.
    latest_milestone = find_latest_milestone(results_folder)
    if latest_milestone is not None:
        print(f"Resuming from checkpoint model-{latest_milestone}.pt")
        trainer.load(latest_milestone)
    else:
        print("No checkpoint found. Starting training from scratch.")

    print("Training on CIFAR-10 images...")
    trainer.train()
    print("Training complete!")

    # Step 4: Sample and save
    diffusion.eval()
    with torch.inference_mode():
        sampled = diffusion.sample(batch_size=1)  # (1, 3, 32, 32) in [0, 1]

    from torchvision.utils import save_image
    out_path = './results_cifar/cifar_sample.png'
    save_image(sampled, out_path)
    print(f"Sampled image saved to {out_path}")
