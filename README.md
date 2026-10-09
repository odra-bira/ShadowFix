# ShadowFix AI — Intelligent Shadow Removal & Image Restoration

A computer vision capstone investigating how **U-Net-based image restoration** removes cast shadows while preserving scene details. This project compares RGB-only, residual, and mask-guided models on the ISTD shadow removal benchmark and includes a Gradio demonstration.

## Results

Evaluation was reported on **540 ISTD test images** at 256 × 256 resolution.

| Method | PSNR (dB) ↑ | SSIM ↑ | Shadow PSNR (dB) ↑ |
|---|---:|---:|---:|
| **Mask-guided U-Net** | **30.189** | **0.9514** | **26.303** |
| RGB U-Net baseline | 27.176 | 0.9324 | 24.097 |
| RGB residual (50 epochs) | 26.589 | 0.9268 | 25.062 |
| RGB residual (continued) | 26.428 | 0.9268 | 25.027 |

*These metrics are reported in the capstone presentation and have not been independently re-run for this repository.*

## Project visuals

### Input
![Shadowed input](assets/shadow_input.png)

### Automatic shadow removal — before and after
![Automatic mode](assets/automatic_before_after.png)

### Guided restoration
![Guided result](assets/guided_before_after.png)

![Mask-guided interface](assets/guided_mask_interface.png)

## Approach

- **Data:** ISTD paired shadow images, shadow masks, and shadow-free ground truth.
- **Architecture:** U-Net-style encoder–decoder, approximately 7.76M parameters in the reported configuration.
- **Training:** PyTorch; 256 × 256 inputs; batch size 8; mixed precision on Tesla T4.
- **Model variants:** RGB U-Net, residual RGB U-Net, mask-guided U-Net.
- **Evaluation:** PSNR, SSIM, and shadow/non-shadow regional PSNR.
- **Demo:** Gradio automatic and guided shadow restoration.

## Repository layout

```text
ShadowFix/
├── assets/
│   ├── shadow_input.png
│   ├── automatic_before_after.png
│   ├── guided_before_after.png
│   └── guided_mask_interface.png
├── notebooks/
│   └── shadow_removal_colab.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

Open the notebook in Google Colab with a GPU runtime. The notebook references an additional project ZIP and checkpoints, which must be supplied separately, along with the ISTD dataset. Adjust Google Drive and local data paths before running.

## Limitations

On images unlike ISTD, shadow traces may remain, very dark regions may appear washed out, and hard edges can introduce halos or color shifts. Strong correction can also affect non-shadow regions.

---

**Project showcase:** Odra Bira

*The accompanying presentation file is intentionally excluded from this repository.*
