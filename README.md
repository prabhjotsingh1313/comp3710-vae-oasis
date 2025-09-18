# COMP3710 – VAE on OASIS (4.3.1 Easy)

**Goal.** Train a **Variational Autoencoder (VAE)** on the preprocessed OASIS brain MRI slices and
visualise the learned latent manifold (UMAP) + reconstructions.

## Result (evidence)

- **Epochs:** 40  
- **Best val loss:** ~0.2636 (rec ~0.2634, KL ~0.0004)  
- **Artifacts:** see `outputs/recons_final.png` and `outputs/latent_umap.png`  
- **Hardware:** Google Colab GPU (Torch CUDA available: `True`)

> Log sample:
> ```
> Epoch 040 | train 0.2639 (rec 0.2636, kl 0.0004) | 
> val 0.2637 (rec 0.2635, kl 0.0004)
> ```

## Repository structure
- `notebooks/oasis_vae_colab.ipynb` – runnable end-to-end Colab notebook
- `src/` – (optional) modularised code: dataset, model, training, viz
- `outputs/` – reconstructions & UMAP (images only; **no datasets** checked in)

## Data (OASIS)
The dataset is on UQ Rangpur under `/home/groups/comp3710/`.  
For Colab, I zipped my local copy into Drive (not included in repo).
Update paths in the notebook:
- `ZIP_PATH = "/content/drive/MyDrive/datasets/oasis/keras_png_slices_data.zip"`
- `EXTRACT_TO = "/content/oasis_png"`

## How to run (Colab)
1. Open `notebooks/oasis_vae_colab.ipynb` in Google Colab.
2. Mount Drive and set the `ZIP_PATH` to your copy of OASIS PNG slices.
3. Run all cells. Artifacts are saved to your Drive path (`/oasis_vae_outputs`).

## Model
- **ConvVAE** with stride-2 conv encoder to 8×8 feature map, `z_dim=64`, deconv decoder.
- **Loss:** `BCEWithLogits` reconstruction + β-KL (β warmup to 0.5).
- **Training:** Adam (`1e-4`), AMP optional, gradient clip=2.0.
- **Viz:** Reconstruction grid (input vs recon); UMAP on latent means `μ`.

## Notes / Limitations
- Dataset **not** distributed in this repo (size/licensing).
- KL is small (dataset & β schedule); you can increase β after warmup for a tighter prior.
- For Rangpur, point the notebook to `/home/groups/comp3710/...` instead of Drive.

## License
MIT – see `LICENSE`.
