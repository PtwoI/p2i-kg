# Latent Diffusion and DiT extension

This additive fragment connects two successive changes in diffusion architecture:

- **Latent Diffusion Model (LDM)** moves denoising from pixels into a learned
  perceptual latent space and introduces general cross-attention conditioning.
- **Diffusion Transformer (DiT)** keeps latent diffusion but replaces the U-Net
  backbone with patch tokens and adaLN-conditioned Transformer blocks.

Primary evidence is the CVPR 2022 LDM paper and the ICCV 2023 DiT paper. Runtime
details are cross-checked against the authors' official `CompVis/latent-diffusion`
and `facebookresearch/DiT` implementations. The executable examples are compact,
randomly initialized structural demonstrations; their exact omissions are recorded
in each model's `runtime_note`.

The fragment is safe for `p2i-kg extend docs/ldm-dit.json`: it only adds new IDs and
relationships, and therefore preserves existing local curation and catalog scope.
