# ComfyUI-Stencil

Training-free composition transfer via attention map injection. A stencil
transfers structure without color or style; this node does the same for
diffusion transformers.

Currently supports: Krea2.

## How it works

Per sampling step, the composition reference is noised to the current sigma
and run through the model. The reference's image-token queries and keys are
captured at the selected blocks and injected into the target generation,
while the target keeps its own values. This is the Plug-and-Play mechanism
(Tumanyan et al., CVPR 2023): the self-attention affinity that governs
spatial structure comes from the reference, but the content aggregated
through V stays the target's, so layout transfers without the reference's
appearance bleeding in.

## Workflow

```
[Comp image] -> VAEEncode -> RF Inversion -> Stencil Composition (Krea2) -> KSampler
                             (or plain LATENT into reference_latent)
```

Either input works:
- `rf_inversion`: output of the RF Inversion node from ComfyUi-Untwisting-RoPE.
  Only the clean latent is read; the trajectory is not needed.
- `reference_latent`: plain VAEEncode output.

Reference must match the generation resolution.

## Parameters

- `qk_strength`: 0-1 blend of reference image-token Q/K into target. 1.0 = full reference structure. Start 1.0.
- `block_start` / `block_end`: DiT block range. Krea2 has blocks 0-27.
- `sigma_high` / `sigma_low`: sigma window for injection. Composition is
  established early (high sigma). Default 1.0 -> 0.45.
- `noise_seed`: seed for the fixed noise used to bring the reference to each
  step's sigma.

## Cost

One extra full model forward per step while inside the sigma window, plus a
one-time calibration forward on the first step. Attention for injected blocks
is computed manually (no flash attention on those blocks).

## Limitations

- Do not combine with node packs that install optimized_attention_override
  (UntwistingRoPE) in the same chain. One override slot exists.
- CFG batching (cond+uncond in one call) is supported; separate-call CFG
  doubles the reference passes.
- Krea2 only for now.
