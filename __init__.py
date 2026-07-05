"""ComfyUI-Stencil: training-free composition transfer via attention map injection.

Mechanism (Krea2, v1):
  Per sampling step, inside a model_function_wrapper:
    1. Noise the clean composition reference analytically to the current sigma
       (flow matching: x_sigma = (1 - sigma) * x0 + sigma * eps, fixed eps per run).
    2. Reference pass: run the noisy reference through the DiT with the target's
       own conditioning. A hook on transformer_options["optimized_attention_override"]
       captures post-RoPE, GQA-expanded Q/K for the active blocks.
    3. Target pass: same override recomputes attention manually for active blocks,
       blends the image-token->image-token attention rows of the reference into the
       target (convex blend + row renormalization), then completes attention with
       the target's V.

Notes / assumptions:
  - Krea2 SingleStreamDiT, 28 blocks, attention routed through ComfyUI's
    optimized_attention with transformer_options support.
  - Block indexing is derived from a per-pass attention call counter. A one-time
    calibration pass counts total attention calls per forward; any extra calls
    (e.g. context refiner) are assumed to run BEFORE the 28 blocks and are treated
    as an offset.
  - Text length per pass is measured by hooking dm.txtmlp.forward (plain nn.Module,
    survives VRAM reloads). Fallback: img tokens = (H//2)*(W//2) from latent dims.
  - Reference and target must have identical latent spatial dims.
  - Do not combine with other packs that install optimized_attention_override
    (e.g. UntwistingRoPE's sdpa fix) in the same run; only one override slot exists.
"""

import math
import torch


# ---------------------------------------------------------------------------
# Locate the Krea2 diffusion model across ComfyUI wrapper variations.
# Mirrors ComfyUi-Untwisting-RoPE's resolver so the txtmlp hook lands.
# ---------------------------------------------------------------------------

_DIFFUSION_ATTR_PATHS = (
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",
    "diffusion_model",
)


def _parse_blocks(spec):
    """Parse a block-range string into a set of ints.

    Accepts comma/space separated single blocks and inclusive ranges, e.g.
    "7-15, 20-27" or "0 3 5-8". Returns an empty set for blank/None input.
    """
    if not spec:
        return set()
    out = set()
    for chunk in str(spec).replace(",", " ").split():
        if "-" in chunk:
            try:
                a, b = chunk.split("-", 1)
                a, b = int(a), int(b)
            except ValueError:
                continue
            if a > b:
                a, b = b, a
            out.update(range(a, b + 1))
        else:
            try:
                out.add(int(chunk))
            except ValueError:
                continue
    return out


def _get_attr_path(root, attr_path):
    obj = root
    for part in attr_path.split("."):
        if obj is None or not hasattr(obj, part):
            return None
        try:
            obj = getattr(obj, part)
        except Exception:
            return None
    return obj


def _find_diffusion_model(model_patcher):
    for path in _DIFFUSION_ATTR_PATHS:
        obj = _get_attr_path(model_patcher, path)
        if obj is not None:
            return obj
    return None


# ---------------------------------------------------------------------------
# Q/K/V normalization: always return BHSD with Q's head count (GQA expanded).
# ---------------------------------------------------------------------------

def _normalize_qkv(q, k, v, heads):
    if q.ndim == 3:
        b, s, hd = q.shape
        d = hd // heads
        q = q.view(b, s, heads, d).transpose(1, 2)
    if k.ndim == 3:
        b, s, hd = k.shape
        d = q.shape[-1]
        hk = hd // d
        k = k.view(b, s, hk, d).transpose(1, 2)
    if v.ndim == 3:
        b, s, hd = v.shape
        d = q.shape[-1]
        hv = hd // d
        v = v.view(b, s, hv, d).transpose(1, 2)
    hq = q.shape[1]
    if k.shape[1] != hq:
        k = k.repeat_interleave(hq // k.shape[1], dim=1)
    if v.shape[1] != hq:
        v = v.repeat_interleave(hq // v.shape[1], dim=1)
    return q, k, v


# ---------------------------------------------------------------------------
# Manual attention with reference attention-map blending.
# Head-chunked to bound memory. Softmax in fp32.
# ---------------------------------------------------------------------------

def _pnp_attention(q_t, k_t, v_t, q_r, k_r, v_r, txt_t, txt_r,
                   qk_strength, value_strength, head_chunk=4):
    """Structure injection for Krea2.

    Two independent levers on image-token positions (text positions untouched):
      qk_strength: blend reference Q/K into target (PnP affinity transfer,
                   Tumanyan et al. CVPR 2023). Transfers routing/structure
                   without content.
      value_strength: blend reference V into target. Transfers content/features
                   directly; strong but overlay-prone at full strength late in
                   sampling. Intended for early-sigma-only schedules.

    Shapes: q_t/k_t/v_t [B,H,S_t,D], S_t = txt_t + img.
            q_r/k_r/v_r [B,H,S_r,D], S_r = txt_r + img.
    img token count must match: (S_t - txt_t) == (S_r - txt_r).
    """
    b, h, s_t, d = q_t.shape
    scale = 1.0 / math.sqrt(d)

    if qk_strength > 0.0:
        q_inj = q_t.clone()
        k_inj = k_t.clone()
        qi_t = q_inj[:, :, txt_t:, :]
        ki_t = k_inj[:, :, txt_t:, :]
        qi_r = q_r[:, :, txt_r:, :].to(q_inj.dtype)
        ki_r = k_r[:, :, txt_r:, :].to(k_inj.dtype)
        q_inj[:, :, txt_t:, :] = (1.0 - qk_strength) * qi_t + qk_strength * qi_r
        k_inj[:, :, txt_t:, :] = (1.0 - qk_strength) * ki_t + qk_strength * ki_r
    else:
        q_inj, k_inj = q_t, k_t

    if value_strength > 0.0 and v_r is not None:
        v_mix = v_t.clone()
        vi_t = v_mix[:, :, txt_t:, :]
        vi_r = v_r[:, :, txt_r:, :].to(v_mix.dtype)
        v_mix[:, :, txt_t:, :] = (1.0 - value_strength) * vi_t + value_strength * vi_r
    else:
        v_mix = v_t

    out = torch.empty((b, h, s_t, d), device=q_t.device, dtype=v_mix.dtype)
    for hs in range(0, h, head_chunk):
        he = min(hs + head_chunk, h)
        logits = torch.matmul(q_inj[:, hs:he].float(),
                              k_inj[:, hs:he].float().transpose(-1, -2)) * scale
        a = torch.softmax(logits, dim=-1)
        del logits
        out[:, hs:he] = torch.matmul(a.to(v_mix.dtype), v_mix[:, hs:he])
        del a
    return out


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class StencilKrea2:
    """Composition transfer for Krea2 via attention map injection.

    Feed a composition reference (RF Inversion output from the UntwistingRoPE
    pack, or a plain LATENT). Per step, the reference is noised to the current
    sigma, passed through the model, its image-token attention maps are captured
    for the selected blocks, and blended into the target's attention.
    """

    CATEGORY = "model_patches/Stencil"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "qk_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blend weight of reference image-token queries and "
                               "keys into the target (PnP structure injection). "
                               "1.0 = full reference structure, values kept from "
                               "target. 0 = off.",
                }),
                "value_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blend weight of reference image-token values into "
                               "the target. Strong structure transfer but overlay-"
                               "prone; best restricted to early steps via a high "
                               "sigma_low (e.g. 0.7-0.8). 0 = off.",
                }),
                "block_start": ("INT", {
                    "default": 0, "min": 0, "max": 999, "step": 1,
                    "tooltip": "First DiT block to inject (inclusive).",
                }),
                "block_end": ("INT", {
                    "default": 27, "min": 0, "max": 999, "step": 1,
                    "tooltip": "Last DiT block to inject (inclusive).",
                }),
                "blocks": ("STRING", {
                    "default": "",
                    "tooltip": "Optional block range override, e.g. \"7-15, 20-27\". "
                               "If non-empty, overrides block_start/block_end. "
                               "If empty, block_start/block_end are used.",
                }),
                "sigma_high": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Inject while sigma <= this value. 1.0 = from the first step.",
                }),
                "sigma_low": ("FLOAT", {
                    "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Stop injecting once sigma drops below this value. "
                               "Composition is set early; late injection fights detail.",
                }),
                "noise_seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31 - 1,
                    "tooltip": "Seed for the fixed noise used to bring the reference "
                               "to each step's sigma.",
                }),
                "verbose": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "rf_inversion": ("*", {
                    "tooltip": "RF Inversion output from the UntwistingRoPE pack. "
                               "The clean reference latent is read from it; the RF "
                               "trajectory itself is not required.",
                }),
                "reference_latent": ("LATENT", {
                    "tooltip": "Alternative to rf_inversion: plain latent of the "
                               "composition reference (from VAEEncode).",
                }),
            },
        }

    def patch(self, model, qk_strength, value_strength, block_start, block_end,
              blocks,
              sigma_high, sigma_low, noise_seed, verbose=False,
              rf_inversion=None, reference_latent=None):

        # --- resolve clean reference latent -------------------------------
        ref_clean = None
        if isinstance(rf_inversion, dict) and torch.is_tensor(rf_inversion.get("untwist_ref_clean", None)):
            # Stored by RFInversion already in model (processed) latent space.
            ref_clean = rf_inversion["untwist_ref_clean"].detach().clone()
        elif isinstance(reference_latent, dict) and torch.is_tensor(reference_latent.get("samples", None)):
            samples = reference_latent["samples"].detach().clone()
            try:
                ref_clean = model.model.process_latent_in(samples)
            except Exception:
                ref_clean = samples
        if ref_clean is None:
            raise RuntimeError(
                "StencilKrea2: connect either rf_inversion (from RF Inversion) "
                "or reference_latent (from VAEEncode)."
            )

        model_clone = model.clone()
        n_blocks_expected = 28  # Krea2 SingleStreamDiT

        state = {
            "mode": "off",          # off | count | record | inject
            "attn_idx": 0,
            "offset": None,          # extra attention calls before block 0
            "ref_qk": {},            # block_idx -> (q_bhsd fp16, k_bhsd fp16)
            "txtlen_current": None,
            "txtlen_ref": None,
            "eps": None,
            "warned": set(),
            "printed_info": False,
        }
        if blocks and blocks.strip():
            active_blocks = _parse_blocks(blocks)
        else:
            active_blocks = set(range(int(block_start), int(block_end) + 1))
        qk_strength = float(qk_strength)
        value_strength = float(value_strength)
        sig_hi = float(sigma_high)
        sig_lo = float(sigma_low)

        def _warn_once(key, msg):
            if key not in state["warned"]:
                state["warned"].add(key)
                print(f"[Stencil] {msg}")

        # --- txtmlp hook: measure text token count per forward ------------
        dm = _find_diffusion_model(model_clone)
        if dm is None:
            _warn_once("dm", "could not locate diffusion_model; text length will "
                             "use the computed fallback.")
        try:
            if dm is not None and hasattr(dm, "txtmlp"):
                if not hasattr(dm, "_stencil_orig_txtmlp_forward"):
                    dm._stencil_orig_txtmlp_forward = dm.txtmlp.forward
                orig_txtmlp_fwd = dm._stencil_orig_txtmlp_forward

                def hooked_txtmlp(x, *a, **kw):
                    try:
                        state["txtlen_current"] = int(x.shape[1])
                    except Exception:
                        pass
                    return orig_txtmlp_fwd(x, *a, **kw)

                dm.txtmlp.forward = hooked_txtmlp
        except Exception:
            _warn_once("txtmlp", "txtmlp hook failed; falling back to computed "
                                 "image-token count for text length.")

        # Fallback image-token count from latent dims (patchify 2x2).
        lh, lw = int(ref_clean.shape[-2]), int(ref_clean.shape[-1])
        img_tokens_fallback = (lh // 2) * (lw // 2)

        # --- attention override --------------------------------------------
        def attention_override(orig_func, *args, **kwargs):
            mode = state["mode"]
            if mode == "off":
                return orig_func(*args, **kwargs)

            idx = state["attn_idx"]
            state["attn_idx"] = idx + 1

            if mode == "count":
                return orig_func(*args, **kwargs)

            offset = state["offset"] or 0
            block_idx = idx - offset
            if block_idx < 0 or block_idx not in active_blocks:
                return orig_func(*args, **kwargs)

            q = args[0]
            k = args[1]
            v = args[2]
            heads = args[3] if len(args) > 3 else kwargs.get("heads")
            mask = kwargs.get("mask", args[4] if len(args) > 4 else None)
            if mask is not None:
                _warn_once("mask", "attention mask present; skipping injection "
                                   "for masked calls.")
                return orig_func(*args, **kwargs)

            if mode == "record":
                qn, kn, vn = _normalize_qkv(q, k, v, heads)
                state["ref_qk"][block_idx] = (
                    qn.detach().to(torch.float16),
                    kn.detach().to(torch.float16),
                    vn.detach().to(torch.float16) if value_strength > 0.0 else None,
                )
                return orig_func(*args, **kwargs)

            # mode == "inject"
            stored = state["ref_qk"].get(block_idx)
            if stored is None:
                return orig_func(*args, **kwargs)

            q_t, k_t, v_t = _normalize_qkv(q, k, v, heads)
            q_r, k_r, v_r = stored

            s_t = q_t.shape[2]
            s_r = q_r.shape[2]
            txt_t = state["txtlen_current"]
            txt_r = state["txtlen_ref"]
            if txt_t is None:
                txt_t = s_t - img_tokens_fallback
            if txt_r is None:
                txt_r = s_r - img_tokens_fallback
            if (s_t - txt_t) != (s_r - txt_r) or txt_t < 0 or txt_r < 0:
                _warn_once("imglen", f"image token mismatch (target {s_t - txt_t}, "
                                     f"ref {s_r - txt_r}); passthrough.")
                return orig_func(*args, **kwargs)

            out = _pnp_attention(
                q_t, k_t, v_t,
                q_r.to(q_t.device), k_r.to(q_t.device),
                v_r.to(q_t.device) if v_r is not None else None,
                int(txt_t), int(txt_r), qk_strength, value_strength,
            )

            if kwargs.get("skip_output_reshape", False):
                return out
            b, h, s, d = out.shape
            return out.transpose(1, 2).reshape(b, s, h * d)

        # --- model function wrapper ----------------------------------------
        def model_function_wrapper(apply_model, wargs):
            input_x = wargs["input"]
            timestep = wargs["timestep"]
            c = wargs["c"].copy()
            to = c.get("transformer_options", {}).copy()
            to["optimized_attention_override"] = attention_override
            c["transformer_options"] = to

            try:
                sigma = float(timestep.flatten()[0].item())
            except Exception:
                sigma = 1.0

            active = ((qk_strength > 0.0) or (value_strength > 0.0)) and (sig_lo <= sigma <= sig_hi)
            if verbose:
                print(f"[Stencil] step sigma={sigma:.4f} injecting={active}")
            if not active:
                state["mode"] = "off"
                return apply_model(input_x, timestep, **c)

            # spatial check
            if ref_clean.shape[-2:] != input_x.shape[-2:]:
                raise RuntimeError(
                    f"StencilKrea2: spatial mismatch. reference latent "
                    f"{tuple(ref_clean.shape[-2:])} vs target "
                    f"{tuple(input_x.shape[-2:])}. Match resolutions."
                )

            ref = ref_clean.to(device=input_x.device, dtype=input_x.dtype)
            if ref.shape[0] != input_x.shape[0]:
                reps = [input_x.shape[0]] + [1] * (ref.ndim - 1)
                ref = ref[:1].repeat(*reps)

            if state["eps"] is None:
                g = torch.Generator(device="cpu").manual_seed(int(noise_seed))
                state["eps"] = torch.randn(
                    ref_clean.shape, generator=g, dtype=torch.float32,
                ).to(device=input_x.device)
            eps = state["eps"].to(device=input_x.device, dtype=input_x.dtype)
            if eps.shape[0] != input_x.shape[0]:
                reps = [input_x.shape[0]] + [1] * (eps.ndim - 1)
                eps = eps[:1].repeat(*reps)

            # flow matching forward noising
            ref_noisy = (1.0 - sigma) * ref + sigma * eps

            # one-time calibration: count attention calls per forward
            if state["offset"] is None:
                state["mode"] = "count"
                state["attn_idx"] = 0
                _ = apply_model(ref_noisy, timestep, **c)
                total = state["attn_idx"]
                state["offset"] = max(0, total - n_blocks_expected)
                if verbose or state["offset"] != 0:
                    print(f"[Stencil] calibration: {total} attention calls per "
                          f"forward, block offset {state['offset']}.")

            # reference pass: record maps
            state["mode"] = "record"
            state["attn_idx"] = 0
            state["ref_qk"] = {}
            _ = apply_model(ref_noisy, timestep, **c)
            state["txtlen_ref"] = state["txtlen_current"]

            if verbose and not state["printed_info"]:
                state["printed_info"] = True
                blocks_stored = sorted(state["ref_qk"].keys())
                print(f"[Stencil] sigma={sigma:.4f} stored blocks="
                      f"{blocks_stored[:3]}..{blocks_stored[-3:] if len(blocks_stored) > 3 else ''} "
                      f"txtlen_ref={state['txtlen_ref']} "
                      f"qk_strength={qk_strength}")

            # target pass: inject
            state["mode"] = "inject"
            state["attn_idx"] = 0
            try:
                out = apply_model(input_x, timestep, **c)
            finally:
                state["mode"] = "off"
                state["ref_qk"] = {}
            return out

        model_clone.set_model_unet_function_wrapper(model_function_wrapper)
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "StencilKrea2": StencilKrea2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StencilKrea2": "Stencil Composition (Krea2)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
