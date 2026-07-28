"""Evaluate the pixel+spectral attributor under the baselines' exact protocol.

Thin wrapper: the whole harness loop lives in patch_attribution.eval.run_eval,
only the model builder and the fused scoring differ.

Usage:
    python -m patch_freq_attribution.eval \\
        --checkpoint $WORK/hyp_fine_tuning/checkpoints/attribution_22cls_patchfreq.pt \\
        --root_dir $FAST/datasets/iab_dataset --level_start 0 --level_end 7 \\
        --log_dir $WORK/outputs/hypclip_patchfreq
"""
from comparison.training.test_hypclip import load_anchors
from patch_attribution.eval import parse_args, run_eval
from patch_freq_attribution.model import PatchFreqAttributionCLIP, fused_logits


def build_patch_freq_model(ckpt, device):
    curv = ckpt.get('curv', 1.0)
    model = PatchFreqAttributionCLIP(
        clip_name=ckpt['clip_name'],
        lora_r=ckpt.get('lora_r', 16),
        lora_alpha=ckpt.get('lora_alpha', 32),
        hyperbolic_dim=ckpt.get('hyperbolic_dim', 128),
        curv=curv,
        patch_size=ckpt.get('patch_size', 112),
    ).to(device)
    model.clip.load_state_dict(ckpt['lora_state'])
    model.projection.load_state_dict(ckpt['projection'])
    model.projection_spec.load_state_dict(ckpt['projection_spec'])
    model.eval()

    # Both anchor sets go through load_anchors so both get the checkpoint-order →
    # harness-order permutation; a shim dict feeds it the spectral tangents.
    x_anc_pix = load_anchors(ckpt, model, curv, device)
    x_anc_spec = load_anchors({'class_names': ckpt['class_names'],
                               'anchor_tangent': ckpt['anchor_tangent_spec']},
                              model, curv, device)
    tau = ckpt['fusion_tau'].to(device)
    print(f"Fusion temperatures: pixel={tau.exp()[0]:.3f}  spectral={tau.exp()[1]:.3f}")

    def logits_fn(pixel):
        x_views, x_spec = model(pixel)
        return fused_logits(x_views.float(), x_spec.float(),
                            x_anc_pix, x_anc_spec, tau, curv=curv)

    return model, logits_fn


if __name__ == '__main__':
    run_eval(parse_args(), build=build_patch_freq_model)
