"""CPU hierarchy checks: python -m tests.test_family_hierarchy (no downloads)."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from geometry.lorentz import exp_map0, half_aperture, oxy_angle, elementwise_dist
from losses.attribution_loss import EntailmentConeLoss
from training.attribution_args import parse_args, validate_args
from training.families import INCREFA_FAMILIES, increfa_mapping, minimum_cone_depth


class FamilyHierarchyTests(unittest.TestCase):
    def test_family_anchor_penalty_uses_worst_pair(self):
        tangent = torch.tensor([[1.2, 0.], [1.4, 0.], [3., 0.]], dtype=torch.float64,
                               requires_grad=True)
        classes = exp_map0(torch.tensor([[4., 0.], [4., .1], [4., -.1]], dtype=torch.float64))
        images = exp_map0(torch.tensor([[5., 0.]], dtype=torch.float64))
        base, _ = EntailmentConeLoss()(images, classes, torch.tensor([0]))
        loss, stats = EntailmentConeLoss(lambda_anchor=.5, target_norm=2)(
            images, classes, torch.tensor([0]), x_fam=exp_map0(tangent), family_of=torch.arange(3))
        # z=1; only the closest family pair (distance .2) violates the margin.
        self.assertTrue(torch.allclose(stats['loss_anchor'], tangent.new_tensor(.8)))
        self.assertTrue(torch.allclose(loss - base, tangent.new_tensor(.4)))
        grad, = torch.autograd.grad(loss - base, tangent)
        self.assertTrue(torch.allclose(grad, tangent.new_tensor([[.5, 0.], [-.5, 0.], [0., 0.]])))

    def test_taxonomy_and_subsets(self):
        names = [n for group in INCREFA_FAMILIES.values() for n in group]
        self.assertEqual(len(names), 29)
        self.assertEqual(len(set(names)), 29)
        self.assertEqual(increfa_mapping(["gpt4o", "real", "E4S", "Imagen3"]),
                         (["real", "gan", "autoregressive", "diffusion"], [2, 0, 1, 3]))
        self.assertEqual(increfa_mapping(["Imagen3", "cogview4"]), (["diffusion"], [0, 0]))
        self.assertEqual(increfa_mapping(["nano-banana", "cogview4", "midjourneyv6", "gpt4o"]),
                         (["autoregressive", "diffusion", "unknown"], [2, 1, 2, 0]))
        self.assertEqual(increfa_mapping(["midjourneyv6", "nano-banana"]),
                         (["unknown"], [0, 0]))
        self.assertEqual([len(group) for group in INCREFA_FAMILIES.values()], [1, 4, 1, 21, 2])
        with self.assertRaisesRegex(ValueError, "no family"):
            increfa_mapping(["unknown"])

    def test_cli_validation(self):
        base = ["--dataset_path", "x", "--hierarchy", "increfa", "--anchor_init", "random"]
        args = parse_args(base, dataset="iabench")
        validate_args(args)
        self.assertEqual((args.family_init_norm, args.family_depth_margin,
                          args.lambda_family, args.lambda_family_depth), (1.2, 0.5, 1, 0.5))
        for target in ("4", "2.5"):
            tied = parse_args(base + ["--target_norm", target], dataset="iabench")
            validate_args(tied)
            self.assertEqual(tied.geodesic_margin, float(target) / 2)
        matching = parse_args(base + ["--target_norm", "4", "--geodesic_margin", "2"], dataset="iabench")
        validate_args(matching)
        cases = [(["--freeze_anchors"], "learnable random"),
                 (["--anchor_init", "text"], "learnable random"),
                 (["--family_init_norm", "0.8", "--min_radius", "0.5"], "valid cones"),
                 (["--family_depth_margin", "1"], "Initial class depth"),
                 (["--lambda_family", "nan"], "finite"),
                 (["--family_depth_margin", "-1"], "non-negative"),
                 (["--target_norm", "4", "--geodesic_margin", "4"], "hierarchy mode"),
                 (["--curv", "0"], "positive curvature"),
                 (["--anchor_norm_range", "0", "3"], "cone domain"),
                 (["--generators", "unknown"], "no family")]
        for options, message in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                validate_args(parse_args(base + options, dataset="iabench"))
        with self.assertRaisesRegex(ValueError, "only for IABench"):
            validate_args(parse_args(base + ["--captions_dir", "x"]))

    def test_family_formula_and_gradients(self):
        for curv in (0.5, 1.0, 2.0):
            with self.subTest(curv=curv):
                families = torch.tensor([[1.2, .03], [1.15, -.05]], dtype=torch.float64,
                                        requires_grad=True)
                classes = torch.tensor([[1.4, .1], [2., -.2], [1.5, -.6]], dtype=torch.float64,
                                       requires_grad=True)
                images = exp_map0(torch.tensor([[3., .1], [3., -.1]], dtype=torch.float64), curv)
                xa, xf = exp_map0(classes, curv), exp_map0(families, curv)
                mapping, labels = torch.tensor([0, 1, 0]), torch.tensor([0, 2])
                loss_fn = EntailmentConeLoss(curv=curv, min_radius=.1, margin=.3,
                    lambda_neg=.7, lambda_anchor=.2, target_norm=3, lambda_family=1.3,
                    lambda_family_depth=.4, family_depth_margin=.5)
                loss, stats = loss_fn(images, xa, labels, x_fam=xf, family_of=mapping)
                baseline, _ = EntailmentConeLoss(curv=curv, min_radius=.1,
                    margin=.3, lambda_neg=.7)(images, xa, labels)
                psi = half_aperture(xf, curv, .1)
                xi = torch.stack([oxy_angle(f.expand_as(xa), xa, curv) for f in xf], dim=1)
                positive = (xi[torch.arange(3), mapping] - psi[mapping]).clamp_min(0).mean()
                wrong = torch.arange(2)[None, :] != mapping[:, None]
                negative = (psi[None, :] + .3 - xi).clamp_min(0)[wrong].mean()
                depth = (.5 + families.norm(dim=-1)[mapping] - classes.norm(dim=-1)).clamp_min(0).square().mean()
                anchor = (1.5 - elementwise_dist(xf[:1], xf[1:], curv, eps=1e-5)).clamp_min(0).mean()
                expected = baseline + 1.3 * (positive + .7 * negative) + .4 * depth + .2 * anchor
                self.assertTrue(torch.allclose(loss, expected))
                self.assertTrue(torch.allclose(stats["loss_family_depth"], depth))
                actual_grad = torch.autograd.grad(loss, (classes, families), retain_graph=True)
                expected_grad = torch.autograd.grad(expected, (classes, families))
                for actual, reference in zip(actual_grad, expected_grad):
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertGreater(actual.norm().item(), 0)
                    self.assertTrue(torch.allclose(actual, reference, atol=1e-8))
                self.assertTrue(all(not v.requires_grad for v in stats.values()))

    def test_anchor_margin_follows_scope(self):
        for curv in (.5, 1., 2.):
            with self.subTest(curv=curv):
                xa = exp_map0(torch.tensor([[2., 0.], [0., 2.]], dtype=torch.float64), curv)
                xf = exp_map0(torch.tensor([[1.2, 0.], [0., 1.2]], dtype=torch.float64), curv)
                images = exp_map0(torch.tensor([[3., 0.], [0., 3.]], dtype=torch.float64), curv)
                labels = torch.arange(2)
                criterion = EntailmentConeLoss(curv=curv, lambda_anchor=1, target_norm=4)
                for inputs, margin, points in (({}, 4., xa),
                        ({"x_fam": xf, "family_of": labels}, 2., xf)):
                    _, stats = criterion(images, xa, labels, **inputs)
                    expected = (margin - elementwise_dist(points[:1], points[1:], curv,
                                                          eps=1e-5)).clamp_min(0).mean()
                    self.assertTrue(torch.allclose(stats["loss_anchor"], expected))
                    _, explicit = EntailmentConeLoss(curv=curv, lambda_anchor=1,
                        target_norm=4, geodesic_margin=margin)(images, xa, labels, **inputs)
                    self.assertTrue(torch.allclose(explicit["loss_anchor"], expected))
                    with self.assertRaisesRegex(ValueError, "must equal"):
                        EntailmentConeLoss(curv=curv, target_norm=4,
                            geodesic_margin=6 - margin)(images, xa, labels, **inputs)

    def test_depth_and_anchor_gradients_are_separate(self):
        classes = torch.tensor([[2., .0], [1., 1.732]], dtype=torch.float64, requires_grad=True)
        families = torch.tensor([[1.2, .0], [1.2, .2]], dtype=torch.float64, requires_grad=True)
        images = exp_map0(torch.tensor([[3., .0], [1.5, 2.598]], dtype=torch.float64))
        mapping = labels = torch.arange(2)
        xa, xf = exp_map0(classes), exp_map0(families)
        baseline, _ = EntailmentConeLoss()(images, xa, labels)
        anchor, stats = EntailmentConeLoss(lambda_anchor=1, target_norm=3)(
            images, xa, labels, x_fam=xf, family_of=mapping)
        class_grad, family_grad = torch.autograd.grad(anchor - baseline, (classes, families))
        self.assertTrue(torch.allclose(class_grad, torch.zeros_like(class_grad)))
        self.assertGreater(family_grad.norm().item(), 0)
        self.assertGreater(stats["loss_anchor"].item(), 0)
        self.assertGreater(stats["mean_fam_cos_sim"].item(), 0)
        for depth, violated in ((1.4, True), (2., False)):
            classes = torch.tensor([[depth, .0], [depth, .0]], dtype=torch.float64, requires_grad=True)
            families = torch.tensor([[1.2, .0]], dtype=torch.float64, requires_grad=True)
            xa, xf = exp_map0(classes), exp_map0(families)
            baseline, _ = EntailmentConeLoss()(images, xa, labels)
            loss, stats = EntailmentConeLoss(lambda_family_depth=1)(
                images, xa, labels, x_fam=xf, family_of=torch.zeros(2, dtype=torch.long))
            gc, gf = torch.autograd.grad(loss - baseline, (classes, families))
            self.assertEqual(stats["loss_family_depth"] > 0, violated)
            if violated:
                self.assertTrue((gc[:, 0] < 0).all())
                self.assertGreater(gf[0, 0].item(), 0)
            else:
                self.assertEqual(gc.count_nonzero(), 0)
                self.assertEqual(gf.count_nonzero(), 0)

    def test_single_family_and_invalid_inputs(self):
        xa = exp_map0(torch.tensor([[2., .0], [2., .1]]))
        xf = exp_map0(torch.tensor([[1.2, .0]]))
        images = exp_map0(torch.tensor([[3., .0], [3., .1]]))
        labels, mapping = torch.arange(2), torch.zeros(2, dtype=torch.long)
        criterion = EntailmentConeLoss(lambda_family=1, lambda_anchor=1, target_norm=3)
        loss, stats = criterion(images, xa, labels, x_fam=xf, family_of=mapping)
        self.assertTrue(torch.isfinite(loss))
        for key in ("loss_anchor", "loss_family_neg", "wrong_family_violations",
                    "mean_fam_cos_sim", "max_fam_cos_sim"):
            self.assertEqual(stats[key], 0)
        for inputs in ({}, {"x_fam": xf},
                       {"x_fam": xf, "family_of": torch.tensor([0, 1])},
                       {"x_fam": xf, "family_of": mapping.float()},
                       {"x_fam": xf, "family_of": mapping[:1]}):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                criterion(images, xa, labels, **inputs)

    def test_valid_nested_cones_transmit_containment(self):
        family, classes, images = [exp_map0(torch.tensor([[r, .0]])) for r in (1.2, 2., 3.)]
        _, stats = EntailmentConeLoss(min_radius=.5, lambda_family=1)(
            images, classes, torch.tensor([0]), x_fam=family, family_of=torch.tensor([0]))
        self.assertEqual(stats["inside_img"], 1)
        self.assertEqual(stats["inside_class_family"], 1)
        self.assertEqual(stats["inside_img_family"], 1)

    def test_anchor_seed_optimizer_order_and_projection(self):
        from tests.test_attribution_training import TinyModel, TinyTokenizer
        from training.anchors import AnchorState

        args = parse_args(["--dataset_path", "x", "--anchor_init", "random",
                           "--hyperbolic_dim", "4", "--min_radius", ".5"], dataset="iabench")
        names = ["real", "E4S", "gpt4o", "Imagen3"]
        model = TinyModel()
        with patch("transformers.CLIPTokenizer.from_pretrained", return_value=TinyTokenizer()):
            torch.manual_seed(5)
            plain = AnchorState(model, None, names, names, args, "cpu")
            rng = torch.get_rng_state()
            args.hierarchy = "increfa"
            torch.manual_seed(5)
            hierarchy = AnchorState(model, None, names, names, args, "cpu")
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(plain.tangent(), hierarchy.tangent()))
        self.assertIs(hierarchy.parameters()[0], hierarchy.anchor_tangent)
        self.assertIs(hierarchy.parameters()[-1], hierarchy.family_tangent)
        minimum = minimum_cone_depth(args.curv, args.min_radius)
        with torch.no_grad():
            hierarchy.family_tangent.mul_(.01)
            hierarchy.anchor_tangent.mul_(.01)
        hierarchy.project_(args)
        for parameter in hierarchy.parameters():
            self.assertTrue(torch.all(parameter.norm(dim=-1) >= minimum - 1e-6))
            self.assertTrue(torch.isfinite(half_aperture(exp_map0(parameter), min_radius=.5)).all())

    def test_family_checkpoint_readout_and_legacy_fallback(self):
        from tests.probe_family_readout import load_family

        tangent = torch.tensor([[1.2, .0], [.0, 1.2]])
        checkpoint = {"hierarchy": "increfa", "class_names": ["real", "Imagen3"],
                      "family_names": ["real", "diffusion"], "family_of": [0, 1],
                      "family_tangent": tangent, "curv": 1., "min_radius": .5}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.pt"
            torch.save(checkpoint, path)
            model = Mock()
            with patch("transformers.CLIPTokenizer.from_pretrained") as tokenize:
                x, _, mapping, names, _ = load_family(path, model, "unused", ["Imagen3", "real"], "cpu")
                tokenize.assert_not_called()
                model.encode_text.assert_not_called()
                self.assertTrue(torch.equal(x, exp_map0(tangent)))
                self.assertEqual(mapping.tolist(), [1, 0])
                self.assertEqual(names, checkpoint["family_names"])
            del checkpoint["family_tangent"]
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "missing learned"):
                load_family(path, model, "unused", checkpoint["class_names"], "cpu")
            checkpoint["hierarchy"] = "legacy"
            torch.save(checkpoint, path)
            from tests.test_attribution_training import TinyTokenizer
            with patch("transformers.CLIPTokenizer.from_pretrained", return_value=TinyTokenizer()):
                model.encode_text.return_value = (exp_map0(tangent), tangent)
                load_family(path, model, "unused", checkpoint["class_names"], "cpu")
                model.encode_text.assert_called_once()


if __name__ == "__main__":
    unittest.main()
