"""PackFlow evaluation: crystal-matching metrics + test-set evaluation.

- :mod:`packflow.evaluation.metrics` -- the per-crystal metric formulas (RMSD,
  matching, density, AMD, ...), unchanged from the paper.
- :mod:`packflow.evaluation.evaluator` -- model loading, lambda sampling, the
  full test-set loop, seed/top-k aggregation and (optional) UMA scoring.
- :mod:`packflow.evaluation.reporting` -- saving results + summary printing.

High-level API::

    from packflow import load_checkpoint
    from packflow.evaluation import evaluate

    model = load_checkpoint("packflow-pa", device="cuda")
    summary = evaluate(model, data_dir="data/processed", n_steps=500,
                       num_seeds_per_crystal=8, lambda_val=1.0)
"""

from __future__ import annotations

from typing import List, Optional

# Heavy deps (torch, pymatgen, rdkit, UMA) are only needed for actual evaluation.
# Import the public names lazily (PEP 562) so lightweight consumers -- e.g.
# ``packflow.evaluation.paper_tables`` (stdlib-only) -- don't pull in that stack.
_LAZY = {
    "compute_all_metrics_for_crystal": ("packflow.evaluation.metrics", "compute_all_metrics_for_crystal"),
    "compute_metrics_batched": ("packflow.evaluation.metrics", "compute_metrics_batched"),
    "get_default_constants": ("packflow.evaluation.metrics", "get_default_constants"),
    "mean_metrics": ("packflow.evaluation.metrics", "mean_metrics"),
    "evaluator": ("packflow.evaluation.evaluator", None),
    "evaluate_test_set": ("packflow.evaluation.evaluator", "evaluate_test_set"),
    "load_pretrained_model": ("packflow.evaluation.evaluator", "load_pretrained_model"),
    "CSP_BLIND_TEST_REFCODES": ("packflow.evaluation.evaluator", "CSP_BLIND_TEST_REFCODES"),
    "save_results": ("packflow.evaluation.reporting", "save_results"),
    "print_summary": ("packflow.evaluation.reporting", "print_summary"),
}


def __getattr__(name: str):
    import importlib

    if name in _LAZY:
        module_name, attr = _LAZY[name]
        module = importlib.import_module(module_name)
        return module if attr is None else getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def evaluate(
    model,
    data_dir: str,
    output_dir: str = "evaluation_results",
    n_steps: int = 500,
    num_seeds_per_crystal: int = 1,
    num_monte_carlo_samples: int = 1,
    lambda_val: float = 1.0,
    max_crystals: Optional[int] = None,
    batch_size: int = 1,
    device: str = "cpu",
    seed: int = 42,
    refcodes: Optional[List[str]] = None,
    csp_blind_test: bool = False,
    compute_uma_metrics: bool = False,
    uma_relaxation_steps: int = 0,
    seed_batch_size: Optional[int] = None,
    save_crystal_data: bool = False,
    save_uma_features: bool = False,
    visualize: bool = False,
    model_name: str = "packflow-pa",
    chunk: bool = False,
):
    """Evaluate a PackFlow model on a processed test set and return summary metrics.

    Args:
        model: a loaded model (from :func:`packflow.load_checkpoint`) or a
            model-zoo name / checkpoint path (loaded with the eval loader).
        data_dir: directory containing ``test.pt``.
        Other args mirror the CLI options of the original evaluation script.

    Returns:
        The mean-metrics dict (also written, with per-crystal details, to
        ``output_dir``).
    """
    from . import evaluator
    from .evaluator import evaluate_test_set, load_pretrained_model, CSP_BLIND_TEST_REFCODES
    from .metrics import get_default_constants
    from .reporting import save_results, print_summary

    evaluator.set_deterministic_seed(seed)

    if isinstance(model, str):
        from packflow.checkpoints import resolve

        ckpt_path = resolve(model, download=False)
        flow_matching = load_pretrained_model(ckpt_path, device=device, model_type=model_name)
    else:
        flow_matching = model
    if hasattr(flow_matching, "model"):
        flow_matching.model.eval()

    refcode_filter = CSP_BLIND_TEST_REFCODES if csp_blind_test else refcodes
    _, test_loader = evaluator.load_test_data(data_dir, batch_size, max_crystals, refcode_filter=refcode_filter)
    atomic_masses, covalent_radii = get_default_constants()

    results = evaluate_test_set(
        flow_matching, test_loader, atomic_masses, covalent_radii,
        n_steps, max_crystals, visualize, output_dir,
        num_seeds_per_crystal, num_monte_carlo_samples, lambda_val,
        compute_uma_metrics, uma_relaxation_steps=uma_relaxation_steps,
        model_name=model_name, device=device, seed_batch_size=seed_batch_size,
        save_crystal_data=save_crystal_data, save_uma_features=save_uma_features,
    )
    if not results["best_metrics_list"]:
        raise RuntimeError("No successful evaluations.")

    summary = save_results(
        results["best_metrics_list"], results["all_crystals_data"],
        output_dir, getattr(flow_matching, "checkpoint_path", model_name),
        n_steps, data_dir, save_crystal_data=save_crystal_data,
        save_uma_features=save_uma_features, chunk=chunk,
    )
    print_summary(summary)
    return summary


__all__ = [
    "evaluate",
    "evaluate_test_set",
    "load_pretrained_model",
    "save_results",
    "print_summary",
    "compute_all_metrics_for_crystal",
    "compute_metrics_batched",
    "get_default_constants",
    "mean_metrics",
    "CSP_BLIND_TEST_REFCODES",
]
