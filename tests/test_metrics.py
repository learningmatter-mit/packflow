"""Regression pins for the evaluation metric formulas.

These lock the relocated ``crystal_metrics`` -> ``packflow.evaluation.metrics``
code to known numeric outputs on small fixed inputs, so the move stays exact.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from packflow.evaluation import metrics as M  # noqa: E402


def test_mean_metrics_simple_average():
    out = M.mean_metrics([{"a": 1.0, "b": 10.0}, {"a": 3.0, "b": 30.0}])
    assert out["mean_a"] == pytest.approx(2.0)
    assert out["mean_b"] == pytest.approx(20.0)
    assert out["num_crystals"] == 2


def test_jsd_identical_is_zero_and_disjoint_is_ln2():
    p = torch.tensor([1.0, 0.0, 0.0])
    assert M.jsd(p, p.clone()) == pytest.approx(0.0, abs=1e-6)
    q = torch.tensor([0.0, 0.0, 1.0])
    # JSD of disjoint distributions -> ln(2).
    assert M.jsd(p, q) == pytest.approx(math.log(2.0), abs=1e-4)


def test_hist_pdf_normalizes():
    vals = torch.tensor([0.5, 0.5, 1.5, 2.5])
    bins = torch.tensor([0.0, 1.0, 2.0, 3.0])
    pdf = M.compute_hist_pdf(vals, bins)
    assert pdf.sum().item() == pytest.approx(1.0, abs=1e-4)
    assert pdf.argmax().item() == 0  # two values land in the first bin


def test_hist_overlap_identical_is_100():
    p = torch.tensor([0.2, 0.3, 0.5])
    assert M.compute_hist_overlap(p, p.clone()) == pytest.approx(100.0, abs=1e-3)


def test_wasserstein_shift():
    # Mass at bin 0 vs mass at bin 2, bin_width 1 -> W1 == 2.
    p = torch.tensor([1.0, 0.0, 0.0])
    q = torch.tensor([0.0, 0.0, 1.0])
    assert M.wasserstein_from_pmf(p, q, bin_width=1.0) == pytest.approx(2.0, abs=1e-4)


def test_density_cubic_cell():
    # 10 A cubic cell (V = 1000 A^3), single carbon (12.011 amu).
    lattice = torch.tensor([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    z = torch.tensor([6])
    masses, _ = M.get_default_constants()
    d = M.compute_density(lattice, z, masses)
    expected = (masses[6] / 1000.0) * 1.660539
    assert float(d) == pytest.approx(expected, rel=1e-4)
