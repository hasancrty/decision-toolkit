"""Core modülü için temel birim testleri (pytest)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from core import ahp, topsis, entropy, wsm, sensitivity


def test_ahp_weights_sum_to_one():
    matrix = [
        [1, 3, 5],
        [1/3, 1, 3],
        [1/5, 1/3, 1],
    ]
    result = ahp.compute_weights(matrix, labels=["A", "B", "C"])
    assert np.isclose(result.weights.sum(), 1.0)
    assert len(result.weights) == 3


def test_ahp_consistent_identity_like_matrix():
    # Tamamen tutarlı bir matris: A çok daha önemli, diğerleri eşit
    matrix = [
        [1, 5, 5],
        [1/5, 1, 1],
        [1/5, 1, 1],
    ]
    result = ahp.compute_weights(matrix)
    assert result.is_consistent


def test_ahp_rejects_non_square_matrix():
    with pytest.raises(ValueError):
        ahp.compute_weights([[1, 2], [0.5, 1], [1, 1]])


def test_topsis_best_alternative_is_dominant():
    # Alternatif A her kriterde en iyi (benefit kriterlerde en yüksek)
    decision_matrix = [
        [10, 10],  # A: dominant
        [5, 5],    # B
        [1, 1],    # C
    ]
    weights = [0.5, 0.5]
    criteria_types = ["benefit", "benefit"]
    result = topsis.rank(decision_matrix, weights, criteria_types, ["A", "B", "C"])
    assert result.ranked_labels()[0] == "A"
    assert result.ranked_labels()[-1] == "C"


def test_topsis_cost_criteria_prefers_lower_values():
    # Tek kriter: maliyet (cost) -> en düşük maliyetli alternatif kazanmalı
    decision_matrix = [[100], [50], [200]]
    weights = [1.0]
    criteria_types = ["cost"]
    result = topsis.rank(decision_matrix, weights, criteria_types, ["Pahalı", "Ucuz", "En Pahalı"])
    assert result.ranked_labels()[0] == "Ucuz"


def test_topsis_scores_between_zero_and_one():
    decision_matrix = [[3, 7], [9, 2], [5, 5]]
    weights = [0.6, 0.4]
    criteria_types = ["benefit", "cost"]
    result = topsis.rank(decision_matrix, weights, criteria_types)
    assert all(0.0 <= s <= 1.0 for s in result.scores)


def test_topsis_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        topsis.rank([[1, 2], [3, 4]], weights=[1.0], criteria_types=["benefit", "cost"])


# ---------------------------------------------------------------------
# Entropy testleri
# ---------------------------------------------------------------------

def test_entropy_weights_sum_to_one():
    decision_matrix = [[250, 7, 12, 8], [400, 9, 7, 9], [300, 8, 9, 6], [280, 6, 15, 7]]
    result = entropy.compute_weights(decision_matrix, criteria_types=["cost", "benefit", "cost", "benefit"])
    assert np.isclose(result.weights.sum(), 1.0)


def test_entropy_identical_column_gets_low_weight():
    # İkinci kriterde tüm alternatifler aynı değere sahip -> ayırt edici değil -> düşük ağırlık
    decision_matrix = [[10, 5], [20, 5], [30, 5]]
    result = entropy.compute_weights(decision_matrix, criteria_types=["benefit", "benefit"])
    assert result.weights[1] < result.weights[0]


def test_combine_weights_alpha_extremes():
    sw = np.array([0.7, 0.3])
    ow = np.array([0.2, 0.8])
    only_subjective = entropy.combine_weights(sw, ow, alpha=1.0)
    only_objective = entropy.combine_weights(sw, ow, alpha=0.0)
    assert np.allclose(only_subjective, sw)
    assert np.allclose(only_objective, ow)


def test_combine_weights_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        entropy.combine_weights([0.5, 0.5], [0.5, 0.5], alpha=1.5)


# ---------------------------------------------------------------------
# WSM / WPM testleri
# ---------------------------------------------------------------------

def test_wsm_dominant_alternative_wins():
    decision_matrix = [[10, 10], [5, 5], [1, 1]]
    result = wsm.wsm_rank(decision_matrix, weights=[0.5, 0.5], criteria_types=["benefit", "benefit"],
                            alternative_labels=["A", "B", "C"])
    assert result.ranked_labels()[0] == "A"


def test_wpm_dominant_alternative_wins():
    decision_matrix = [[10, 10], [5, 5], [1, 1]]
    result = wsm.wpm_rank(decision_matrix, weights=[0.5, 0.5], criteria_types=["benefit", "benefit"],
                            alternative_labels=["A", "B", "C"])
    assert result.ranked_labels()[0] == "A"


def test_compare_methods_returns_all_three():
    decision_matrix = [[250, 7], [400, 9], [300, 8]]
    comparison = wsm.compare_methods(decision_matrix, weights=[0.5, 0.5],
                                       criteria_types=["cost", "benefit"],
                                       alternative_labels=["A", "B", "C"])
    assert set(comparison.keys()) == {"WSM", "WPM", "TOPSIS"}
    for ranking in comparison.values():
        assert set(ranking) == {"A", "B", "C"}


# ---------------------------------------------------------------------
# Duyarlılık analizi testleri
# ---------------------------------------------------------------------

def test_vary_single_weight_extremes_match_pure_criterion():
    # Ağırlık 1.0 iken kazanan, o kriterde en iyi olan alternatif olmalı
    decision_matrix = [[10, 1], [1, 10], [5, 5]]
    result = sensitivity.vary_single_weight(
        decision_matrix, base_weights=[0.5, 0.5], criteria_types=["benefit", "benefit"],
        criterion_index=0, alternative_labels=["A", "B", "C"], steps=5,
    )
    assert result.top_alternative_per_step[-1] == "A"  # ağırlık=1.0 -> kriter 0 (A en iyi)
    assert result.top_alternative_per_step[0] == "B"   # ağırlık=0.0 -> kriter 1 (B en iyi)


def test_vary_single_weight_rejects_bad_index():
    with pytest.raises(ValueError):
        sensitivity.vary_single_weight([[1, 2]], base_weights=[0.5, 0.5],
                                         criteria_types=["benefit", "benefit"], criterion_index=5)


def test_weight_perturbation_summary_returns_all_criteria():
    decision_matrix = [[250, 7, 12], [400, 9, 7], [300, 8, 9]]
    summary = sensitivity.weight_perturbation_summary(
        decision_matrix, base_weights=[0.4, 0.4, 0.2], criteria_types=["cost", "benefit", "cost"],
        criteria_labels=["Fiyat", "Kalite", "Süre"],
    )
    assert set(summary.keys()) == {"Fiyat", "Kalite", "Süre"}
    assert all(isinstance(v, (bool, np.bool_)) for v in summary.values())
