"""
WSM (Weighted Sum Model) ve WPM (Weighted Product Model).

En basit ve en eski MCDA yöntemleridir. TOPSIS'e göre daha az sofistike olsa
da, sonuçları TOPSIS ile karşılaştırarak sıralamanın yöntemden bağımsız ne
kadar "sağlam" (robust) olduğunu görmek için idealdirler. İki yöntem aynı
alternatifi 1. sıraya koyuyorsa, bu sonuca güven artar.

WSM: her alternatifin skoru = kriterlerin ağırlıklı toplamı.
    S_i = sum_j(w_j * r_ij)
WPM: her alternatifin skoru = kriterlerin ağırlıklı çarpımı (üstel).
    S_i = prod_j(r_ij ^ w_j)
    WPM, oransal karşılaştırmalara WSM'den daha az duyarlıdır ve birim
    değişimlerinden etkilenmez (dimensionless).

Referans:
    Triantaphyllou, E. (2000). Multi-Criteria Decision Making Methods:
    A Comparative Study. Springer.
"""

from __future__ import annotations
import numpy as np


class RankingResult:
    """WSM/WPM ortak sonuç sınıfı (TOPSISResult ile aynı arayüz)."""

    def __init__(self, scores: np.ndarray, ranking: list[int], alternative_labels: list[str], method: str):
        self.scores = scores
        self.ranking = ranking
        self.alternative_labels = alternative_labels
        self.method = method

    def ranked_labels(self) -> list[str]:
        return [self.alternative_labels[i] for i in self.ranking]

    def as_dict(self) -> dict[str, float]:
        return {label: float(s) for label, s in zip(self.alternative_labels, self.scores)}

    def __repr__(self) -> str:
        lines = [f"  {r}. {self.alternative_labels[i]}  (skor={self.scores[i]:.4f})"
                  for r, i in enumerate(self.ranking, start=1)]
        return f"RankingResult[{self.method}](\n" + "\n".join(lines) + "\n)"


def _normalize_for_ratio(matrix: np.ndarray, criteria_types: list[str]) -> np.ndarray:
    """Oran (ratio) normalizasyonu: her sütunu kendi maksimumuna böler.
    Cost kriterlerde ters çevirir ki her zaman 'yüksek = iyi' olsun."""
    normalized = matrix.copy()
    n = matrix.shape[1]
    for j in range(n):
        col = matrix[:, j]
        if criteria_types[j] == "benefit":
            col_max = col.max()
            normalized[:, j] = col / col_max if col_max != 0 else 0
        else:  # cost
            col_min = col.min()
            col_min = col_min if col_min != 0 else 1e-12
            normalized[:, j] = col_min / col
    return normalized


def wsm_rank(
    decision_matrix: list[list[float]],
    weights: list[float],
    criteria_types: list[str],
    alternative_labels: list[str] | None = None,
) -> RankingResult:
    """Weighted Sum Model ile alternatifleri sıralar."""
    matrix = np.array(decision_matrix, dtype=float)
    m, n = matrix.shape
    if alternative_labels is None:
        alternative_labels = [f"A{i+1}" for i in range(m)]

    normalized = _normalize_for_ratio(matrix, criteria_types)
    w = np.array(weights, dtype=float)
    scores = (normalized * w).sum(axis=1)
    ranking = list(np.argsort(-scores))
    return RankingResult(scores, ranking, alternative_labels, method="WSM")


def wpm_rank(
    decision_matrix: list[list[float]],
    weights: list[float],
    criteria_types: list[str],
    alternative_labels: list[str] | None = None,
) -> RankingResult:
    """Weighted Product Model ile alternatifleri sıralar."""
    matrix = np.array(decision_matrix, dtype=float)
    m, n = matrix.shape
    if alternative_labels is None:
        alternative_labels = [f"A{i+1}" for i in range(m)]

    normalized = _normalize_for_ratio(matrix, criteria_types)
    normalized[normalized <= 0] = 1e-12  # 0^w veya negatif üs koruması
    w = np.array(weights, dtype=float)
    scores = np.prod(normalized ** w, axis=1)
    ranking = list(np.argsort(-scores))
    return RankingResult(scores, ranking, alternative_labels, method="WPM")


def compare_methods(
    decision_matrix: list[list[float]],
    weights: list[float],
    criteria_types: list[str],
    alternative_labels: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Aynı problemi WSM, WPM ve (topsis modülünden) TOPSIS ile çözüp
    sıralamaları yan yana karşılaştırma için döner.

    Returns:
        dict: {"WSM": [...sıralı isimler...], "WPM": [...], "TOPSIS": [...]}
    """
    from . import topsis as topsis_module  # local import: circular import önleme

    wsm_result = wsm_rank(decision_matrix, weights, criteria_types, alternative_labels)
    wpm_result = wpm_rank(decision_matrix, weights, criteria_types, alternative_labels)
    topsis_result = topsis_module.rank(decision_matrix, weights, criteria_types, alternative_labels)

    return {
        "WSM": wsm_result.ranked_labels(),
        "WPM": wpm_result.ranked_labels(),
        "TOPSIS": topsis_result.ranked_labels(),
    }
