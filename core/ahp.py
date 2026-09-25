"""
AHP (Analytic Hierarchy Process) implementasyonu.

Saaty'nin geliştirdiği ikili karşılaştırma (pairwise comparison) yöntemine
dayanır. Karar vericiler kriterleri birbirine göre 1-9 skalasında karşılaştırır,
bu modül de kriter ağırlıklarını ve tutarlılık oranını (Consistency Ratio) hesaplar.

Referans:
    Saaty, T. L. (1980). The Analytic Hierarchy Process. McGraw-Hill.
"""

from __future__ import annotations
import numpy as np


# Saaty'nin rastgele tutarlılık indeksi (Random Index) tablosu, n=1..10
_RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}


class AHPResult:
    """AHP hesaplama sonucunu tutan basit veri sınıfı."""

    def __init__(self, weights: np.ndarray, consistency_ratio: float, labels: list[str]):
        self.weights = weights
        self.consistency_ratio = consistency_ratio
        self.labels = labels

    @property
    def is_consistent(self) -> bool:
        """CR < 0.10 ise karşılaştırmalar tutarlı kabul edilir (Saaty kriteri)."""
        return self.consistency_ratio < 0.10

    def as_dict(self) -> dict[str, float]:
        return {label: float(w) for label, w in zip(self.labels, self.weights)}

    def __repr__(self) -> str:
        lines = [f"  {label}: {w:.4f}" for label, w in zip(self.labels, self.weights)]
        status = "tutarlı ✅" if self.is_consistent else "TUTARSIZ ⚠️"
        return (
            "AHPResult(\n"
            + "\n".join(lines)
            + f"\n  CR={self.consistency_ratio:.4f} ({status})\n)"
        )


def compute_weights(comparison_matrix: list[list[float]], labels: list[str] | None = None) -> AHPResult:
    """
    İkili karşılaştırma matrisinden kriter ağırlıklarını hesaplar.

    Args:
        comparison_matrix: n x n boyutunda ikili karşılaştırma matrisi.
            matrix[i][j], i. kriterin j. kritere göre önemini ifade eder
            (Saaty skalası: 1=eşit önemde ... 9=aşırı derecede önemli).
            Matris karşıt-simetrik olmalıdır: matrix[j][i] = 1 / matrix[i][j].
        labels: Kriter isimleri (opsiyonel, verilmezse "C1", "C2", ... kullanılır).

    Returns:
        AHPResult: normalize edilmiş ağırlıklar ve tutarlılık oranı.
    """
    matrix = np.array(comparison_matrix, dtype=float)
    n = matrix.shape[0]

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Karşılaştırma matrisi kare (n x n) olmalıdır.")
    if labels is None:
        labels = [f"C{i+1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels uzunluğu matris boyutuyla eşleşmiyor.")

    # Eigenvalue yöntemi: normalize edilmiş sütunların satır ortalaması
    col_sums = matrix.sum(axis=0)
    normalized = matrix / col_sums
    weights = normalized.mean(axis=1)

    # Tutarlılık kontrolü (lambda_max, CI, CR)
    weighted_sum = matrix @ weights
    lambda_max = (weighted_sum / weights).mean()
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = _RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0

    return AHPResult(weights=weights, consistency_ratio=cr, labels=labels)
