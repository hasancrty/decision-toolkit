"""
Entropy Ağırlıklandırma Yöntemi (Shannon Entropy Weighting).

AHP, karar vericinin SUBJEKTIF yargılarına (ikili karşılaştırmalar) dayanırken,
Entropy yöntemi kriter ağırlıklarını doğrudan VERİDEN, yani karar matrisindeki
değerlerin dağılımından OBJEKTIF olarak hesaplar.

Mantık: Bir kriterde alternatifler arasındaki değerler birbirine çok yakınsa
(düşük "bilgi içeriği"/entropi yüksekse), o kriter alternatifleri ayırt etmede
az işe yarar → düşük ağırlık alır. Değerler arasında büyük farklar varsa
(entropi düşükse), o kriter ayırt edici bilgi taşır → yüksek ağırlık alır.

Kullanım senaryosu: AHP ile hesaplanan subjektif ağırlıkları, veriden gelen
objektif ağırlıklarla karşılaştırarak veya ikisini birleştirerek (bkz.
`combine_weights`) daha sağlam bir ağırlık seti elde edebilirsiniz.

Referans:
    Shannon, C. E. (1948). A Mathematical Theory of Communication.
    Zeleny, M. (1982). Multiple Criteria Decision Making. McGraw-Hill.
"""

from __future__ import annotations
import numpy as np


class EntropyResult:
    """Entropy ağırlıklandırma sonucunu tutan veri sınıfı."""

    def __init__(self, weights: np.ndarray, entropy: np.ndarray, labels: list[str]):
        self.weights = weights
        self.entropy = entropy  # her kriter için ham entropi değeri (0-1 arası)
        self.labels = labels

    def as_dict(self) -> dict[str, float]:
        return {label: float(w) for label, w in zip(self.labels, self.weights)}

    def __repr__(self) -> str:
        lines = [
            f"  {label}: ağırlık={w:.4f}  (entropi={e:.4f})"
            for label, w, e in zip(self.labels, self.weights, self.entropy)
        ]
        return "EntropyResult(\n" + "\n".join(lines) + "\n)"


def compute_weights(
    decision_matrix: list[list[float]],
    criteria_types: list[str] | None = None,
    labels: list[str] | None = None,
) -> EntropyResult:
    """
    Karar matrisinden Shannon entropi yöntemiyle objektif kriter ağırlıkları hesaplar.

    Args:
        decision_matrix: m x n boyutunda matris (m=alternatif, n=kriter).
        criteria_types: n uzunluğunda "benefit"/"cost" listesi. Verilirse "cost"
            kriterleri önce normalize edilirken ters çevrilir (düşük değer
            yüksek fayda olarak ele alınır). Verilmezse tüm kriterler "benefit"
            gibi ham haliyle işlenir.
        labels: Kriter isimleri (opsiyonel).

    Returns:
        EntropyResult: her kriter için hesaplanan ağırlık ve entropi değeri.
    """
    matrix = np.array(decision_matrix, dtype=float)
    m, n = matrix.shape

    if labels is None:
        labels = [f"C{i+1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels uzunluğu kriter sayısıyla eşleşmiyor.")

    work_matrix = matrix.copy()
    if criteria_types is not None:
        if len(criteria_types) != n:
            raise ValueError("criteria_types uzunluğu kriter sayısıyla eşleşmiyor.")
        for j, ctype in enumerate(criteria_types):
            if ctype == "cost":
                col = work_matrix[:, j]
                col_max = col.max()
                # cost kriterini benefit'e çevir: yüksek maliyet -> düşük skor
                work_matrix[:, j] = (col_max - col) + 1e-12

    # 1. Normalizasyon: her kriter sütununu toplamı 1 olacak şekilde ölçekle
    col_sums = work_matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1e-12
    p = work_matrix / col_sums
    p[p == 0] = 1e-12  # log(0) koruması

    # 2. Entropi hesabı: e_j = -k * sum_i(p_ij * ln(p_ij)), k = 1/ln(m)
    k = 1.0 / np.log(m) if m > 1 else 0.0
    entropy = -k * (p * np.log(p)).sum(axis=0)
    entropy = np.clip(entropy, 0.0, 1.0)

    # 3. Çeşitlilik derecesi (degree of diversification) ve ağırlıklar
    diversification = 1.0 - entropy
    total_diversification = diversification.sum()
    if total_diversification == 0:
        # Tüm kriterler eşit derecede "bilgi vermiyor" -> eşit ağırlık ver
        weights = np.ones(n) / n
    else:
        weights = diversification / total_diversification

    return EntropyResult(weights=weights, entropy=entropy, labels=labels)


def combine_weights(
    subjective_weights: list[float],
    objective_weights: list[float],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    AHP'den gelen subjektif ağırlıklarla Entropy'den gelen objektif ağırlıkları
    birleştirir (weighted combination).

    Args:
        subjective_weights: örn. AHP'den gelen ağırlıklar.
        objective_weights: örn. Entropy'den gelen ağırlıklar.
        alpha: subjektif ağırlığın toplam içindeki payı (0-1 arası).
            alpha=0.5 -> ikisine eşit ağırlık verilir.
            alpha=1.0 -> yalnızca subjektif (AHP) kullanılır.
            alpha=0.0 -> yalnızca objektif (Entropy) kullanılır.

    Returns:
        np.ndarray: birleştirilmiş ve normalize edilmiş ağırlıklar (toplamı 1).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 0 ile 1 arasında olmalıdır.")

    sw = np.array(subjective_weights, dtype=float)
    ow = np.array(objective_weights, dtype=float)
    if sw.shape != ow.shape:
        raise ValueError("subjective_weights ve objective_weights aynı uzunlukta olmalı.")

    combined = alpha * sw + (1 - alpha) * ow
    combined = combined / combined.sum()
    return combined
