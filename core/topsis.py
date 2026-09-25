"""
TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).

Alternatifleri, "ideal çözüme" olan yakınlıklarına ve "negatif ideal çözümden"
olan uzaklıklarına göre sıralar. Kriter ağırlıkları genellikle AHP gibi bir
yöntemle önceden hesaplanır ve bu modüle girdi olarak verilir.

Referans:
    Hwang, C. L., & Yoon, K. (1981). Multiple Attribute Decision Making.
    Springer-Verlag.
"""

from __future__ import annotations
import numpy as np


class TOPSISResult:
    """TOPSIS hesaplama sonucunu tutan veri sınıfı."""

    def __init__(self, scores: np.ndarray, ranking: list[int], alternative_labels: list[str]):
        self.scores = scores
        self.ranking = ranking  # en iyiden en kötüye alternatif indeksleri
        self.alternative_labels = alternative_labels

    def ranked_labels(self) -> list[str]:
        """Alternatifleri en iyiden en kötüye sıralanmış isim listesi olarak döner."""
        return [self.alternative_labels[i] for i in self.ranking]

    def as_dict(self) -> dict[str, float]:
        return {label: float(s) for label, s in zip(self.alternative_labels, self.scores)}

    def __repr__(self) -> str:
        lines = []
        for rank, idx in enumerate(self.ranking, start=1):
            lines.append(f"  {rank}. {self.alternative_labels[idx]}  (skor={self.scores[idx]:.4f})")
        return "TOPSISResult(\n" + "\n".join(lines) + "\n)"


def rank(
    decision_matrix: list[list[float]],
    weights: list[float],
    criteria_types: list[str],
    alternative_labels: list[str] | None = None,
) -> TOPSISResult:
    """
    Alternatifleri TOPSIS yöntemiyle sıralar.

    Args:
        decision_matrix: m x n boyutunda matris (m=alternatif, n=kriter).
            Her satır bir alternatifin kriterlere göre ham puanlarını içerir.
        weights: n uzunluğunda kriter ağırlıkları (toplamı 1 olması önerilir,
            AHP'den elde edilebilir).
        criteria_types: n uzunluğunda liste, her kriter için "benefit" (yüksek
            iyi, örn. kalite) veya "cost" (düşük iyi, örn. fiyat) değerini alır.
        alternative_labels: Alternatif isimleri (opsiyonel).

    Returns:
        TOPSISResult: her alternatifin skoru ve en iyiden en kötüye sıralaması.
    """
    matrix = np.array(decision_matrix, dtype=float)
    w = np.array(weights, dtype=float)
    m, n = matrix.shape

    if len(weights) != n:
        raise ValueError("weights uzunluğu kriter sayısıyla eşleşmiyor.")
    if len(criteria_types) != n:
        raise ValueError("criteria_types uzunluğu kriter sayısıyla eşleşmiyor.")
    if alternative_labels is None:
        alternative_labels = [f"A{i+1}" for i in range(m)]
    if len(alternative_labels) != m:
        raise ValueError("alternative_labels uzunluğu alternatif sayısıyla eşleşmiyor.")

    # 1. Vektör normalizasyonu
    norm = np.sqrt((matrix ** 2).sum(axis=0))
    norm[norm == 0] = 1e-12  # sıfıra bölme koruması
    normalized = matrix / norm

    # 2. Ağırlıklandırma
    weighted = normalized * w

    # 3. İdeal ve negatif-ideal çözümler
    ideal_best = np.zeros(n)
    ideal_worst = np.zeros(n)
    for j, ctype in enumerate(criteria_types):
        if ctype not in ("benefit", "cost"):
            raise ValueError(f"criteria_types[{j}] 'benefit' veya 'cost' olmalı, alınan: {ctype}")
        if ctype == "benefit":
            ideal_best[j] = weighted[:, j].max()
            ideal_worst[j] = weighted[:, j].min()
        else:  # cost
            ideal_best[j] = weighted[:, j].min()
            ideal_worst[j] = weighted[:, j].max()

    # 4. Öklid mesafeleri
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    # 5. Yakınlık katsayısı (closeness coefficient)
    denom = dist_best + dist_worst
    denom[denom == 0] = 1e-12
    scores = dist_worst / denom

    ranking = list(np.argsort(-scores))  # yüksekten düşüğe

    return TOPSISResult(scores=scores, ranking=ranking, alternative_labels=alternative_labels)
