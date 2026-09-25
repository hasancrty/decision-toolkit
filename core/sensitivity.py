"""
Duyarlılık Analizi (Sensitivity Analysis).

"En iyi alternatif, kriter ağırlıklarındaki küçük değişikliklere karşı ne kadar
dayanıklı (robust)?" sorusuna cevap arar. Gerçek kararlarda ağırlıklar genellikle
kesin değildir (örn. "Kalite %40 mı %35 mi önemli, tam emin değiliz") — bu
modül, bir kriterin ağırlığını kademeli olarak değiştirip sonucun ne zaman
değiştiğini gösterir.

İki analiz türü sunar:
1. `vary_single_weight`: Tek bir kriterin ağırlığını 0'dan 1'e tarar, diğerlerini
   orantılı olarak yeniden ölçekler, her adımda TOPSIS sıralamasını hesaplar.
2. `weight_perturbation_summary`: Tüm kriterleri sırayla küçük miktarlarda
   (+/- delta) değiştirip en iyi alternatifin kaç kez değiştiğini sayar.
"""

from __future__ import annotations
import numpy as np
from . import topsis as topsis_module


class SensitivityResult:
    """Tek kriter için duyarlılık taraması sonucu."""

    def __init__(self, criterion_label: str, weight_values: np.ndarray,
                 top_alternative_per_step: list[str], scores_per_step: np.ndarray,
                 alternative_labels: list[str]):
        self.criterion_label = criterion_label
        self.weight_values = weight_values
        self.top_alternative_per_step = top_alternative_per_step
        self.scores_per_step = scores_per_step  # shape: (steps, n_alternatives)
        self.alternative_labels = alternative_labels

    @property
    def is_stable(self) -> bool:
        """Taranan aralıkta en iyi alternatif hiç değişmediyse True."""
        return len(set(self.top_alternative_per_step)) == 1

    def change_points(self) -> list[tuple[float, str, str]]:
        """En iyi alternatifin değiştiği ağırlık noktalarını döner:
        [(ağırlık_değeri, önceki_kazanan, yeni_kazanan), ...]"""
        changes = []
        for i in range(1, len(self.top_alternative_per_step)):
            prev = self.top_alternative_per_step[i - 1]
            curr = self.top_alternative_per_step[i]
            if prev != curr:
                changes.append((float(self.weight_values[i]), prev, curr))
        return changes

    def __repr__(self) -> str:
        status = "SAĞLAM ✅ (kazanan hiç değişmiyor)" if self.is_stable else "HASSAS ⚠️ (kazanan değişiyor)"
        lines = [f"Kriter: {self.criterion_label} — {status}"]
        for point, prev, curr in self.change_points():
            lines.append(f"  ağırlık={point:.2f} noktasında kazanan '{prev}' -> '{curr}' değişti")
        return "\n".join(lines)


def vary_single_weight(
    decision_matrix: list[list[float]],
    base_weights: list[float],
    criteria_types: list[str],
    criterion_index: int,
    alternative_labels: list[str] | None = None,
    criteria_labels: list[str] | None = None,
    steps: int = 21,
) -> SensitivityResult:
    """
    Bir kriterin ağırlığını 0'dan 1'e tarayıp diğer kriterlerin ağırlıklarını
    orantılı olarak yeniden ölçekleyerek TOPSIS sonucunun nasıl değiştiğini izler.

    Args:
        decision_matrix: karar matrisi (m alternatif x n kriter).
        base_weights: başlangıç ağırlıkları (n uzunluğunda, toplamı 1 olmalı).
        criteria_types: her kriter için "benefit"/"cost".
        criterion_index: hangi kriterin ağırlığının tarama yapılacağı (0-indeksli).
        alternative_labels: alternatif isimleri (opsiyonel).
        criteria_labels: kriter isimleri (opsiyonel, sadece raporlama için).
        steps: taranacak nokta sayısı (varsayılan 21 -> 0.00, 0.05, ..., 1.00).

    Returns:
        SensitivityResult: her ağırlık değerinde kazanan alternatif ve skorlar.
    """
    n = len(base_weights)
    if not 0 <= criterion_index < n:
        raise ValueError("criterion_index geçersiz.")

    label = (criteria_labels[criterion_index] if criteria_labels else f"C{criterion_index+1}")
    other_indices = [i for i in range(n) if i != criterion_index]
    other_base_sum = sum(base_weights[i] for i in other_indices)

    weight_values = np.linspace(0.0, 1.0, steps)
    top_alt_per_step = []
    all_scores = []

    m = len(decision_matrix)
    if alternative_labels is None:
        alternative_labels = [f"A{i+1}" for i in range(m)]

    for target_w in weight_values:
        new_weights = list(base_weights)
        remaining = 1.0 - target_w
        if other_base_sum > 0:
            for i in other_indices:
                new_weights[i] = base_weights[i] / other_base_sum * remaining
        else:
            for i in other_indices:
                new_weights[i] = remaining / len(other_indices) if other_indices else 0
        new_weights[criterion_index] = target_w

        result = topsis_module.rank(decision_matrix, new_weights, criteria_types, alternative_labels)
        top_alt_per_step.append(result.ranked_labels()[0])
        # skorları orijinal alternatif sırasına göre topla
        score_row = [result.as_dict()[lab] for lab in alternative_labels]
        all_scores.append(score_row)

    return SensitivityResult(
        criterion_label=label,
        weight_values=weight_values,
        top_alternative_per_step=top_alt_per_step,
        scores_per_step=np.array(all_scores),
        alternative_labels=alternative_labels,
    )


def weight_perturbation_summary(
    decision_matrix: list[list[float]],
    base_weights: list[float],
    criteria_types: list[str],
    alternative_labels: list[str] | None = None,
    criteria_labels: list[str] | None = None,
    delta: float = 0.10,
) -> dict[str, bool]:
    """
    Her kriterin ağırlığını +/- delta kadar oynatıp (diğerlerini orantılı
    yeniden ölçekleyerek) en iyi alternatifin değişip değişmediğini kontrol eder.
    Hızlı bir "hangi kriter kararı en çok etkiliyor" özeti sağlar.

    Returns:
        dict: {kriter_adı: is_robust (bool)} — True ise o kriterdeki +/-delta
            oynama en iyi alternatifi DEĞİŞTİRMİYOR (sonuç o kritere karşı sağlam).
    """
    n = len(base_weights)
    if criteria_labels is None:
        criteria_labels = [f"C{i+1}" for i in range(n)]

    base_result = topsis_module.rank(decision_matrix, base_weights, criteria_types, alternative_labels)
    base_winner = base_result.ranked_labels()[0]

    summary = {}
    for idx, label in enumerate(criteria_labels):
        robust = True
        for direction in (+1, -1):
            target = np.clip(base_weights[idx] + direction * delta, 0.0, 1.0)
            other_indices = [i for i in range(n) if i != idx]
            other_sum = sum(base_weights[i] for i in other_indices)
            new_weights = list(base_weights)
            remaining = 1.0 - target
            if other_sum > 0:
                for i in other_indices:
                    new_weights[i] = base_weights[i] / other_sum * remaining
            new_weights[idx] = target

            result = topsis_module.rank(decision_matrix, new_weights, criteria_types, alternative_labels)
            if result.ranked_labels()[0] != base_winner:
                robust = False
                break
        summary[label] = robust

    return summary
