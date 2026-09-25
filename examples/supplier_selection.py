"""
Örnek: Tedarikçi Seçimi (Genişletilmiş Analiz)

Bir üretim şirketinin 4 aday tedarikçi arasından seçim yapması senaryosu.
Bu örnek, kütüphanenin tüm bileşenlerini uçtan uca gösterir:

  Adım 1 (AHP)          — Karar vericinin ikili karşılaştırmalarından
                           SUBJEKTİF kriter ağırlıkları hesaplanır.
  Adım 2 (Entropy)       — Aynı ağırlıklar, karar matrisindeki veri
                           dağılımından OBJEKTİF olarak da hesaplanır.
                           İkisi karşılaştırılıp harmanlanır.
  Adım 3 (TOPSIS/WSM/WPM)— Üç farklı sıralama yöntemi aynı ağırlıklarla
                           çalıştırılıp sonuçlar karşılaştırılır (yöntemler
                           hemfikirse sonuca güven artar).
  Adım 4 (Duyarlılık)    — En kritik kriterin ağırlığı 0'dan 1'e taranarak
                           kazananın ne zaman değiştiği bulunur.
  Adım 5 (Grafik)        — Sonuçlar bir çubuk grafik ve duyarlılık eğrisi
                           olarak examples/output/ klasörüne kaydedilir.

Çalıştırmak için:
    python examples/supplier_selection.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import ahp, entropy, topsis, wsm, sensitivity

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    criteria = ["Fiyat", "Kalite", "Teslimat Süresi", "Finansal İstikrar"]
    suppliers = ["Tedarikçi A", "Tedarikçi B", "Tedarikçi C", "Tedarikçi D"]

    # [Fiyat ($), Kalite (1-10), Teslimat Süresi (gün), Finansal İstikrar (1-10)]
    decision_matrix = [
        [250, 7, 12, 8],   # Tedarikçi A: ucuz ama orta kalite
        [400, 9, 7,  9],   # Tedarikçi B: pahalı ama yüksek kalite/hız
        [300, 8, 9,  6],   # Tedarikçi C: dengeli
        [280, 6, 15, 7],   # Tedarikçi D: ucuz ama yavaş teslimat
    ]
    criteria_types = ["cost", "benefit", "cost", "benefit"]

    # -----------------------------------------------------------------
    # Adım 1: Subjektif ağırlıklar (AHP)
    # -----------------------------------------------------------------
    section("ADIM 1: AHP ile Subjektif Kriter Ağırlıkları")

    comparison_matrix = [
        [1,     1/3,  2,   3],
        [3,     1,    4,   5],
        [1/2,   1/4,  1,   2],
        [1/3,   1/5,  1/2, 1],
    ]
    ahp_result = ahp.compute_weights(comparison_matrix, labels=criteria)
    print(ahp_result)
    if not ahp_result.is_consistent:
        print("⚠️  Uyarı: karşılaştırma matrisi tutarsız (CR >= 0.10). Gerçek bir")
        print("    uygulamada ikili karşılaştırmalar gözden geçirilmelidir.")

    # -----------------------------------------------------------------
    # Adım 2: Objektif ağırlıklar (Entropy) ve harmanlama
    # -----------------------------------------------------------------
    section("ADIM 2: Entropy ile Objektif Kriter Ağırlıkları")

    entropy_result = entropy.compute_weights(decision_matrix, criteria_types, labels=criteria)
    print(entropy_result)

    print("\nAHP (subjektif) ve Entropy (objektif) ağırlıkları karşılaştırması:")
    print(f"  {'Kriter':<20}{'AHP':>10}{'Entropy':>12}")
    for label, sw, ow in zip(criteria, ahp_result.weights, entropy_result.weights):
        print(f"  {label:<20}{sw:>10.4f}{ow:>12.4f}")

    combined_weights = entropy.combine_weights(
        ahp_result.weights, entropy_result.weights, alpha=0.6
    )
    print("\nHarmanlanmış ağırlıklar (alpha=0.6, yani %60 AHP + %40 Entropy):")
    for label, w in zip(criteria, combined_weights):
        print(f"  {label:<20}{w:.4f}")

    # -----------------------------------------------------------------
    # Adım 3: Üç yöntemle sıralama ve karşılaştırma
    # -----------------------------------------------------------------
    section("ADIM 3: TOPSIS / WSM / WPM Karşılaştırması")

    topsis_result = topsis.rank(decision_matrix, combined_weights, criteria_types, suppliers)
    print(topsis_result)

    comparison = wsm.compare_methods(decision_matrix, combined_weights, criteria_types, suppliers)
    print("\nÜç yöntemin sıralamaları:")
    for method, ranking in comparison.items():
        print(f"  {method:8s}: {' > '.join(ranking)}")

    winners = {ranking[0] for ranking in comparison.values()}
    if len(winners) == 1:
        print(f"\n✅ Üç yöntem de aynı kazananda hemfikir: {winners.pop()} — sonuca güven yüksek.")
    else:
        print(f"\n⚠️  Yöntemler farklı kazananlar buluyor: {winners} — sonuç ağırlıklara duyarlı olabilir.")

    # -----------------------------------------------------------------
    # Adım 4: Duyarlılık analizi
    # -----------------------------------------------------------------
    section("ADIM 4: Duyarlılık Analizi")

    robustness = sensitivity.weight_perturbation_summary(
        decision_matrix, combined_weights, criteria_types, suppliers, criteria, delta=0.10
    )
    print("Her kriterde +/- 0.10 ağırlık oynamasına karşı kazanan sağlam mı?")
    for crit, is_robust in robustness.items():
        print(f"  {crit:<20} {'sağlam ✅' if is_robust else 'HASSAS ⚠️'}")

    # En hassas kriteri tam tarama ile incele
    most_sensitive_idx = criteria.index(
        min(robustness, key=lambda c: robustness[c])  # False (hassas) önce gelir
    ) if not all(robustness.values()) else 0

    sens_result = sensitivity.vary_single_weight(
        decision_matrix, combined_weights, criteria_types,
        criterion_index=most_sensitive_idx,
        alternative_labels=suppliers, criteria_labels=criteria, steps=21,
    )
    print(f"\nEn hassas kriter için tam tarama ({sens_result.criterion_label}):")
    print(sens_result)

    # -----------------------------------------------------------------
    # Adım 5: Grafikler
    # -----------------------------------------------------------------
    section("ADIM 5: Grafik Üretimi")

    try:
        _plot_ranking(topsis_result, os.path.join(OUTPUT_DIR, "supplier_ranking.png"))
        _plot_sensitivity(sens_result, os.path.join(OUTPUT_DIR, "supplier_sensitivity.png"))
        print(f"Grafikler kaydedildi: {OUTPUT_DIR}/")
    except ImportError:
        print("matplotlib kurulu değil, grafikler atlandı. Kurmak için: pip install matplotlib")

    best = topsis_result.ranked_labels()[0]
    print(f"\n✅ Önerilen tedarikçi: {best}")


def _plot_ranking(result, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = result.ranked_labels()
    scores = [result.as_dict()[label] for label in labels]

    plt.figure(figsize=(7, 4))
    plt.barh(labels[::-1], scores[::-1], color="#4C72B0")
    plt.xlabel("TOPSIS Skoru")
    plt.title("Tedarikçi Sıralaması")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_sensitivity(sens_result, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for i, label in enumerate(sens_result.alternative_labels):
        plt.plot(sens_result.weight_values, sens_result.scores_per_step[:, i], label=label, marker="o", markersize=3)

    plt.xlabel(f"'{sens_result.criterion_label}' Ağırlığı")
    plt.ylabel("TOPSIS Skoru")
    plt.title(f"Duyarlılık Analizi: {sens_result.criterion_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
