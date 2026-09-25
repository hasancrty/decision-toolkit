"""
Örnek: Yatırım Alternatifi Seçimi

Getiri, risk, likidite ve vade kriterlerine göre 3 yatırım aracı arasında
seçim yapan kısa bir TOPSIS örneği (ağırlıklar bu örnekte doğrudan verilmiştir).

Çalıştırmak için:
    python examples/investment_choice.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import topsis


def main():
    options = ["Hisse Senedi Fonu", "Tahvil", "Gayrimenkul Sertifikası"]

    # [Beklenen Getiri (%), Risk (1-10), Likidite (1-10), Vade (yıl)]
    decision_matrix = [
        [12, 8, 7, 1],
        [5,  2, 9, 3],
        [8,  5, 3, 5],
    ]

    weights = [0.40, 0.30, 0.20, 0.10]
    criteria_types = ["benefit", "cost", "benefit", "cost"]

    result = topsis.rank(decision_matrix, weights, criteria_types, options)

    print("Yatırım Alternatifi Sıralaması")
    print("-" * 35)
    print(result)
    print(f"\n✅ Önerilen: {result.ranked_labels()[0]}")


if __name__ == "__main__":
    main()
