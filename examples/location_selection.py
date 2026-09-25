"""
Örnek: Ofis Lokasyonu Seçimi

Kira maliyeti, ulaşım erişimi, ofis alanı ve çalışan memnuniyet anketi
puanına göre 3 aday lokasyon arasında seçim yapan kısa bir TOPSIS örneği.

Çalıştırmak için:
    python examples/location_selection.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import topsis


def main():
    locations = ["Merkez Ofis Binası", "Teknopark Kampüsü", "Şehir Dışı Plaza"]

    # [Kira ($/ay), Ulaşım Erişimi (1-10), Alan (m²), Çalışan Memnuniyeti (1-10)]
    decision_matrix = [
        [8000, 9, 450, 7],
        [6000, 7, 600, 8],
        [3500, 4, 800, 6],
    ]

    weights = [0.35, 0.25, 0.20, 0.20]
    criteria_types = ["cost", "benefit", "benefit", "benefit"]

    result = topsis.rank(decision_matrix, weights, criteria_types, locations)

    print("Ofis Lokasyonu Sıralaması")
    print("-" * 35)
    print(result)
    print(f"\n✅ Önerilen: {result.ranked_labels()[0]}")


if __name__ == "__main__":
    main()
