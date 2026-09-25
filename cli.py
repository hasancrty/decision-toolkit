"""
decision-toolkit CLI — JSON dosyasından karar problemi okuyup sonucu terminale
ve isteğe bağlı olarak bir PNG grafiğine yazan komut satırı aracı.

Kullanım:
    python cli.py problem.json
    python cli.py problem.json --method topsis --chart out.png
    python cli.py problem.json --compare          # WSM/WPM/TOPSIS karşılaştır
    python cli.py problem.json --sensitivity       # tüm kriterler için duyarlılık özeti

JSON formatı için examples/sample_problem.json dosyasına bakın.
"""

from __future__ import annotations
import argparse
import json
import sys

from core import ahp, topsis, wsm, sensitivity


def load_problem(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_weights(problem: dict) -> list[float]:
    """
    Problem tanımında ya doğrudan 'weights' verilmiş olabilir, ya da
    'comparison_matrix' (AHP girdisi) verilip ağırlıklar hesaplanabilir.
    """
    if "weights" in problem:
        return problem["weights"]
    if "comparison_matrix" in problem:
        result = ahp.compute_weights(
            problem["comparison_matrix"], labels=problem.get("criteria")
        )
        print("[AHP] Hesaplanan ağırlıklar:")
        print(result)
        if not result.is_consistent:
            print("⚠️  Uyarı: karşılaştırma matrisi tutarsız (CR >= 0.10).\n")
        return list(result.weights)
    raise ValueError("Problem dosyasında 'weights' veya 'comparison_matrix' bulunmalı.")


def cmd_rank(problem: dict, args: argparse.Namespace) -> None:
    weights = resolve_weights(problem)
    dm = problem["decision_matrix"]
    ctypes = problem["criteria_types"]
    alts = problem.get("alternatives")

    if args.method == "wsm":
        result = wsm.wsm_rank(dm, weights, ctypes, alts)
    elif args.method == "wpm":
        result = wsm.wpm_rank(dm, weights, ctypes, alts)
    else:
        result = topsis.rank(dm, weights, ctypes, alts)

    print(f"\n[{args.method.upper()}] Sıralama:")
    print(result)

    if args.chart:
        _save_chart(result, args.chart)
        print(f"\nGrafik kaydedildi: {args.chart}")


def cmd_compare(problem: dict) -> None:
    weights = resolve_weights(problem)
    dm = problem["decision_matrix"]
    ctypes = problem["criteria_types"]
    alts = problem.get("alternatives")

    comparison = wsm.compare_methods(dm, weights, ctypes, alts)
    print("\nYöntem Karşılaştırması (en iyiden en kötüye):")
    for method, ranking in comparison.items():
        print(f"  {method:8s}: {' > '.join(ranking)}")

    winners = {ranking[0] for ranking in comparison.values()}
    if len(winners) == 1:
        print(f"\n✅ Tüm yöntemler aynı kazananda hemfikir: {winners.pop()}")
    else:
        print(f"\n⚠️  Yöntemler farklı kazananlar buluyor: {winners} — sonucu dikkatli yorumlayın.")


def cmd_sensitivity(problem: dict) -> None:
    weights = resolve_weights(problem)
    dm = problem["decision_matrix"]
    ctypes = problem["criteria_types"]
    alts = problem.get("alternatives")
    crit_labels = problem.get("criteria")

    summary = sensitivity.weight_perturbation_summary(
        dm, weights, ctypes, alts, crit_labels, delta=0.10
    )
    print("\nDuyarlılık Özeti (+/- 0.10 ağırlık oynamasına karşı kazanan sağlam mı?):")
    for criterion, is_robust in summary.items():
        status = "sağlam ✅" if is_robust else "HASSAS ⚠️" 
        print(f"  {criterion}: {status}")


def _save_chart(result, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Grafik için matplotlib gerekli: pip install matplotlib", file=sys.stderr)
        return

    labels = result.ranked_labels()
    scores = [result.as_dict()[label] for label in labels]

    plt.figure(figsize=(7, 4))
    bars = plt.barh(labels[::-1], scores[::-1], color="#4C72B0")
    plt.xlabel("Skor")
    plt.title(f"{result.method if hasattr(result, 'method') else 'TOPSIS'} Sıralaması")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="decision-toolkit CLI")
    parser.add_argument("problem_file", help="Karar problemini içeren JSON dosyası")
    parser.add_argument("--method", choices=["topsis", "wsm", "wpm"], default="topsis",
                         help="Kullanılacak sıralama yöntemi (varsayılan: topsis)")
    parser.add_argument("--chart", metavar="PATH", help="Sıralama grafiğini PNG olarak kaydet")
    parser.add_argument("--compare", action="store_true",
                         help="WSM/WPM/TOPSIS sonuçlarını yan yana karşılaştır")
    parser.add_argument("--sensitivity", action="store_true",
                         help="Ağırlık duyarlılık analizi özeti göster")
    args = parser.parse_args()

    problem = load_problem(args.problem_file)

    if args.compare:
        cmd_compare(problem)
    elif args.sensitivity:
        cmd_sensitivity(problem)
    else:
        cmd_rank(problem, args)


if __name__ == "__main__":
    main()
