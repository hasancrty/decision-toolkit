# decision-toolkit

Çok kriterli karar analizi (Multi-Criteria Decision Analysis / MCDA) için
bağımsız, test edilmiş bir Python kütüphanesi ve komut satırı aracı.

Karar vericilerin sübjektif önceliklerinden (**AHP**) veya doğrudan veriden
(**Entropy**) kriter ağırlıkları çıkarır; alternatifleri üç farklı yöntemle
(**TOPSIS**, **WSM**, **WPM**) sıralar; ve sonucun ağırlıklardaki belirsizliğe
karşı ne kadar **sağlam** olduğunu **duyarlılık analizi** ile ölçer.

Herhangi bir "birden fazla kritere göre en iyi seçeneği bul" problemine
uygulanabilir: tedarikçi seçimi, yatırım kararı, lokasyon seçimi, ürün
karşılaştırması, işe alım, proje önceliklendirme ve daha fazlası.

## Neden bu kütüphane?

Gerçek dünyadaki kararlar nadiren tek bir sayıya indirgenebilir — genellikle
birbiriyle çelişen birden fazla kriteri (maliyet, kalite, risk, süre...) aynı
anda değerlendirmek gerekir. Çoğu basit "puanlama tablosu" yaklaşımının iki
büyük eksiği vardır: **ağırlıklar nereden geliyor?** ve **sonuç ne kadar
güvenilir?** `decision-toolkit` tam olarak bu iki soruyu cevaplamak için
tasarlandı:

- **Ağırlıklar nereden geliyor?** → AHP (insan yargısı) ve/veya Entropy
  (veriden objektif çıkarım), ikisini harmanlama desteğiyle.
- **Sonuç ne kadar güvenilir?** → Üç bağımsız sıralama yöntemi (TOPSIS/WSM/WPM)
  birbirini doğrular; duyarlılık analizi hangi kriterin kararı en çok
  etkilediğini gösterir.

Matematiksel arkaplan için bkz. [METHODOLOGY.md](METHODOLOGY.md).

## Özellikler

- ✅ **AHP** — ikili karşılaştırmalardan ağırlık + otomatik tutarlılık kontrolü (CR)
- ✅ **Entropy** — karar matrisinden objektif ağırlık çıkarımı
- ✅ **Ağırlık harmanlama** — subjektif ve objektif ağırlıkları birleştirme
- ✅ **TOPSIS** — ideal çözüme yakınlık tabanlı sıralama
- ✅ **WSM / WPM** — sonucu doğrulamak için basit karşılaştırma yöntemleri
- ✅ **Duyarlılık analizi** — ağırlık taraması, kritik geçiş noktaları, sağlamlık özeti
- ✅ **CLI aracı** — JSON dosyasından çalıştır, terminalde veya grafik olarak sonuç al
- ✅ **Grafik üretimi** — sıralama ve duyarlılık eğrileri (matplotlib, PNG)
- ✅ **21 birim/entegrasyon testi** (pytest)

## Kurulum

```bash
git clone https://github.com/<kullanici-adi>/decision-toolkit.git
cd decision-toolkit
pip install -r requirements.txt
```

## Hızlı Başlangıç (Python API)

```python
from core import ahp, entropy, topsis, wsm, sensitivity

# 1. Subjektif ağırlıkları AHP ile hesapla
comparison_matrix = [
    [1,   3,   5],
    [1/3, 1,   3],
    [1/5, 1/3, 1],
]
ahp_result = ahp.compute_weights(comparison_matrix, labels=["Maliyet", "Kalite", "Hız"])
print(ahp_result)  # CR < 0.10 ise tutarlı

# 2. Karar matrisi
decision_matrix = [
    [250, 7, 12],   # Alternatif A
    [400, 9, 7],    # Alternatif B
    [300, 8, 9],    # Alternatif C
]
criteria_types = ["cost", "benefit", "cost"]

# 3. (Opsiyonel) Objektif ağırlıkları Entropy ile hesapla ve harmanla
entropy_result = entropy.compute_weights(decision_matrix, criteria_types)
final_weights = entropy.combine_weights(ahp_result.weights, entropy_result.weights, alpha=0.6)

# 4. TOPSIS ile sırala
ranking = topsis.rank(decision_matrix, final_weights, criteria_types, ["A", "B", "C"])
print(ranking)
print("En iyi seçenek:", ranking.ranked_labels()[0])

# 5. WSM/WPM ile doğrula
comparison = wsm.compare_methods(decision_matrix, final_weights, criteria_types, ["A", "B", "C"])
print(comparison)  # {"WSM": [...], "WPM": [...], "TOPSIS": [...]}

# 6. Duyarlılık analizi: sonuç ağırlıklara karşı sağlam mı?
robustness = sensitivity.weight_perturbation_summary(
    decision_matrix, final_weights, criteria_types,
    alternative_labels=["A", "B", "C"], criteria_labels=["Maliyet", "Kalite", "Hız"],
)
print(robustness)  # {"Maliyet": True, "Kalite": False, ...}
```

## Komut Satırı (CLI) Kullanımı

Karar problemini bir JSON dosyasına yazıp doğrudan terminalden çalıştırabilirsiniz.
Format için `examples/sample_problem.json` dosyasına bakın.

```bash
# Basit sıralama (TOPSIS varsayılan)
python cli.py examples/sample_problem.json

# Farklı yöntem seç
python cli.py examples/sample_problem.json --method wsm

# Sonucu grafik olarak kaydet
python cli.py examples/sample_problem.json --chart ranking.png

# Üç yöntemi karşılaştır (sonuca güven kontrolü)
python cli.py examples/sample_problem.json --compare

# Duyarlılık analizi özeti
python cli.py examples/sample_problem.json --sensitivity
```

JSON dosyasında `weights` doğrudan verilebilir, ya da `comparison_matrix`
verilerek CLI'nin AHP ile ağırlıkları otomatik hesaplaması sağlanabilir.

## Örnekler

`examples/` klasöründe farklı domainlerden çalışan senaryolar bulunur:

| Dosya | Senaryo | Kapsam |
|---|---|---|
| `supplier_selection.py` | Tedarikçi seçimi | **Tam kapsamlı**: AHP + Entropy + harmanlama + TOPSIS/WSM/WPM karşılaştırması + duyarlılık analizi + grafik üretimi |
| `investment_choice.py` | Yatırım aracı seçimi | TOPSIS ile hızlı örnek |
| `location_selection.py` | Ofis lokasyonu seçimi | TOPSIS ile hızlı örnek |
| `sample_problem.json` | CLI için hazır problem dosyası | — |

Çalıştırmak için:

```bash
python examples/supplier_selection.py
```

Bu, `examples/output/` klasörüne iki grafik üretir:
- `supplier_ranking.png` — TOPSIS skorlarına göre çubuk grafik
- `supplier_sensitivity.png` — en kritik kriterin ağırlığına göre skor eğrileri

### Örnek çıktı (özet)

```
ADIM 3: TOPSIS / WSM / WPM Karşılaştırması
TOPSISResult(
  1. Tedarikçi C  (skor=0.7014)
  2. Tedarikçi B  (skor=0.6210)
  3. Tedarikçi A  (skor=0.5287)
  4. Tedarikçi D  (skor=0.3274)
)

Üç yöntemin sıralamaları:
  WSM     : Tedarikçi B > Tedarikçi C > Tedarikçi A > Tedarikçi D
  WPM     : Tedarikçi B > Tedarikçi C > Tedarikçi A > Tedarikçi D
  TOPSIS  : Tedarikçi C > Tedarikçi B > Tedarikçi A > Tedarikçi D

⚠️  Yöntemler farklı kazananlar buluyor: {'Tedarikçi B', 'Tedarikçi C'}

ADIM 4: Duyarlılık Analizi
  Fiyat                HASSAS ⚠️
  Kalite               sağlam ✅
  Teslimat Süresi      sağlam ✅
  Finansal İstikrar    sağlam ✅
```

Bu örnek, kütüphanenin sadece "bir sayı üretmediğini", kararın ne kadar
güvenilir olduğu hakkında da bilgi verdiğini gösterir — Fiyat kriterinin
ağırlığı biraz değişse kazanan değişebiliyor, bu da karar vericiye "bu
kriter üzerinde daha dikkatli düşün" sinyali veriyor.

## Test Çalıştırma

```bash
pip install -r requirements.txt
pytest tests/ -v
```

21 test, tüm çekirdek modülleri (AHP, Entropy, TOPSIS, WSM/WPM, Duyarlılık
Analizi) ve CLI'yi (subprocess entegrasyon testleri) kapsar.

## Proje Yapısı

```
decision-toolkit/
├── core/
│   ├── ahp.py           # AHP: subjektif ağırlıklandırma
│   ├── entropy.py        # Entropy: objektif ağırlıklandırma + harmanlama
│   ├── topsis.py         # TOPSIS: ideal-çözüm tabanlı sıralama
│   ├── wsm.py             # WSM/WPM: basit sıralama + yöntem karşılaştırma
│   └── sensitivity.py    # Duyarlılık analizi
├── examples/
│   ├── supplier_selection.py   # Tam kapsamlı demo
│   ├── investment_choice.py
│   ├── location_selection.py
│   └── sample_problem.json     # CLI için örnek girdi
├── tests/
│   ├── test_core.py      # 17 birim testi
│   └── test_cli.py       # 4 CLI entegrasyon testi
├── cli.py                 # Komut satırı aracı
├── METHODOLOGY.md         # Matematiksel arkaplan ve yöntem seçim rehberi
├── requirements.txt
└── README.md
```

## Yöntemler Hakkında (Özet)

Detaylı matematiksel açıklamalar için [METHODOLOGY.md](METHODOLOGY.md) dosyasına bakın.

- **AHP**: karar vericinin ikili karşılaştırmalarından ağırlık çıkarır,
  tutarlılığı otomatik doğrular (CR < 0.10).
- **Entropy**: ağırlıkları doğrudan veri dağılımından hesaplar, sübjektif
  girdi gerektirmez.
- **TOPSIS**: alternatifleri "ideale yakınlık / kötüden uzaklık" oranına göre
  sıralar.
- **WSM/WPM**: basit ağırlıklı toplam/çarpım — TOPSIS sonucunu doğrulamak için.
- **Duyarlılık Analizi**: ağırlıklardaki belirsizliğin sonucu ne kadar
  etkilediğini ölçer.

## Referanslar

- Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.
- Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making*. Springer-Verlag.
- Zeleny, M. (1982). *Multiple Criteria Decision Making*. McGraw-Hill.
- Triantaphyllou, E. (2000). *Multi-Criteria Decision Making Methods: A Comparative Study*. Springer.

## Lisans

MIT
