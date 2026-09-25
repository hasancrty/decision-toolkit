# Metodoloji

Bu doküman, `decision-toolkit` içindeki her yöntemin matematiksel temelini,
ne zaman kullanılması gerektiğini ve birbirleriyle nasıl ilişkili olduklarını
açıklar.

## İçindekiler

1. [Genel Akış](#genel-akış)
2. [AHP — Subjektif Ağırlıklandırma](#ahp--subjektif-ağırlıklandırma)
3. [Entropy — Objektif Ağırlıklandırma](#entropy--objektif-ağırlıklandırma)
4. [Ağırlıkları Harmanlama](#ağırlıkları-harmanlama)
5. [TOPSIS](#topsis)
6. [WSM ve WPM](#wsm-ve-wpm)
7. [Duyarlılık Analizi](#duyarlılık-analizi)
8. [Hangi Yöntemi Ne Zaman Kullanmalı?](#hangi-yöntemi-ne-zaman-kullanmalı)

---

## Genel Akış

Bir MCDA (Çok Kriterli Karar Analizi) problemi tipik olarak şu adımlardan geçer:

```
[1] Kriterleri belirle  →  [2] Kriter ağırlıklarını hesapla  →  [3] Alternatifleri
    (Fiyat, Kalite...)       (AHP ve/veya Entropy)                puanla ve sırala
                                                                   (TOPSIS/WSM/WPM)
                                                                        ↓
                                                              [4] Duyarlılık analizi
                                                                  (sonuç ne kadar sağlam?)
```

`decision-toolkit`'in her modülü bu akışın bir adımına karşılık gelir.

---

## AHP — Subjektif Ağırlıklandırma

**Dosya:** `core/ahp.py`

AHP, karar vericinin kriterleri ikili olarak karşılaştırmasını ister:
"Kalite, Fiyat'a göre ne kadar önemli?" gibi sorulara Saaty'nin 1-9 skalasında
cevap verilir:

| Değer | Anlamı |
|---|---|
| 1 | Eşit önemde |
| 3 | Biraz daha önemli |
| 5 | Kuvvetle daha önemli |
| 7 | Çok kuvvetle daha önemli |
| 9 | Aşırı derecede daha önemli |
| 2,4,6,8 | Ara değerler |

Bu cevaplar bir **n×n ikili karşılaştırma matrisi** oluşturur (n = kriter sayısı).
Matris karşıt-simetriktir: `matrix[j][i] = 1 / matrix[i][j]`.

### Ağırlık Hesabı (Eigenvalue Yöntemi)

1. Her sütunu kendi toplamına böl (normalize et).
2. Her satırın ortalamasını al → bu, o kriterin ağırlığıdır.

### Tutarlılık Kontrolü

İnsan yargıları her zaman mantıksal olarak tutarlı olmayabilir (örn. A, B'den
önemli; B, C'den önemli; ama C, A'dan önemli denirse çelişki oluşur). AHP bunu
**Tutarlılık Oranı (Consistency Ratio, CR)** ile ölçer:

```
CI = (λ_max - n) / (n - 1)       [Consistency Index]
CR = CI / RI                      [RI: Saaty'nin rastgele indeks tablosu]
```

**CR < 0.10** ise karşılaştırmalar kabul edilebilir tutarlılıktadır. Değilse,
karar vericinin ikili karşılaştırmaları gözden geçirmesi önerilir.

---

## Entropy — Objektif Ağırlıklandırma

**Dosya:** `core/entropy.py`

AHP tamamen insan yargısına dayanırken, Entropy yöntemi ağırlıkları
**doğrudan veriden** çıkarır — hiçbir sübjektif girdi gerekmez.

**Temel fikir:** Bir kriterde tüm alternatifler birbirine çok benziyorsa, o
kriter alternatifleri ayırt etmede işe yaramaz → düşük ağırlık. Değerler
arasında büyük farklar varsa, o kriter ayırt edici bilgi taşır → yüksek ağırlık.

### Formül

```
p_ij = x_ij / Σx_ij                              [normalizasyon]
e_j = -k · Σ(p_ij · ln(p_ij)),  k = 1/ln(m)        [entropi, m=alternatif sayısı]
d_j = 1 - e_j                                     [çeşitlilik derecesi]
w_j = d_j / Σd_j                                  [nihai ağırlık]
```

**Ne zaman kullanılır:** Kriterler hakkında güvenilir sayısal veri var ama
karar vericilerin sübjektif önceliklerini toplamak zor/pahalıysa (örn. büyük
ölçekli, tekrarlanan kararlar).

---

## Ağırlıkları Harmanlama

**Fonksiyon:** `entropy.combine_weights(subjective, objective, alpha)`

AHP (insan bilgisi) ve Entropy (veri) genellikle farklı ağırlıklar üretir —
bu normaldir, çünkü ikisi farklı bilgi kaynaklarını yakalar. Çoğu pratik
uygulamada ikisinin **ağırlıklı ortalaması** alınır:

```
w_combined = α · w_AHP + (1-α) · w_Entropy
```

`α` (alpha) parametresi, insan yargısına ne kadar güvenildiğini kontrol eder.
`α=0.5` her ikisine eşit ağırlık verir; `α=1.0` yalnızca AHP'yi kullanır.

---

## TOPSIS

**Dosya:** `core/topsis.py`

Her alternatifi, **ideal çözüme** (her kriterde en iyi değer) olan yakınlığı
ve **negatif ideal çözümden** (her kriterde en kötü değer) olan uzaklığı
üzerinden değerlendirir.

### Adımlar

1. **Vektör normalizasyonu:** `r_ij = x_ij / √(Σx_ij²)`
2. **Ağırlıklandırma:** `v_ij = w_j · r_ij`
3. **İdeal/negatif-ideal çözümler:** benefit kriterlerde max/min, cost
   kriterlerde min/max alınır.
4. **Öklid mesafeleri:** `D+_i` (ideale uzaklık), `D-_i` (negatif ideale uzaklık)
5. **Yakınlık katsayısı:** `C_i = D-_i / (D+_i + D-_i)`, 0-1 arası, yüksek=iyi.

**Avantajı:** Hem "en iyiye yakın olma" hem "en kötüden uzak olma" durumunu
aynı anda dikkate alır, bu da onu WSM'den daha dengeli yapar.

---

## WSM ve WPM

**Dosya:** `core/wsm.py`

En basit iki MCDA yöntemi. TOPSIS sonucunu **doğrulamak** için kullanılırlar:
eğer üç yöntem de aynı kazananı buluyorsa, sonuca duyulan güven artar.

- **WSM (Weighted Sum Model):** `S_i = Σ(w_j · r_ij)` — ağırlıklı toplam.
  Kriterler arasında **tam ikame edilebilirlik** varsayar (bir kriterdeki
  düşüklük başka kriterdeki yükseklikle tam telafi edilebilir).
- **WPM (Weighted Product Model):** `S_i = Π(r_ij ^ w_j)` — ağırlıklı çarpım.
  Birimlerden bağımsızdır (dimensionless) ve büyük farkları WSM'den daha
  fazla cezalandırır/ödüllendirir (çarpımsal etki).

`wsm.compare_methods()` üç yöntemi birden çalıştırıp sonuçları yan yana verir.

---

## Duyarlılık Analizi

**Dosya:** `core/sensitivity.py`

Kriter ağırlıkları nadiren %100 kesindir. Duyarlılık analizi şu soruyu
cevaplar: *"Ağırlıklar biraz değişse, kazanan alternatif değişir mi?"*

### `vary_single_weight`
Bir kriterin ağırlığını 0'dan 1'e tarar (diğerlerini orantılı olarak yeniden
ölçekler) ve her adımda TOPSIS'i yeniden çalıştırır. Kazananın **hangi ağırlık
değerinde değiştiğini** (`change_points()`) bulur. Sonuç, ağırlığa karşı skor
eğrileri şeklinde çizilebilir (bkz. `examples/supplier_selection.py`).

### `weight_perturbation_summary`
Her kriteri sırayla ±delta (varsayılan 0.10) kadar oynatıp kazananın değişip
değişmediğine bakar — hızlı bir "hangi kriter en kritik?" özeti verir.

**Neden önemli:** Eğer kazanan, bir kriterin ağırlığındaki küçük bir
değişiklikle kolayca değişiyorsa, o karar "hassas"tır ve karar vericinin o
kriterin ağırlığı üzerinde daha fazla düşünmesi gerekir.

---

## Hangi Yöntemi Ne Zaman Kullanmalı?

| Durum | Önerilen Yaklaşım |
|---|---|
| Karar vericinin net öncelikleri var, veri az | AHP ile ağırlık → TOPSIS |
| Bol miktarda geçmiş veri var, sübjektif girdi zor | Entropy ile ağırlık → TOPSIS |
| Her ikisi de mevcut | AHP + Entropy → `combine_weights` → TOPSIS |
| Sonucun ne kadar güvenilir olduğunu bilmek istiyorum | `wsm.compare_methods` + `sensitivity` |
| Kararı yöneticilere/paydaşlara sunacağım | TOPSIS sonucu + duyarlılık grafiği (en ikna edici kombinasyon) |
