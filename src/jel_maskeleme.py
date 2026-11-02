# -*- coding: utf-8 -*-
"""
jel_maskeleme.py -- jel karelerindeki EL YAZISI hasta tanitici bilgisini siler
==============================================================================
NEDEN
------
Arsiv jellerinin bir kisminda, seritlerin ustundeki bos kenarda hastanin bas
harfleri el yazisiyla duruyor (iki-uc harflik kisa notlar). Goruntuler
yayimlanacak veya paylasilacaksa bu serit once temizlenmeli.

Model bu kenara zaten sanstan az bakiyor (Bolum 3.9: yazili karelerde Grad-CAM
kutlesinin %3.9'u, yazisizlarda %1.8'i, duz dagilimda beklenen %7.8). Yani bu
script bir PAYLASIM araci; egitim orijinal karelerle yapildi ve maskeli set
modeli degistirmez. Maskeli seti egitimde kullanmak istersen bu ayri bir karar.

NASIL CALISIR
--------------
  1. Her karenin ust kenar seridine bakar (varsayilan: satir %0-16,
     sutun %30-100; sol %30 disarida cunku ELP uygulama noktasi orada ve
     koyudur).
  2. Seritte arka plandan koyu baglantili bilesenleri bulur.
  3. El yazisi mi, jel icerigi mi ayirir:
       - cok kucuk  -> gurultu, atlanir
       - cok buyuk  -> jel kenari/golge, atlanir
       - seridin alt kenarina degiyorsa -> asagi devam eden bant, DOKUNULMAZ
       - doluluk orani yuksek ve genis/alcak ise -> bant gorunumlu, DOKUNULMAZ
     Kalanlar el yazisi sayilir (ince, kopuk, dusuk doluluklu konturlar).
  4. Kalan bilesenleri birkac piksel genisletip yerel arka plan rengiyle doldurur.

VARSAYILAN: KURU KOSU. Hicbir goruntu yazilmaz; sadece rapor ve ornek sayfa
uretilir. Yazmak icin --uygula ver.

KULLANIM
---------
    python jel_maskeleme.py                          # kuru kosu + ornek sayfa
    python jel_maskeleme.py --ornek-sayfa 24         # daha fazla ornek goster
    python jel_maskeleme.py --uygula                 # maskeli kopyayi yaz
    python jel_maskeleme.py --uygula --yalnizca-yazili   # sadece degisenleri yaz

Orijinal klasor HICBIR durumda degistirilmez; cikti ayri bir klasore yazilir.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage


# --------------------------------------------------------------- tespit
def serit_kutusu(sek, oran, sol_oran):
    H, W = sek
    return 0, max(1, int(round(oran * H))), int(round(sol_oran * W)), W


def yazi_maskesi(gri, ust_oran, sol_oran, esik, min_alan_orani, maks_alan_orani,
                 analiz_oran, dolu_esik=0.70):
    """gri: float 0-1. Donus: (bool maske tam boyutta, bilesen listesi).

    Kural: el yazisi, seritlerin BASLADIGI cizginin (ust_oran) tamamen
    ustunde kalan koyu bilesendir. Bir bilesen bu cizginin altina tasiyorsa
    asagi devam eden gercek jel icerigi sayilir ve DOKUNULMAZ. Bu yuzden
    bilesenler daha genis bir pencerede (analiz_oran) aranir -- yoksa
    cizgide kesilen bir bant, kesik parcasi yazi sanilir.
    """
    r0, r1, c0, c1 = serit_kutusu(gri.shape, analiz_oran, sol_oran)
    pencere = gri[r0:r1, c0:c1]
    if pencere.size == 0:
        return np.zeros_like(gri, bool), []

    H = gri.shape[0]
    sinir = max(1, int(round(ust_oran * H)))        # yazinin kalmasi gereken cizgi
    arka = float(np.median(pencere))
    koyu = pencere < (arka - esik / 255.0)
    if not koyu.any():
        return np.zeros_like(gri, bool), []

    etiket, n = ndimage.label(koyu, structure=np.ones((3, 3), bool))
    ph, pw = pencere.shape
    min_alan = max(12.0, min_alan_orani * sinir * pw)
    maks_alan = maks_alan_orani * sinir * pw

    tutulan = np.zeros_like(koyu, bool)
    bilesenler = []
    dilimler = list(ndimage.find_objects(etiket))
    elenen = []            # ikinci gecis icin: cizgiyi asan bilesenler
    for sl, idx in zip(dilimler, range(1, n + 1)):
        if sl is None:
            continue
        par = etiket[sl] == idx
        alan = float(par.sum())
        yuk = sl[0].stop - sl[0].start
        gen = sl[1].stop - sl[1].start
        doluluk = alan / float(yuk * gen)

        if alan < min_alan:
            sebep = "kucuk"
        elif sl[0].stop > sinir:
            # cizginin altina tasiyor -> serit icerigi, dokunma
            sebep = "serit-icerigi"
        elif alan > maks_alan:
            sebep = "buyuk"
        elif doluluk > dolu_esik:
            # kutusunu neredeyse tamamen dolduruyor: bant/leke gorunumu.
            # El yazisi konturdur, kutusunun yarisindan azini doldurur.
            sebep = "bant-gorunumlu"
        else:
            sebep = None

        bilesenler.append(dict(alan=alan, yuk=yuk, gen=gen, doluluk=doluluk,
                               sebep=sebep or "YAZI"))
        if sebep is None:
            tutulan[sl][par] = True
        elif sebep == "serit-icerigi" and doluluk <= dolu_esik and alan <= maks_alan:
            elenen.append((sl, idx, par))

    # Ikinci gecis. Birinci gecis yaziyi bulduysa, bazi harfler cizginin bir
    # miktar altina sarkmis olabilir (ornegin iki harfli bir notun ikinci harfi) ve birinci gecis
    # onlari serit icerigi sanmistir. Yazinin GERCEKTE kapladigi kutuya
    # bakarak ayni yukseklikteki ve sagindaki konturlari da al. Kutuya
    # bagli kaldigi icin seritlere inmez.
    if tutulan.any():
        ys, xs = np.where(tutulan)
        y_alt = ys.max() + int(round(0.04 * H))
        x_sol = xs.min() - int(round(0.03 * pw))
        for sl, idx, par in elenen:
            if sl[0].stop <= y_alt and sl[1].start >= x_sol:
                tutulan[sl][par] = True
                bilesenler.append(dict(alan=float(par.sum()),
                                       yuk=sl[0].stop - sl[0].start,
                                       gen=sl[1].stop - sl[1].start,
                                       doluluk=0.0, sebep="YAZI-2"))

    tam = np.zeros_like(gri, bool)
    tam[r0:r1, c0:c1] = tutulan
    return tam, bilesenler


def maskele(im, ust_oran, sol_oran, esik, min_alan_orani, maks_alan_orani,
            genislet, analiz_oran, dolu_esik=0.70):
    """im: PIL RGB. Donus: (maskeli PIL, yazi_piksel_sayisi, bilesenler)."""
    a = np.asarray(im).astype(np.float32) / 255.0
    gri = a.mean(axis=2) if a.ndim == 3 else a
    m, bilesenler = yazi_maskesi(gri, ust_oran, sol_oran, esik,
                                 min_alan_orani, maks_alan_orani, analiz_oran,
                                 dolu_esik)
    if not m.any():
        return im, 0, bilesenler

    if genislet > 0:
        m = ndimage.binary_dilation(m, iterations=genislet)

    r0, r1, c0, c1 = serit_kutusu(gri.shape, ust_oran, sol_oran)
    serit = gri[r0:r1, c0:c1]
    temiz = ~m[r0:r1, c0:c1]
    arka = float(np.median(serit[temiz])) if temiz.any() else float(np.median(serit))

    cik = np.asarray(im).copy()
    if cik.ndim == 2:
        cik[m] = int(round(arka * 255))
    else:
        cik[m] = np.array([int(round(arka * 255))] * cik.shape[2],
                          dtype=cik.dtype)[:cik.shape[2]]
    return Image.fromarray(cik), int(m.sum()), bilesenler


# --------------------------------------------------------------- ornek sayfa
def ornek_sayfa(ciftler, yol, sutun=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(ciftler)
    if n == 0:
        return None
    satir = int(np.ceil(n / sutun))
    fig, axs = plt.subplots(satir * 2, sutun,
                            figsize=(3.1 * sutun, 3.0 * satir * 2))
    axs = np.atleast_2d(axs)
    for ax in axs.ravel():
        ax.axis("off")
    for k, (ad, once, sonra) in enumerate(ciftler):
        s, c = divmod(k, sutun)
        axs[2 * s, c].imshow(once, cmap="gray")
        axs[2 * s, c].set_title(os.path.basename(ad)[:34] + "\noncesi", fontsize=7)
        axs[2 * s + 1, c].imshow(sonra, cmap="gray")
        axs[2 * s + 1, c].set_title("sonrasi", fontsize=7)
    fig.tight_layout()
    fig.savefig(yol, dpi=110)
    plt.close(fig)
    return yol


# --------------------------------------------------------------- ana
def main():
    ap = argparse.ArgumentParser(
        description="Jel karelerindeki el yazisi tanitici bilgiyi maskeler.")
    ap.add_argument("--kaynak", default="Cropped_Data_EN",
                    help="Orijinal kare klasoru (okunur, degistirilmez)")
    ap.add_argument("--cikti", default="Cropped_Data_EN_maskeli",
                    help="Maskeli kopyanin yazilacagi klasor (--uygula ile)")
    ap.add_argument("--rapor", default="maskeleme_raporu.csv")
    ap.add_argument("--uygula", action="store_true",
                    help="Goruntuleri gercekten yaz (varsayilan: kuru kosu)")
    ap.add_argument("--yalnizca-yazili", action="store_true",
                    help="--uygula ile: sadece yazi bulunan kareleri yaz")
    ap.add_argument("--ornek-sayfa", type=int, default=12,
                    help="Kac kare icin oncesi/sonrasi sayfa uretilsin (0 = uretme)")
    ap.add_argument("--ornek-cikti", default="maskeleme_ornekleri.png")
    ap.add_argument("--ust-oran", type=float, default=0.22,
                    help="Yazi bu cizginin tamamen ustunde kalmali (yukseklik orani)")
    ap.add_argument("--dolu-esik", type=float, default=0.70,
                    help="Kutusunu bu orandan fazla dolduran bilesen bant sayilir")
    ap.add_argument("--sol-oran", type=float, default=0.30)
    ap.add_argument("--esik", type=float, default=28.0,
                    help="Arka plandan kac tonluk koyuluk yazi sayilsin (0-255)")
    ap.add_argument("--min-alan", type=float, default=2.5e-4,
                    help="Serit alanina oran: bundan kucuk bilesen gurultu")
    ap.add_argument("--maks-alan", type=float, default=0.45,
                    help="Serit alanina oran: bundan buyuk bilesen jel icerigi")
    ap.add_argument("--analiz-oran", type=float, default=0.42,
                    help="Bilesenlerin arandigi pencere (ust-oran'dan buyuk olmali)")
    ap.add_argument("--genislet", type=int, default=3,
                    help="Maskeyi kac piksel genislet")
    ap.add_argument("--kontrol-ornegi", type=int, default=0,
                    help="--uygula ile: kac maskeli kare gozle kontrol icin kopyalansin")
    ap.add_argument("--kontrol-dizin", default="sonuc_maskeleme_kontrol",
                    help="--kontrol-ornegi ile: orneklerin kopyalanacagi klasor")
    ap.add_argument("--tohum", type=int, default=20261101)
    a = ap.parse_args()

    if not os.path.isdir(a.kaynak):
        sys.exit(f"kaynak klasor yok: {a.kaynak}")
    if a.uygula and os.path.abspath(a.cikti) == os.path.abspath(a.kaynak):
        sys.exit("cikti kaynakla ayni olamaz; orijinaller korunmali")

    dosyalar = []
    for kok, _, adlar in os.walk(a.kaynak):
        for ad in sorted(adlar):
            if ad.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                dosyalar.append(os.path.join(kok, ad))
    dosyalar.sort()
    print(f"{len(dosyalar)} kare taranacak: {a.kaynak}")
    print("KURU KOSU -- hicbir goruntu yazilmayacak" if not a.uygula
          else f"UYGULA -- maskeli kopyalar: {a.cikti}")

    rng = np.random.default_rng(a.tohum)
    satirlar, ornekler = [], []
    yazili = 0
    for k, yol in enumerate(dosyalar, 1):
        rel = os.path.relpath(yol, a.kaynak)
        try:
            im = Image.open(yol).convert("RGB")
        except Exception as e:                       # bozuk dosya atlanir
            satirlar.append(dict(dosya=rel, durum=f"okunamadi: {e}",
                                 yazi_pikseli=0, bilesen=0))
            continue

        yeni, piksel, bilesenler = maskele(
            im, a.ust_oran, a.sol_oran, a.esik, a.min_alan, a.maks_alan,
            a.genislet, a.analiz_oran, a.dolu_esik)
        nyazi = sum(1 for b in bilesenler if b["sebep"] == "YAZI")
        if piksel > 0:
            yazili += 1
        satirlar.append(dict(dosya=rel, durum="yazi" if piksel else "temiz",
                             yazi_pikseli=piksel, bilesen=nyazi))

        if a.uygula and (piksel > 0 or not a.yalnizca_yazili):
            hedef = os.path.join(a.cikti, rel)
            os.makedirs(os.path.dirname(hedef), exist_ok=True)
            yeni.save(hedef)

        if piksel > 0 and len(ornekler) < a.ornek_sayfa:
            ornekler.append((rel, np.asarray(im.convert("L")),
                             np.asarray(yeni.convert("L"))))
        if k % 500 == 0:
            print(f"  ...{k}/{len(dosyalar)}  yazili {yazili}", flush=True)

    if os.path.dirname(a.rapor):
        os.makedirs(os.path.dirname(a.rapor), exist_ok=True)
    with open(a.rapor, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dosya", "durum", "yazi_pikseli",
                                           "bilesen"])
        w.writeheader()
        w.writerows(satirlar)

    print(f"\n{len(dosyalar)} kare | yazi bulunan {yazili} "
          f"(%{100.0*yazili/max(1,len(dosyalar)):.1f}) | rapor: {a.rapor}")
    if a.ornek_sayfa and ornekler:
        p = ornek_sayfa(ornekler, a.ornek_cikti)
        print(f"ornek sayfa: {p}  ({len(ornekler)} kare, oncesi/sonrasi)")
    if a.kontrol_ornegi and os.path.isdir(a.cikti):
        import shutil
        yazili = [r["dosya"] for r in satirlar if r["durum"] == "yazi"]
        if not yazili:
            print("kontrol ornegi: yazi bulunan kare yok, kopyalanmadi")
        else:
            sec = rng.choice(len(yazili), size=min(a.kontrol_ornegi, len(yazili)),
                             replace=False)
            os.makedirs(a.kontrol_dizin, exist_ok=True)
            for k, idx in enumerate(sorted(sec), 1):
                rel = yazili[idx]
                kaynak = os.path.join(a.cikti, rel)
                hedef = os.path.join(a.kontrol_dizin,
                                     f"{k:02d}__" + rel.replace(os.sep, "__"))
                shutil.copy2(kaynak, hedef)
            print(f"kontrol ornegi: {min(a.kontrol_ornegi, len(yazili))} maskeli kare -> "
                  f"{a.kontrol_dizin}/ (yalniz yazi bulunan karelerden)")

    if not a.uygula:
        print("\nKuru kosuydu. Ornek sayfayi kontrol et; dogruysa --uygula ile calistir.")


if __name__ == "__main__":
    main()
