# -*- coding: utf-8 -*-
"""
bitti_isaretle.py -- toplu "bitti" isaretleme
==============================================
ife_annotator.py artik SADECE sizin "bitti" dediginiz taramalari atliyor.
Ilk calistirmada bu liste bos olacagi icin 147 Interlab taramasi da
karsiniza gelir. Bu script, belirttiginiz taramalari toplu olarak bitti
listesine ekler.

Interlab'in tamami bittiyse:
    python bitti_isaretle.py --cihaz INTERLAB

Belirli bir esigin uzerinde kayit iceren taramalari bitti say:
    python bitti_isaretle.py --en-az 10

Listeyi gormek icin:
    python bitti_isaretle.py --listele
"""
import argparse
from pathlib import Path
import pandas as pd

MASTER = "annotations_master.csv"
DONE_FILE = Path("tamamlanan_taramalar.txt")


def mevcut() -> set:
    if not DONE_FILE.exists():
        return set()
    return {l.strip() for l in DONE_FILE.read_text(encoding="utf-8").splitlines() if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=MASTER)
    ap.add_argument("--cihaz", choices=["INTERLAB", "HELENA"],
                    help="Bu cihazin tum kayitli taramalarini bitti say")
    ap.add_argument("--en-az", type=int, default=None,
                    help="En az bu kadar kayit iceren taramalari bitti say")
    ap.add_argument("--listele", action="store_true")
    ap.add_argument("--cikar-eksikli", action="store_true",
                    help="denetim_eksik_paneller.csv'de gecen taramalari bitti "
                         "listesinden CIKAR (tekrar acilsinlar)")
    args = ap.parse_args()

    var = mevcut()
    if args.listele:
        print(f"{len(var)} tarama bitti isaretli:")
        for a in sorted(var):
            print("  ", a)
        return

    if args.cikar_eksikli:
        # Eksik panel listesi iki kaynaktan gelebilir:
        #   harita_sorunlari.csv       (tarama_haritasi.py -- guncel)
        #   denetim_eksik_paneller.csv (eski kirpma_denetimi.py)
        eksikli = set()
        h = Path("harita_sorunlari.csv")
        e = Path("denetim_eksik_paneller.csv")
        if h.exists():
            t = pd.read_csv(h)
            if "sorun" in t.columns:
                t = t[t["sorun"].astype(str).isin(["BOS", "TEKRAR"])]
            eksikli |= set(t["kaynak_tarama"].astype(str))
            print(f"harita_sorunlari.csv okundu: {len(eksikli)} tarama")
        elif e.exists():
            eksikli = set(pd.read_csv(e)["kaynak_tarama"].astype(str))
            print(f"denetim_eksik_paneller.csv okundu: {len(eksikli)} tarama")
        else:
            raise SystemExit(
                "Sorunlu tarama listesi yok.\n"
                "Once:  python tarama_haritasi.py --taramalar A4_Taramalar\n"
                "Ya da tum taramalari yeniden acmak icin "
                "tamamlanan_taramalar.txt dosyasini silin.")
        kalan = var - eksikli
        cikan = var & eksikli
        DONE_FILE.write_text("\n".join(sorted(kalan)) + ("\n" if kalan else ""),
                             encoding="utf-8")
        print(f"{len(cikan)} tarama bitti listesinden cikarildi "
              f"(eksik panel adayi var). Kalan bitti: {len(kalan)}")
        print("Simdi: python ife_annotator.py  -> sari cerceveler nereyi "
              "kesmeniz gerektigini gosterir.")
        return

    df = pd.read_csv(args.input)
    sec = df
    if args.cihaz:
        sec = sec[sec["device"].astype(str) == args.cihaz]
    sayim = sec["source_file"].astype(str).value_counts()
    if args.en_az:
        sayim = sayim[sayim >= args.en_az]
    yeni = set(sayim.index) - var

    if not yeni:
        print("Eklenecek yeni tarama yok.")
        return
    with open(DONE_FILE, "a", encoding="utf-8") as f:
        for a in sorted(yeni):
            f.write(a + "\n")
    print(f"{len(yeni)} tarama bitti listesine eklendi. Toplam: {len(var) + len(yeni)}")
    print(f"Geri almak icin {DONE_FILE} dosyasindan ilgili satirlari silin.")


if __name__ == "__main__":
    main()
