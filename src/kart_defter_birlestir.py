# -*- coding: utf-8 -*-
"""
kart_defter_birlestir.py -- kirpma <-> defter kaydi (kart ID + IFE no)
========================================================================
ZINCIR
-------
    kart_id_oku.py    -> her kirpma icin  kart_id + ife_no
    defter_oku.py     -> her defter sayfasi icin  sayfa_no + 4 kayit (sira 1-4)

    kart_id  ==  sayfa_no      (dogrulandi: ayni seri)
    ife_no   ==  sira          (IFE 1-4  <->  defter 1-4)

DOGRULAMA
----------
Her eslesmede jeldeki EL YAZISI BAS HARF ile defterdeki AD SOYAD'in bas
harfleri karsilastirilir. Jeldeki bas harf kirpmanin uzerinde yazili;
kirpma_basharf.csv varsa (defter_oku.py --kirpmalar ile uretilir) kullanilir.
Uyusmayanlar "KONTROL" olarak isaretlenir -- silinmez, size birakilir.

CIKTI
------
    birlesik_kayitlar.xlsx  (Excel'de dogrudan acilir)
    her satir: kirpma, kart_id, ife_no, ad, soyad, barkod, defterdeki tani,
               SIZIN taniniz, eslesme durumu

KULLANIM
---------
    python kart_defter_birlestir.py
    python kart_defter_birlestir.py --basharf kirpma_basharf.csv
"""
from __future__ import annotations

import argparse
import unicodedata

import numpy as np
import pandas as pd

_TR = str.maketrans({"İ": "I", "I": "I", "ı": "I", "i": "I", "Ş": "S", "ş": "S",
                     "Ğ": "G", "ğ": "G", "Ü": "U", "ü": "U", "Ö": "O", "ö": "O",
                     "Ç": "C", "ç": "C"})


def norm(s) -> str:
    if not isinstance(s, str):
        return ""
    s = s.translate(_TR)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()


def ad_basharf(ad, soyad) -> str:
    """'Yusuf' + 'Yilmaz' -> 'YY';  'Serif Mehmet' + 'Cetin' -> 'SMC'"""
    parcalar = [k for k in (norm(ad) + " " + norm(soyad)).split() if k]
    return "".join(k[0] for k in parcalar)


def jel_basharf(s) -> str:
    """'Y.Y' / 'E.Ozlen Fenerbak' -> harfler.  Soyad tam yazildiysa bas harfi alinir."""
    s = norm(s)
    if not s:
        return ""
    parcalar = [k for k in s.replace(".", " ").replace("-", " ").split() if k]
    return "".join(k[0] for k in parcalar)


def uyum(jel: str, defter: str) -> str:
    """'tam' | 'ilk-son' | 'zayif' | 'uyusmuyor' | ''"""
    if not jel or not defter:
        return ""
    if jel == defter:
        return "tam"
    if len(jel) >= 2 and len(defter) >= 2 and jel[0] == defter[0] and jel[-1] == defter[-1]:
        return "ilk-son"
    if jel[0] == defter[0]:
        return "zayif"
    return "uyusmuyor"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kart", default="kart_id_eslestirme.csv")
    ap.add_argument("--defter", default="defter_okunan.csv")
    ap.add_argument("--master", default="annotations_master.csv")
    ap.add_argument("--basharf", default="kirpma_basharf.csv",
                    help="kirpma -> jeldeki el yazisi bas harf (varsa)")
    ap.add_argument("--cikti", default="birlesik_kayitlar.xlsx")
    a = ap.parse_args()

    kart = pd.read_csv(a.kart, dtype={"kart_id": str})
    defter = pd.read_csv(a.defter, dtype={"sayfa_no": str, "barkod": str})
    master = pd.read_csv(a.master)

    # kart_id_oku.py bos defter_* sutunlari yazar; birlesmede cakismasin
    kart = kart.drop(columns=[c for c in kart.columns if c.startswith("defter_")],
                     errors="ignore")
    kart["kart_id"] = kart["kart_id"].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    defter["sayfa_no"] = defter["sayfa_no"].fillna("").astype(str).str.strip()
    defter["sira"] = pd.to_numeric(defter["sira"], errors="coerce").fillna(0).astype(int)
    # Sayfa kimligi = FOTOGRAF (klasor + dosya). Helena'da sayfa_no (1,2,3...) kart ID degil, yillar arasinda
    # da ayni sayfa numarasi tekrar edebilir; (sayfa_no, sira) ile ayiklamak gercek kayitlari siler.
    defter["_foto"] = defter["tarama_klasoru"].astype(str) + "|" + defter["defter_fotograf"].astype(str)
    d_once = len(defter)
    defter = defter.drop_duplicates(["_foto", "sira"], keep="first")
    if len(defter) < d_once:
        print(f"Defterde tekrar eden (fotograf, sira): {d_once - len(defter)} satir atildi")
    # Ayni kart ID'si (Interlab) iki ayri fotografta okunmussa (sayfa iki kez cekilmis) ilkini tut
    interlab = ~defter["tarama_klasoru"].astype(str).str.upper().str.startswith("HELENA")
    d_once = len(defter)
    defter = pd.concat([defter[interlab].drop_duplicates(["sayfa_no", "sira"], keep="first"), defter[~interlab]])
    if len(defter) < d_once:
        print(f"Interlab'da ayni kart ID'si iki fotografta: {d_once - len(defter)} satir atildi")

    # --- IDRAR kayitlarini ayikla ve sirayi YENIDEN numarala ---
    # Defter sayfasinda idrar ornekleri de sirayla yazili; jel kartinda 4 SERUM paneli var.
    # Sayfada 5-6 kayit varsa "sira 1-4 = panel 1-4" kayar. Idrar satirlari cikarilip
    # kalan serum kayitlari sayfa icinde 1..n olarak yeniden numaralanir.
    def _idrar(t):
        t = norm(t)
        return any(k in t for k in ("IDRAR", "BENCE", "BJP", "BJ ", "URINE", "SPOT"))
    idrar_m = defter["defter_tani"].apply(_idrar)
    print(f"Defterde idrar kaydi: {int(idrar_m.sum())} (sira atamasindan cikarildi)")
    defter = defter[~idrar_m].copy()
    defter = defter.sort_values(["_foto", "sira"])
    defter["sira_orijinal"] = defter["sira"]
    defter["sira"] = defter.groupby("_foto").cumcount() + 1
    yeniden = int((defter["sira"] != defter["sira_orijinal"]).sum())
    print(f"Idrar cikarilinca sirasi degisen serum kaydi: {yeniden}")
    fazla = defter.groupby("_foto")["sira"].max()
    print(f"Idrar cikarildiktan sonra 4'ten fazla kaydi olan sayfa: {int((fazla > 4).sum())} "
          f"(bunlar iki karta yayiliyor olabilir; sira>4 kayitlar eslesmez)")
    kart["ife_no"] = pd.to_numeric(kart["ife_no"], errors="coerce").fillna(0).astype(int)

    # Ayni kirpma birden fazla satirda olabilir (script tekrar calistirilinca
    # eklenmis). Kart ID'si OLAN satiri tercih ederek tekillestir.
    once = len(kart)
    kart = (kart.assign(_idli=kart["kart_id"].ne(""))
                .sort_values("_idli", ascending=False)
                .drop_duplicates("kirpma_dosyasi", keep="first")
                .drop(columns="_idli"))
    if len(kart) < once:
        print(f"Tekrarli satir temizlendi: {once} -> {len(kart)}")
    print(f"Kirpma : {len(kart)}  (kart ID'li: {int(kart['kart_id'].ne('').sum())})")
    print(f"Defter : {len(defter)} kayit, {defter['sayfa_no'].nunique()} sayfa")

    # --- Kendi taniniz ---
    kendi = master[["filename", "pattern", "bands", "all_diagnoses_tr", "device"]].rename(
        columns={"filename": "kirpma_dosyasi", "pattern": "benim_patern",
                 "all_diagnoses_tr": "benim_tani_tr"})
    b = kart.merge(kendi, on="kirpma_dosyasi", how="left")

    # --- Defterle birlestir: kart_id == sayfa_no AND ife_no == sira ---
    # Kart ID uzerinden birlestirme YALNIZ Interlab (basili kart ID'si olan) kayitlar icin:
    # sayfa_no'su bos veya Helena olan defter satirlari buraya girmez (bos ID'ler kart ID'si
    # okunamayan 913 kirpma ile carpisip binlerce sahte eslesme uretir).
    defter_kart = defter[(defter["sayfa_no"].astype(str).str.strip() != "") &
                         ~defter["tarama_klasoru"].astype(str).str.upper().str.startswith("HELENA")]
    d2 = defter_kart.rename(columns={"sayfa_no": "kart_id", "sira": "ife_no"})
    d2 = d2[["kart_id", "ife_no", "ad", "soyad", "barkod", "defter_tani",
             "sayfa_tarih", "guven", "defter_fotograf"]]
    b = b.merge(d2, on=["kart_id", "ife_no"], how="left")

    # --- Jeldeki bas harf (varsa) ---
    b["jel_basharf"] = ""
    try:
        bh = pd.read_csv(a.basharf, dtype=str).fillna("")
        # Ayni kirpma icin birden cok satir olabilir (tekrarlanan okumalar): dolu olani tut, tek satira indir
        bh = bh.sort_values("jel_basharf", key=lambda c: c.eq(""), kind="stable").drop_duplicates("kirpma_dosyasi", keep="first")
        bh = bh[["kirpma_dosyasi", "jel_basharf"]].drop_duplicates("kirpma_dosyasi")
        b = b.drop(columns=["jel_basharf"]).merge(bh, on="kirpma_dosyasi", how="left")
        b["jel_basharf"] = b["jel_basharf"].fillna("")
        print(f"Jel bas harfi olan kirpma: {int(b['jel_basharf'].ne('').sum())}")
    except FileNotFoundError:
        print("(kirpma_basharf.csv yok -- bas harf dogrulamasi atlanacak)")

    # --- Dogrulama ---
    b["defter_basharf"] = [ad_basharf(x, y) for x, y in zip(b["ad"], b["soyad"])]
    b["_j"] = b["jel_basharf"].apply(jel_basharf)
    b["basharf_uyum"] = [uyum(j, d) for j, d in zip(b["_j"], b["defter_basharf"])]

    def durum(r):
        if not r["kart_id"]:
            # Helena kartlarinda basili ID yok -- bunlar bas harf + tarihle
            # eslesecek, OCR hatasi degil.
            if str(r.get("device", "")).upper() == "HELENA":
                return "HELENA (isimle eslesecek)"
            return "KART ID OKUNAMADI"
        if pd.isna(r.get("ad")) and pd.isna(r.get("barkod")):
            return "DEFTERDE YOK"
        u = r["basharf_uyum"]
        if u in ("tam", "ilk-son"):
            return "ESLESTI"
        if u == "":
            return "ESLESTI (bas harf dogrulanmadi)"
        if u == "zayif":
            return "KONTROL (bas harf kismen uyuyor)"
        return "KONTROL (bas harf UYUSMUYOR)"

    b["durum"] = b.apply(durum, axis=1)

    # --- Bas harf uyusmayan panelleri ayni sayfada bas harfi TUTAN kayitla degistir ---
    d_kol = ["ad", "soyad", "barkod", "defter_tani", "sayfa_tarih", "guven", "defter_fotograf"]
    sayfa_kayit = {k: g for k, g in d2.groupby("kart_id")}
    duzeltilen = 0
    for i, r in b[b["durum"].str.startswith("KONTROL") & b["_j"].ne("")].iterrows():
        g = sayfa_kayit.get(r["kart_id"])
        if g is None:
            continue
        aday = [(j, x) for j, x in g.iterrows()
                if uyum(r["_j"], ad_basharf(x["ad"], x["soyad"])) in ("tam", "ilk-son")]
        if len(aday) == 1:
            _, x = aday[0]
            for c in d_kol:
                b.at[i, c] = x[c]
            b.at[i, "defter_basharf"] = ad_basharf(x["ad"], x["soyad"])
            b.at[i, "basharf_uyum"] = uyum(r["_j"], b.at[i, "defter_basharf"])
            b.at[i, "durum"] = "ESLESTI (bas harfle duzeltildi)"
            duzeltilen += 1
    print(f"Bas harfe gore ayni sayfada baska kayda tasinan panel: {duzeltilen}")

    # =====================================================================
    # HELENA: kartta basili ID yok. Hasta adi kartin kenarinda dikey yazili;
    # defter_oku.py --helena-kenar bunu TARAMADAN okuyup kirpma_basharf.csv'ye
    # jel_basharf (= "E. Senol") ve jel_tarih olarak yazar. Eslestirme:
    # bas harf + soyad benzerligi + tarih yakinligi, defterin HELENA sayfalari icinde.
    # =====================================================================
    hel_idx = b.index[b["durum"] == "HELENA (isimle eslesecek)"]
    hd = defter[defter["tarama_klasoru"].astype(str).str.upper().str.startswith("HELENA")].copy()
    if len(hel_idx) and len(hd):
        import difflib
        try:
            bh_ek = pd.read_csv(a.basharf, dtype=str).fillna("")
            bh_ek = bh_ek.drop_duplicates("kirpma_dosyasi").set_index("kirpma_dosyasi")
        except FileNotFoundError:
            bh_ek = pd.DataFrame()
        def tarih(v):
            for f in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%Y-%m-%d"):
                try: return pd.to_datetime(str(v).strip(), format=f)
                except Exception: pass
            return pd.NaT
        hd["_t"] = hd["sayfa_tarih"].apply(tarih)
        hd["_soyad"] = hd["soyad"].apply(norm); hd["_ad1"] = hd["ad"].apply(lambda v: norm(v)[:1])
        def ayir(ad_metni):
            t = norm(ad_metni).replace(".", " ").replace("-", " ").split()
            if not t: return "", ""
            if len(t) == 1: return "", t[0]
            return t[0][:1], t[-1]
        adaylar = []
        for i in hel_idx:
            fn = b.at[i, "kirpma_dosyasi"]
            jel_ad = b.at[i, "jel_basharf"] if b.at[i, "jel_basharf"] else ""
            jel_t = tarih(bh_ek.at[fn, "jel_tarih"]) if (len(bh_ek) and fn in bh_ek.index and "jel_tarih" in bh_ek.columns) else pd.NaT
            ilk, soy = ayir(jel_ad)
            if not soy: continue
            for j, r in hd.iterrows():
                oran = difflib.SequenceMatcher(None, soy, r["_soyad"]).ratio()
                ilk_ok = bool(ilk) and ilk == r["_ad1"]
                if not ((ilk_ok and oran >= 0.70) or oran >= 1.0): continue
                gun = abs((jel_t - r["_t"]).days) if (pd.notna(jel_t) and pd.notna(r["_t"])) else None
                # Tarih sadece BONUS (defter yillari OCR'da sik bozuk); dislama kriteri degil
                skor = oran + (0.3 if ilk_ok else 0) + (0.5 if gun == 0 else 0.25 if (gun is not None and gun <= 14) else 0)
                adaylar.append((skor, i, j, oran, gun, ilk_ok))
        adaylar.sort(reverse=True)
        kullanilan_d, atanan = set(), {}
        for skor, i, j, oran, gun, ilk_ok in adaylar:
            if i in atanan or j in kullanilan_d: continue
            atanan[i] = (j, oran, gun, ilk_ok); kullanilan_d.add(j)
        d_kol = ["ad", "soyad", "barkod", "defter_tani", "sayfa_tarih", "guven", "defter_fotograf"]
        for i, (j, oran, gun, ilk_ok) in atanan.items():
            x = hd.loc[j]
            for c in d_kol: b.at[i, c] = x[c]
            b.at[i, "defter_basharf"] = ad_basharf(x["ad"], x["soyad"])
            b.at[i, "basharf_uyum"] = f"soyad {oran:.2f}" + (f", {gun} gun" if gun is not None else ", tarih yok")
            # A: bas harf + soyad>=0.85 -> guvenli. B: bas harf + 0.70-0.85 -> tarih uyuyorsa guvenli, yoksa kontrol.
            # C: soyad birebir ama bas harf farkli/yok -> kontrol (yaygin soyadlar).
            if ilk_ok and oran >= 0.85:
                b.at[i, "durum"] = "HELENA ESLESTI"
            elif ilk_ok and gun is not None and gun <= 14:
                b.at[i, "durum"] = "HELENA ESLESTI"
            else:
                b.at[i, "durum"] = "HELENA ESLESTI (kontrol)"
        b.loc[b["durum"] == "HELENA (isimle eslesecek)", "durum"] = "HELENA ESLESMEDI"
        print(f"HELENA: {len(hel_idx)} panel, {len(hd)} defter kaydi; eslesen {len(atanan)} "
              f"(guvenli {int((b.loc[list(atanan)] ['durum'] == 'HELENA ESLESTI').sum())})")
        for i, (j, oran, gun, ilk_ok) in list(atanan.items())[:6]:
            print(f"    jel '{b.at[i, 'jel_basharf']}' <- defter '{hd.at[j, 'ad']} {hd.at[j, 'soyad']}' (soyad {oran:.2f}, {gun} gun)")
    elif len(hel_idx):
        print("HELENA: defter sayfasi (Defter/Helena*) okunmamis veya kenar adlari okunmamis; eslestirme atlandi")
    b = b.drop(columns=["_j"])

    kolonlar = ["durum", "kirpma_dosyasi", "kaynak_tarama", "kart_id", "ife_no",
                "jel_basharf", "defter_basharf", "basharf_uyum",
                "ad", "soyad", "barkod", "sayfa_tarih",
                "defter_tani", "benim_patern", "benim_tani_tr", "bands",
                "guven", "defter_fotograf", "device"]
    b = b[[c for c in kolonlar if c in b.columns]]
    for c in ("kart_id", "barkod", "ife_no"):
        if c in b.columns:
            b[c] = b[c].apply(lambda v: "" if pd.isna(v) else str(v).replace(".0", "") if isinstance(v, float) else str(v))
    b = b.sort_values(["durum", "kaynak_tarama", "kart_id", "ife_no"])

    try:
        b.to_excel(a.cikti, index=False)
        print(f"\n-> {a.cikti}")
    except Exception as e:
        csv = a.cikti.rsplit(".", 1)[0] + ".csv"
        b.to_csv(csv, index=False, encoding="utf-8-sig")
        print(f"\nExcel yazilamadi ({e}); -> {csv}")

    print("\nESLESME DURUMU")
    for d, n in b["durum"].value_counts().items():
        print(f"  {n:5d}  {d}")

    es = b[b["durum"].str.startswith("ESLESTI")]
    if len(es):
        print(f"\n{len(es)} kayit defterle eslesti. Bunlarda:")
        print(f"  barkodu olan       : {int(es['barkod'].fillna('').astype(str).str.strip().ne('').sum())}")
        print(f"  defter tanisi olan : {int(es['defter_tani'].fillna('').astype(str).str.strip().ne('').sum())}")
    print("\n'KONTROL' satirlarina gozle bakin; 'KART ID OKUNAMADI' olanlar icin")
    print("kart_id_eslestirme.csv'de kart_id sutununu elle doldurup tekrar calistirin.")


if __name__ == "__main__":
    main()
