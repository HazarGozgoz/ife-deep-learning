# -*- coding: utf-8 -*-
"""
lab_birlestir.py -- eslesen kayitlara laboratuvar degerlerini ekler
=====================================================================
Defter barkodlari lab dosyasindaki protokol numaralariyla AYNI DEGIL
(kontrol edildi: 320 barkodun sifiri eslesiyor). Ama ikisinde de HASTA
ADI ve TARIH var; eslestirme onunla yapilir.

NEDEN +-30 GUN
---------------
Gercek verinizde olculdu: jel tarihi ile lab testi arasinda haftalar var.
    +-7 gun  ->  88 / 329 eslesme
    +-30 gun -> 238 / 329 eslesme  (%72)
    +-90 gun -> 239 / 329          (daha genis pencere kazandirmiyor)
IFE toplu calisildigi icin kan alinma tarihi ile jel tarihi ayrisiyor.
Tam AD SOYAD ile eslestigi icin genis pencere cakisma riski yaratmiyor
(bas harf degil, tam isim).

BULANIK ESLESME
----------------
Defter el yazisindan okundugu icin isimde 1-2 harf hatasi olabilir
("Muride" -> "Murude"). En fazla 2 harf farkla eslesmeye izin verilir ve
bu eslesmeler "bulanik" olarak isaretlenir -- gozle kontrol edin.

EKLENEN DEGERLER
-----------------
Gamma%, Albumin%, Alfa/Beta%, IgG, IgA, IgM, serbest kappa/lambda,
K/L orani, M protein%, total protein, yas, cinsiyet -- jel tarihine EN
YAKIN ziyaretten.

KULLANIM
---------
    python lab_birlestir.py --lab "Hasta_sonuc_lari_tunahan.xlsx"
    python lab_birlestir.py --lab ... --pencere 45
"""
from __future__ import annotations

import argparse
import unicodedata

import pandas as pd

_TR = str.maketrans({"İ": "I", "I": "I", "ı": "I", "i": "I", "Ş": "S", "ş": "S",
                     "Ğ": "G", "ğ": "G", "Ü": "U", "ü": "U", "Ö": "O", "ö": "O",
                     "Ç": "C", "ç": "C"})

TEST_MAP = {
    "Gamma%": "gamma_pct", "Albumin%": "albumin_pct", "Alfa 1%": "alfa1_pct",
    "Alfa 2%": "alfa2_pct", "Beta 1%": "beta1_pct", "Beta 2%": "beta2_pct",
    "A/G Ratio": "ag_orani", "M Protein%": "m_protein_pct",
    "* IgG": "igg", "* IgA": "iga", "IgM": "igm",
    "SERBEST KAPPA HAFIF ZINCIR (KAN)": "serbest_kappa",
    "SERBEST LAMBDA HAFIF ZINCIR (KAN)": "serbest_lambda",
    "KAPPA / LAMBDA ORANI": "kl_orani",
    "* Total Protein": "total_protein", "* Albumin": "albumin",
    "* Kreatinin": "kreatinin", "* Kalsiyum": "kalsiyum",
    "* Laktat dehidrogenaz (LDH)": "ldh", "* CRP": "crp",
}


def norm(s) -> str:
    if not isinstance(s, str):
        return ""
    s = s.translate(_TR)
    s = unicodedata.normalize("NFKD", s)
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).upper().split())


def lev(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 3:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sayi(v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip().replace(",", ".")
    for k in ("<", ">", "=", "~"):
        s = s.replace(k, "")
    try:
        return float(s)
    except ValueError:
        return pd.NA


# ---------------------------------------------------------------------------
# LAB KARSILASTIRMASI (Supplemental Table 8)
# ---------------------------------------------------------------------------
# birlesik_lab.xlsx uzerinden, +-30 gun icinde lab sonucu olan panellerde:
#   hipogama bayragi   -> IgG, gamma%
#   poliklonal bayragi -> gamma%, IgG
#   izotip (IgG/IgA/IgM bandi olan vs diger bant-pozitif) -> ilgili Ig
# Ortanca, n ve iki yonlu Mann-Whitney U testi P degeri. Cikti: lab_karsilastirma.csv
# KULLANIM:  python lab_birlestir.py --karsilastir            (birlesik_lab.xlsx okur)
#            python lab_birlestir.py --karsilastir --girdi-lab baska.xlsx
def karsilastir(dosya: str = "birlesik_lab.xlsx", pencere: int = 30, cikti: str = "lab_karsilastirma.csv"):
    from scipy.stats import mannwhitneyu
    d = pd.read_excel(dosya)
    w = d[pd.to_numeric(d["lab_gun_farki"], errors="coerce").abs() <= pencere].copy()
    t = w["benim_tani_tr"].fillna("")
    hyp = t.str.contains("Hipogama"); poly = t.str.contains("Poliklonal")
    bands = w["bands"].fillna("")
    bp = w[bands.str.contains("definite")]
    rows = []
    def cmp(df, mask, col, etiket, birim):
        a = pd.to_numeric(df.loc[mask, col], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[~mask, col], errors="coerce").dropna()
        if len(a) < 3 or len(b) < 3:
            return
        pval = mannwhitneyu(a, b, alternative="two-sided").pvalue
        rows.append({"karsilastirma": etiket, "olcum": col, "birim": birim,
                     "ortanca_var": round(float(a.median()), 1), "ortanca_yok": round(float(b.median()), 1),
                     "n_var": len(a), "n_yok": len(b), "p_mann_whitney": pval})
    cmp(w, hyp, "igg", "hipogamaglobulinemi bayragi", "mg/dL")
    cmp(w, hyp, "gamma_pct", "hipogamaglobulinemi bayragi", "%")
    cmp(w, poly, "gamma_pct", "poliklonal artis bayragi", "%")
    cmp(w, poly, "igg", "poliklonal artis bayragi", "mg/dL")
    for iso, col in (("IGG", "igg"), ("IGA", "iga"), ("IGM", "igm")):
        m = bp["bands"].str.contains(iso + "_")
        cmp(bp, m, col, f"{iso} bandi vs diger bant-pozitif paneller", "mg/dL")
    out = pd.DataFrame(rows)
    gf = pd.to_numeric(w["lab_gun_farki"], errors="coerce").abs()
    print(f"+-{pencere} gun icinde lab sonucu olan panel: {len(w)} (gun farki ortanca {gf.median():.0f})")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}"))
    out.to_csv(cikti, index=False, encoding="utf-8-sig")
    print(f"-> {cikti}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi", default="birlesik_kayitlar.xlsx")
    ap.add_argument("--lab", help="LIS sonuc dokumu (.xlsx); --karsilastir icin gerekmez")
    ap.add_argument("--pencere", type=int, default=30)
    ap.add_argument("--harf-tolerans", type=int, default=2)
    ap.add_argument("--cikti", default="birlesik_lab.xlsx")
    ap.add_argument("--tani-listesi", action="store_true",
                    help="LIS'e verilecek KLINIK TANI ISTEK LISTESI'ni yaz (klinik_tani_istek_listesi.xlsx): "
                         "defterle eslesen her panel icin ad, soyad, IFE tarihi, cihaz, bulunabilen LIS "
                         "protokol/hasta no ve doldurulacak bos tani sutunlari.")
    ap.add_argument("--karsilastir", action="store_true",
                    help="birlesik_lab.xlsx uzerinden lab karsilastirmasi (Supplemental Table 8)")
    ap.add_argument("--girdi-lab", default="birlesik_lab.xlsx", help="--karsilastir icin girdi dosyasi")
    a = ap.parse_args()
    if a.karsilastir:
        karsilastir(a.girdi_lab, a.pencere)
        return

    if not a.lab:
        ap.error("--lab gerekli (ya da --karsilastir kullanin)")
    print("Lab dosyasi okunuyor...")
    d = pd.read_excel(a.lab, header=None)
    d.columns = ["id1", "protokol", "hasta_adi", "yas", "cinsiyet", "medeni",
                 "tarih", "test", "sonuc"][:d.shape[1]]
    d["t"] = pd.to_datetime(d["tarih"], errors="coerce").dt.normalize()
    d = d[d["t"].notna()].copy()
    d["ad_n"] = d["hasta_adi"].apply(norm)
    d["kisa"] = d["test"].map(TEST_MAP)
    d = d[d["kisa"].notna()].copy()
    d["deger"] = d["sonuc"].apply(sayi)

    ziyaret = d.pivot_table(index=["ad_n", "t"], columns="kisa", values="deger",
                            aggfunc="first").reset_index()
    kimlik = d.groupby(["ad_n", "t"]).agg(protokol=("protokol", "first"),
                                         lis_hasta_id=("id1", "first"),
                                         yas=("yas", "first"),
                                         cinsiyet=("cinsiyet", "first")).reset_index()
    ziyaret = kimlik.merge(ziyaret, on=["ad_n", "t"], how="left")
    print(f"  {len(ziyaret):,} hasta-gun ziyareti, {ziyaret['ad_n'].nunique():,} hasta")
    tum_adlar = ziyaret["ad_n"].unique().tolist()

    b = pd.read_excel(a.girdi, dtype=str)
    hedef = b["durum"].fillna("").str.startswith(("ESLESTI", "KONTROL", "HELENA ESLESTI"))
    b["_t"] = pd.to_datetime(b["sayfa_tarih"], errors="coerce", dayfirst=True)
    b["_ad"] = (b["ad"].fillna("") + " " + b["soyad"].fillna("")).apply(norm)
    print(f"Defterle eslesen kayit: {int(hedef.sum())}")

    pencere = pd.Timedelta(days=a.pencere)
    lab_kol = [c for c in ziyaret.columns if c not in ("ad_n", "t")]
    sonuc = {k: [] for k in ["lab_durum", "lab_ad", "lab_gun_farki"] + lab_kol}
    ad_cache: dict[str, list[str]] = {}

    for _, r in b.iterrows():
        bos = {k: pd.NA for k in lab_kol}
        if not hedef.loc[r.name] or pd.isna(r["_t"]) or not r["_ad"]:
            sonuc["lab_durum"].append(""); sonuc["lab_ad"].append(""); sonuc["lab_gun_farki"].append(pd.NA)
            for k in lab_kol:
                sonuc[k].append(bos[k])
            continue
        # aday isimler: once tam, sonra bulanik
        if r["_ad"] not in ad_cache:
            if r["_ad"] in set(tum_adlar):
                ad_cache[r["_ad"]] = [(r["_ad"], "tam")]
            else:
                ad_cache[r["_ad"]] = [(x, "bulanik") for x in tum_adlar
                                      if x[:1] == r["_ad"][:1] and lev(r["_ad"], x) <= a.harf_tolerans]
        adaylar = ad_cache[r["_ad"]]
        secim = None
        for ad_l, tip in adaylar:
            z = ziyaret[(ziyaret["ad_n"] == ad_l) & ((ziyaret["t"] - r["_t"]).abs() <= pencere)]
            if len(z):
                z = z.assign(_fark=(z["t"] - r["_t"]).abs()).sort_values("_fark")
                # Elektroforezi olan ziyareti tercih et
                zg = z[z["gamma_pct"].notna()] if "gamma_pct" in z.columns else z
                sec = (zg if len(zg) else z).iloc[0]
                secim = (sec, tip, ad_l)
                break
        if secim is None:
            durum = "LAB BULUNAMADI" if adaylar else "ISIM LAB'DA YOK"
            sonuc["lab_durum"].append(durum); sonuc["lab_ad"].append(""); sonuc["lab_gun_farki"].append(pd.NA)
            for k in lab_kol:
                sonuc[k].append(bos[k])
            continue
        sec, tip, ad_l = secim
        sonuc["lab_durum"].append("LAB ESLESTI" if tip == "tam" else "LAB ESLESTI (bulanik isim)")
        sonuc["lab_ad"].append(ad_l)
        sonuc["lab_gun_farki"].append(int((sec["t"] - r["_t"]).days))
        for k in lab_kol:
            sonuc[k].append(sec[k])

    for k, v in sonuc.items():
        b[k] = v

    # --- DEMOGRAFI: yas/cinsiyet icin pencere sarti gevsetilir ---------------
    # Lab degerleri icin +-30 gun sart (IFE'deki tabloyu yansitmali), ama cinsiyet
    # degismez ve yas hesaplanabilir. Pencere icinde ziyareti olmayan ama adi
    # LIS'te bulunan panellere en yakin ziyaretin yas/cinsiyeti yazilir; yas,
    # ziyaret ile IFE tarihi arasindaki yil farki kadar duzeltilir.
    if "yas" in b.columns and "cinsiyet" in b.columns:
        b["demografi_kaynak"] = ""
        pencere_ici = b["lab_durum"].fillna("").str.startswith("LAB ESLESTI")
        b.loc[pencere_ici, "demografi_kaynak"] = "pencere"
        ad_index = {ad_l: g for ad_l, g in ziyaret.groupby("ad_n")}
        for i, r in b[hedef & ~pencere_ici].iterrows():
            if pd.isna(r["_t"]) or not r["_ad"]:
                continue
            for ad_l, tip in ad_cache.get(r["_ad"], []):
                g = ad_index.get(ad_l)
                if g is None or not len(g):
                    continue
                fark = (g["t"] - r["_t"]).abs()
                j = fark.idxmin()
                yas = pd.to_numeric(g.loc[j, "yas"], errors="coerce")
                if pd.notna(yas):
                    yas = int(round(yas - (g.loc[j, "t"] - r["_t"]).days / 365.25))
                b.at[i, "yas"] = yas
                b.at[i, "cinsiyet"] = g.loc[j, "cinsiyet"]
                b.at[i, "lab_ad"] = ad_l
                b.at[i, "demografi_kaynak"] = (f"en yakin ziyaret ({int(fark.loc[j].days)} gun"
                                              f"{', bulanik isim' if tip != 'tam' else ''})")
                break

        dem = b[hedef & b["cinsiyet"].notna() & (b["cinsiyet"].astype(str).str.strip() != "")]
        dev_col = dem["device"].astype(str).str.upper() if "device" in dem.columns else pd.Series("INTERLAB", index=dem.index)
        print("\nDEMOGRAFI (defterle eslesen paneller icinde yas/cinsiyeti LIS'ten bulunanlar):")
        rows = []
        for grup, g in [("Tum", dem)] + list(dem.groupby(dev_col)):
            if not len(g):
                continue
            y = pd.to_numeric(g["yas"], errors="coerce")
            k = int((g["cinsiyet"].astype(str) == "Kadın").sum())
            n = len(g)
            rows.append({"grup": grup, "n_panel": n, "n_hasta": g["_ad"].nunique(),
                         "yas_ortanca": y.median(), "yas_q1": y.quantile(.25), "yas_q3": y.quantile(.75),
                         "yas_min": y.min(), "yas_max": y.max(), "kadin_n": k,
                         "kadin_pct": round(100 * k / n, 1),
                         "pencere_ici_n": int((g["demografi_kaynak"] == "pencere").sum())})
            print(f"  {grup:<10} n={n:4d} panel ({g['_ad'].nunique()} hasta) | yas ortanca {y.median():.0f} "
                  f"(IQR {y.quantile(.25):.0f}-{y.quantile(.75):.0f}; {y.min():.0f}-{y.max():.0f}) | "
                  f"kadin {k} (%{100 * k / n:.0f}) | +-{a.pencere} gun icinde: {int((g['demografi_kaynak'] == 'pencere').sum())}")
        eksik = b[hedef & ~b.index.isin(dem.index)]
        if len(eksik):
            print(f"  Yas/cinsiyeti bulunamayan defter-eslesmis panel: {len(eksik)} "
                  f"(LIS dokumu {ziyaret['t'].min().date()} - {ziyaret['t'].max().date()}; disindaki tarihler ve Helena icin yeni dokum gerekir)")
        pd.DataFrame(rows).to_csv("demografi_ozeti.csv", index=False, encoding="utf-8-sig")
        print("  -> demografi_ozeti.csv")

    # --- KLINIK TANI ISTEK LISTESI (LIS'e verilir) -----------------------------
    if a.tani_listesi:
        h = b[hedef].copy()
        h["_ife_tarih"] = pd.to_datetime(h["sayfa_tarih"], errors="coerce", dayfirst=True)
        lst = pd.DataFrame({
            "sira": range(1, len(h) + 1),
            "ad": h["ad"].fillna("").astype(str).str.strip(),
            "soyad": h["soyad"].fillna("").astype(str).str.strip(),
            "IFE_tarihi": h["_ife_tarih"].dt.strftime("%d.%m.%Y"),
            "cihaz": h["device"].astype(str) if "device" in h.columns else "",
            "defter_barkod": h["barkod"].fillna("").astype(str) if "barkod" in h.columns else "",
            "LIS_protokol": h["protokol"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True) if "protokol" in h.columns else "",
            "LIS_hasta_id": h["lis_hasta_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True) if "lis_hasta_id" in h.columns else "",
            "LIS_eslesme": h["lab_durum"].fillna("").astype(str),
            "kirpma_dosyasi": h["kirpma_dosyasi"].astype(str) if "kirpma_dosyasi" in h.columns else "",
            "KLINIK_TANI_ICD10 (IFE tarihi ±12 ay)": "",
            "TANI_TARIHI": "",
            "HEMATOLOJI_NOTU": "",
        }).sort_values(["soyad", "ad", "IFE_tarihi"]).reset_index(drop=True)
        lst["sira"] = range(1, len(lst) + 1)
        kapsam = pd.DataFrame({
            "olcut": ["panel (defterle eslesen)", "benzersiz hasta (ad+soyad)", "LIS'te bulunan (protokol var)",
                      "LIS'te bulunamayan (yeni dokum gerek)"],
            "deger": [len(lst), int((lst["ad"] + "|" + lst["soyad"]).nunique()),
                      int((lst["LIS_protokol"] != "").sum()), int((lst["LIS_protokol"] == "").sum())]})
        with pd.ExcelWriter("klinik_tani_istek_listesi.xlsx") as w:
            lst.to_excel(w, sheet_name="istek_listesi", index=False)
            kapsam.to_excel(w, sheet_name="kapsam", index=False)
            pd.DataFrame({"not": [
                "LIS birimine: her satir bir IFE paneli; ayni hasta birden fazla satirda olabilir.",
                "Istenen: IFE tarihinden 12 ay once ve 12 ay sonra girilmis hematolojik/ilgili ICD-10 tanilari "
                "(D47.2 MGUS, C90.0 multipl miyelom, C88.0 Waldenstrom, E85 amiloidoz, C90.1 plazma hucreli losemi, "
                "C90.2 ekstramedüller plazmasitom, D89.- vb.) ve tani tarihi.",
                "LIS_protokol / LIS_hasta_id dolu satirlar dogrudan bulunur; bos satirlar icin ad+soyad+tarih ile arama.",
                "Ayni dokumde YAS ve CINSIYET de istensin (2015 Ocak-Ekim ve 2020-2021 panelleri icin eksik).",
            ]}).to_excel(w, sheet_name="LIS_notu", index=False)
        print(f"\nKLINIK TANI ISTEK LISTESI -> klinik_tani_istek_listesi.xlsx: {len(lst)} panel, "
              f"{kapsam['deger'][1]} hasta, LIS'te bulunan {kapsam['deger'][2]}, bulunamayan {kapsam['deger'][3]}")

    b = b.drop(columns=["_t", "_ad"])
    b.to_excel(a.cikti, index=False)

    print(f"\n-> {a.cikti}")
    print("\nLAB ESLESME DURUMU (defterle eslesenler icinde):")
    for k, n in b.loc[hedef, "lab_durum"].value_counts().items():
        print(f"  {n:5d}  {k}")
    ok = b[b["lab_durum"].fillna("").str.startswith("LAB ESLESTI")]
    if len(ok):
        gf = pd.to_numeric(ok["lab_gun_farki"], errors="coerce")
        print(f"\n  Gun farki: ortanca {gf.abs().median():.0f}, en fazla {gf.abs().max():.0f}")
        for c in ("gamma_pct", "igg", "iga", "igm", "kl_orani"):
            if c in ok.columns:
                print(f"  {c:<12} dolu: {int(pd.to_numeric(ok[c], errors='coerce').notna().sum())}")
        bul = ok[ok["lab_durum"].str.contains("bulanik")]
        if len(bul):
            print(f"\n  {len(bul)} bulanik isim eslesmesi -- 'lab_ad' sutunundan kontrol edin:")
            for _, r in bul.head(5).iterrows():
                print(f"    defter: {r['ad']} {r['soyad']:<14} -> lab: {r['lab_ad']}")


if __name__ == "__main__":
    main()
