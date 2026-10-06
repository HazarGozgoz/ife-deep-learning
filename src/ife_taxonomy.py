# -*- coding: utf-8 -*-
"""
ife_taxonomy.py  --  v2.0  ÜÇ EKSENLİ TAKSONOMİ
=================================================
Tüm araçların (kesme, gözden geçirme, hata ayıklama, veri hazırlama,
eğitim) tek ortak tanı kaynağı.

NEDEN YENİDEN TASARLANDI
-------------------------
Eski 21 sınıflık DÜZ liste, aslında ÜÇ AYRI EKSENİ tek isme sıkıştırıyordu:
    SUSPICIOUS_BAND_ON_POLYCLONAL_BACKGROUND
    = kesinlik (şüpheli) + zemin (poliklonal) + bant (belirtilemiyor)
Bunun üç somut sonucu oldu (gerçek veride ölçüldü, n=2922 serum):

  1) ÇOKLUK KAYBI. Checkbox bir KÜME'dir; aynı kutu iki kez işaretlenemez.
     "İki ayrı IgA-Kappa bandı" kaydedilemiyordu. Sadece biklonalde değil:
       SUSPICIOUS_BAND_IgG_Kappa,_IgG_Kappa  -> 14 kayıt
       SUSPICIOUS_BAND_Kappa,_Kappa          ->  6 kayıt
  2) TİPLEME KAYBI. Serbest metne yazılan bant tipi, "gestalt" sınıflarda
     bilinçli olarak yok sayılıyordu (haklı bir sebeple: metinden bant
     çıkarmak yanlış pozitif üretiyordu). Ama bu, gerçek bilgiyi de attı:
       SUSPICIOUS_BAND         : 533 kaydın 314'ünde (%59) tipleme vardı
       POLYCLONAL_HYPERGAMMA.  : 252 kaydın 198'inde (%79) lane bilgisi vardı
  3) EKSEN KARIŞMASI. Hipogamaglobulinemi 848 kaydın 424'ünde BİR BANTLA
     BİRLİKTE (immünoparezi + M-protein: klasik myelom tablosu). Yani
     hipogama, klonalite ekseninin bir değeri değil, AYRI bir bulgudur.

YENİ YAPI (dört alan, birbirinden bağımsız)
--------------------------------------------
  A) pattern        : TEK seçim  -- klonalite paterni
  B) bands          : SIRALI liste, TEKRAR EDEBİLİR -- her bant için
                      (tip, kesinlik, bölge notu)
  C) flags          : çoklu -- hipogamaglobulinemi, poliklonal zemin
  D) poly_lanes     : çoklu -- poliklonal artışın görüldüğü lane'ler
                      (BANT DEĞİLDİR; ayrı bir çıktı ekseni)

KULLANICI KARARLARI (bu çalışmaya özgü, bilinçli seçimler)
------------------------------------------------------------
  * Şüpheli bantlar, bant hedefinde KESİN bantlarla AYNI sayılır
    (duyarlılık öncelikli). Kesinlik bilgisi yine de saklanır, böylece
    makalede "sadece kesin bantlar" alt analizi yapılabilir.
  * Poliklonal lane bilgisi modelin AYRI bir çıktı ekseni olarak tahmin
    edilir; bant hedefine karıştırılmaz.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ===========================================================================
# EKSEN A -- PATERN (tek seçim)
# ===========================================================================
# PATERN ARTIK ELLE SEÇİLMEZ -- BANTLARDAN TÜRETİLİR.
#
# Önceki sürümde patern ayrı bir radyo düğmesiydi, bantlar ayrı bir listeydi.
# İkisi çelişebildiği için "Biklonal seçildi ama 1 bant kaydedildi" gibi
# uyarılar çıkıyordu. Ama patern zaten bant sayısının kendisidir:
#     1 bant = monoklonal, 2 bant = biklonal, 3+ bant = oligoklonal
# Aynı bilgiyi iki kez sormak, yalnızca tutarsızlık üretiyordu. Göçte
# işaretlenen 116 kaydın 68'i tam olarak bu sahte çelişkiydi.
#
# Kullanıcı jele bakıp NE GÖRDÜĞÜNÜ girer (bantlar + genel bulgular);
# patern otomatik hesaplanır. Model tarafında patern yine ayrı bir çıktı
# başlığıdır -- CNN görüntüden kendisi tahmin eder.
#
# İsimler standart IFE raporlama diline geri alındı: "biklonal gammopati"
# ve "oligoklonal bantlar" hem klinik hem literatür karşılığıdır. Yalnızca
# ağır zincir sınıfı jel diline çevrildi (aşağıda BAND_CLASSES), çünkü
# "ağır zincir hastalığı" jelin tek başına kuramayacağı bir tanıdır.
PATTERN_CLASSES: list[str] = [
    "NO_MONOCLONAL_BAND",
    "POLYCLONAL_HYPERGAMMAGLOBULINEMIA",
    "SUSPICIOUS_BAND",
    "MONOCLONAL_GAMMOPATHY",
    "BICLONAL_GAMMOPATHY",
    "OLIGOCLONAL_BANDS",
    "ATYPICAL_PATTERN",
    "UNINTERPRETABLE_TECHNICAL_ISSUE",
]
PATTERN_TR: dict[str, str] = {
    "NO_MONOCLONAL_BAND": "Monoklonal bant izlenmedi",
    "POLYCLONAL_HYPERGAMMAGLOBULINEMIA": "Poliklonal hipergammaglobulinemi",
    "SUSPICIOUS_BAND": "Şüpheli bant",
    "MONOCLONAL_GAMMOPATHY": "Monoklonal gammopati (1 bant)",
    "BICLONAL_GAMMOPATHY": "Biklonal gammopati (2 bant)",
    "OLIGOCLONAL_BANDS": "Oligoklonal bantlar (≥3 bant)",
    "ATYPICAL_PATTERN": "Atipik patern",
    "UNINTERPRETABLE_TECHNICAL_ISSUE": "Yorumlanamadı (teknik neden)",
}

PATTERN_EN: dict[str, str] = {
    "NO_MONOCLONAL_BAND": "No monoclonal band",
    "POLYCLONAL_HYPERGAMMAGLOBULINEMIA": "Polyclonal hypergammaglobulinaemia",
    "SUSPICIOUS_BAND": "Suspicious band",
    "MONOCLONAL_GAMMOPATHY": "Monoclonal gammopathy",
    "BICLONAL_GAMMOPATHY": "Biclonal gammopathy",
    "OLIGOCLONAL_BANDS": "Oligoclonal bands",
    "ATYPICAL_PATTERN": "Atypical pattern",
    "UNINTERPRETABLE_TECHNICAL_ISSUE": "Uninterpretable (technical)",
}
FLAG_EN: dict[str, str] = {
    "HYPOGAMMAGLOBULINEMIA": "Hypogammaglobulinaemia",
    "POLYCLONAL_INCREASE": "Polyclonal increase",
}
BAND_EN_SHORT: dict[str, str] = {
    "IGG_KAPPA": "IgG-κ", "IGG_LAMBDA": "IgG-λ", "IGA_KAPPA": "IgA-κ", "IGA_LAMBDA": "IgA-λ",
    "IGM_KAPPA": "IgM-κ", "IGM_LAMBDA": "IgM-λ",
    "IGG_NO_LIGHT_CHAIN": "IgG without light chain", "IGA_NO_LIGHT_CHAIN": "IgA without light chain",
    "IGM_NO_LIGHT_CHAIN": "IgM without light chain",
    "FREE_KAPPA": "Free κ", "FREE_LAMBDA": "Free λ",
}

# ===========================================================================
# EKSEN B -- BANTLAR (sıralı liste, tekrar edebilir)
# ===========================================================================
BAND_CLASSES: list[str] = [
    "IGG_KAPPA", "IGG_LAMBDA",
    "IGA_KAPPA", "IGA_LAMBDA",
    "IGM_KAPPA", "IGM_LAMBDA",
    "IGG_NO_LIGHT_CHAIN", "IGA_NO_LIGHT_CHAIN", "IGM_NO_LIGHT_CHAIN",
    "FREE_KAPPA", "FREE_LAMBDA",
]
# ---------------------------------------------------------------------------
# BİR BANT, GÖRÜNDÜĞÜ LANE'LERLE TANIMLANIR
# ---------------------------------------------------------------------------
# Jelde bir bant, hizalı olarak hangi antiserum lane'lerinde çöküyorsa o
# lane'lerle tanımlanır:
#     {IgG, κ} -> IgG kappa monoklonal bant
#     {κ}      -> serbest kappa bandı
#     {IgG}    -> hafif zincir karşılığı izlenmeyen IgG bandı
#
# KESİNLİK ADI DEĞİŞTİRİR. Şüpheli bir bulguda klon kimliği İDDİA EDİLMEZ;
# yalnızca şüphenin görüldüğü lane söylenir -- literatürdeki kullanım da
# budur ("κ staining", "discrete IgG and kappa bands"). Yani:
#     {IgG} + kesin    -> "IgG bandı, hafif zincir karşılığı izlenmedi"
#     {IgG} + şüpheli  -> "IgG'de şüpheli bant"        (klon iddiası YOK)
#     {κ}   + şüpheli  -> "Kappa'da şüpheli bant"
# Eskiden şüpheli bantlar da klon adıyla yazılıyordu ("Serbest Kappa
# (Şüpheli)") -- bu, jelin söylemediği bir şeyi iddia ediyordu.
LANES: list[str] = ["IGG", "IGA", "IGM", "KAPPA", "LAMBDA"]
LANE_TR: dict[str, str] = {"IGG": "IgG", "IGA": "IgA", "IGM": "IgM",
                           "KAPPA": "Kappa", "LAMBDA": "Lambda"}
# Turkce bulunma eki, sesli uyumuna gore (IgG "i-ge" -> de, IgA "i-ge-a" -> da)
LANE_LOC_SUFFIX: dict[str, str] = {"IGG": "'de", "IGA": "'da", "IGM": "'de",
                                   "KAPPA": "'da", "LAMBDA": "'da"}

BAND_LANES: dict[str, tuple[str, ...]] = {
    "IGG_KAPPA": ("IGG", "KAPPA"), "IGG_LAMBDA": ("IGG", "LAMBDA"),
    "IGA_KAPPA": ("IGA", "KAPPA"), "IGA_LAMBDA": ("IGA", "LAMBDA"),
    "IGM_KAPPA": ("IGM", "KAPPA"), "IGM_LAMBDA": ("IGM", "LAMBDA"),
    "IGG_NO_LIGHT_CHAIN": ("IGG",), "IGA_NO_LIGHT_CHAIN": ("IGA",),
    "IGM_NO_LIGHT_CHAIN": ("IGM",), "FREE_KAPPA": ("KAPPA",),
    "FREE_LAMBDA": ("LAMBDA",),
}

# Kesin bantların adı (klon kimliği iddia edilebilir)
BAND_TR: dict[str, str] = {
    "IGG_KAPPA": "IgG Kappa", "IGG_LAMBDA": "IgG Lambda",
    "IGA_KAPPA": "IgA Kappa", "IGA_LAMBDA": "IgA Lambda",
    "IGM_KAPPA": "IgM Kappa", "IGM_LAMBDA": "IgM Lambda",
    "IGG_NO_LIGHT_CHAIN": "IgG bandı — hafif zincir karşılığı izlenmedi",
    "IGA_NO_LIGHT_CHAIN": "IgA bandı — hafif zincir karşılığı izlenmedi",
    "IGM_NO_LIGHT_CHAIN": "IgM bandı — hafif zincir karşılığı izlenmedi",
    "FREE_KAPPA": "Serbest Kappa", "FREE_LAMBDA": "Serbest Lambda",
}

# Yayın için İngilizce karşılıklar (literatür dili)
BAND_EN: dict[str, str] = {
    "IGG_KAPPA": "IgG kappa restricted band", "IGG_LAMBDA": "IgG lambda restricted band",
    "IGA_KAPPA": "IgA kappa restricted band", "IGA_LAMBDA": "IgA lambda restricted band",
    "IGM_KAPPA": "IgM kappa restricted band", "IGM_LAMBDA": "IgM lambda restricted band",
    "IGG_NO_LIGHT_CHAIN": "IgG band without light chain counterpart",
    "IGA_NO_LIGHT_CHAIN": "IgA band without light chain counterpart",
    "IGM_NO_LIGHT_CHAIN": "IgM band without light chain counterpart",
    "FREE_KAPPA": "free kappa restricted band", "FREE_LAMBDA": "free lambda restricted band",
}


def band_label(band_class: str, certainty: str = "definite", lang: str = "tr") -> str:
    """
    Bandın raporlanacak adı. KESİNLİĞE GÖRE DEĞİŞİR.

    Şüpheli bantta klon kimliği iddia edilmez; şüphenin görüldüğü lane(ler)
    söylenir:  {IGG} + şüpheli -> "IgG'de şüpheli bant"
    """
    if certainty == "definite":
        return (BAND_TR if lang == "tr" else BAND_EN).get(band_class, band_class)
    lanes = BAND_LANES.get(band_class, ())
    if not lanes:
        return "Şüpheli bant" if lang == "tr" else "equivocal band"
    if lang == "tr":
        govde = " ve ".join(LANE_TR[l] for l in lanes)
        return f"{govde}{LANE_LOC_SUFFIX[lanes[-1]]} şüpheli bant"
    return "faint/equivocal band in " + " and ".join(
        LANE_TR[l] for l in lanes) + (" lanes" if len(lanes) > 1 else " lane")
CERTAINTY_VALUES: tuple[str, str] = ("definite", "suspicious")
CERTAINTY_TR: dict[str, str] = {"definite": "Kesin", "suspicious": "Şüpheli"}

# ===========================================================================
# EKSEN C -- BAYRAKLAR (çoklu, klonaliteden bağımsız)
# ===========================================================================
# Bantların DIŞINDA işaretlenen bulgular. Bunlar elle seçilir; patern
# bunlardan ve bant sayısından otomatik türetilir.
# --- KLİNİK BULGULAR: modelin TAHMİN ETTİĞİ bayraklar ---
# "Poliklonal zemin" ve "poliklonal hipergammaglobulinemi" TEK KAVRAMDA
# birleştirildi. Verinizde ikisi aynı jel bulgusunun iki bağlamıydı:
# hipergammaglobulinemi bulgunun KENDİSİ olduğunda (221 kayıt bantsız),
# zemin ise bir bandın ARKA PLANI olduğunda (18 kayıt bantlı) işaretlenmişti.
# Patern artık otomatik hesaplandığı için bu ayrımı bayrağın taşımasına gerek
# yok: bant yoksa patern zaten "poliklonal hipergammaglobulinemi" olur, bant
# varsa patern bandı yansıtır ve bayrak zemini bildirir. Birleştirme, modele
# 252+28 ayrık yerine tek ve 274 örnekli bir sınıf verir.
FLAG_CLASSES: list[str] = [
    "HYPOGAMMAGLOBULINEMIA",
    "POLYCLONAL_INCREASE",
]
FLAG_TR: dict[str, str] = {
    "HYPOGAMMAGLOBULINEMIA": "Hipogamaglobulinemi",
    "POLYCLONAL_INCREASE": "Poliklonal (diffüz) artış",
}

# --- PATERNİ BELİRLEYEN SEÇİMLER ---
# Bunlar pencerede işaretlenir ve paterni belirler; AYRI bir model hedefi
# DEĞİLDİR (patern başlığı zaten tahmin ediyor). Eskiden hem patern hem
# bayrak listesindeydiler ve model aynı şeyi iki kez öğreniyordu.
PATTERN_SELECT_FLAGS: list[str] = ["ATYPICAL_PATTERN", "UNINTERPRETABLE_TECHNICAL_ISSUE"]
PATTERN_SELECT_TR: dict[str, str] = {
    "ATYPICAL_PATTERN": "Atipik patern",
    "UNINTERPRETABLE_TECHNICAL_ISSUE": "Yorumlanamadı (teknik neden)",
}

# --- KAYIT İŞARETLERİ: klinik bulgu DEĞİL, ASLA model hedefi olmaz ---
META_FLAGS: list[str] = ["UNTYPED_SUSPICIOUS_BAND", "BAND_COUNT_INCOMPLETE"]
META_TR: dict[str, str] = {
    "UNTYPED_SUSPICIOUS_BAND": "Şüpheli bant (tipi belirlenemedi)",
    "BAND_COUNT_INCOMPLETE": "Bant sayısı eksik kaydedilmiş",
}

ALL_FLAGS: list[str] = FLAG_CLASSES + PATTERN_SELECT_FLAGS + META_FLAGS
ALL_FLAG_TR: dict[str, str] = {**FLAG_TR, **PATTERN_SELECT_TR, **META_TR}

# ===========================================================================
# EKSEN D -- POLİKLONAL ARTIŞIN LANE'LERİ (çoklu; BANT DEĞİL)
# ===========================================================================
POLY_LANE_CLASSES: list[str] = ["IGG", "IGA", "IGM", "KAPPA", "LAMBDA"]
POLY_LANE_TR: dict[str, str] = {
    "IGG": "IgG", "IGA": "IgA", "IGM": "IgM", "KAPPA": "Kappa", "LAMBDA": "Lambda",
}

# Klinik olarak kritik bantlar: hepsi monoklonal süreç işareti taşır.
CRITICAL_BANDS: frozenset[str] = frozenset(BAND_CLASSES)
CRITICAL_PATTERNS: frozenset[str] = frozenset({
    "MONOCLONAL_GAMMOPATHY", "BICLONAL_GAMMOPATHY", "OLIGOCLONAL_BANDS"})

# (ağır, hafif) imzası -- klon sayımı için
_CLONE_SIGNATURES: dict[str, tuple[str | None, str | None]] = {
    "IGG_KAPPA": ("IGG", "KAPPA"), "IGG_LAMBDA": ("IGG", "LAMBDA"),
    "IGA_KAPPA": ("IGA", "KAPPA"), "IGA_LAMBDA": ("IGA", "LAMBDA"),
    "IGM_KAPPA": ("IGM", "KAPPA"), "IGM_LAMBDA": ("IGM", "LAMBDA"),
    "IGG_NO_LIGHT_CHAIN": ("IGG", None),
    "IGA_NO_LIGHT_CHAIN": ("IGA", None),
    "IGM_NO_LIGHT_CHAIN": ("IGM", None),
    "FREE_KAPPA": (None, "KAPPA"), "FREE_LAMBDA": (None, "LAMBDA"),
}


# ===========================================================================
# KAYIT NESNESİ
# ===========================================================================
@dataclass
class Band:
    band_class: str
    certainty: str = "definite"
    note: str = ""

    def to_token(self) -> str:
        t = f"{self.band_class}:{self.certainty}"
        return f"{t}:{self.note.replace('|', '/').replace(':', '-')}" if self.note else t

    @staticmethod
    def from_token(tok: str) -> "Band | None":
        parts = tok.split(":")
        if not parts or parts[0].strip() not in BAND_CLASSES:
            return None
        cert = parts[1].strip() if len(parts) > 1 and parts[1].strip() in CERTAINTY_VALUES else "definite"
        note = parts[2].strip() if len(parts) > 2 else ""
        return Band(parts[0].strip(), cert, note)


# Patern siralamasi: uzman beyani ile hesaplanan degeri karsilastirmak icin.
_PATTERN_RANK = {"NO_MONOCLONAL_BAND": 0, "POLYCLONAL_HYPERGAMMAGLOBULINEMIA": 0,
                 "SUSPICIOUS_BAND": 1, "MONOCLONAL_GAMMOPATHY": 2,
                 "BICLONAL_GAMMOPATHY": 3, "OLIGOCLONAL_BANDS": 4}


@dataclass
class IFERecord:
    # DİKKAT: elle atanmaz. IFERecord.resolve_pattern() ile hesaplanır.
    pattern: str = "NO_MONOCLONAL_BAND"
    bands: list[Band] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    poly_lanes: set[str] = field(default_factory=set)
    note: str = ""
    # Uzmanın ESKİ kayıtta açıkça yazdığı patern (varsa). Bant listesinden
    # hesaplanan değerle çelişirse UZMAN BEYANI KAZANIR -- ikinci bandın
    # kimliği kaydedilmemiş olabilir, ama uzman jelde iki bant görmüştür.
    declared_pattern: str = ""

    # --- serileştirme ---
    def bands_str(self) -> str:
        return "|".join(b.to_token() for b in self.bands)

    def flags_str(self) -> str:
        return "|".join(sorted(self.flags))

    def poly_lanes_str(self) -> str:
        return "|".join(sorted(self.poly_lanes))

    # --- model hedefleri ---
    def band_target(self) -> set[str]:
        """Bant çoklu-etiket hedefi. KULLANICI KARARI: şüpheli bantlar kesin
        bantlarla AYNI sayılır (duyarlılık öncelikli)."""
        return {b.band_class for b in self.bands}

    def band_target_definite_only(self) -> set[str]:
        """Makalede 'sadece kesin bantlar' duyarlılık analizi için."""
        return {b.band_class for b in self.bands if b.certainty == "definite"}

    def n_bands(self) -> int:
        return len(self.bands)

    def resolve_pattern(self) -> str:
        """
        Paterni bantlardan/bayraklardan hesaplar.

        AMA uzman eski kayıtta açıkça "biklonal" ya da "oligoklonal" dediyse
        ve hesaplanan değer bundan DÜŞÜKSE, uzmanın beyanı korunur: eski araç
        ikinci bandın kimliğini sormuyordu, bu yüzden bant listesi eksik
        olabilir -- ama uzman jelde o bantları görmüştür. Kayıt
        BAND_COUNT_INCOMPLETE ile işaretlenir ki eksik bant sonradan
        girilebilsin.
        """
        hesaplanan = derive_pattern(self.bands, self.flags)
        d = self.declared_pattern
        if (d in ("BICLONAL_GAMMOPATHY", "OLIGOCLONAL_BANDS")
                and _PATTERN_RANK.get(hesaplanan, 0) < _PATTERN_RANK.get(d, 0)):
            self.pattern = d
            self.flags.add("BAND_COUNT_INCOMPLETE")
        else:
            self.pattern = hesaplanan
            # Bantlar artik yeterli: eksiklik bayragi KENDILIGINDEN duser.
            # Yoksa gecmiste konmus bayrak, siz eksik bandi ekledikten sonra
            # bile uyari uretmeye devam ederdi ("2 bant isaretlenmis ama
            # yalnizca 2 bant kaydedilmis" gibi anlamsiz bir mesaj).
            self.flags.discard("BAND_COUNT_INCOMPLETE")
        return self.pattern

    def has_suspicious(self) -> bool:
        return any(b.certainty == "suspicious" for b in self.bands)


def parse_bands(text) -> list[Band]:
    if not isinstance(text, str) or not text.strip():
        return []
    out = []
    for tok in text.split("|"):
        b = Band.from_token(tok.strip())
        if b:
            out.append(b)
    return out


def parse_set_field(text, allowed: list[str]) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    return {t.strip().upper() for t in text.split("|") if t.strip().upper() in allowed}


# ===========================================================================
# KLON SAYIMI VE PATERN TÜRETME
# ===========================================================================
def count_distinct_clones(bands: list[Band]) -> int:
    """
    Kaç FARKLI klon var?

    Kural 1: Aynı hafif zincire sahip serbest hafif zincir + intakt Ig
             (IgG-Kappa + serbest Kappa) TEK klondur -- bir klonun hem intakt
             hem fazla serbest hafif zincir üretmesi sık görülür. Eski kod
             bunu yanlışlıkla 2 klon sayıp oligoklonal kategorisini şişiriyordu.
    Kural 2: Aynı ağır zincir + FARKLI hafif zincir (IgG-Kappa + IgG-Lambda)
             2 klondur. Eski kod "2 farklı ağır zincir" şartı aradığı için
             bunu kaçırıyordu; oysa biklonal olguların önemli bir kısmı tam
             olarak bu paterndedir (Bakshi & Anand, Lab Med 2019).
    Kural 3: AYNI tipten iki ayrı bant (iki IgA-Kappa, farklı mobilite) da
             2 klondur -- artık kaydedilebiliyor, bu yüzden sayılabiliyor.
    """
    sigs = [_CLONE_SIGNATURES[b.band_class] for b in bands if b.band_class in _CLONE_SIGNATURES]
    intact = [s for s in sigs if s[0] is not None]
    free_only = [s for s in sigs if s[0] is None]
    intact_lcs = {s[1] for s in intact if s[1] is not None}
    n = len(intact)  # tekrarlar KORUNUR (Kural 3)
    for s in free_only:
        if s[1] not in intact_lcs:
            n += 1
    return n


def derive_pattern(bands: list[Band], flags: set[str]) -> str:
    """
    Paterni bantlardan ve bayraklardan HESAPLAR. Elle seçilmez.

    Öncelik: teknik sorun > klonal bant sayısı > poliklonal > normal.
    Yorumlanamayan bir jelde bant sayısı anlamsızdır, o yüzden en üstte.
    """
    if "UNINTERPRETABLE_TECHNICAL_ISSUE" in flags:
        return "UNINTERPRETABLE_TECHNICAL_ISSUE"

    # KRİTİK: monoklonal/biklonal/oligoklonal SADECE KESİN bantlardan sayılır.
    # Şüpheli bant, tanımı gereği "monoklonal diyemediğim bant"tır; onu klon
    # olarak saymak yorumu olduğundan güçlü gösterir. Standart IFE
    # raporlamasında "şüpheli bant" kendi başına bir kategoridir.
    definite = [b for b in bands if b.certainty == "definite"]
    n = count_distinct_clones(definite)
    if n >= 3:
        return "OLIGOCLONAL_BANDS"
    if n == 2:
        return "BICLONAL_GAMMOPATHY"
    if n == 1:
        return "MONOCLONAL_GAMMOPATHY"

    # Kesin bant yok ama şüpheli bant var
    # POLİKLONAL ZEMİN PATERNE GİRMEZ -- bağımsız bir bayraktır ve HER
    # paternle birlikte bulunabilir (MGUS'ta da, biklonalde de, oligoklonalde
    # de). Eskiden yalnızca şüpheli bantta patern adına gömülüydü; bu tutarsızdı.
    if any(b.certainty == "suspicious" for b in bands) or "UNTYPED_SUSPICIOUS_BAND" in flags:
        return "SUSPICIOUS_BAND"

    if "ATYPICAL_PATTERN" in flags:
        return "ATYPICAL_PATTERN"
    if "POLYCLONAL_INCREASE" in flags:
        return "POLYCLONAL_HYPERGAMMAGLOBULINEMIA"
    return "NO_MONOCLONAL_BAND"


def check_consistency(rec: IFERecord) -> str:
    """
    Patern artık bantlardan türetildiği için "patern ile bant sayısı
    çelişiyor" diye bir durum KALMADI -- o çelişki, aynı bilgiyi iki kez
    sormanın yan etkisiydi. Geriye tek bir gerçek eksiklik kontrolü kalıyor.
    """
    # Tipi belirlenemeyen şüpheli bant EKSİK KAYIT DEĞİLDİR: bant şüpheli
    # olduğu için zaten tiplenememiştir, "şüpheli bant, tip belirlenemedi"
    # geçerli ve tamamlanmış bir yorumdur. Gözden geçirme listesine alınmaz.
    # Bayragin VARLIGINA degil, gercek bant sayisina bak: bayrak gecmiste
    # konmus olabilir ama siz eksik bandi eklemis olabilirsiniz.
    _gerekli = {"BICLONAL_GAMMOPATHY": 2, "OLIGOCLONAL_BANDS": 3}
    if rec.pattern in _gerekli:
        n = count_distinct_clones([b for b in rec.bands if b.certainty == "definite"])
        if n < _gerekli[rec.pattern]:
            return (f"'{PATTERN_TR[rec.pattern]}' işaretlenmiş ama {n} kesin bant "
                    f"kaydedilmiş — eksik bandı jele bakıp ekleyin.")
    if "POLYCLONAL_INCREASE" in rec.flags and not rec.poly_lanes:
        return "Poliklonal artış işaretli ama hangi lane'lerde olduğu belirtilmedi."
    return ""


# ===========================================================================
# ESKİ (v1, 21 düz sınıf) VERİYİ YENİ YAPIYA ÇEVİRME
# ===========================================================================
# Bu bölüm, sizin mevcut 2922 serum kaydınızı YENİDEN İŞARETLEMEDEN yeni
# yapıya taşır. Serbest metinde saklı kalan tipleme/lane bilgisini de geri
# kazanır (SUSPICIOUS_BAND'lerin %59'u, POLYCLONAL'lerin %79'u).

_LEGACY_PREFIXES = ("SERUM_", "MONOCLONAL_", "URINE_")

# Eski düz sınıf adları (uzundan kısaya -- kısa olan uzunla eşleşmesin)
_LEGACY_BASES = sorted([
    "NON_SPECIFIC_FINDINGS", "POLYCLONAL_HYPERGAMMAGLOBULINEMIA", "HYPOGAMMAGLOBULINEMIA",
    "IGG_HEAVY_CHAIN_ONLY", "IGA_HEAVY_CHAIN_ONLY", "IGM_HEAVY_CHAIN_ONLY",
    "IGG_NO_LIGHT_CHAIN", "IGA_NO_LIGHT_CHAIN", "IGM_NO_LIGHT_CHAIN",
    "IGG_KAPPA", "IGG_LAMBDA", "IGA_KAPPA", "IGA_LAMBDA", "IGM_KAPPA", "IGM_LAMBDA",
    "FREE_KAPPA", "FREE_LAMBDA", "BICLONAL_GAMMOPATHY", "OLIGOCLONAL_BANDS",
    "SUSPICIOUS_BAND_ON_POLYCLONAL_BACKGROUND", "SUSPICIOUS_BAND",
    "ATYPICAL_PATTERN", "UNINTERPRETABLE_TECHNICAL_ISSUE", "OTHER",
], key=len, reverse=True)

# Serbest metindeki bant tiplemesi -> kanonik bant sınıfı.
# Sıra önemli: iki anahtarlı desenler tek anahtarlılardan ÖNCE denenir,
# yoksa "IGG KAPPA" yanlışlıkla "IGG (ağır zincir tek başına)" olur.
_TEXT_BAND_PATTERNS: list[tuple[tuple[str, ...], str]] = [
    (("IGG", "KAPPA"), "IGG_KAPPA"), (("IGG", "LAMBDA"), "IGG_LAMBDA"),
    (("IGA", "KAPPA"), "IGA_KAPPA"), (("IGA", "LAMBDA"), "IGA_LAMBDA"),
    (("IGM", "KAPPA"), "IGM_KAPPA"), (("IGM", "LAMBDA"), "IGM_LAMBDA"),
    (("FREE", "KAPPA"), "FREE_KAPPA"), (("FREE", "LAMBDA"), "FREE_LAMBDA"),
    (("SERBEST", "KAPPA"), "FREE_KAPPA"), (("SERBEST", "LAMBDA"), "FREE_LAMBDA"),
    (("IGG",), "IGG_NO_LIGHT_CHAIN"),
    (("IGA",), "IGA_NO_LIGHT_CHAIN"),
    (("IGM",), "IGM_NO_LIGHT_CHAIN"),
    (("KAPPA",), "FREE_KAPPA"), (("LAMBDA",), "FREE_LAMBDA"),
]

# NOT: Eskiden burada "sadece ağır zincir yazılmışsa belirsiz" diye bir uyarı
# vardı. O uyarı YANLIŞTI: bu çalışma JEL YORUMLUYOR, hastalık teşhis etmiyor.
# "IgG bandı var, hafif zincir karşılığı izlenmedi" jelde görülen şeyin tam
# kendisidir; belirsiz değildir. Belirsizlik, sınıfın İngilizce adının
# ("heavy chain only") jelin kuramayacağı bir hastalığı ima etmesinden
# kaynaklanıyordu. Ad düzeltildi, uyarı kaldırıldı -- 153 kayıt gözden
# geçirme listesinden çıktı.


# Eski sınıf adı -> yeni jel-dili adı. Eski CSV'deki "IGG_HEAVY_CHAIN_ONLY"
# etiketleri bu haritayla yeni "IGG_NO_LIGHT_CHAIN" bandına düşer; olmazsa
# sessizce kaybolurlardı.
LEGACY_BAND_RENAME: dict[str, str] = {
    "IGG_HEAVY_CHAIN_ONLY": "IGG_NO_LIGHT_CHAIN",
    "IGA_HEAVY_CHAIN_ONLY": "IGA_NO_LIGHT_CHAIN",
    "IGM_HEAVY_CHAIN_ONLY": "IGM_NO_LIGHT_CHAIN",
}


def _strip_prefixes(token: str) -> str:
    t = token.strip().upper().replace(" ", "_")
    changed = True
    while changed:
        changed = False
        for pre in _LEGACY_PREFIXES:
            if t.startswith(pre):
                t = t[len(pre):]
                changed = True
    return t


def split_legacy_token(token: str) -> tuple[str | None, str]:
    """'SERUM_SUSPICIOUS_BAND_IgG_Kappa' -> ('SUSPICIOUS_BAND', 'IGG_KAPPA')"""
    t = _strip_prefixes(token)
    for base in _LEGACY_BASES:
        if t == base:
            return base, ""
        if t.startswith(base + "_"):
            return base, t[len(base) + 1:]
    return None, t


def parse_band_text(text: str) -> list[str]:
    """
    Serbest metinden SIRALI bant listesi çıkarır. Virgül bantları ayırır;
    TEKRARLAR KORUNUR ('IgA_Kappa,_IgA_Kappa' -> iki ayrı IgA-Kappa bandı).
    """
    if not text:
        return []
    body = text.upper().replace("_", " ").replace("-", " ")
    out: list[str] = []
    for chunk in body.split(","):
        chunk = " " + " ".join(chunk.split()) + " "
        if not chunk.strip():
            continue
        for keys, cls in _TEXT_BAND_PATTERNS:
            if all(f" {k} " in chunk or chunk.strip() == k or k in chunk for k in keys):
                out.append(cls)
                break
    return out


def parse_lane_text(text: str) -> set[str]:
    """Poliklonal lane bilgisi ('IGG_KAPPA' -> {IGG, KAPPA}). Bant DEĞİL."""
    if not text:
        return set()
    u = text.upper()
    return {lane for lane in POLY_LANE_CLASSES if lane in u}


UNTYPED_BAND_NOTE = "tipi belirtilmemiş şüpheli bant"


def from_legacy(all_diagnoses, primary_diagnosis="") -> tuple[IFERecord, list[str]]:
    """
    Eski 'all_diagnoses' hücresini (';' ayraçlı) yeni IFERecord'a çevirir.
    primary_diagnosis, düzeltme sonrası bayatlamış olabilir ama içindeki
    serbest metin bazen all_diagnoses'ta olmayan bilgi taşır -- bant/lane
    metni orada bulunursa YEDEK kaynak olarak kullanılır.

    Döner: (kayıt, kurtarma notları)
    """
    rec = IFERecord()
    notes: list[str] = []
    definite_bands: list[str] = []
    suspicious_bands: list[str] = []
    saw_suspicious_class = False

    tokens = [t.strip() for t in str(all_diagnoses).split(";") if t.strip()]
    for tok in tokens:
        base, detail = split_legacy_token(tok)
        if base is None:
            continue
        if base == "NON_SPECIFIC_FINDINGS":
            pass  # bant yoksa patern zaten NO_MONOCLONAL_BAND olur
        elif base == "HYPOGAMMAGLOBULINEMIA":
            rec.flags.add("HYPOGAMMAGLOBULINEMIA")
        elif base == "POLYCLONAL_HYPERGAMMAGLOBULINEMIA":
            rec.flags.add("POLYCLONAL_INCREASE")
            rec.poly_lanes |= parse_lane_text(detail)   # <- 198 kayıt kurtarılır
            if detail:
                notes.append("poliklonal lane metinden")
        elif base == "SUSPICIOUS_BAND_ON_POLYCLONAL_BACKGROUND":
            saw_suspicious_class = True
            rec.flags.add("POLYCLONAL_INCREASE")  # zemin = artış (birleştirildi)
            got = parse_band_text(detail)
            suspicious_bands.extend(got)
            if got:
                notes.append("şüpheli bant tiplemesi metinden")
        elif base == "SUSPICIOUS_BAND":
            saw_suspicious_class = True
            got = parse_band_text(detail)               # <- 314 kayıt kurtarılır
            suspicious_bands.extend(got)
            if got:
                notes.append("şüpheli bant tiplemesi metinden")
        elif base == "BICLONAL_GAMMOPATHY":
            rec.declared_pattern = "BICLONAL_GAMMOPATHY"
            got = parse_band_text(detail)
            if got:
                definite_bands.extend(got)
                notes.append("biklonal klonları metinden")
        elif base == "OLIGOCLONAL_BANDS":
            rec.declared_pattern = "OLIGOCLONAL_BANDS"
            got = parse_band_text(detail)
            if got:
                definite_bands.extend(got)
        elif base == "ATYPICAL_PATTERN":
            rec.flags.add("ATYPICAL_PATTERN")
        elif base == "UNINTERPRETABLE_TECHNICAL_ISSUE":
            rec.flags.add("UNINTERPRETABLE_TECHNICAL_ISSUE")
        elif base == "OTHER":
            rec.note = (rec.note + " | OTHER").strip(" |")
        elif base in BAND_CLASSES or base in LEGACY_BAND_RENAME:
            definite_bands.append(LEGACY_BAND_RENAME.get(base, base))

    # Yedek kaynak: bayatlamış primary_diagnosis içindeki serbest metin
    if not definite_bands and not suspicious_bands and primary_diagnosis:
        pbase, pdetail = split_legacy_token(str(primary_diagnosis))
        if pbase in ("BICLONAL_GAMMOPATHY", "OLIGOCLONAL_BANDS", "SUSPICIOUS_BAND"):
            got = parse_band_text(pdetail)
            if got:
                (suspicious_bands if pbase == "SUSPICIOUS_BAND" else definite_bands).extend(got)
                notes.append("eski primary_diagnosis metninden")

    # Şüpheli sınıf işaretli ama hiç tip çıkmadıysa, tipi bilinmeyen bir
    # şüpheli bant olduğunu KAYBETMEMEK için not düşülür.
    if saw_suspicious_class and not suspicious_bands and not definite_bands:
        # Tipi hiç yazılmamış şüpheli bant. Bant listesi boş kalır ama
        # paternin "şüpheli bant" olduğunu KAYBETMEMEK için bayrak konur.
        rec.note = (rec.note + " | " + UNTYPED_BAND_NOTE).strip(" |")
        rec.flags.add("UNTYPED_SUSPICIOUS_BAND")

    # Aynı bant hem kesin hem şüpheli geldiyse KESİN kazanır (tekrarı önler)
    for b in definite_bands:
        rec.bands.append(Band(b, "definite"))
    definite_set = set(definite_bands)
    for b in suspicious_bands:
        if b not in definite_set:
            rec.bands.append(Band(b, "suspicious"))

    # Patern HER ZAMAN hesaplanır -- eski kayıtta ne yazdığından bağımsız.
    rec.resolve_pattern()
    return rec, notes


def to_legacy_labels(rec: IFERecord) -> set[str]:
    """Geriye dönük uyumluluk: yeni kayıttan eski düz sınıf kümesi üretir."""
    out: set[str] = set()
    if rec.pattern == "NO_MONOCLONAL_BAND":
        out.add("NON_SPECIFIC_FINDINGS")
    elif rec.pattern != "MONOCLONAL_GAMMOPATHY":
        out.add(rec.pattern)
    out |= rec.band_target()
    if "HYPOGAMMAGLOBULINEMIA" in rec.flags:
        out.add("HYPOGAMMAGLOBULINEMIA")
    if "POLYCLONAL_INCREASE" in rec.flags:
        out.add("POLYCLONAL_HYPERGAMMAGLOBULINEMIA")
    if rec.has_suspicious():
        out.add("SUSPICIOUS_BAND_ON_POLYCLONAL_BACKGROUND"
                if "POLYCLONAL_BACKGROUND" in rec.flags else "SUSPICIOUS_BAND")
    return out


# ===========================================================================
# KENDİ KENDİNİ TEST  (python ife_taxonomy.py)
# ===========================================================================
if __name__ == "__main__":
    # --- eksen ayrımı: hipogama bantla birlikte olabilmeli ---
    r, _ = from_legacy("HYPOGAMMAGLOBULINEMIA;IGG_KAPPA")
    assert r.pattern == "MONOCLONAL_GAMMOPATHY", r.pattern
    assert "HYPOGAMMAGLOBULINEMIA" in r.flags
    assert r.band_target() == {"IGG_KAPPA"}

    # --- 314 kayıtlık kurtarma: şüpheli bandın tiplemesi geri geliyor ---
    r, notes = from_legacy("SERUM_SUSPICIOUS_BAND_IgG_Kappa")
    assert r.band_target() == {"IGG_KAPPA"}, r.band_target()
    assert r.bands[0].certainty == "suspicious"
    # ŞÜPHELİ BANT MONOKLONAL GAMMOPATİ DEĞİLDİR -- kendi kategorisidir
    assert r.pattern == "SUSPICIOUS_BAND", r.pattern

    # --- çokluk: iki ayrı aynı-tip şüpheli bant ---
    r, _ = from_legacy("SERUM_SUSPICIOUS_BAND_IgG_Kappa,_IgG_Kappa")
    assert r.n_bands() == 2, r.bands
    assert r.pattern == "SUSPICIOUS_BAND", r.pattern  # ikisi de şüpheli

    # --- 198 kayıtlık kurtarma: poliklonal lane bilgisi BANT DEĞİL ---
    r, _ = from_legacy("SERUM_POLYCLONAL_HYPERGAMMAGLOBULINEMIA_IgG_Kappa")
    assert r.poly_lanes == {"IGG", "KAPPA"}, r.poly_lanes
    assert r.band_target() == set(), "poliklonal lane BANT olarak okunmamalı!"
    assert r.pattern == "POLYCLONAL_HYPERGAMMAGLOBULINEMIA"

    # --- poliklonal zemin ayrı bir bayrak ---
    r, _ = from_legacy("SUSPICIOUS_BAND_ON_POLYCLONAL_BACKGROUND_IgG_Kappa")
    assert "POLYCLONAL_INCREASE" in r.flags
    assert r.band_target() == {"IGG_KAPPA"}
    assert r.pattern == "SUSPICIOUS_BAND", r.pattern   # zemin ayrı bayrak
    # Kesin bant + şüpheli bant: patern KESİN banda göre
    r, _ = from_legacy("IGG_KAPPA;SERUM_SUSPICIOUS_BAND_IgA_Lambda")
    assert r.pattern == "MONOCLONAL_GAMMOPATHY", r.pattern
    assert r.band_target() == {"IGG_KAPPA", "IGA_LAMBDA"}

    # --- biklonal: aynı ağır zincir + farklı hafif zincir = 2 klon ---
    r, _ = from_legacy("BICLONAL_GAMMOPATHY;IGG_KAPPA;IGG_LAMBDA")
    assert count_distinct_clones(r.bands) == 2

    # --- aynı klonun serbest hafif zinciri 2. klon SAYILMAZ ---
    r, _ = from_legacy("IGG_KAPPA;FREE_KAPPA")
    assert count_distinct_clones(r.bands) == 1, r.bands
    assert r.pattern == "MONOCLONAL_GAMMOPATHY"
    r2, _ = from_legacy("SERUM_SUSPICIOUS_BAND")
    assert r2.pattern == "SUSPICIOUS_BAND", r2.pattern

    # --- eşleşmeyen serbest hafif zincir 2. klon SAYILIR ---
    r, _ = from_legacy("IGG_KAPPA;FREE_LAMBDA")
    assert count_distinct_clones(r.bands) == 2

    # --- kesin bant, aynı tipteki şüpheliyi bastırır (tekrar üretmez) ---
    r, _ = from_legacy("IGG_KAPPA;SERUM_SUSPICIOUS_BAND_IgG_Kappa")
    assert r.n_bands() == 1 and r.bands[0].certainty == "definite"

    # --- KULLANICI KARARI: şüpheli bantlar hedefte kesinlerle aynı sayılır ---
    r, _ = from_legacy("SERUM_SUSPICIOUS_BAND_IgA")
    assert r.band_target() == {"IGA_NO_LIGHT_CHAIN"}
    assert r.band_target_definite_only() == set()

    # --- serileştirme gidiş-dönüş, tekrarlar korunarak ---
    r, _ = from_legacy("SERUM_SUSPICIOUS_BAND_Kappa,_Kappa")
    assert [b.band_class for b in parse_bands(r.bands_str())] == ["FREE_KAPPA", "FREE_KAPPA"]

    # --- tutarlılık denetimi ---
    # Patern türetildiği için "patern-bant çelişkisi" ARTIK OLUŞAMAZ
    r = IFERecord(bands=[Band("IGG_KAPPA")]); r.resolve_pattern()
    assert r.pattern == "MONOCLONAL_GAMMOPATHY" and check_consistency(r) == ""
    r = IFERecord(bands=[Band("IGA_KAPPA"), Band("IGA_KAPPA")]); r.resolve_pattern()
    assert r.pattern == "BICLONAL_GAMMOPATHY", r.pattern
    r = IFERecord(bands=[Band("IGG_KAPPA"), Band("IGA_LAMBDA"), Band("IGM_KAPPA")])
    r.resolve_pattern()
    assert r.pattern == "OLIGOCLONAL_BANDS", r.pattern
    # Yorumlanamayan jelde bant sayısı paterni belirlemez
    r = IFERecord(bands=[Band("IGG_KAPPA")], flags={"UNINTERPRETABLE_TECHNICAL_ISSUE"})
    assert r.resolve_pattern() == "UNINTERPRETABLE_TECHNICAL_ISSUE"
    # Tek gerçek eksiklik kontrolü: poliklonal işaretli ama lane yok
    r = IFERecord(flags={"POLYCLONAL_INCREASE"})
    assert check_consistency(r) != ""
    # Tipi belirlenemeyen şüpheli bant GEÇERLİ bir yorumdur, listeye düşmez
    r2, _ = from_legacy("SERUM_SUSPICIOUS_BAND")
    assert r2.pattern == "SUSPICIOUS_BAND"
    assert check_consistency(r2) == "", "tipsiz şüpheli bant eksik sayılmamalı"
    # UZMAN BEYANI hesaplamayı ezer: biklonal dendi, tek bant kaydedilmiş
    r, _ = from_legacy("BICLONAL_GAMMOPATHY;IGA_KAPPA")
    assert r.pattern == "BICLONAL_GAMMOPATHY", r.pattern
    assert "BAND_COUNT_INCOMPLETE" in r.flags
    assert check_consistency(r) != ""
    # Bantlar tamsa bayrak KONMAZ
    r, _ = from_legacy("BICLONAL_GAMMOPATHY;IGA_KAPPA;IGG_LAMBDA")
    assert r.pattern == "BICLONAL_GAMMOPATHY"
    assert "BAND_COUNT_INCOMPLETE" not in r.flags
    # Bant sayisi beyandan FAZLAysa hesaplanan kazanir (oligoklonal)
    r, _ = from_legacy("BICLONAL_GAMMOPATHY;IGG_KAPPA;IGA_LAMBDA;IGM_KAPPA")
    assert r.pattern == "OLIGOCLONAL_BANDS", r.pattern

    # Ayni tipten IKI KESIN bant = gercek biklonal, uyari VERMEMELI
    _b1 = IFERecord(bands=[Band("IGA_KAPPA", "definite"), Band("IGA_KAPPA", "definite")])
    _b1.resolve_pattern()
    assert _b1.pattern == "BICLONAL_GAMMOPATHY", _b1.pattern
    assert check_consistency(_b1) == "", check_consistency(_b1)
    # Gecmisten kalan eksiklik bayragi, bantlar tamamlaninca DUSMELI
    _b2 = IFERecord(bands=[Band("IGA_KAPPA", "definite"), Band("IGA_KAPPA", "definite")],
                    flags={"BAND_COUNT_INCOMPLETE"})
    _b2.declared_pattern = "BICLONAL_GAMMOPATHY"
    _b2.resolve_pattern()
    assert "BAND_COUNT_INCOMPLETE" not in _b2.flags, _b2.flags
    assert check_consistency(_b2) == "", check_consistency(_b2)
    # Gercekten eksikse uyari SURMELI
    _b3 = IFERecord(bands=[Band("IGA_KAPPA", "definite")])
    _b3.declared_pattern = "BICLONAL_GAMMOPATHY"
    _b3.resolve_pattern()
    assert check_consistency(_b3) != ""

    # Poliklonal zemin HER paternle birlikte olabilir
    r3 = IFERecord(bands=[Band("IGG_KAPPA")], flags={"POLYCLONAL_INCREASE"})
    assert r3.resolve_pattern() == "MONOCLONAL_GAMMOPATHY"
    r4 = IFERecord(bands=[Band("IGG_KAPPA"), Band("IGA_LAMBDA")],
                   flags={"POLYCLONAL_INCREASE"}, poly_lanes={"IGG"})
    assert r4.resolve_pattern() == "BICLONAL_GAMMOPATHY"
    assert check_consistency(r4) == ""
    r.poly_lanes = {"IGG"}
    assert check_consistency(r) == ""

    # --- geriye dönük uyumluluk ---
    r, _ = from_legacy("HYPOGAMMAGLOBULINEMIA;IGG_KAPPA")
    assert to_legacy_labels(r) == {"HYPOGAMMAGLOBULINEMIA", "IGG_KAPPA"}

    # --- JEL DİLİ: "IgG bandı, hafif zincir izlenmedi" belirsiz DEĞİLDİR ---
    # (Bu bir jel yorumlama çalışması; ağır zincir hastalığı iddiası yok.)
    r, _ = from_legacy("SERUM_SUSPICIOUS_BAND_IgG")
    assert r.band_target() == {"IGG_NO_LIGHT_CHAIN"}
    assert check_consistency(r) == "", "jel bulgusu belirsiz sayılmamalı"
    assert r.pattern == "SUSPICIOUS_BAND"

    # --- eski ad -> yeni ad: kesin bantlar kaybolmamalı ---
    r, _ = from_legacy("MONOCLONAL_IGG_HEAVY_CHAIN_ONLY")
    assert r.band_target() == {"IGG_NO_LIGHT_CHAIN"}, r.band_target()
    assert r.bands[0].certainty == "definite"
    r, _ = from_legacy("SERUM_HYPOGAMMAGLOBULINEMIA;IGA_HEAVY_CHAIN_ONLY")
    assert r.band_target() == {"IGA_NO_LIGHT_CHAIN"}
    assert "HYPOGAMMAGLOBULINEMIA" in r.flags

    # --- hiçbir sınıf adı hastalık iddia etmemeli ---
    # Ağır zincir sınıfı jel dilinde kalmalı (kullanıcı düzeltmesi)
    for _c in BAND_CLASSES:
        assert "HEAVY_CHAIN_ONLY" not in _c, f"hastalık dili: {_c}"
    # Kayıt işaretleri ve patern seçimleri model hedefine SIZMAMALI
    assert set(FLAG_CLASSES).isdisjoint(META_FLAGS)
    assert set(FLAG_CLASSES).isdisjoint(PATTERN_SELECT_FLAGS)
    assert len(FLAG_CLASSES) == 2, FLAG_CLASSES
    # Poliklonal zemin -> artış olarak birleşti
    r, _ = from_legacy("SUSPICIOUS_BAND_ON_POLYCLONAL_BACKGROUND_IgG_Kappa")
    assert r.flags == {"POLYCLONAL_INCREASE"}, r.flags
    r, _ = from_legacy("SERUM_POLYCLONAL_HYPERGAMMAGLOBULINEMIA_IgG_Kappa")
    assert r.flags == {"POLYCLONAL_INCREASE"}

    print("ife_taxonomy.py v2.0: tüm kendi-kendini-testler geçti.")
    print(f"  Patern: {len(PATTERN_CLASSES)} | Bant: {len(BAND_CLASSES)} | "
          f"Bayrak: {len(FLAG_CLASSES)} | Poliklonal lane: {len(POLY_LANE_CLASSES)}")
