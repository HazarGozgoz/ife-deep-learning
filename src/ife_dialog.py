# -*- coding: utf-8 -*-
"""
ife_dialog.py -- ÜÇ EKSENLİ tanı giriş penceresi
==================================================
Hem ife_annotator.py (kesme aracı) hem ife_review_tool.py (gözden geçirme)
AYNI pencereyi kullanır. Tek kopya, tek davranış -- eski koddaki en büyük
sorun her aracın kendi tanı sözlüğünü ve kendi diyalog kodunu taşımasıydı.

ESKİ PENCERENİN SORUNU
-----------------------
Tek bir uzun checkbox listesi vardı. Checkbox bir KÜME oluşturur:
  * Aynı kutuyu iki kez işaretleyemezsiniz -> "iki ayrı IgA-Kappa bandı"
    kaydedilemiyordu.
  * Bant tipi, kesinlik ve zemin tek isme sıkışıyordu
    (SUSPICIOUS_BAND_ON_POLYCLONAL_BACKGROUND).
  * Tipleme sadece serbest metne yazılabiliyordu, o da sonra atılıyordu.

YENİ PENCERE
-------------
Dört bölüm, dördü de birbirinden bağımsız:
  1) PATERN      : radyo düğmesi, tek seçim
  2) BANTLAR     : satır satır; her satırda tip + kesinlik + bölge notu.
                   "+ Bant ekle" ile istediğiniz kadar, AYNI TİP TEKRAR
                   EDİLEBİLİR.
  3) BULGULAR    : hipogamaglobulinemi / poliklonal zemin (bağımsız kutular)
  4) POLİKLONAL  : artışın görüldüğü lane'ler (sadece poliklonal seçilince
                   aktifleşir; bunlar BANT DEĞİLDİR)

Alt kısımda CANLI tutarlılık uyarısı vardır: "Biklonal seçtiniz ama 1 bant
girdiniz" gibi durumlar kaydetmeden ÖNCE görünür.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QScrollArea, QVBoxLayout, QWidget)

from ife_taxonomy import (ALL_FLAGS, BAND_CLASSES, BAND_TR, CERTAINTY_TR,
                          CERTAINTY_VALUES, FLAG_CLASSES, FLAG_TR, META_FLAGS,
                          PATTERN_SELECT_FLAGS, PATTERN_SELECT_TR, PATTERN_TR,
                          POLY_LANE_CLASSES, POLY_LANE_TR, Band, IFERecord,
                          band_label, check_consistency, count_distinct_clones,
                          derive_pattern)

MAX_BANDS = 8


class BandRow(QWidget):
    """Tek bir bant satırı: tip + kesinlik + bölge notu + sil."""

    def __init__(self, parent_dialog, band: Band | None = None):
        super().__init__()
        self.parent_dialog = parent_dialog
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)

        self.type_cb = QComboBox()
        self.type_cb.addItem("— bant tipi seçiniz —", "")
        for cls in BAND_CLASSES:
            self.type_cb.addItem(BAND_TR[cls], cls)
        lay.addWidget(self.type_cb, 4)
        self.preview = QLabel("")
        self.preview.setStyleSheet("color:#036; font-style:italic;")

        self.cert_cb = QComboBox()
        for c in CERTAINTY_VALUES:
            self.cert_cb.addItem(CERTAINTY_TR[c], c)
        lay.addWidget(self.cert_cb, 2)

        self.note_le = QLineEdit()
        self.note_le.setPlaceholderText("bölge / not (isteğe bağlı)")
        lay.addWidget(self.note_le, 3)

        btn_del = QPushButton("✕")
        btn_del.setFixedWidth(30)
        btn_del.setToolTip("Bu bandı sil")
        btn_del.clicked.connect(self.remove_self)
        lay.addWidget(btn_del)

        if band:
            i = self.type_cb.findData(band.band_class)
            if i >= 0:
                self.type_cb.setCurrentIndex(i)
            j = self.cert_cb.findData(band.certainty)
            if j >= 0:
                self.cert_cb.setCurrentIndex(j)
            self.note_le.setText(band.note)

        lay.addWidget(self.preview, 3)
        self.type_cb.currentIndexChanged.connect(self._changed)
        self.cert_cb.currentIndexChanged.connect(self._changed)
        self._changed()

    def _changed(self):
        """Kesinlik 'Şüpheli' yapılınca ad LANE diline döner -- şüpheli bantta
        klon kimliği iddia edilmez."""
        cls = self.type_cb.currentData()
        if cls:
            self.preview.setText("→ " + band_label(cls, self.cert_cb.currentData()))
        else:
            self.preview.setText("")
        self.parent_dialog.refresh_warning()

    def remove_self(self):
        self.parent_dialog.remove_band_row(self)

    def get_band(self) -> Band | None:
        cls = self.type_cb.currentData()
        if not cls:
            return None
        return Band(cls, self.cert_cb.currentData(), self.note_le.text().strip())


class IFEDiagnosisDialog(QDialog):
    """Üç eksenli tanı giriş penceresi. record=None ise boş başlar."""

    def __init__(self, record: IFERecord | None = None, parent=None, title="Tanı Girişi"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(760, 720)
        self.band_rows: list[BandRow] = []
        self._meta_flags: set[str] = set()

        root = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # ---------- 1) BANTLAR (ana bölüm) ----------
        gb_band = QGroupBox("1. Jelde gördüğünüz bantlar  (aynı tipi birden çok kez ekleyebilirsiniz)")
        bl = QVBoxLayout(gb_band)
        hint = QLabel("İki ayrı IgA-Kappa bandı gördüyseniz, IgA-Kappa'yı İKİ KEZ ekleyin.\n"
                      "Bant yoksa bu bölümü boş bırakın.")
        hint.setStyleSheet("color:#555; font-style:italic; padding:2px;")
        bl.addWidget(hint)
        self.bands_layout = QVBoxLayout()
        bl.addLayout(self.bands_layout)
        btn_add = QPushButton("+ Bant ekle")
        btn_add.clicked.connect(lambda: self.add_band_row())
        bl.addWidget(btn_add)
        self.clone_label = QLabel("")
        self.clone_label.setStyleSheet("color:#555; padding:3px;")
        bl.addWidget(self.clone_label)
        lay.addWidget(gb_band)

        # ---------- 3) EK BULGULAR ----------
        gb_flag = QGroupBox("2. Diğer bulgular")
        fl = QVBoxLayout(gb_flag)
        self.flag_boxes: dict[str, QCheckBox] = {}
        for f in FLAG_CLASSES:
            cb = QCheckBox(FLAG_TR[f])
            self.flag_boxes[f] = cb
            cb.toggled.connect(self.on_flags_changed)
            fl.addWidget(cb)
        # Paterni belirleyen seçimler -- ayrı model hedefi değil
        fl.addSpacing(6)
        for f in PATTERN_SELECT_FLAGS:
            cb = QCheckBox(PATTERN_SELECT_TR[f])
            self.flag_boxes[f] = cb
            cb.toggled.connect(self.on_flags_changed)
            fl.addWidget(cb)
        note = QLabel("Hipogamaglobulinemi bir bantla BİRLİKTE işaretlenebilir\n"
                      "(immünoparezi + M-protein: klasik myelom tablosu).")
        note.setStyleSheet("color:#555; font-style:italic;")
        fl.addWidget(note)
        lay.addWidget(gb_flag)

        # ---------- 4) POLİKLONAL LANE'LER ----------
        self.gb_poly = QGroupBox("3. Poliklonal artışın görüldüğü lane'ler  (bant DEĞİLDİR)")
        ql = QHBoxLayout(self.gb_poly)
        self.lane_boxes: dict[str, QCheckBox] = {}
        for l in POLY_LANE_CLASSES:
            cb = QCheckBox(POLY_LANE_TR[l])
            self.lane_boxes[l] = cb
            cb.toggled.connect(self.refresh_warning)
            ql.addWidget(cb)
        lay.addWidget(self.gb_poly)

        # ---------- serbest not ----------
        lay.addWidget(QLabel("Serbest not (isteğe bağlı):"))
        self.note_le = QLineEdit()
        self.note_le.setPlaceholderText("Sınıf adı olarak KULLANILMAZ, sadece not olarak saklanır.")
        lay.addWidget(self.note_le)

        # ---------- uyarı + butonlar ----------
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(line)

        # Patern ELLE SEÇİLMEZ -- girdiklerinizden hesaplanır ve burada
        # gösterilir. (Eskiden ayrı bir radyo düğmesiydi ve bantlarla
        # çelişebiliyordu; o çelişki uyarılarının tamamı bu yüzden çıkıyordu.)
        self.pattern_label = QLabel("")
        self.pattern_label.setStyleSheet(
            "font-size:16px; font-weight:bold; color:#036; padding:8px; "
            "background:#eef4ff; border:1px solid #ccd;")
        self.pattern_label.setWordWrap(True)
        root.addWidget(self.pattern_label)
        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet("color:#a00; font-weight:bold; padding:6px;")
        root.addWidget(self.warn_label)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self.load_record(record or IFERecord())

    # ------------------------------------------------------------------
    def load_record(self, rec: IFERecord):
        self._meta_flags = {f for f in rec.flags if f in META_FLAGS}
        for b in rec.bands:
            self.add_band_row(b)
        for f, cb in self.flag_boxes.items():
            cb.setChecked(f in rec.flags)
        for l, cb in self.lane_boxes.items():
            cb.setChecked(l in rec.poly_lanes)
        self.note_le.setText(rec.note)
        self.on_flags_changed()

    def add_band_row(self, band: Band | None = None):
        if len(self.band_rows) >= MAX_BANDS:
            return
        row = BandRow(self, band)
        self.band_rows.append(row)
        self.bands_layout.addWidget(row)
        self.refresh_warning()

    def remove_band_row(self, row: BandRow):
        if row in self.band_rows:
            self.band_rows.remove(row)
            row.setParent(None)
            row.deleteLater()
        self.refresh_warning()

    def on_flags_changed(self):
        is_poly = self.flag_boxes["POLYCLONAL_INCREASE"].isChecked()
        self.gb_poly.setEnabled(is_poly)
        if not is_poly:
            for cb in self.lane_boxes.values():
                cb.setChecked(False)
        self.refresh_warning()

    # ------------------------------------------------------------------
    def get_record(self) -> IFERecord:
        bands = [r.get_band() for r in self.band_rows]
        bands = [b for b in bands if b is not None]
        rec = IFERecord(
            bands=bands,
            # Kayıt işaretleri (META_FLAGS) kullanıcıya gösterilmez ama
            # mevcut kayıtta varsa korunur -- elle düzeltilince temizlenir.
            flags=({f for f, cb in self.flag_boxes.items() if cb.isChecked()}
                   | (self._meta_flags - {"BAND_COUNT_INCOMPLETE",
                                          "UNTYPED_SUSPICIOUS_BAND"})),
            poly_lanes={l for l, cb in self.lane_boxes.items() if cb.isChecked()},
            note=self.note_le.text().strip(),
        )
        rec.resolve_pattern()   # patern HER ZAMAN hesaplanır
        return rec

    def refresh_warning(self):
        rec = self.get_record()
        n = count_distinct_clones(rec.bands)
        self.clone_label.setText(f"Girilen bant: {len(rec.bands)}   |   Ayrı klon: {n}")
        self.pattern_label.setText(f"Yorum:  {PATTERN_TR[rec.pattern]}")
        self.warn_label.setText(check_consistency(rec))
