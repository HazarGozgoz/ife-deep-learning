# -*- coding: utf-8 -*-
"""
ife_annotator.py  (IFE Pro Studio - Serum-Only, v10.0)
========================================================
cropping_script.py (v9.3) + cropping_script_helena.py (v9.4) +
cropping_script_2.py (v9.7) dosyalarının TEK, güncel, sadece-serum sürümü.

Bu üç dosya zamanla birbirinden kopyalanarak farklılaşmıştı (bkz. bu dosyaların
diff'i) -- bu da tanı isimlerinin sürüme göre değişmesine (SERUM_ önekli /
öneksiz) yol açan asıl nedenlerden biriydi. Artık TEK dosya var ve etiketler
ife_taxonomy.py'den geliyor.

v9.7'ye göre değişenler
-----------------------
1) İDRAR TAMAMEN KALDIRILDI: SampleTypeDialog ve URINE_DIAGNOSIS_MAP yok.
   Her kırpma otomatik olarak "SERUM" kabul edilir -- artık her seçimden önce
   "Serum mu İdrar mı?" sormuyor, bu da işaretleme hızını da artırır.
2) Tanı sözlüğü artık ife_taxonomy.DIAGNOSIS_MAP -- başka hiçbir dosyada
   kopyası yok.
3) Cihaz bilgisi artık dosya adına "helena_" önekini ZORLA yapıştırarak değil,
   ayrı bir "device" kolonuna açıkça yazılarak tutuluyor (bkz. DEVICE_NAME
   sabiti aşağıda). Bu, hem daha temiz hem de yayın.py'deki
   "'helena' in path.lower()" gibi kırılgan dosya-adı-koklama mantığını
   gereksiz kılıyor.
4) Kaynak taramaya göre gruplama (sızıntı önleme) hâlâ "source_file" kolonuyla
   yapılıyor; asıl bölme mantığı artık veri_hazirlama.py'de TEK yerde.
5) TIFF desteği + QImageReader RAM sınırı kaldırma (v9.7'nin faydalı düzeltmesi)
   ve satır-satır anlık CSV yazımı (v9.7) korundu.

KULLANIM
--------
Her tarama grubunu (ör. bir günlük Interlab taramaları ya da bir Helena
partisi) işaretlemeden ÖNCE aşağıdaki iki sabiti o partiye göre güncelleyin:
    INPUT_FOLDER = Path("A4_Taramalar")   # ya da "A4_Taramalar_Helena" vb.
    DEVICE_NAME  = "INTERLAB"             # ya da "HELENA"
"""
import sys
from datetime import datetime
from math import floor, ceil
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import (QAction, QColor, QFontMetricsF, QImageReader, QImageWriter,
                          QKeySequence, QPainter, QPen, QPixmap, QTransform)
from PyQt6.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox,
                              QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
                              QComboBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                              QMainWindow, QMessageBox, QPushButton,
                              QScrollArea, QStatusBar, QToolBar, QVBoxLayout, QWidget)

from ife_dialog import IFEDiagnosisDialog
from ife_taxonomy import (CERTAINTY_TR, FLAG_TR, IFERecord, PATTERN_TR, band_label,
                          check_consistency, count_distinct_clones, to_legacy_labels)


def record_to_turkish(rec) -> str:
    """Jelin uzerine cizilecek kisa Turkce ozet."""
    parts = [PATTERN_TR.get(rec.pattern, rec.pattern)]
    for b in rec.bands:
        # band_label kesinliğe göre doğru adı üretir: şüpheli bantta
        # "Kappa'da şüpheli bant", kesin bantta "Serbest Kappa".
        parts.append(band_label(b.band_class, b.certainty))
    for f in sorted(rec.flags):
        if f in FLAG_TR:
            parts.append(FLAG_TR[f])
    return "; ".join(parts)

# --- KONFİGÜRASYON (her parti için güncelleyin) ---
INPUT_FOLDER = Path("A4_Taramalar")     # taranmış A4 sayfalarının klasörü

# --- CİHAZ BELİRLEME ---
# Helena ve Interlab taramaları AYNI klasörde duruyor; Helena dosyaları
# adında "helena" geçiyor (ör. "helena mart006.tif"). Bu yüzden cihaz her
# dosya için ADINDAN okunur; parti başına elle ayar gerekmez.
# Elle sabitlemek isterseniz DEVICE_OVERRIDE = "HELENA" yapın.
DEVICE_OVERRIDE = None                  # None = otomatik | "HELENA" | "INTERLAB"
HELENA_NAME_PATTERNS = ("helena",)      # dosya adında (küçük harfe çevrilip) aranır

# Hangi taramaların atlanacağı, SİZİN "bitti" dediğiniz listeye göre
# belirlenir. Eskiden "bu taramadan herhangi bir kayıt var mı" diye
# bakılıyordu; yarım bıraktığınız taramalar da atlanıyordu.
SKIP_ANNOTATED = True
DONE_FILE = Path("tamamlanan_taramalar.txt")
# kirpma_denetimi.py'nin urettigi "hic kesilmemis olabilir" listesi.
# Varsa, acik taramada bu bolgeler SARI cerceveyle gosterilir.
EKSIK_FILE = Path("denetim_eksik_paneller.csv")


def device_for_file(path: Path) -> str:
    """Tarama dosyasının hangi cihazdan geldiğini adından belirler."""
    if DEVICE_OVERRIDE:
        return DEVICE_OVERRIDE
    name = path.name.lower()
    return "HELENA" if any(p in name for p in HELENA_NAME_PATTERNS) else "INTERLAB"
OUTPUT_FOLDER = Path("Cropped_Data_EN")
ANNOTATIONS_FILE = Path("annotations_master.csv")  # goc sonrasi TEK master dosya
TOOL_VERSION = "10.0_Serum_Only_Unified"
ANNOTATOR_NAME = "Dr. Hazar Gozgoz"
PROJECT_ID = "IFE_AI_PROJECT_2025"
SESSION_ID = datetime.now().strftime('%Y%m%d%H%M%S')
MIN_ZOOM, MAX_ZOOM = 0.1, 15.0
MIN_SELECTION_SIZE = 16  # minimum 16x16 piksel seçim

# etiket_gocu.py'nin urettigi master semasiyla BIREBIR ayni olmali ki yeni
# Helena kayitlari eski Interlab kayitlarinin yanina sorunsuz eklensin.
EXPECTED_COLUMNS = [
    "timestamp", "annotator", "tool_version", "project_id", "session_id",
    "source_file", "device", "sample_type", "filename",
    # --- ÜÇ EKSENLİ ALANLAR ---
    "pattern", "bands", "n_bands", "n_clones", "flags", "poly_lanes",
    "band_target", "band_target_definite", "has_suspicious", "detail_note",
    # --- yol + geriye dönük uyum ---
    "all_diagnoses_tr", "img_relpath", "primary_diagnosis", "primary_diagnosis_eski",
    "all_diagnoses", "all_diagnoses_canonical", "review_reason",
    "x", "y", "w", "h", "source_image_width", "source_image_height",
]


class MultiDiagnosisDialog(QDialog):
    def __init__(self, diagnosis_map, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tanı/Bulgu Seçimi")
        self.setMinimumWidth(420)
        self.diagnosis_map = diagnosis_map
        content_layout = QVBoxLayout()

        self.checkboxes = [QCheckBox(name) for name in self.diagnosis_map if name != "Diger..."]
        for cb in self.checkboxes:
            content_layout.addWidget(cb)

        self.other_cb = QCheckBox("Diger...")
        self.other_le = QLineEdit()
        self.other_le.setPlaceholderText("Diğer tanıyı İngilizce, virgülle ayırarak yazın...")
        self.other_le.hide()
        self.other_cb.toggled.connect(self.other_le.setVisible)
        content_layout.addWidget(self.other_cb)
        content_layout.addWidget(self.other_le)

        sw = QWidget()
        sw.setLayout(content_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sw)

        ml = QVBoxLayout(self)
        ml.addWidget(scroll)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        ml.addWidget(btns)

    def get_selected_turkish(self):
        return [cb.text() for cb in self.checkboxes if cb.isChecked()]

    def get_other_text(self):
        return self.other_le.text().strip() if self.other_cb.isChecked() else ""



class CloneEntryDialog(QDialog):
    """
    Biklonal / oligoklonal için YAPISAL klon girişi.

    Neden gerekli: checkbox'lar bir KÜME oluşturur, aynı kutuyu iki kez
    işaretleyemezsiniz. Bu yüzden "iki ayrı IgA-Kappa bandı" gibi gerçek bir
    biklonal patern eski araçta kaydedilemiyordu. Burada her klon AYRI BİR
    SATIR; aynı tip iki kez seçilebilir ve her birine bandın yerini
    yazabilirsiniz.
    """

    def __init__(self, gestalt_name: str, n_start: int = 2, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{gestalt_name} — Klonları Belirtin")
        self.setMinimumWidth(560)
        self.rows: list[tuple[QComboBox, QLineEdit]] = []

        outer = QVBoxLayout(self)
        info = QLabel(
            "Her bant için bir satır doldurun. AYNI tipi iki kez seçebilirsiniz\n"
            "(ör. iki ayrı IgA-Kappa bandı) — bu gerçek bir biklonal paterndir."
        )
        info.setStyleSheet("color:#333; padding:6px; background:#eef; border:1px solid #ccd;")
        outer.addWidget(info)

        self.rows_layout = QVBoxLayout()
        outer.addLayout(self.rows_layout)

        btn_add = QPushButton("+ Klon ekle")
        btn_add.clicked.connect(lambda: self.add_row())
        outer.addWidget(btn_add)

        self.warn_label = QLabel("")
        self.warn_label.setStyleSheet("color:#a00; font-weight:bold; padding:4px;")
        self.warn_label.setWordWrap(True)
        outer.addWidget(self.warn_label)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        for _ in range(n_start):
            self.add_row()

    def add_row(self):
        if len(self.rows) >= 8:
            return
        row = QHBoxLayout()
        idx = len(self.rows) + 1
        row.addWidget(QLabel(f"Klon {idx}:"))
        combo = QComboBox()
        combo.addItem("— seçiniz —", "")
        for label, cls in CLONE_OPTIONS:
            combo.addItem(label, cls)
        row.addWidget(combo, 3)
        note = QLineEdit()
        note.setPlaceholderText("bandın yeri / notu (isteğe bağlı)")
        row.addWidget(note, 2)
        container = QWidget()
        container.setLayout(row)
        self.rows_layout.addWidget(container)
        self.rows.append((combo, note))

    def get_clones(self) -> tuple[list[str], str]:
        """(kanonik sınıf listesi -- SIRALI ve TEKRAR EDEBİLİR, not metni)"""
        clones, notes = [], []
        for i, (combo, note) in enumerate(self.rows, start=1):
            cls = combo.currentData()
            if not cls:
                continue
            clones.append(cls)
            txt = note.text().strip()
            if txt:
                notes.append(f"Klon {i} ({combo.currentText()}): {txt}")
        return clones, " | ".join(notes)


class PhotoViewer(QGraphicsView):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.main_scene = QGraphicsScene(self)
        self.photo_item = QGraphicsPixmapItem()
        self.main_scene.addItem(self.photo_item)
        self.setScene(self.main_scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.rubber_band = None
        self.start_point = None
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def set_photo(self, pixmap):
        pixmap.setDevicePixelRatio(1.0)
        self.photo_item.setPixmap(pixmap)
        self.main_scene.setSceneRect(self.photo_item.boundingRect())
        self.fitInView(self.main_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if not event.angleDelta().isNull():
            current_zoom = self.transform().m11()
            factor = 1.15 ** (event.angleDelta().y() / 120.0)
            if (current_zoom * factor < MIN_ZOOM and factor < 1.0) or (
                    current_zoom * factor > MAX_ZOOM and factor > 1.0):
                return
            self.scale(factor, factor)
            return
        if not event.pixelDelta().isNull():
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - event.pixelDelta().x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - event.pixelDelta().y())

    def _cancel_selection(self):
        if self.rubber_band:
            self.main_scene.removeItem(self.rubber_band)
            self.rubber_band = None
        self.start_point = None
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.main_window.update_status_bar()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._cancel_selection()
            return
        mods = QApplication.keyboardModifiers()
        is_pan_key = bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        if is_pan_key and self.dragMode() == QGraphicsView.DragMode.NoDrag:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        mods = QApplication.keyboardModifiers()
        is_pan_key = bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        if not is_pan_key and self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._cancel_selection()
            return
        mods = QApplication.keyboardModifiers()
        is_panning_modifier = bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        if event.button() == Qt.MouseButton.LeftButton:
            if is_panning_modifier:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                super().mousePressEvent(event)
            else:
                self.start_point = self.mapToScene(event.pos())
                if self.rubber_band:
                    self.main_scene.removeItem(self.rubber_band)
                self.rubber_band = self.main_scene.addRect(
                    QRectF(self.start_point, self.start_point),
                    QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine))
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.start_point and self.dragMode() == QGraphicsView.DragMode.NoDrag:
            rect = QRectF(self.start_point, self.mapToScene(event.pos())).normalized()
            self.rubber_band.setRect(rect)
            x, y, w, h = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
            self.main_window.update_status_bar(f"Selection: {w}x{h}px @ ({x},{y})")
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.start_point and self.dragMode() == QGraphicsView.DragMode.NoDrag:
            self.start_point = None
            self.main_window.update_status_bar()
            if self.rubber_band and (
                    self.rubber_band.rect().width() < MIN_SELECTION_SIZE or
                    self.rubber_band.rect().height() < MIN_SELECTION_SIZE):
                # Kucuk tiklama: yeni kirpma degil, MEVCUT bir kirpmayi SECME
                r = self.rubber_band.rect()
                self.main_window.secim_tikla(r.center())
            elif self.rubber_band:
                r = self.rubber_band.rect()
                aligned_rect = QRectF(floor(r.x()), floor(r.y()), ceil(r.width()), ceil(r.height()))
                self.main_window.selection_made(aligned_rect)
            if self.rubber_band:
                self.main_scene.removeItem(self.rubber_band)
                self.rubber_band = None
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"IFE Pro Studio v{TOOL_VERSION} (Serum-Only)")
        self.setGeometry(100, 100, 1200, 800)

        valid_ext = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        all_files = sorted(p for p in INPUT_FOLDER.iterdir() if p.suffix.lower() in valid_ext)

        # Mevcut kayıtları tarama başına say -- hangi taramada kaç şerit
        # kırpmışsınız, ekranda göreceksiniz.
        self.existing_counts: dict[str, int] = {}
        self.existing_by_file: dict[str, list] = {}
        if ANNOTATIONS_FILE.exists():
            try:
                prev = pd.read_csv(ANNOTATIONS_FILE)
                self.existing_counts = prev["source_file"].astype(str).value_counts().to_dict()
                if {"x", "y", "w", "h"}.issubset(prev.columns):
                    ok = prev[["x", "y", "w", "h"]].notna().all(axis=1)
                    for ad, grp in prev[ok].groupby(prev["source_file"].astype(str)):
                        self.existing_by_file[ad] = grp[
                            ["x", "y", "w", "h", "all_diagnoses_tr"]].values.tolist()
            except Exception as e:
                print(f"UYARI: önceki etiketler okunamadı ({e}).")

        # Eksik panel onerileri (varsa) -- tarama basina
        self.eksik_by_file: dict[str, list] = {}
        if EKSIK_FILE.exists():
            try:
                ek = pd.read_csv(EKSIK_FILE)
                for ad, grp in ek.groupby(ek["kaynak_tarama"].astype(str)):
                    self.eksik_by_file[ad] = grp[["yaklasik_x", "yaklasik_y",
                                                  "yaklasik_w", "yaklasik_h"]].values.tolist()
                print(f"INFO: {len(ek)} aday eksik panel yuklendi "
                      f"({len(self.eksik_by_file)} tarama) -- SARI cerceveyle gosterilecek.")
            except Exception as e:
                print(f"UYARI: eksik panel listesi okunamadi ({e}).")

        # SADECE "bitti" dediğiniz taramalar atlanır.
        done: set[str] = set()
        if SKIP_ANNOTATED and DONE_FILE.exists():
            done = {l.strip() for l in DONE_FILE.read_text(encoding="utf-8").splitlines()
                    if l.strip()}
        self.image_files = [p for p in all_files if p.name not in done]
        skipped = len(all_files) - len(self.image_files)
        if skipped:
            print(f"INFO: {skipped} tarama 'bitti' işaretli, atlandı. "
                  f"({len(self.image_files)} tarama kaldı)")
        yarim = [p.name for p in self.image_files if self.existing_counts.get(p.name, 0) > 0]
        if yarim:
            print(f"INFO: {len(yarim)} tarama YARIM -- daha önce kırptığınız şeritler "
                  f"ekranda MAVİ çerçeveyle gösterilecek.")
        n_hel = sum(1 for p in self.image_files if device_for_file(p) == "HELENA")
        print(f"INFO: Etiketlenecek: {n_hel} HELENA + "
              f"{len(self.image_files) - n_hel} INTERLAB taraması")
        if not self.image_files:
            sys.exit(f"'{INPUT_FOLDER}' içinde etiketlenecek tarama kalmadı.\n"
                     f"(Hepsini 'bitti' işaretlediyseniz beklenen durumdur. "
                     f"Yeniden açmak için {DONE_FILE} dosyasını silin.)")

        self.current_image_index = -1
        self.annotations_this_image = []
        self.dataframe = self.load_or_create_dataframe()

        self.viewer = PhotoViewer(self)
        self.setCentralWidget(self.viewer)
        self.setStatusBar(QStatusBar(self))
        self.create_toolbar_and_shortcuts()
        self.next_image()

    def load_or_create_dataframe(self) -> pd.DataFrame:
        if ANNOTATIONS_FILE.exists():
            try:
                df = pd.read_csv(ANNOTATIONS_FILE)
                missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
                if missing:
                    print(f"INFO: CSV'ye eksik kolonlar ekleniyor: {missing}")
                    df[missing] = pd.NA
                    df = df[EXPECTED_COLUMNS + [c for c in df.columns if c not in EXPECTED_COLUMNS]]
                    df.to_csv(ANNOTATIONS_FILE, index=False, sep=',', encoding='utf-8-sig')
                return df
            except Exception as e:
                print(f"WARNING: CSV okunamadı: {e}. Yedek alınıp yeni dosya oluşturuluyor.")
                try:
                    ANNOTATIONS_FILE.rename(ANNOTATIONS_FILE.with_suffix('.csv.bak'))
                except OSError:
                    pass
                return pd.DataFrame(columns=EXPECTED_COLUMNS)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    def _mk_shortcut(self, key, slot):
        act = QAction(self)
        act.setShortcut(QKeySequence(key))
        act.triggered.connect(slot)
        return act

    def create_toolbar_and_shortcuts(self):
        toolbar = QToolBar("Tools")
        self.addToolBar(toolbar)

        def add(text, slot, shortcut):
            act = QAction(text, self)
            act.triggered.connect(slot)
            act.setShortcut(QKeySequence(shortcut))
            toolbar.addAction(act)

        add("Previous (P)", self.previous_image, "P")
        add("Next (N)", self.next_image, "N")
        add("Undo (Z)", self.undo_last_annotation, "Z")
        add("Fit (F)", self.fit_to_view, "F")
        toolbar.addSeparator()
        add("Rotate Left (L)", self.rotate_left, "L")
        add("Rotate Right (R)", self.rotate_right, "R")
        add("Rotate 180 (X)", self.rotate_180, "X")
        add("Flip Horz (H)", self.flip_horizontal, "H")
        add("Flip Vert (V)", self.flip_vertical, "V")
        toolbar.addSeparator()
        add("Reset Orientation (O)", self.reset_orientation, "O")
        add("Save Rotated Copy (S)", self.save_rotated_copy, "S")

        act_sil = QAction("Seçili kırpmayı SİL (Delete)", self)
        act_sil.triggered.connect(self.secili_sil)
        act_sil.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        toolbar.addAction(act_sil)
        self.addAction(self._mk_shortcut(Qt.Key.Key_Backspace, self.secili_sil))

        act_done = QAction("Bu taramayı BİTTİ işaretle (B)", self)
        act_done.triggered.connect(self.mark_done)
        act_done.setShortcut(QKeySequence("B"))
        toolbar.addAction(act_done)

        self.addAction(self._mk_shortcut(Qt.Key.Key_Left, self.previous_image))
        self.addAction(self._mk_shortcut(Qt.Key.Key_Right, self.next_image))

    def update_status_bar(self, message=None):
        if message:
            self.statusBar().showMessage(message)
            return
        if not (0 <= self.current_image_index < len(self.image_files)):
            self.statusBar().clearMessage()
            return
        filepath = self.image_files[self.current_image_index]
        self.statusBar().showMessage(
            f"[{self.current_image_index + 1}/{len(self.image_files)}] {filepath.name} | "
            f"kayıtlı: {self.existing_counts.get(filepath.name, 0)} | "
            f"bos aday: {len(getattr(self, 'suggested_boxes', []))} | "
            f"Cihaz: {self.current_device()} | Ctrl/Cmd+Drag=Pan | Drag=Select | Wheel=Zoom | "
            f"F=Fit | L/R/X/H/V=Transform | S=Save Copy")

    def change_image(self, direction):
        new_index = self.current_image_index + direction
        if not (0 <= new_index < len(self.image_files)):
            print(f"INFO: Listenin {'sonuna' if direction > 0 else 'başına'} ulaşıldı.")
            return
        self.current_image_index = new_index
        filepath = self.image_files[self.current_image_index]
        self.current_pixmap = QPixmap(str(filepath))
        self.annotations_this_image = []
        # Daha önce bu taramada kırptıklarınız (CSV'den) -- mavi çizilecek
        self.previous_boxes = self.existing_by_file.get(filepath.name, [])
        self.suggested_boxes = self._bosluk_bul(self.previous_boxes)
        self.secili_kutu = None
        self.tekrar_idx = self._tekrar_bul(self.previous_boxes)
        if self.current_pixmap.isNull():
            self.change_image(direction)
            return
        self.viewer.set_photo(self.current_pixmap)
        # ONEMLI: tarama acilir acilmaz mevcut kirpmalari CIZ. Bu cagri
        # eksik oldugu icin tarama bombos aciliyor, neyi kestiginizi
        # goremiyordunuz.
        self.redraw_annotations()
        self.update_status_bar()
        print(f"\nNow processing: {filepath.name}  "
              f"(kayitli: {len(self.previous_boxes)}, "
              f"bos aday: {len(self.suggested_boxes)})")

    def next_image(self):
        self.change_image(1)

    def previous_image(self):
        self.change_image(-1)

    def selection_made(self, rect):
        # ÜÇ EKSENLİ tanı penceresi (ife_dialog.py). Kesme aracı ve gözden
        # geçirme aracı AYNI pencereyi kullanır -- tek kopya, tek davranış.
        dlg = IFEDiagnosisDialog(parent=self, title="Tanı Girişi — Yeni Kırpma")
        if not dlg.exec():
            return
        rec = dlg.get_record()

        # Hiçbir şey seçilmediyse kaydetme
        if (rec.pattern == "NO_MONOCLONAL_BAND" and not rec.bands
                and not rec.flags and not rec.note):
            reply = QMessageBox.question(
                self, "Boş kayıt",
                "Patern 'Non-spesifik' ve hiçbir bant/bulgu girilmedi.\n"
                "Bu şerit gerçekten normal mi?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Pencerede zaten canlı uyarı var; yine de kaydetmeden önce son kontrol
        warn = check_consistency(rec)
        if warn:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Tutarlılık Uyarısı")
            box.setText(warn)
            box.setInformativeText("Yine de bu şekilde kaydedilsin mi?")
            box.setStandardButtons(QMessageBox.StandardButton.Save |
                                   QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(QMessageBox.StandardButton.Cancel)
            if box.exec() != QMessageBox.StandardButton.Save:
                return

        self._pending_record = rec
        legacy = to_legacy_labels(rec)
        primary_diag = rec.pattern if not rec.bands else sorted(rec.band_target())[0]
        all_en_str = ";".join(sorted(legacy)) if legacy else rec.pattern

        image_rect_f = QRectF(self.current_pixmap.rect())
        crop_rect_f = rect.intersected(image_rect_f)
        if crop_rect_f.isEmpty():
            return
        crop_rect_int = QRectF(
            floor(crop_rect_f.x()), floor(crop_rect_f.y()),
            ceil(crop_rect_f.width()), ceil(crop_rect_f.height())
        ).toRect().intersected(self.current_pixmap.rect())
        if crop_rect_int.isEmpty():
            return

        try:
            diag_folder = OUTPUT_FOLDER / primary_diag
            diag_folder.mkdir(parents=True, exist_ok=True)

            base_fn = self.image_files[self.current_image_index].stem
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            x, y, w, h = crop_rect_int.x(), crop_rect_int.y(), crop_rect_int.width(), crop_rect_int.height()
            output_fn = f"{base_fn}__{timestamp}__x{x}_y{y}_w{w}_h{h}.png"
            output_path = diag_folder / output_fn
            self.current_pixmap.copy(crop_rect_int).save(str(output_path))
        except Exception as e:
            print(f"!!! FILE SAVE ERROR: {e}")
            return

        rec = self._pending_record
        new_annotation = {
            "timestamp": datetime.now().isoformat(),
            "annotator": ANNOTATOR_NAME,
            "tool_version": TOOL_VERSION,
            "project_id": PROJECT_ID,
            "session_id": SESSION_ID,
            "source_file": self.image_files[self.current_image_index].name,
            "device": self.current_device(),
            "sample_type": "SERUM",
            "filename": output_fn,
            # --- üç eksen ---
            "pattern": rec.pattern,
            "bands": rec.bands_str(),
            "n_bands": rec.n_bands(),
            "n_clones": count_distinct_clones(rec.bands),
            "flags": rec.flags_str(),
            "poly_lanes": rec.poly_lanes_str(),
            "band_target": ";".join(sorted(rec.band_target())),
            "band_target_definite": ";".join(sorted(rec.band_target_definite_only())),
            "has_suspicious": int(rec.has_suspicious()),
            "detail_note": rec.note,
            "all_diagnoses_tr": record_to_turkish(rec),
            # --- yol + geriye dönük uyum ---
            "img_relpath": f"{primary_diag}/{output_fn}",
            "primary_diagnosis": primary_diag,
            "primary_diagnosis_eski": primary_diag,
            "all_diagnoses": all_en_str,
            "all_diagnoses_canonical": all_en_str,
            "review_reason": check_consistency(rec),
            "x": x, "y": y, "w": w, "h": h,
            "source_image_width": self.current_pixmap.width(),
            "source_image_height": self.current_pixmap.height(),
        }

        self.save_single_row_to_csv(new_annotation)
        self.annotations_this_image.append(
            {'annotation_data': new_annotation, 'crop_rect_draw': QRectF(crop_rect_f),
             'filepath_to_delete': output_path})
        self.redraw_annotations()
        print(f"SAVED & WRITTEN TO CSV: '{output_fn}' -> {all_en_str}")

    @staticmethod
    def _ortusme(a, b) -> float:
        ax, ay, aw, ah = [float(v) for v in a[:4]]
        bx, by, bw, bh = [float(v) for v in b[:4]]
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        k = (x2 - x1) * (y2 - y1)
        return k / (aw * ah + bw * bh - k)

    @staticmethod
    def _kumele(degerler, tol):
        """Yakin degerleri ayni gruba toplar (satir/sutun bulmak icin)."""
        sira = sorted(range(len(degerler)), key=lambda i: degerler[i])
        grup = [0] * len(degerler)
        g = 0
        for k in range(1, len(sira)):
            if degerler[sira[k]] - degerler[sira[k - 1]] > tol:
                g += 1
            grup[sira[k]] = g
        return grup

    def _bosluk_bul(self, kutular):
        """
        Kirpma merkezlerinden satir/sutun izgarasini cikarir, BOS hucreleri
        dondurur. Disaridan CSV okumaz -- her taramada anlik hesaplanir.
        Bu bir ADAY listesidir: jel kartinda gercekten bos pozisyon olabilir.
        """
        if len(kutular) < 4:
            return []
        try:
            xs = [float(b[0]) for b in kutular]
            ys = [float(b[1]) for b in kutular]
            ws = [float(b[2]) for b in kutular]
            hs = [float(b[3]) for b in kutular]
        except Exception:
            return []
        gw = sorted(ws)[len(ws) // 2]
        gh = sorted(hs)[len(hs) // 2]
        cx = [x + w / 2 for x, w in zip(xs, ws)]
        cy = [y + h / 2 for y, h in zip(ys, hs)]
        satir = self._kumele(cy, gh * 0.5)
        sutun = self._kumele(cx, gw * 0.5)
        satir_y, sutun_x = {}, {}
        for r, v in zip(satir, cy):
            satir_y.setdefault(r, []).append(v)
        for c, v in zip(sutun, cx):
            sutun_x.setdefault(c, []).append(v)
        satir_y = {k: sorted(v)[len(v) // 2] for k, v in satir_y.items()}
        sutun_x = {k: sorted(v)[len(v) // 2] for k, v in sutun_x.items()}
        dolu = set(zip(satir, sutun))
        bos = []
        for r in sorted(satir_y):
            for c in sorted(sutun_x):
                if (r, c) not in dolu:
                    bos.append([sutun_x[c] - gw / 2, satir_y[r] - gh / 2, gw, gh])
        return bos

    def _tekrar_bul(self, kutular) -> set:
        """Birbirini buyuk olcude ortusen kirpmalarin indeksleri."""
        tek = set()
        for i in range(len(kutular)):
            for j in range(i + 1, len(kutular)):
                if self._ortusme(kutular[i], kutular[j]) > 0.5:
                    tek.add(i); tek.add(j)
        return tek

    def secim_tikla(self, nokta):
        """Tiklanan noktadaki mevcut kirpmayi secer (silmek icin)."""
        x, y = nokta.x(), nokta.y()
        bulunan = None
        for i, bx in enumerate(self.previous_boxes):
            try:
                bxx, byy, bww, bhh = [float(v) for v in bx[:4]]
            except Exception:
                continue
            if bxx <= x <= bxx + bww and byy <= y <= byy + bhh:
                bulunan = i
                break
        if bulunan is None:
            self.secili_kutu = None
            self.statusBar().showMessage(
                "Burada kayitli kirpma yok. Yeni kirpma icin SURUKLEYIN.", 3000)
        else:
            self.secili_kutu = bulunan
            etiket = str(self.previous_boxes[bulunan][4]) if len(
                self.previous_boxes[bulunan]) > 4 else ""
            self.statusBar().showMessage(
                f"SECILDI: {etiket[:60]}   —   silmek icin DELETE tusuna basin", 8000)
        self.redraw_annotations()

    def secili_sil(self):
        """Secili kirpmayi CSV'den siler, goruntusunu _SILINENLER klasorune tasir."""
        if self.secili_kutu is None:
            self.statusBar().showMessage(
                "Once silmek istediginiz kirpmanin uzerine TIKLAYIN.", 3000)
            return
        kutu = self.previous_boxes[self.secili_kutu]
        x, y, w, h = [int(float(v)) for v in kutu[:4]]
        kaynak = self.image_files[self.current_image_index].name

        try:
            tum = pd.read_csv(ANNOTATIONS_FILE)
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"CSV okunamadi: {e}")
            return
        hedef = (tum["source_file"].astype(str) == kaynak)
        for kol, deger in (("x", x), ("y", y), ("w", w), ("h", h)):
            hedef &= pd.to_numeric(tum[kol], errors="coerce").round().eq(deger)
        if not hedef.any():
            QMessageBox.warning(self, "Bulunamadi",
                                "Bu kirpma CSV'de bulunamadi.")
            return

        satir = tum[hedef].iloc[0]
        ad = str(satir.get("filename", ""))
        cevap = QMessageBox.question(
            self, "Silinsin mi?",
            f"{ad}\n\n{str(satir.get('all_diagnoses_tr', ''))[:80]}\n\n"
            f"Bu kayit silinecek. Goruntu dosyasi silinmez, "
            f"'{OUTPUT_FOLDER}/_SILINENLER' klasorune tasinir.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if cevap != QMessageBox.StandardButton.Yes:
            return

        # Goruntuyu cop klasorune tasi (geri alinabilsin)
        rel = str(satir.get("img_relpath", "") or "")
        kaynak_yol = OUTPUT_FOLDER / rel if rel else None
        if kaynak_yol and kaynak_yol.exists():
            cop = OUTPUT_FOLDER / "_SILINENLER"
            cop.mkdir(parents=True, exist_ok=True)
            try:
                kaynak_yol.rename(cop / kaynak_yol.name)
            except Exception as e:
                print(f"UYARI: goruntu tasinamadi ({e})")

        tum[~hedef].to_csv(ANNOTATIONS_FILE, index=False, sep=",",
                           encoding="utf-8-sig")
        print(f"SILINDI: {ad}")

        # Ekrani guncelle
        del self.previous_boxes[self.secili_kutu]
        self.secili_kutu = None
        self.tekrar_idx = self._tekrar_bul(self.previous_boxes)
        self.existing_counts[kaynak] = self.existing_counts.get(kaynak, 1) - 1
        self.existing_by_file[kaynak] = self.previous_boxes
        self.redraw_annotations()
        self.update_status_bar()

    def mark_done(self):
        """Bu taramayı 'bitti' listesine ekler; bir daha açılışta gelmez."""
        if not (0 <= self.current_image_index < len(self.image_files)):
            return
        ad = self.image_files[self.current_image_index].name
        mevcut = set()
        if DONE_FILE.exists():
            mevcut = {l.strip() for l in DONE_FILE.read_text(encoding="utf-8").splitlines()
                      if l.strip()}
        if ad not in mevcut:
            with open(DONE_FILE, "a", encoding="utf-8") as f:
                f.write(ad + "\n")
            print(f"BİTTİ işaretlendi: {ad}  (toplam {len(mevcut) + 1})")
        else:
            print(f"Zaten bitti işaretli: {ad}")
        self.next_image()

    def current_device(self) -> str:
        """Şu an açık olan taramanın cihazı (dosya adından)."""
        if not self.image_files:
            return "INTERLAB"
        return device_for_file(self.image_files[self.current_image_index])

    def save_single_row_to_csv(self, row_data: dict):
        """
        Satiri CSV'ye ekler.

        KRITIK: mode='a' ile yazarken pandas, sozlugun KENDI sirasini kullanir;
        dosyadaki basligin sirasini dikkate almaz. Ikisi farkliysa degerler
        yanlis sutunlara yazilir ve bu SESSIZCE olur -- tanilar 'device'
        sutununa, cihaz adi 'sample_type' sutununa duser. Bu yuzden once
        dosyanin mevcut basligi okunur ve satir ona gore hizalanir.
        """
        try:
            if not ANNOTATIONS_FILE.exists():
                pd.DataFrame([row_data]).reindex(columns=EXPECTED_COLUMNS).to_csv(
                    ANNOTATIONS_FILE, index=False, sep=',', encoding='utf-8-sig')
                self.dataframe = pd.concat(
                    [self.dataframe, pd.DataFrame([row_data])], ignore_index=True)
                return

            # Dosyanin gercek basligini oku
            basliklar = list(pd.read_csv(ANNOTATIONS_FILE, nrows=0).columns)
            fazla = [k for k in row_data if k not in basliklar]
            if fazla:
                # Yeni alan(lar) var: dosyayi bir kez genisleterek yeniden yaz.
                # Ekleme moduyla devam etmek kaymaya yol acardi.
                print(f"INFO: CSV'ye yeni sutun ekleniyor: {fazla}")
                tum = pd.read_csv(ANNOTATIONS_FILE)
                for k in fazla:
                    tum[k] = pd.NA
                basliklar = list(tum.columns)
                tum = pd.concat([tum, pd.DataFrame([row_data]).reindex(columns=basliklar)],
                                ignore_index=True)
                tum.to_csv(ANNOTATIONS_FILE, index=False, sep=',', encoding='utf-8-sig')
                self.dataframe = tum
                return

            satir = pd.DataFrame([row_data]).reindex(columns=basliklar)
            satir.to_csv(ANNOTATIONS_FILE, mode='a', header=False,
                         index=False, sep=',', encoding='utf-8-sig')
            self.dataframe = pd.concat([self.dataframe, satir], ignore_index=True)
        except Exception as e:
            print(f"!!! CRITICAL ERROR WRITING TO CSV: {e}")

    def redraw_annotations(self):
        temp_pixmap = self.current_pixmap.copy()
        painter = QPainter(temp_pixmap)
        font = painter.font()
        zoom = max(0.1, self.viewer.transform().m11())
        font.setPointSize(max(8, min(24, int(12 / zoom))))
        painter.setFont(font)

        # Aday eksik paneller: SARI (buralari kesmeyi unutmus olabilirsiniz)
        for bx in getattr(self, "suggested_boxes", []):
            try:
                x, y, w, h = [float(v) for v in bx]
                painter.setPen(QPen(QColor(235, 170, 0), max(3, int(9 / zoom))))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRectF(x, y, w, h))
                painter.setPen(QColor(200, 120, 0))
                painter.drawText(QPointF(x + 8, y + 40 / zoom), "EKSIK?")
            except Exception:
                continue

        # Önceki oturumlarda kırpılanlar: MAVİ (aynı şeridi tekrar kesmeyin)
        for i, bx in enumerate(getattr(self, "previous_boxes", [])):
            try:
                x, y, w, h = [float(v) for v in bx[:4]]
            except Exception:
                continue
            if i == getattr(self, "secili_kutu", None):
                renk, kalinlik = QColor(150, 40, 200), max(4, int(14 / zoom))
            elif i in getattr(self, "tekrar_idx", set()):
                renk, kalinlik = QColor(220, 30, 30), max(3, int(9 / zoom))
            else:
                renk, kalinlik = QColor(40, 90, 220), max(2, int(6 / zoom))
            painter.setPen(QPen(renk, kalinlik))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(x, y, w, h))
            if i in getattr(self, "tekrar_idx", set()):
                painter.setPen(QColor(200, 20, 20))
                painter.drawText(QPointF(x + 8, y + 70 / zoom), "TEKRAR")

        for ann_item in self.annotations_this_image:
            pen_width = max(2, int(6 / zoom))
            painter.setPen(QPen(Qt.GlobalColor.green, pen_width))
            r = ann_item['crop_rect_draw']
            painter.drawRect(r)
            text = str(ann_item['annotation_data'].get('all_diagnoses_tr') or "?")
            fm = QFontMetricsF(font)
            text_rect = fm.boundingRect(text)
            tl = r.topLeft()
            label_pos = QPointF(tl.x(), tl.y() - 4)
            bg_rect = QRectF(label_pos.x() - 2, label_pos.y() - text_rect.height(),
                             text_rect.width() + 4, text_rect.height() + 2)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 180))
            painter.drawRect(bg_rect)
            painter.setPen(QPen(Qt.GlobalColor.black))
            painter.drawText(label_pos, text)
        painter.end()
        self.viewer.photo_item.setPixmap(temp_pixmap)

    def undo_last_annotation(self):
        if not self.annotations_this_image:
            print("INFO: Bu resim için geri alınacak işlem yok.")
            return
        last_ann_item = self.annotations_this_image.pop()

        fp = last_ann_item['filepath_to_delete']
        try:
            if fp.exists():
                fp.unlink()
                print(f"DELETED FILE: '{fp.name}'")
        except Exception as e:
            print(f"!!! DELETE FILE ERROR: {e}")

        try:
            df = pd.read_csv(ANNOTATIONS_FILE)
            target_filename = last_ann_item['annotation_data']['filename']
            df = df[df['filename'] != target_filename]
            df.to_csv(ANNOTATIONS_FILE, index=False, sep=',', encoding='utf-8-sig')
            self.dataframe = df
            print(f"REMOVED FROM CSV: {target_filename}")
        except Exception as e:
            print(f"!!! CSV UNDO ERROR: {e}")

        self.redraw_annotations()

    def _apply_transform(self, transform_func, info_str):
        if hasattr(self, "current_pixmap") and not self.current_pixmap.isNull():
            self.current_pixmap = transform_func(self.current_pixmap)
            self.annotations_this_image = []
            self.viewer.set_photo(self.current_pixmap)
            self.fit_to_view()
            self.update_status_bar(f"{info_str}. Annotations cleared (geometry changed).")
            print(f"Image {info_str}. Annotations for this image have been cleared.")
        else:
            print("INFO: No image loaded.")

    def rotate_right(self):
        self._apply_transform(
            lambda pix: pix.transformed(QTransform().rotate(90), Qt.TransformationMode.SmoothTransformation),
            "Rotated 90 CW")

    def rotate_left(self):
        self._apply_transform(
            lambda pix: pix.transformed(QTransform().rotate(-90), Qt.TransformationMode.SmoothTransformation),
            "Rotated 90 CCW")

    def rotate_180(self):
        self._apply_transform(
            lambda pix: pix.transformed(QTransform().rotate(180), Qt.TransformationMode.SmoothTransformation),
            "Rotated 180")

    def flip_horizontal(self):
        self._apply_transform(lambda pix: QPixmap.fromImage(pix.toImage().mirrored(True, False)),
                              "Flipped Horizontally")

    def flip_vertical(self):
        self._apply_transform(lambda pix: QPixmap.fromImage(pix.toImage().mirrored(False, True)),
                              "Flipped Vertically")

    def reset_orientation(self):
        if hasattr(self, "current_image_index") and 0 <= self.current_image_index < len(self.image_files):
            path = self.image_files[self.current_image_index]
            pm = QPixmap(str(path))
            if not pm.isNull():
                self.current_pixmap = pm
                self.annotations_this_image = []
                self.viewer.set_photo(self.current_pixmap)
                self.fit_to_view()
                self.update_status_bar("Orientation reset.")
                print("Image orientation has been reset.")

    def fit_to_view(self):
        self.viewer.fitInView(self.viewer.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def save_rotated_copy(self):
        if hasattr(self, "current_pixmap") and not self.current_pixmap.isNull():
            try:
                src = self.image_files[self.current_image_index]
                out_dir = OUTPUT_FOLDER / "_ROTATED_FULL"
                out_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
                out_path = out_dir / f"{src.stem}__rot_{stamp}.png"
                img = self.current_pixmap.toImage()
                writer = QImageWriter(str(out_path), b"png")
                writer.setCompression(6)
                if not writer.write(img):
                    print(f"!!! ERROR saving rotated copy: {writer.errorString()}")
                else:
                    print(f"Saved rotated copy: {out_path.name}")
                    self.statusBar().showMessage(f"Saved rotated copy to {out_dir.name}", 3000)
            except Exception as e:
                print(f"!!! ERROR saving rotated copy: {e}")


if __name__ == "__main__":
    if not INPUT_FOLDER.exists():
        INPUT_FOLDER.mkdir()
        print(f"INFO: '{INPUT_FOLDER}' klasörü oluşturuldu. Taramalarınızı buraya atıp tekrar çalıştırın.")
        sys.exit()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Büyük TIFF taramaları Qt'nin varsayılan bellek sınırına takılmasın (v9.7 düzeltmesi)
    QImageReader.setAllocationLimit(0)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
