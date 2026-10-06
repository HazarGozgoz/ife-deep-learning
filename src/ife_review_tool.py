# -*- coding: utf-8 -*-
"""
ife_review_tool.py  (IFE Review Studio - Serum-Only, v3.0)
============================================================
go_zden_gecirme_aracı.py (Review_2.0_Grouped) dosyasının güncellenmiş sürümü.
Zaten etiketlenmiş kırpmaları primary_diagnosis'e göre GRUPLAYIP art arda
gösterir (aynı sınıftaki hatalar göze daha kolay çarpar), 'D' tuşuyla
düzeltme penceresi açar.

Değişiklikler
-------------
1) İdrar tamamen kaldırıldı (SampleTypeDialog + URINE_DIAGNOSIS_MAP yok).
2) Tanı sözlüğü artık ife_taxonomy.DIAGNOSIS_MAP (tek kaynak).
3) ÖNEMLİ HATA DÜZELTMESİ: Orijinal dosyada "Diğer..." kutusuna girilen
   serbest metin, virgülle bölünüp OLDUĞU GİBİ sınıf adı olarak
   ekleniyordu:

       other_en_list = [s.strip().replace(" ", "_") for s in other_text.split(',') ...]
       final_en.extend(other_en_list)

   Yani biri "Diğer" kutusuna "hemoliz, tekrar bak" yazarsa, all_diagnoses
   sütununa gerçek bir sınıfmış gibi "HEMOLIZ" ve "TEKRAR_BAK" diye iki yeni
   "sınıf" ekleniyordu -- bu da nadir sınıfları daha da seyrekleştiren,
   fark edilmesi zor bir kaynaktı. Artık annotator/annotasyon aracıyla
   TUTARLI şekilde: serbest metin her zaman tek bir kanonik "OTHER" sınıfına
   düşer, kendisi ayrı bir not olarak saklanır.
"""
import sys
from datetime import datetime
import pandas as pd
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
                              QGraphicsPixmapItem, QDialog, QCheckBox, QVBoxLayout, QHBoxLayout,
                              QDialogButtonBox, QToolBar, QInputDialog, QScrollArea, QLineEdit,
                              QStatusBar, QLabel, QMessageBox, QComboBox, QPushButton)
from PyQt6.QtGui import QAction, QPainter, QPixmap, QKeySequence
from PyQt6.QtCore import Qt

import argparse
def _txt(v) -> str:
    """NaN-guvenli metin okuma. pandas bos hucreyi NaN (float) verir ve
    float('nan') Python'da DOGRU sayilir; `str(v or "")` bu yuzden "nan"
    uretir. Bu yardimci onu bos stringe cevirir."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    t = str(v).strip()
    return "" if t.lower() == "nan" else t


from ife_dialog import IFEDiagnosisDialog
from ife_taxonomy import (Band, IFERecord, PATTERN_TR, check_consistency,
                          count_distinct_clones, parse_bands, parse_set_field,
                          FLAG_CLASSES, POLY_LANE_CLASSES, to_legacy_labels)

# --- KONFİGÜRASYON ---
CROPPED_FOLDER = Path("Cropped_Data_EN")
ANNOTATIONS_FILE = Path("annotations_master.csv")
TOOL_VERSION = "Review_3.0_Serum_Only"
ANNOTATOR_NAME = "Dr. Hazar Gozgoz"
MIN_ZOOM, MAX_ZOOM = 0.1, 15.0
FLAGGED_ONLY = False  # --only-flagged ile acilir
HATA_LISTESI = None   # --hata-listesi ai_suspected_errors.csv ile acilir
# Hata avi kararlari OTOMATIK kaydedilir: her kayit icin "duzeltildi" ya da
# "uzman hakli" satiri. Makaledeki "N celiskiden X'inde etiket duzeltildi"
# sayisi buradan cikar; elle not tutmaniza gerek yok.
KARAR_LOG = Path("hata_avi_kararlari.csv")
# --- KOR IKINCI OKUYUCU MODU (--kor-okuyucu "Dr. Ad Soyad") ---------------
# Gozlemciler arasi uyum (kappa) icin ikinci bir uzman, rastgele secilmis
# panelleri ETIKETI GORMEDEN okur. Master CSV'ye HIC yazilmaz; sonuclar
# ayri dosyaya gider. Ornek listesi ilk calistirmada cekilir ve sabitlenir;
# sonraki acilislarda kaldigi yerden devam eder.
KOR_OKUYUCU = None
KOR_ORNEK_N = 300
KOR_SEED = 42
KOR_ORNEK_FILE = Path("ikinci_okuyucu_ornek.csv")
KOR_OUT = Path("ikinci_okuyucu_annotations.csv")
KOR_TOOL_VERSION = "Review_3.1_Blind_Reader"


class ReviewPhotoViewer(QGraphicsView):
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
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def set_photo(self, pixmap):
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
        if not event.pixelDelta().isNull():
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - event.pixelDelta().x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - event.pixelDelta().y())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.main_window.trigger_correction()
        super().mousePressEvent(event)


class ReviewWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"IFE Review Studio v{TOOL_VERSION} (Serum-Only, Grup Modlu)")
        self.setGeometry(50, 50, 1200, 900)

        self.dataframe = self.load_and_prep_dataframe()
        if self.dataframe is None:
            sys.exit("Hata: CSV dosyası yüklenemedi.")

        if not KOR_OKUYUCU:
            self.dataframe.sort_values(by=['pattern', 'bands', 'filename'], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)
        self.kor_etiketli: set[str] = set()
        if KOR_OKUYUCU:
            self._kor_hazirla()

        # ÖNEMLİ: idrar satırlarını dataframe'den ATMIYORUZ, sadece bu
        # inceleme oturumunda GEZİLECEK satırların indekslerini ayırıyoruz.
        # Böylece self.dataframe her zaman TAM veriyi tutar ve geri
        # kaydedildiğinde (to_csv) hiçbir satır sessizce kaybolmaz.
        if "sample_type" in self.dataframe.columns:
            is_serum = self.dataframe["sample_type"].astype(str).str.upper() == "SERUM"
            sel = is_serum.copy()
            # --only-flagged: sadece elden gecirilmesi gereken kayitlari gezer.
            # etiket_gocu.py'nin isaretledigi biklonal/klon tutarsizliklari +
            # QC bayraklari. Boylece 2900 kaydin tamamini gezmek zorunda
            # kalmadan, sadece gercekten bakilmasi gereken satirlari acarsiniz.
            if FLAGGED_ONLY:
                # etiket_gocu.py 'review_reason' kolonunu yazar. DİKKAT: boş
                # hücreler CSV'den NaN olarak geri okunur; fillna("") olmadan
                # str(NaN) == "nan" olur ve HER kayıt işaretli sayılırdı.
                if "review_reason" not in self.dataframe.columns:
                    print("UYARI: 'review_reason' kolonu yok. Önce etiket_gocu.py "
                          "çalıştırın. Tüm kayıtlar gösteriliyor.")
                else:
                    flag = self.dataframe["review_reason"].fillna("").astype(str).str.strip().ne("")
                    sel = sel & flag
            self.review_indices = list(self.dataframe.index[sel])

            # --hata-listesi: modelin celistigi kayitlar, supheye gore sirali
            self.ai_claims: dict[str, str] = {}
            if HATA_LISTESI:
                try:
                    h = pd.read_csv(HATA_LISTESI)
                    h = h.sort_values("ScoreNorm", ascending=False)
                    sira = []
                    for _, r in h.iterrows():
                        ad = str(r.get("Dosya_Adi", ""))
                        if ad not in self.ai_claims:
                            self.ai_claims[ad] = (
                                f"YZ: {r.get('Durum', '')}  |  hedef: {r.get('Bant_TR', r.get('Bulgu', ''))}"
                                f"  (%{float(r.get('Model_Prob', 0)) * 100:.0f})\n"
                                f"YZ'nin gordugu: {r.get('YZ_Tam_Tahmini', '')}")
                            sira.append(ad)
                    fn = self.dataframe["filename"].astype(str)
                    konum = {ad: i for i, ad in zip(self.dataframe.index, fn)}
                    self.review_indices = [konum[ad] for ad in sira if ad in konum]
                    print(f"INFO: --hata-listesi: {len(self.review_indices)} kayit, "
                          f"supheden en yuksek basta.")
                except Exception as e:
                    print(f"UYARI: hata listesi okunamadi ({e}).")
            n_urine = int((~is_serum).sum())
            if n_urine:
                print(f"INFO: {n_urine} idrar kaydı gösterilmiyor (dosyadan SİLİNMEDİ).")
            if FLAGGED_ONLY:
                print(f"INFO: --only-flagged: {len(self.review_indices)} işaretli kayıt "
                      f"gezilecek (toplam {int(is_serum.sum())} serum kaydı içinden).")
                if not self.review_indices:
                    print("      İşaretli kayıt kalmamış - düzeltilecek bir şey yok.")
        else:
            self.review_indices = list(self.dataframe.index)

        if KOR_OKUYUCU:
            konum = {str(f): i for i, f in zip(self.dataframe.index, self.dataframe["filename"].astype(str))}
            self.review_indices = [konum[f] for f in self.kor_ornek if f in konum]
        self.total_records = len(self.review_indices)
        self.current_index = -1
        if KOR_OKUYUCU and self.kor_etiketli:
            # kaldigi yerden: ilk etiketsiz panelin bir oncesine kur
            for k, i in enumerate(self.review_indices):
                if str(self.dataframe.at[i, "filename"]) not in self.kor_etiketli:
                    self.current_index = k - 1
                    break
            else:
                self.current_index = self.total_records - 2

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.info_label = QLabel("Yükleniyor...")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: darkblue; padding: 10px; "
            "background-color: #f0f0f0; border: 2px solid #ccc;")
        self.info_label.setWordWrap(True)
        main_layout.addWidget(self.info_label)

        self.viewer = ReviewPhotoViewer(self)
        main_layout.addWidget(self.viewer)

        self.setStatusBar(QStatusBar(self))
        self.create_toolbar_and_shortcuts()
        self.next_image()

    def _kor_hazirla(self):
        """Rastgele ornek listesini olustur/oku ve daha once etiketlenenleri yukle."""
        d = self.dataframe
        if "sample_type" in d.columns:
            d = d[d["sample_type"].astype(str).str.upper() == "SERUM"]
        if KOR_ORNEK_FILE.exists():
            self.kor_ornek = list(pd.read_csv(KOR_ORNEK_FILE)["filename"].astype(str))
            print(f"INFO: kor ornek listesi okundu: {KOR_ORNEK_FILE} ({len(self.kor_ornek)} panel)")
        else:
            n = min(KOR_ORNEK_N, len(d))
            ornek = d.sample(n=n, random_state=KOR_SEED)
            ornek[["filename", "img_relpath", "device", "source_file"]].to_csv(
                KOR_ORNEK_FILE, index=False, encoding="utf-8-sig")
            self.kor_ornek = list(ornek["filename"].astype(str))
            print(f"INFO: {n} panellik rastgele ornek cekildi (seed {KOR_SEED}) -> {KOR_ORNEK_FILE}")
        if KOR_OUT.exists():
            try:
                self.kor_etiketli = set(pd.read_csv(KOR_OUT)["filename"].astype(str))
            except Exception:
                self.kor_etiketli = set()
        print(f"INFO: KOR OKUYUCU = {KOR_OKUYUCU} | etiketli {len(self.kor_etiketli)} / {len(self.kor_ornek)}")

    def _kor_kaydet(self, record, new_rec):
        """Ikinci okuyucunun kararini ayri CSV'ye yazar (ayni panel yeniden okunursa ustune yazar)."""
        satir = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "annotator": KOR_OKUYUCU, "tool_version": KOR_TOOL_VERSION,
            "filename": str(record["filename"]), "img_relpath": _txt(record.get("img_relpath")),
            "device": _txt(record.get("device")), "source_file": _txt(record.get("source_file")),
            "pattern": new_rec.pattern, "bands": new_rec.bands_str(),
            "n_bands": new_rec.n_bands(), "n_clones": count_distinct_clones(new_rec.bands),
            "flags": new_rec.flags_str(), "poly_lanes": new_rec.poly_lanes_str(),
            "band_target": ";".join(sorted(new_rec.band_target())),
            "band_target_definite": ";".join(sorted(new_rec.band_target_definite_only())),
            "has_suspicious": int(new_rec.has_suspicious()), "detail_note": new_rec.note,
        }
        if KOR_OUT.exists():
            eski = pd.read_csv(KOR_OUT, dtype=str)
            eski = eski[eski["filename"].astype(str) != satir["filename"]]
            yeni = pd.concat([eski, pd.DataFrame([satir])], ignore_index=True)
        else:
            yeni = pd.DataFrame([satir])
        yeni.to_csv(KOR_OUT, index=False, encoding="utf-8-sig")
        self.kor_etiketli.add(satir["filename"])

    def load_and_prep_dataframe(self):
        if not ANNOTATIONS_FILE.exists():
            return None
        try:
            df = pd.read_csv(ANNOTATIONS_FILE)
            try:
                df.to_csv(ANNOTATIONS_FILE.with_suffix('.review_bak.csv'), index=False, encoding='utf-8-sig')
            except Exception:
                pass
            return df
        except Exception as e:
            print(f"CSV Hata: {e}")
            return None

    def create_toolbar_and_shortcuts(self):
        toolbar = QToolBar("Tools")
        self.addToolBar(toolbar)
        prev = QAction("Previous (P)", self)
        prev.triggered.connect(self.previous_image)
        prev.setShortcut(QKeySequence("P"))
        toolbar.addAction(prev)
        next_act = QAction("Next (N)", self)
        next_act.triggered.connect(self.next_image)
        next_act.setShortcut(QKeySequence("N"))
        toolbar.addAction(next_act)
        correct_act = QAction("DÜZELT (D)", self)
        correct_act.triggered.connect(self.trigger_correction)
        correct_act.setShortcut(QKeySequence("D"))
        toolbar.addAction(correct_act)

        self.addAction(self._mk_shortcut(Qt.Key.Key_Left, self.previous_image))
        self.addAction(self._mk_shortcut(Qt.Key.Key_Right, self.next_image))
        # D kisayolu zaten toolbar action'inda tanimli; ikinci kez
        # eklemek Qt'de "Ambiguous shortcut overload" uyarisi uretiyordu.

    def _mk_shortcut(self, key, slot):
        act = QAction(self)
        act.setShortcut(QKeySequence(key))
        act.triggered.connect(slot)
        return act

    def update_ui_info(self):
        if not (0 <= self.current_index < self.total_records):
            return
        record = self.dataframe.loc[self.review_indices[self.current_index]]
        if KOR_OKUYUCU:
            etiketli = str(record["filename"]) in self.kor_etiketli
            self.info_label.setText(f"KÖR OKUMA — Panel {self.current_index + 1} / {self.total_records}"
                                    f"{'   (etiketlendi ✓)' if etiketli else ''}\n"
                                    "Tanı girmek için görüntüye tıklayın ya da D")
            self.info_label.setStyleSheet("font-size: 17px; font-weight: bold; color: #333; "
                                          "background-color: #f0f0f0; padding: 10px; border-bottom: 3px solid #333;")
            self.statusBar().showMessage(f"Etiketli: {len(self.kor_etiketli)} / {self.total_records}")
            return
        pat = PATTERN_TR.get(str(record.get("pattern", "")), str(record.get("pattern", "")))
        bands = str(record.get("bands", "") or "—")
        flags = str(record.get("flags", "") or "—")
        reason = _txt(record.get("review_reason"))

        color = "#a00" if reason else "darkblue"
        claim = getattr(self, "ai_claims", {}).get(str(record["filename"]), "")
        self.info_label.setText(f"SİZ: {pat}\nBantlar: {bands}\nBulgular: {flags}"
                                + (f"\n⚠ {reason}" if reason else "")
                                + (f"\n\n{claim}" if claim else ""))
        self.info_label.setStyleSheet(
            f"font-size: 17px; font-weight: bold; color: {color}; background-color: #f0f0f0; "
            f"padding: 10px; border-bottom: 3px solid {color};")
        self.statusBar().showMessage(
            f"Kayıt: {self.current_index + 1} / {self.total_records} | "
            f"{record['filename']} | {record.get('device', '?')}")

    def change_image(self, direction):
        new_index = self.current_index + direction
        if not (0 <= new_index < self.total_records):
            print("Liste sonu.")
            return
        self.current_index = new_index
        record = self.dataframe.loc[self.review_indices[self.current_index]]
        rel = _txt(record.get('img_relpath'))
        if not rel:
            rel = f"{record['primary_diagnosis']}/{record['filename']}"
        filepath = CROPPED_FOLDER / rel
        if not filepath.exists():
            print(f"Resim bulunamadı: {filepath}, geçiliyor...")
            self.change_image(direction)
            return
        self.viewer.set_photo(QPixmap(str(filepath)))
        self.update_ui_info()

    def _karar_yaz(self, karar: str, eski: str = "", yeni: str = ""):
        """Hata avi modunda her karari CSV'ye ekler (yalnizca --hata-listesi ile)."""
        if not HATA_LISTESI or not (0 <= self.current_index < len(self.review_indices)):
            return
        rec = self.dataframe.loc[self.review_indices[self.current_index]]
        ad = str(rec["filename"])
        satir = {"zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "kirpma_dosyasi": ad, "karar": karar,
                 "yz_iddiasi": self.ai_claims.get(ad, "").replace("\n", " | "),
                 "eski_etiket": eski, "yeni_etiket": yeni}
        yeni_dosya = not KARAR_LOG.exists()
        pd.DataFrame([satir]).to_csv(KARAR_LOG, mode="a", header=yeni_dosya,
                                     index=False, encoding="utf-8-sig")

    def next_image(self):
        if KOR_OKUYUCU and 0 <= self.current_index < self.total_records:
            rec = self.dataframe.loc[self.review_indices[self.current_index]]
            if str(rec["filename"]) not in self.kor_etiketli:
                QMessageBox.warning(self, "Kör okuma", "Bu paneli önce etiketleyin (görüntüye tıklayın ya da D).")
                return
            if self.current_index == self.total_records - 1:
                QMessageBox.information(self, "Tamamlandı",
                                        f"Tüm paneller etiketlendi: {len(self.kor_etiketli)} / {self.total_records}\n"
                                        f"Sonuç dosyası: {KOR_OUT}")
                return
        # Hata avinda "N" = uzman hakli, model yanildi
        if HATA_LISTESI and not getattr(self, "_son_duzeltme_index", None) == self.current_index:
            rec = self.dataframe.loc[self.review_indices[self.current_index]]
            self._karar_yaz("UZMAN HAKLI", eski=str(rec.get("all_diagnoses_tr", "")))
        self.change_image(1)

    def previous_image(self):
        self.change_image(-1)

    def trigger_correction(self):
        if not (0 <= self.current_index < self.total_records):
            return
        row_idx = self.review_indices[self.current_index]
        record = self.dataframe.loc[row_idx]
        if KOR_OKUYUCU:
            dlg = IFEDiagnosisDialog(record=None, parent=self,
                                     title=f"Kör okuma — Panel {self.current_index + 1} / {self.total_records}")
            if not dlg.exec():
                return
            new_rec = dlg.get_record()
            warn = check_consistency(new_rec)
            if warn and QMessageBox.question(self, "Tutarlılık Uyarısı", warn + "\n\nYine de kaydedilsin mi?",
                                             QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                                             QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Save:
                return
            self._kor_kaydet(record, new_rec)
            self.update_ui_info()
            if self.current_index < self.total_records - 1:
                self.change_image(1)
            else:
                QMessageBox.information(self, "Tamamlandı",
                                        f"Tüm paneller etiketlendi: {len(self.kor_etiketli)} / {self.total_records}\n"
                                        f"Sonuç dosyası: {KOR_OUT}")
            return
        idx = row_idx
        eski_tr = str(record.get("all_diagnoses_tr", ""))

        # Mevcut kaydı üç eksenli yapıya yükle -> pencerede ÖNCEDEN DOLU gelir,
        # sadece düzeltmeniz gereken yeri değiştirirsiniz.
        rec = IFERecord(
            bands=parse_bands(record.get("bands", "")),
            flags=parse_set_field(record.get("flags", ""), FLAG_CLASSES),
            poly_lanes=parse_set_field(record.get("poly_lanes", ""), POLY_LANE_CLASSES),
            note=_txt(record.get("detail_note")),
        )
        rec.resolve_pattern()

        reason = _txt(record.get("review_reason"))
        title = "Tanı Düzeltme"
        if reason:
            title += "  —  " + reason[:70]
        dlg = IFEDiagnosisDialog(record=rec, parent=self, title=title)
        if not dlg.exec():
            return
        new_rec = dlg.get_record()

        warn = check_consistency(new_rec)
        if warn:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Tutarlılık Uyarısı")
            box.setText(warn)
            box.setInformativeText("Yine de kaydedilsin mi?")
            box.setStandardButtons(QMessageBox.StandardButton.Save |
                                   QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(QMessageBox.StandardButton.Cancel)
            if box.exec() != QMessageBox.StandardButton.Save:
                return

        # NOT: görüntü dosyası DİSKTE OYNATILMAZ. img_relpath sabit kalır.
        legacy = ";".join(sorted(to_legacy_labels(new_rec)))
        d = self.dataframe
        d.at[row_idx, "pattern"] = new_rec.pattern
        d.at[row_idx, "bands"] = new_rec.bands_str()
        d.at[row_idx, "n_bands"] = new_rec.n_bands()
        d.at[row_idx, "n_clones"] = count_distinct_clones(new_rec.bands)
        d.at[row_idx, "flags"] = new_rec.flags_str()
        d.at[row_idx, "poly_lanes"] = new_rec.poly_lanes_str()
        d.at[row_idx, "band_target"] = ";".join(sorted(new_rec.band_target()))
        d.at[row_idx, "band_target_definite"] = ";".join(
            sorted(new_rec.band_target_definite_only()))
        d.at[row_idx, "has_suspicious"] = int(new_rec.has_suspicious())
        d.at[row_idx, "detail_note"] = new_rec.note
        d.at[row_idx, "all_diagnoses_canonical"] = legacy
        d.at[row_idx, "review_reason"] = ""      # elle onaylandı
        d.at[row_idx, "annotator"] = f"{ANNOTATOR_NAME} (Düzeltme)"

        self.dataframe.to_csv(ANNOTATIONS_FILE, index=False, sep=",", encoding="utf-8-sig")
        # Hata avi karari: model hakliydi, etiket duzeltildi
        self._karar_yaz("DUZELTILDI (model hakli)", eski=eski_tr,
                        yeni=str(self.dataframe.at[idx, "all_diagnoses_tr"]))
        self._son_duzeltme_index = self.current_index
        self.update_ui_info()
        QMessageBox.information(
            self, "Kaydedildi",
            f"Patern: {PATTERN_TR.get(new_rec.pattern, new_rec.pattern)}\n"
            f"Bant: {new_rec.n_bands()} | Klon: {count_distinct_clones(new_rec.bands)}")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--only-flagged", action="store_true",
                     help="Sadece etiket_gocu.py'nin isaretledigi sorunlu kayitlari goster")
    _ap.add_argument("--hata-listesi", default=None,
                     help="ai_suspected_errors.csv: modelin sizinle celistigi kayitlari "
                          "supheden en yuksekten baslayarak gezer")
    _ap.add_argument("--kor-okuyucu", default=None, metavar="AD",
                     help="KOR IKINCI OKUYUCU: rastgele N paneli etiketi gormeden okur; sonuc "
                          "ikinci_okuyucu_annotations.csv'ye yazilir, master'a DOKUNULMAZ.")
    _ap.add_argument("--ornek", type=int, default=KOR_ORNEK_N, help="kor okuma icin panel sayisi (vars. 300)")
    _ap.add_argument("--seed", type=int, default=KOR_SEED)
    _args, _rest = _ap.parse_known_args()
    FLAGGED_ONLY = _args.only_flagged
    HATA_LISTESI = _args.hata_listesi
    KOR_OKUYUCU = _args.kor_okuyucu
    KOR_ORNEK_N = _args.ornek
    KOR_SEED = _args.seed

    app = QApplication(sys.argv[:1] + _rest)
    window = ReviewWindow()
    window.show()
    sys.exit(app.exec())
