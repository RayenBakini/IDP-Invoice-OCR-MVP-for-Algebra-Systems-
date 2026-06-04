import io
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

# =========================
# Configuration Windows
# =========================
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

if pytesseract is not None:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# =========================
# Data models
# =========================
@dataclass
class OCRBlock:
    text: str
    x: int
    y: int
    w: int
    h: int
    confidence: float


# =========================
# preprocessing
# =========================
def pil_to_cv2(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preprocess_image(image: Image.Image) -> Image.Image:
    img = pil_to_cv2(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return Image.fromarray(thr)


def load_pages_from_upload(uploaded_file) -> List[Image.Image]:
    suffix = uploaded_file.name.lower()
    data = uploaded_file.read()

    if suffix.endswith(".pdf"):
        if convert_from_bytes is None:
            raise RuntimeError("Le paquet python 'pdf2image' n'est pas installé.")
        if not os.path.exists(POPPLER_PATH):
            raise RuntimeError(f"Poppler est introuvable au chemin : {POPPLER_PATH}")
        try:
            return convert_from_bytes(data, dpi=220, poppler_path=POPPLER_PATH)
        except Exception as e:
            raise RuntimeError(f"Erreur de conversion PDF avec Poppler : {e}")

    return [Image.open(io.BytesIO(data)).convert("RGB")]


# =========================
# OCR — deux passes : bruts + fusionnés
# =========================
def run_tesseract_ocr_raw(image: Image.Image) -> List[OCRBlock]:
    """Retourne les blocs OCR bruts (sans fusion)."""
    if pytesseract is None:
        raise RuntimeError("Le paquet python 'pytesseract' n'est pas installé.")
    if not os.path.exists(TESSERACT_CMD):
        raise RuntimeError(f"Tesseract est introuvable : {TESSERACT_CMD}")

    try:
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            # psm 6 = bloc de texte uniforme
            # psm 11 = sparse text — utile pour les tableaux / mise en page complexe
            config="--oem 3 --psm 6 -l spa+eng",
        )
    except Exception as e:
        raise RuntimeError(f"Tesseract a échoué : {e}")

    blocks: List[OCRBlock] = []
    n = len(data["text"])

    for i in range(n):
        text = str(data["text"][i]).strip()
        conf_raw = str(data["conf"][i]).strip()

        if not text:
            continue

        try:
            conf = max(0.0, min(1.0, float(conf_raw) / 100.0))
        except Exception:
            conf = 0.0

        blocks.append(OCRBlock(
            text=text,
            x=int(data["left"][i]),
            y=int(data["top"][i]),
            w=int(data["width"][i]),
            h=int(data["height"][i]),
            confidence=conf,
        ))

    return blocks


def run_tesseract_ocr(image: Image.Image) -> Tuple[List[OCRBlock], List[OCRBlock]]:
    """
    nretourniw (blocs_bruts, blocs_fusionnés).
    """
    raw = run_tesseract_ocr_raw(image)
    merged = merge_nearby_words(raw, y_tol=12, x_gap=80)
    return raw, merged


def merge_nearby_words(
        blocks: List[OCRBlock],
        y_tol: int = 12,
        x_gap: int = 80,
) -> List[OCRBlock]:
    """nfusionniw les mots proches sur la même ligne."""
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda b: (b.y, b.x))
    merged: List[OCRBlock] = []
    current = blocks[0]

    for b in blocks[1:]:
        same_line = abs(b.y - current.y) <= y_tol
        near_x = b.x <= current.x + current.w + x_gap

        if same_line and near_x:
            new_text = f"{current.text} {b.text}".strip()
            x1 = min(current.x, b.x)
            y1 = min(current.y, b.y)
            x2 = max(current.x + current.w, b.x + b.w)
            y2 = max(current.y + current.h, b.y + b.h)
            new_conf = (current.confidence + b.confidence) / 2.0
            current = OCRBlock(new_text, x1, y1, x2 - x1, y2 - y1, new_conf)
        else:
            merged.append(current)
            current = b

    merged.append(current)
    return merged


# =========================
# Visualization
# =========================
def draw_boxes(
        image: Image.Image,
        blocks: List[OCRBlock],
        min_conf: float = 0.0,
) -> Image.Image:
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)

    for block in blocks:
        if block.confidence < min_conf:
            continue
        draw.rectangle(
            [block.x, block.y, block.x + block.w, block.y + block.h],
            outline=(255, 0, 0),
            width=2,
        )
    return vis


# =========================
# Anchors
# =========================
ANCHORS: dict[str, List[str]] = {
    "protocol_number": [
        "nº protocolo", "no protocolo", "n° protocolo",
        "num protocolo", "protocolo",
        "n protocolo",
    ],
    "invoice_number": [
        # Notaire : bloc "FACTURA" suivi de "22600767-A"
        "factura",
        # Facture 1 : "Nº Factura: B 1474"
        "nº factura", "no factura", "n° factura",
        "numero factura", "número factura", "n factura",
        # Facture 2 : "NÚMERO: 1.726"
        "número:", "numero:", "número :", "numero :",
        "factura serie",
    ],
    "invoice_date": [
        # Notaire : date dans le bloc FACTURA ou Protocolo
        "factura", "protocolo",
        # Standard
        "fecha factura", "fecha firma",
        "fecha:", "fecha :", "fecha",
        "del:", "del :",
    ],
    "net_amount": [
        # Notaire : "Liquido:  881,59 €"
        "liquido:", "liquido :", "líquido:", "líquido :",
        "liquido", "líquido",
        # Facture 2 : "TOTAL A COBRAR: 558,22 €"
        "total a cobrar",
        # Facture 1
        "importe neto", "importe total",
        # Générique
        "total",
    ],
}


# =========================
# Normalisation
# =========================
def normalize_text(text: str) -> str:

    t = text.lower()
    for src, dst in [
        ("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"),
        ("ú", "u"), ("ü", "u"), ("ñ", "n"),
        # Caractères parasites courants en OCR
        ("\u00ba", "o"),  # º → o
        ("\u00aa", "a"),  # ª → a
        ("°", "o"),
    ]:
        t = t.replace(src, dst)
    # Normalise les tirets OCR en tiret ASCII
    t = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\ufe58\ufe63\uff0d]", "-", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_amount(text: str) -> str:
    """nahiw symboles monétaires, espaces et caractères parasites."""
    return re.sub(r"[€$£\s\u00a0]", "", text).strip()


# =========================
# Scoring bel fallback token-par-token
# =========================
def anchor_score(text: str, anchors: List[str]) -> float:
    t = normalize_text(text)
    for a in anchors:
        na = normalize_text(a)
        if na in t:
            return 1.0
    return 0.0


def format_score(field_name: str, candidate: str) -> float:
    """
    nchoufou ke, le texte candidat correspond au format attendu.
    NOUVEAU : essaie aussi chaque token individuel si le texte complet échoue.
    """
    score = _format_score_single(field_name, candidate.strip())
    if score > 0:
        return score

    # njarbou kol token du candidat séparément
    tokens = candidate.strip().split()
    for tok in tokens:
        s = _format_score_single(field_name, tok.strip())
        if s > 0:
            return s * 0.9  # naamlou pénalité khtr extraction approximative

    return 0.0


def _format_score_single(field_name: str, text: str) -> float:
    """nverifiw l format taa texte unique ."""
    text = text.strip()

    if field_name == "invoice_date":
        if re.search(r"\d{2}/\d{2}/\d{2,4}", text):
            return 1.0
        if re.search(r"\d{1,2}-\d{1,2}-\d{2,4}", text):
            return 1.0
        return 0.0

    if field_name == "net_amount":
        cleaned = clean_amount(text)
        if re.match(r"^[-+]?\d{1,3}(?:[.,]\d{3})*[.,]\d{2}$", cleaned):
            return 1.0
        if re.match(r"^\d+[,.]\d{2}$", cleaned):
            return 1.0
        return 0.0

    if field_name == "protocol_number":
        normalized = re.sub(r"[.\s:]", "", text)
        if re.fullmatch(r"\d{2,10}", normalized):
            return 1.0
        return 0.0

    if field_name == "invoice_number":
        normalized = text.replace(":", "").strip()

        if re.fullmatch(r"\d{4,10}-[A-Za-z]{1,4}", normalized):
            return 1.0

        if re.fullmatch(r"[A-Za-z]{1,4}\s?\d{1,10}", normalized):
            return 1.0

        if re.fullmatch(r"[A-Za-z]{0,4}\s?\d{1,3}(?:\.\d{3})*", normalized):
            return 1.0
        if re.fullmatch(r"\d{1,10}", normalized.replace(".", "")):
            return 1.0
        return 0.0

    return 0.0


def bad_candidate_penalty(text: str) -> float:
    t = normalize_text(text)
    penalties = [
        "calle", "cl ", "palma", "baleares", "union", "fax", "email",
        "otros", "papel", "copias", "honorarios", "conceptos",
        "notario", "norma", "cuenta", "banca", "sabadell", "santander",
        "avda", "avd.", "telefono", "tf.", "nif", "c.i.f",
        "doctor", "oviedo", "asturias",
    ]
    for p in penalties:
        if p in t:
            return 0.0
    return 1.0


# =========================
# Extracteurs inline
# =========================
def extract_value_from_same_block(
        anchor_text: str, field_name: str
) -> Optional[str]:
    t = anchor_text.strip()

    patterns: dict[str, List[str]] = {
        "protocol_number": [
            r"(?:n[ºo°°]?\s*protocolo)\s*[:\-]?\s*([0-9][0-9.]*)",
            r"(?:protocolo)\s*[:\-]?\s*([0-9][0-9.]*)",
        ],
        "invoice_number": [
            # "FACTURA  22600767-A  24/03/26" → capture "22600767-A"
            r"(?:factura)\s+(\d{4,10}-[A-Za-z]{1,4})",
            # "Nº Factura: B 1474"
            r"(?:n[ºo°°]?\s*factura)\s*[:\-]?\s*([A-Za-z]{1,4}\s?\d{1,10})",
            r"(?:n[ºo°°]?\s*factura)\s*[:\-]?\s*(\d{1,3}(?:\.\d{3})*)",
            # "NÚMERO: 1.726"
            r"(?:n[úu]mero)\s*[:\-]?\s*([A-Za-z]{0,4}\s?\d{1,3}(?:\.\d{3})*)",
            r"(?:factura\s*serie\s*[a-z]?)\s*(?:n[úu]mero)\s*[:\-]?\s*(\d{1,3}(?:\.\d{3})*)",
        ],
        "invoice_date": [
            # "FACTURA  22600767-A  24/03/26" → capture la date
            r"(?:factura|protocolo)\s+\S+\s+(\d{2}/\d{2}/\d{2,4})",
            # Protocolo seul sur même ligne : "Protocolo 22600767  24/03/26"
            r"(?:protocolo)\s+\d+\s+(\d{2}/\d{2}/\d{2,4})",
            r"(?:fecha\s*factura)\s*[:\-]?\s*(\d{2}/\d{2}/\d{2,4})",
            r"(?:fecha\s*firma)\s*[:\-]?\s*(\d{2}/\d{2}/\d{2,4})",
            r"(?:del|fecha)\s*[:\-]?\s*(\d{2}/\d{2}/\d{2,4})",
            r"(?:del|fecha)\s*[:\-]?\s*(\d{1,2}-\d{1,2}-\d{2,4})",
            r"(\d{2}/\d{2}/\d{2,4})",
        ],
        "net_amount": [
            r"(?:l[íi]quido)\s*[:\-]?\s*([-+]?\d{1,3}(?:[.,]\d{3})*[.,]\d{2})\s*€?",
            r"(?:total\s*a\s*cobrar)\s*[:\-]?\s*([-+]?\d{1,3}(?:[.,]\d{3})*[.,]\d{2})",
            r"(?:importe\s*(?:neto|total))\s*€?\s*[:\-]?\s*([-+]?\d{1,3}(?:[.,]\d{3})*[.,]\d{2})",
            r"([-+]?\d{1,3}(?:[.,]\d{3})*[.,]\d{2})\s*€",
        ],
    }

    for p in patterns.get(field_name, []):
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return None


# =========================
# Recherche spatiale — MULTI-STRATÉGIE
# Stratégies :
#   1. Inline dans le même bloc
#   2. Blocs à droite (même ligne)
#   3. Blocs en-dessous (colonne)
#   4. Reconstruction ligne virtuelle
#   5. Reconstruction colonne virtuelle
# =========================
def find_candidate_near_anchor(
        raw_blocks: List[OCRBlock],
        merged_blocks: List[OCRBlock],
        field_name: str,
) -> Tuple[Optional[str], float]:
    """
    nlawjou la valeur d'un champ en combinant blocs bruts et fusionnés,
    b 5 stratégies spatiales différentes.
    """
    anchors = ANCHORS[field_name]
    best_value: Optional[str] = None
    best_score: float = -1.0

    # nlawjou ala les blocs fusionnés ET les blocs bruts li les ancres
    for source_blocks in (merged_blocks, raw_blocks):
        for anchor_block in source_blocks:
            a_score = anchor_score(anchor_block.text, anchors)
            if a_score <= 0:
                continue

            # Stratégie 1 : extraction inline
            same_block_value = extract_value_from_same_block(anchor_block.text, field_name)
            if same_block_value is not None:
                f_score = format_score(field_name, same_block_value)
                if f_score > 0:
                    score = 0.45 * anchor_block.confidence + 0.30 * a_score + 0.25 * f_score
                    if score > best_score:
                        best_score = score
                        best_value = same_block_value

            # nlawjou parmi tous les blocs candidats (bruts + fusionnés)
            all_candidates = list(raw_blocks) + list(merged_blocks)

            for candidate in all_candidates:
                if candidate is anchor_block:
                    continue

                dx = candidate.x - anchor_block.x
                dy = candidate.y - anchor_block.y

                #  Stratégie 2 : à droite sur la même ligne
                right_side = (0 <= dx <= 700) and (abs(dy) <= 35)

                #  Stratégie 3 : en-dessous dans la même colonne
                # hethi pour les tableaux notariats où le label est
                # en haut et la valeur en-dessous
                same_column = (abs(dx) <= 80) and (0 < dy <= 100)

                # Stratégie 4 : légèrement en-dessous et à droite
                below_right = (0 <= dx <= 400) and (0 < dy <= 80)

                if not (right_side or same_column or below_right):
                    continue

                f_score = format_score(field_name, candidate.text)
                if f_score <= 0:
                    continue

                penalty = bad_candidate_penalty(candidate.text)
                if penalty <= 0:
                    continue

                distance = abs(dx) + abs(dy)

                if right_side:
                    max_dist = 735.0
                    spatial = max(0.0, 1.0 - distance / max_dist)
                    layout_bonus = 1.0
                elif same_column:
                    max_dist = 180.0
                    spatial = max(0.0, 1.0 - distance / max_dist)
                    layout_bonus = 0.85  # légère pénalité car moins fiable
                else:  # below_right
                    max_dist = 480.0
                    spatial = max(0.0, 1.0 - distance / max_dist)
                    layout_bonus = 0.90

                score = (
                    0.35 * candidate.confidence
                    + 0.25 * a_score
                    + 0.25 * spatial
                    + 0.15 * f_score
                ) * penalty * layout_bonus

                if score > best_score:
                    best_score = score
                    best_value = _extract_best_token(field_name, candidate.text)

            # Stratégie 5 : reconstruction de la ligne virtuelle
            if best_value is None:
                same_row_text = " ".join(
                    b.text for b in sorted(raw_blocks + merged_blocks, key=lambda b: b.x)
                    if abs(b.y - anchor_block.y) <= 20
                )
                inline = extract_value_from_same_block(same_row_text, field_name)
                if inline:
                    f_score = format_score(field_name, inline)
                    if f_score > 0:
                        score = 0.40 * anchor_block.confidence + 0.35 * a_score + 0.25 * f_score
                        if score > best_score:
                            best_score = score
                            best_value = inline

            # Stratégie 6 : reconstruction de la colonne virtuelle
            # nregroupiw les blocs lkol f la même plage X kif l'ancre
            if best_value is None:
                ax_center = anchor_block.x + anchor_block.w // 2
                col_tolerance_x = max(anchor_block.w // 2, 60)
                col_blocks = [
                    b for b in sorted(raw_blocks + merged_blocks, key=lambda b: b.y)
                    if abs((b.x + b.w // 2) - ax_center) <= col_tolerance_x
                    and b.y > anchor_block.y
                ]
                for col_block in col_blocks[:5]:  # nlimitiw à 5 blocs sous l'ancre
                    col_inline = extract_value_from_same_block(col_block.text, field_name)
                    candidate_text = col_inline if col_inline else col_block.text
                    f_score = format_score(field_name, candidate_text)
                    if f_score > 0:
                        score = (
                            0.40 * col_block.confidence
                            + 0.30 * a_score
                            + 0.20 * f_score
                            + 0.10 * max(0.0, 1.0 - (col_block.y - anchor_block.y) / 300)
                        )
                        if score > best_score:
                            best_score = score
                            best_value = candidate_text

    if best_value is None:
        return None, 0.0
    return best_value, min(best_score, 1.0)


def _extract_best_token(field_name: str, text: str) -> str:
    """
    Si le bloc contient plusieurs tokens, retourne celui qui correspond
    au format attendu (ex : "22600767-A 24/03/26" → "22600767-A" pour invoice_number).
    Sinon retourne le texte complet.
    """
    tokens = text.strip().split()
    for tok in tokens:
        if _format_score_single(field_name, tok) > 0:
            return tok
    return text


# =========================
# Post-traitement taa le valeurs extraites
# =========================
def postprocess(field_name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    if field_name == "net_amount":
        return clean_amount(value)

    if field_name in ("protocol_number", "invoice_number"):
        return value.strip(" :.-")

    if field_name == "invoice_date":
        m = re.search(r"\d{2}/\d{2}/\d{2,4}", value)
        if m:
            return m.group(0)
        m = re.search(r"\d{1,2}-\d{1,2}-\d{2,4}", value)
        if m:
            return m.group(0)
        return value

    return value


# =========================
# Agrégation des scores
# =========================
def average_scores(scores: List[float]) -> float:
    valid = [s for s in scores if s is not None]
    if not valid:
        return 0.0
    return float(sum(valid) / len(valid))


# =========================
# lihne naamlou extraction taa MVP principale
# =========================
def extract_mvp_fields(raw_blocks: List[OCRBlock], merged_blocks: List[OCRBlock]) -> dict:
    protocol_number, protocol_number_conf = find_candidate_near_anchor(raw_blocks, merged_blocks, "protocol_number")
    invoice_number, invoice_number_conf = find_candidate_near_anchor(raw_blocks, merged_blocks, "invoice_number")
    invoice_date, invoice_date_conf = find_candidate_near_anchor(raw_blocks, merged_blocks, "invoice_date")
    net_amount, net_amount_conf = find_candidate_near_anchor(raw_blocks, merged_blocks, "net_amount")

    protocol_number = postprocess("protocol_number", protocol_number)
    invoice_number = postprocess("invoice_number", invoice_number)
    invoice_date = postprocess("invoice_date", invoice_date)
    net_amount = postprocess("net_amount", net_amount)

    invoice_ref_block = average_scores([protocol_number_conf, invoice_number_conf, invoice_date_conf])
    financial_block = average_scores([net_amount_conf])
    global_conf = average_scores([invoice_ref_block, financial_block])

    return {
        "invoice_reference": {
            "protocol_number": {"value": protocol_number, "confidence": round(protocol_number_conf, 3)},
            "invoice_number": {"value": invoice_number, "confidence": round(invoice_number_conf, 3)},
            "invoice_date": {"value": invoice_date, "confidence": round(invoice_date_conf, 3)},
            "block_confidence": round(invoice_ref_block, 3),
        },
        "financial": {
            "net_amount": {"value": net_amount, "confidence": round(net_amount_conf, 3)},
            "block_confidence": round(financial_block, 3),
        },
        "global_confidence": round(global_conf, 3),
    }


# =========================
# Debug helper
# =========================
def debug_show_all_blocks(raw_blocks: List[OCRBlock], merged_blocks: List[OCRBlock]) -> None:
    st.subheader(" Debug — blocs bruts OCR")
    rows_raw = [{"text": b.text, "x": b.x, "y": b.y, "w": b.w, "h": b.h, "confidence": round(b.confidence, 3)} for b in raw_blocks]
    st.dataframe(pd.DataFrame(rows_raw), use_container_width=True)

    st.subheader(" Debug — blocs fusionnés OCR")
    rows_merged = [{"text": b.text, "x": b.x, "y": b.y, "w": b.w, "h": b.h, "confidence": round(b.confidence, 3)} for b in merged_blocks]
    st.dataframe(pd.DataFrame(rows_merged), use_container_width=True)

    st.subheader(" Debug — scores ancres par champ (blocs fusionnés)")
    for field, anchors in ANCHORS.items():
        st.write(f"**{field}**")
        hits = [
            (b.text, round(anchor_score(b.text, anchors), 2), round(b.confidence, 2))
            for b in merged_blocks
            if anchor_score(b.text, anchors) > 0
        ]
        if hits:
            st.dataframe(pd.DataFrame(hits, columns=["bloc", "anchor_score", "conf"]), use_container_width=True)
        else:
            st.warning(f"Aucun ancre trouvé pour `{field}` dans les blocs fusionnés")

    st.subheader(" Debug — test format_score sur chaque bloc")
    for field in ANCHORS.keys():
        hits = [
            (b.text, round(format_score(field, b.text), 2))
            for b in raw_blocks
            if format_score(field, b.text) > 0
        ]
        if hits:
            st.write(f"**{field}** — candidats valides :")
            st.dataframe(pd.DataFrame(hits, columns=["bloc", "format_score"]), use_container_width=True)



# =========================
# ngeneriw PDF report
# =========================
def generate_extraction_pdf(extraction: dict) -> bytes:
    """
    Generate a clean PDF report with extracted invoice information.
    Confidence scores are intentionally hidden from the exported report.
    """
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("IDP Invoice OCR - Extraction Report", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 0.5 * cm))



    ref = extraction.get("invoice_reference", {})
    fin = extraction.get("financial", {})

    data = [
        ["Field", "Extracted value"],
        ["Protocol number", ref.get("protocol_number", {}).get("value") or "Not detected"],
        ["Invoice number", ref.get("invoice_number", {}).get("value") or "Not detected"],
        ["Invoice date", ref.get("invoice_date", {}).get("value") or "Not detected"],
        ["Net amount", fin.get("net_amount", {}).get("value") or "Not detected"],
    ]

    table = Table(data, colWidths=[6 * cm, 9 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
            ]
        )
    )

    story.append(table)

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes

# =========================
# lihne aana Streamlit app
# =========================
st.set_page_config(page_title="IDP MVP - OCR Blocks", layout="wide")
st.title("IDP MVP — OCR structuré + visualisation")
st.caption("Lecture de facture · Blocs OCR · Extraction 4 champs MVP · Multi-stratégie")

with st.sidebar:
    st.header("Paramètres")
    apply_preprocessing = st.checkbox("Prétraitement OCR", value=True)
    min_conf_display = st.slider("Confiance minimale affichée", 0.0, 1.0, 0.2, 0.05)
    show_debug = st.checkbox("Afficher debug ancres", value=False)
    show_raw = st.checkbox("Afficher blocs bruts (non fusionnés)", value=False)

uploaded_file = st.file_uploader(
    "Charge une facture (PDF, PNG, JPG, JPEG)",
    type=["pdf", "png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.info("Commence par charger une facture pour lancer le MVP.")
    st.stop()

try:
    pages = load_pages_from_upload(uploaded_file)
except Exception as e:
    st.error(f"Erreur au chargement du document : {e}")
    st.stop()

page_index = st.selectbox(
    "Page",
    options=list(range(len(pages))),
    format_func=lambda x: f"Page {x + 1}",
)
original = pages[page_index]
processed = preprocess_image(original) if apply_preprocessing else original

try:
    raw_blocks, merged_blocks = run_tesseract_ocr(processed)
except Exception as e:
    st.error(f"Erreur OCR : {e}")
    st.stop()

# nvisualisiw b les blocs fusionnés
vis = draw_boxes(original, merged_blocks, min_conf=min_conf_display)
extraction = extract_mvp_fields(raw_blocks, merged_blocks)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visualisation", "Blocs OCR", "Extraction MVP", "Debug", "Bruts vs Fusionnés"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Document")
        st.image(original, use_container_width=True)
    with col2:
        st.subheader("Blocs détectés")
        st.image(vis, use_container_width=True)

with tab2:
    display_blocks = raw_blocks if show_raw else merged_blocks
    rows = [asdict(b) for b in display_blocks if b.confidence >= min_conf_display]
    df = pd.DataFrame(rows)
    st.caption(f"Mode : {'blocs bruts' if show_raw else 'blocs fusionnés'} — {len(rows)} blocs")
    st.dataframe(df, use_container_width=True)

with tab3:
    st.json(extraction)
    st.subheader("Lecture rapide")
    ref = extraction["invoice_reference"]
    fin = extraction["financial"]
    st.write(f"**Nº Protocole** : {ref['protocol_number']['value']}  (conf: {ref['protocol_number']['confidence']})")
    st.write(f"**Nº Facture**   : {ref['invoice_number']['value']}   (conf: {ref['invoice_number']['confidence']})")
    st.write(f"**Date Facture** : {ref['invoice_date']['value']}     (conf: {ref['invoice_date']['confidence']})")
    st.write(f"**Montant Net**  : {fin['net_amount']['value']}        (conf: {fin['net_amount']['confidence']})")
    st.markdown("---")
    st.write(f"Score bloc références : **{ref['block_confidence']}**")
    st.write(f"Score bloc financier  : **{fin['block_confidence']}**")
    st.write(f"Score global          : **{extraction['global_confidence']}**")

    # =========================
    # lihne zidit PDF export button
    # =========================
    st.markdown("---")
    st.subheader("Export PDF")

    pdf_report = generate_extraction_pdf(extraction)

    st.download_button(
        label="Download extraction PDF report",
        data=pdf_report,
        file_name="invoice_extraction_report.pdf",
        mime="application/pdf",
    )

    if extraction["global_confidence"] < 0.5:
        st.warning(
            " Confiance globale faible (< 0.5). "
        )

with tab4:
    if show_debug:
        debug_show_all_blocks(raw_blocks, merged_blocks)
    else:
        st.info("Coche 'Afficher debug ancres' dans la barre latérale pour voir le diagnostic.")

with tab5:
    st.subheader("Comparaison : bruts vs fusionnés")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Blocs bruts** ({len(raw_blocks)} blocs)")
        st.dataframe(pd.DataFrame([asdict(b) for b in raw_blocks]), use_container_width=True)
    with c2:
        st.write(f"**Blocs fusionnés** ({len(merged_blocks)} blocs)")
        st.dataframe(pd.DataFrame([asdict(b) for b in merged_blocks]), use_container_width=True)

st.markdown("---")
