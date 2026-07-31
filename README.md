# IDP Invoice OCR MVP

This project is a robust Intelligent Document Processing MVP for extracting structured information from Spanish notary and property registry invoices.

## Main features

- PDF and image upload
- Native PDF text extraction using PyMuPDF
- Tesseract OCR fallback for scanned documents
- Automatic document family detection
- Separate extraction logic for:
  - Notary invoices
  - Property registry invoices
- Multi-page document processing
- Financial consistency validation
- OCR block visualization
- PDF report export

## Extracted fields

### Invoice reference

- Notary or registrar name
- NIF / CIF
- Protocol number
- Invoice number
- Invoice date

### Financial information

- Taxable base
- Retention base
- IVA percentage
- Retention or IRPF percentage
- IVA amount
- Retention amount
- Non-taxable amount
- Final net amount

## Business validation

The application verifies:

```text
IVA amount = Taxable base × IVA percentage

Retention amount = Retention base × Retention percentage

Net amount = Taxable base + IVA amount
             + Non-taxable amount - Retention amount

## Run

```bash
pip install -r requirements.txt
streamlit run idp_mvp_streamlit.py
