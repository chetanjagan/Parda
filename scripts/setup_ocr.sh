#!/usr/bin/env bash
# Phase 2 setup on Kaggle: Tesseract (+Hindi, Kannada), EasyOCR, rapidfuzz. Needs Internet ON.
set -u
echo ">> tesseract + Hindi + Kannada language packs"
apt-get -qq update >/dev/null 2>&1 || true
apt-get -qq install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-kan >/dev/null 2>&1 || echo "   apt failed"
tesseract --list-langs 2>/dev/null | tr '\n' ' '; echo
echo ">> python packages"
pip install -q -r requirements.txt
pip install -q easyocr
python -c "import easyocr, rapidfuzz; print('easyocr', easyocr.__version__, '| rapidfuzz ok')"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (turn on GPU T4)')"
