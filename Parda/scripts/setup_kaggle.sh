#!/usr/bin/env bash
# One-time setup per Kaggle session: fonts for Hindi/Kannada + text shaping.
# Needs Notebook Settings -> Internet: ON
set -u
echo ">> installing fonts (Noto: Latin, Devanagari, Kannada) + fribidi/raqm for Indic shaping"
apt-get -qq update >/dev/null 2>&1 || true
apt-get -qq install -y fonts-noto-core libfribidi0 libraqm0 >/dev/null 2>&1 || echo "   apt install failed, will try direct download"

# Fallback: download fonts directly if apt didn't provide them
FONT_DIR=/kaggle/working/fonts
mkdir -p "$FONT_DIR"
BASE=https://github.com/notofonts/notofonts.github.io/raw/main/fonts
for f in NotoSans/hinted/ttf/NotoSans-Regular.ttf NotoSans/hinted/ttf/NotoSans-Bold.ttf \
         NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf \
         NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Bold.ttf \
         NotoSansKannada/hinted/ttf/NotoSansKannada-Regular.ttf \
         NotoSansKannada/hinted/ttf/NotoSansKannada-Bold.ttf; do
  name=$(basename "$f")
  if ! fc-list 2>/dev/null | grep -q "$name" && [ ! -f "$FONT_DIR/$name" ]; then
    wget -q -O "$FONT_DIR/$name" "$BASE/$f" || rm -f "$FONT_DIR/$name"
  fi
done

echo ">> python packages"
pip install -q -r requirements.txt

python - <<'PY'
from PIL import features
from parda.synth.canvas import available_scripts
print("RAQM text shaping:", features.check("raqm"))
print("fonts available  :", available_scripts())
PY
