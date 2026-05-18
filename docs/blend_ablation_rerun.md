# Blend Ablation 재실행 가이드 (Colab)

hash 버그 수정(hashlib.md5) 이후, casda_no_blend와 casda_composed를
동일한 배경·jitter로 재생성하기 위한 명령이다.
두 step을 **순서대로** 실행해야 한다 (bg-cache를 공유하므로 Step 1 완료 후 Step 2 실행).

---

## 0. 환경변수 설정

Colab 셀에서 경로를 실제 환경에 맞게 수정 후 실행:

```bash
export GENERATED_DIR="/content/drive/MyDrive/data/Severstal/outputs/generated"
export HINT_DIR="/content/drive/MyDrive/data/Severstal/data/processed/controlnet_dataset/hints"
export METADATA_CSV="/content/drive/MyDrive/data/Severstal/data/processed/controlnet_dataset/packaged_roi_metadata.csv"
export SUMMARY_JSON="/content/drive/MyDrive/data/Severstal/outputs/generation_summary.json"
export TRAIN_IMAGES_DIR="/content/drive/MyDrive/data/Severstal/train_images"
export TRAIN_CSV="/content/drive/MyDrive/data/Severstal/train.csv"
export CASDA_BASE="/content/drive/MyDrive/data/Severstal/data/augmented"
export BG_CACHE="/content/drive/MyDrive/data/Severstal/data/cache/bg_types.json"
```

---

## 1. casda_no_blend 재생성

```bash
python scripts/compose_casda_images.py \
  --generated-dir "$GENERATED_DIR" \
  --hint-dir "$HINT_DIR" \
  --metadata-csv "$METADATA_CSV" \
  --summary-json "$SUMMARY_JSON" \
  --clean-images-dir "$TRAIN_IMAGES_DIR" \
  --train-csv "$TRAIN_CSV" \
  --output-dir "$CASDA_BASE/casda_no_blend" \
  --seed 42 \
  --compositions-per-roi 1 \
  --no-blend \
  --workers -1 \
  --bg-cache "$BG_CACHE" \
  --png-compression 1
```

---

## 2. casda_composed 재생성

```bash
python scripts/compose_casda_images.py \
  --generated-dir "$GENERATED_DIR" \
  --hint-dir "$HINT_DIR" \
  --metadata-csv "$METADATA_CSV" \
  --summary-json "$SUMMARY_JSON" \
  --clean-images-dir "$TRAIN_IMAGES_DIR" \
  --train-csv "$TRAIN_CSV" \
  --output-dir "$CASDA_BASE/casda_composed" \
  --seed 42 \
  --compositions-per-roi 1 \
  --workers -1 \
  --bg-cache "$BG_CACHE" \
  --png-compression 1
```

---

## 3. 비교 Figure 생성

두 variant 재생성 완료 후 실행:

```bash
python scripts/visualize_blend_comparison.py \
  --no-blend-dir "$CASDA_BASE/casda_no_blend" \
  --composed-dir "$CASDA_BASE/casda_composed" \
  --train-images-dir "$TRAIN_IMAGES_DIR" \
  --generated-dir "$GENERATED_DIR" \
  --output figures/blend_comparison.png \
  --dpi 300
```

---

## 검증

각 Step 완료 후 metadata.json의 `source_background` 값이 같은 파일명에서 동일한지 확인:

```python
import json, os
CASDA_BASE = os.environ["CASDA_BASE"]
nb = json.load(open(f"{CASDA_BASE}/casda_no_blend/metadata.json"))
cp = json.load(open(f"{CASDA_BASE}/casda_composed/metadata.json"))
nb_map = {e['source_generated']: e['source_background'] for e in nb}
cp_map = {e['source_generated']: e['source_background'] for e in cp}
mismatches = [(k, nb_map[k], cp_map[k]) for k in nb_map if k in cp_map and nb_map[k] != cp_map[k]]
print(f"배경 불일치: {len(mismatches)}개 (0이어야 함)")
```
