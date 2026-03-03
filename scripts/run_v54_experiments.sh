#!/bin/bash
# ============================================================================
# CASDA v5.4 벤치마크 실험 실행 스크립트 (Colab 용)
# ============================================================================
#
# v5.3 분석 결과에 기반한 후속 실험:
#   Phase 1: baseline_trad (H1 가설 검증)
#   Phase 2: CASDA ratio 실험 (10%, 20%, 30%)
#   Phase 3: YOLO-MFD CASDA-Pruning 재실행
#
# 사용법 (Colab에서):
#   # 1) 환경 변수 설정
#   export DATA_DIR="/content/drive/MyDrive/data/Severstal/train_images"
#   export CSV_PATH="/content/drive/MyDrive/data/Severstal/train.csv"
#   export SPLIT_CSV="/content/drive/MyDrive/data/Severstal/casda/splits/split_70_15_15_seed42.csv"
#   export YOLO_DIR="/content/yolo_datasets"
#   export CASDA_DIR="/content/drive/MyDrive/data/Severstal/data/augmented_v4_dataset"
#   export OUTPUT_DIR="/content/drive/MyDrive/data/Severstal/casda/v5.4"
#   export CONFIG="/content/severstal-steel-defect-detection/configs/benchmark_experiment.yaml"
#
#   # 2) 전체 실행
#   bash scripts/run_v54_experiments.sh all
#
#   # 2-1) Phase별 실행
#   bash scripts/run_v54_experiments.sh phase1    # baseline_trad만
#   bash scripts/run_v54_experiments.sh phase2    # ratio 실험만
#   bash scripts/run_v54_experiments.sh phase3    # YOLO-MFD Pruning 재실행
#
# ============================================================================

set -e

# 기본값 (Colab 경로 — 환경 변수로 override 가능)
DATA_DIR="${DATA_DIR:-/content/drive/MyDrive/data/Severstal/train_images}"
CSV_PATH="${CSV_PATH:-/content/drive/MyDrive/data/Severstal/train.csv}"
SPLIT_CSV="${SPLIT_CSV:-/content/drive/MyDrive/data/Severstal/casda/splits/split_70_15_15_seed42.csv}"
YOLO_DIR="${YOLO_DIR:-/content/yolo_datasets}"
CASDA_DIR="${CASDA_DIR:-/content/drive/MyDrive/data/Severstal/data/augmented_v4_dataset}"
OUTPUT_BASE="${OUTPUT_DIR:-/content/drive/MyDrive/data/Severstal/casda/v5.4}"
CONFIG="${CONFIG:-configs/benchmark_experiment.yaml}"

# 공통 인자
COMMON_ARGS="--config $CONFIG \
    --data-dir $DATA_DIR \
    --csv $CSV_PATH \
    --split-csv $SPLIT_CSV \
    --yolo-dir $YOLO_DIR \
    --casda-dir $CASDA_DIR"

PHASE="${1:-all}"

echo "============================================================================"
echo "  CASDA v5.4 Benchmark Experiments"
echo "============================================================================"
echo "  Phase: $PHASE"
echo "  Data dir: $DATA_DIR"
echo "  Output base: $OUTPUT_BASE"
echo "  Config: $CONFIG"
echo "============================================================================"

# ============================================================================
# Phase 0: Bbox 라벨 품질 사전 검증 (빠른 분석, GPU 불필요)
# ============================================================================
# 목적: Detection 실패 원인 중 bbox 품질 문제 확인
#
run_phase0() {
    echo ""
    echo "================================================================"
    echo "  Phase 0: Bbox Label Quality Verification"
    echo "  (No GPU needed — quick analysis)"
    echo "================================================================"

    python scripts/verify_bbox_quality.py \
        --csv "$CSV_PATH" \
        --image-dir "$DATA_DIR" \
        --casda-dir "$CASDA_DIR/casda_full" \
        --output-dir "$OUTPUT_BASE/phase0_bbox_analysis"

    echo "Phase 0 완료: $OUTPUT_BASE/phase0_bbox_analysis"
}

# ============================================================================
# Phase 1: baseline_trad — H1 가설 검증
# ============================================================================
# 목적: CASDA vs Traditional Augmentation 비교
# 3 모델 (EB-YOLOv8, YOLO-MFD, DeepLabV3+) x baseline_trad = 3 실험
#
run_phase1() {
    echo ""
    echo "================================================================"
    echo "  Phase 1: baseline_trad (H1 가설 검증)"
    echo "  3 models x baseline_trad = 3 experiments"
    echo "================================================================"

    python scripts/run_benchmark.py $COMMON_ARGS \
        --output-dir "$OUTPUT_BASE/phase1_trad" \
        --groups trad \
        --models eb_yolov8 yolo_mfd deeplabv3plus

    echo "Phase 1 완료: $OUTPUT_BASE/phase1_trad"
}

# ============================================================================
# Phase 2: CASDA Ratio 실험 — 주입 비율 최적화
# ============================================================================
# 목적: 합성 데이터 비율이 성능에 미치는 영향 파악
# 3 모델 x 3 비율 (10%, 20%, 30%) = 9 실험
#
# v5.3 결과:
#   - 50% (casda_full) → Detection 성능 -12.4%~-45.0% 하락
#   - 30% (casda_pruning ~2000장) → 더 심각한 하락 (EB-YOLOv8 -89.1%)
#   - 가설: 소량 고품질 합성 데이터(10~20%)가 더 효과적
#
run_phase2() {
    echo ""
    echo "================================================================"
    echo "  Phase 2: CASDA Ratio Experiments (10%, 20%, 30%)"
    echo "  3 models x 3 ratios = 9 experiments"
    echo "  (baseline_raw는 v5.3에서 이미 완료 — 비교용으로 결과 참조)"
    echo "================================================================"

    # --groups 없이 --casda-ratio만 지정하면
    # ratio 그룹만 실행됨 (기존 그룹은 실행하지 않음)
    # baseline_raw 결과는 v5.3 outputs에서 참조
    python scripts/run_benchmark.py $COMMON_ARGS \
        --output-dir "$OUTPUT_BASE/phase2_ratio" \
        --casda-ratio 0.1 0.2 0.3 \
        --models eb_yolov8 yolo_mfd deeplabv3plus

    echo "Phase 2 완료: $OUTPUT_BASE/phase2_ratio"
}

# ============================================================================
# Phase 3: YOLO-MFD CASDA-Pruning 재실행
# ============================================================================
# 목적: v5.3에서 체크포인트 손상으로 크래시한 실험 재실행
#
run_phase3() {
    echo ""
    echo "================================================================"
    echo "  Phase 3: YOLO-MFD + CASDA-Pruning 재실행"
    echo "  1 model x 1 group = 1 experiment"
    echo "================================================================"

    python scripts/run_benchmark.py $COMMON_ARGS \
        --output-dir "$OUTPUT_BASE/phase3_rerun" \
        --groups pruning \
        --models yolo_mfd

    echo "Phase 3 완료: $OUTPUT_BASE/phase3_rerun"
}

# ============================================================================
# Phase 실행
# ============================================================================
case "$PHASE" in
    phase0|0|bbox)
        run_phase0
        ;;
    phase1|1|trad)
        run_phase1
        ;;
    phase2|2|ratio)
        run_phase2
        ;;
    phase3|3|rerun)
        run_phase3
        ;;
    all)
        run_phase0
        run_phase1
        run_phase2
        run_phase3
        ;;
    *)
        echo "Usage: $0 {phase0|phase1|phase2|phase3|all}"
        echo ""
        echo "  phase0 (bbox)  : Bbox label quality verification (no GPU)"
        echo "  phase1 (trad)  : baseline_trad 3 models — H1 hypothesis"
        echo "  phase2 (ratio) : CASDA ratio 10/20/30% x 3 models"
        echo "  phase3 (rerun) : YOLO-MFD CASDA-Pruning rerun"
        echo "  all            : Run all phases sequentially"
        exit 1
        ;;
esac

echo ""
echo "============================================================================"
echo "  v5.4 Experiments Complete"
echo "  Results: $OUTPUT_BASE"
echo "============================================================================"
