#!/usr/bin/env python3
"""
Train/Val/Test Split 비율 실험 오케스트레이터.

여러 split 비율(예: 50/25/25, 60/20/20, 70/15/15, 80/10/10)로
DeepLabV3+ 모델을 순차 학습하여, 최적 split 비율 선택의 근거를 마련한다.

각 비율에 대해:
  1) create_dataset_split.py 로 split CSV 생성
  2) run_benchmark.py 로 DeepLabV3+ 학습 + 평가
  3) benchmark_results.json 에서 Dice 등 메트릭 수집

전체 완료 후 비교 리포트(split_comparison.json + 콘솔 테이블) 생성.

사용 예시 (Colab):
  python scripts/run_split_experiment.py \\
      --csv /content/drive/MyDrive/data/Severstal/train.csv \\
      --data-dir /content/drive/MyDrive/data/Severstal/train_images \\
      --config configs/benchmark_experiment.yaml \\
      --output-dir /content/drive/MyDrive/data/Severstal/casda/outputs/split_experiment \\
      --ratios "50/25/25,60/20/20,70/15/15,80/10/10" \\
      --epochs 100 --seed 42

CASDA v5.6 파이프라인 — Split 비율 실험 스크립트
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ============================================================================
# Split Ratio 파싱
# ============================================================================

def parse_ratio_string(ratio_str: str) -> Tuple[float, float, float]:
    """
    'train/val/test' 문자열을 (train_ratio, val_ratio, test_ratio) 튜플로 변환.

    지원 형식:
      - '70/15/15' → (0.70, 0.15, 0.15)  — 합이 100인 경우 자동 정규화
      - '0.7/0.15/0.15' → (0.70, 0.15, 0.15)  — 이미 소수인 경우 그대로 사용

    Args:
        ratio_str: 'train/val/test' 형식 문자열

    Returns:
        (train_ratio, val_ratio, test_ratio)

    Raises:
        ValueError: 파싱 실패 또는 비율 합 불일치
    """
    parts = ratio_str.strip().split('/')
    if len(parts) != 3:
        raise ValueError(
            f"비율 형식이 올바르지 않습니다: '{ratio_str}' "
            f"— 'train/val/test' 형식 필요 (예: '70/15/15')"
        )

    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"비율 값을 숫자로 변환할 수 없습니다: '{ratio_str}'")

    # 합이 100에 가까우면 자동 정규화 (예: 70/15/15 → 0.70/0.15/0.15)
    total = sum(values)
    if abs(total - 100.0) < 1e-6:
        values = [v / 100.0 for v in values]
    elif abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"비율의 합이 1.0(또는 100)이 아닙니다: "
            f"{parts[0]} + {parts[1]} + {parts[2]} = {total}"
        )

    train_r, val_r, test_r = values
    for name, r in [('train', train_r), ('val', val_r), ('test', test_r)]:
        if not (0.0 < r < 1.0):
            raise ValueError(f"{name} 비율이 유효 범위(0~1)를 벗어났습니다: {r}")

    return (round(train_r, 4), round(val_r, 4), round(test_r, 4))


def ratio_to_label(train_r: float, val_r: float, test_r: float) -> str:
    """비율 튜플을 라벨 문자열로 변환 (예: 'split_70_15_15')."""
    t = int(round(train_r * 100))
    v = int(round(val_r * 100))
    te = int(round(test_r * 100))
    return f"split_{t}_{v}_{te}"


# ============================================================================
# Step 1: Split CSV 생성
# ============================================================================

def create_split_csv(
    annotation_csv: str,
    output_path: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> bool:
    """
    create_dataset_split.py를 subprocess로 호출하여 split CSV를 생성.

    이미 존재하는 CSV는 건너뛴다 (idempotent).

    Returns:
        True: 성공, False: 실패
    """
    if os.path.exists(output_path):
        logger.info(f"Split CSV 이미 존재, 건너뜀: {output_path}")
        return True

    # 스크립트 경로 (이 파일과 같은 디렉토리)
    script_dir = Path(__file__).resolve().parent
    create_script = script_dir / 'create_dataset_split.py'
    if not create_script.exists():
        logger.error(f"create_dataset_split.py를 찾을 수 없습니다: {create_script}")
        return False

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        sys.executable, str(create_script),
        '--csv', annotation_csv,
        '--output', output_path,
        '--train-ratio', str(train_ratio),
        '--val-ratio', str(val_ratio),
        '--test-ratio', str(test_ratio),
        '--seed', str(seed),
    ]

    logger.info(f"Split CSV 생성 중: {train_ratio}/{val_ratio}/{test_ratio}")
    logger.info(f"  명령: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"Split CSV 생성 실패 (exit code {result.returncode})")
            if result.stderr:
                logger.error(f"  stderr: {result.stderr.strip()}")
            return False

        if not os.path.exists(output_path):
            logger.error(f"Split CSV 생성 후 파일이 존재하지 않음: {output_path}")
            return False

        logger.info(f"Split CSV 생성 완료: {output_path}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Split CSV 생성 타임아웃 (120초)")
        return False
    except Exception as e:
        logger.error(f"Split CSV 생성 중 예외: {e}")
        return False


# ============================================================================
# Step 2: DeepLabV3+ 벤치마크 실행
# ============================================================================

def run_benchmark(
    config_path: str,
    split_csv_path: str,
    output_dir: str,
    epochs: int,
    seed: int,
    data_dir: Optional[str] = None,
    annotation_csv: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> bool:
    """
    run_benchmark.py를 subprocess로 호출하여 DeepLabV3+ 학습 + 평가.

    Args:
        config_path: benchmark_experiment.yaml 경로
        split_csv_path: split CSV 파일 경로
        output_dir: 실험 출력 디렉토리
        epochs: 학습 에폭 수
        seed: 랜덤 시드
        data_dir: 학습 이미지 디렉토리 (--data-dir로 전달)
        annotation_csv: annotation CSV 경로 (--csv로 전달)
        extra_args: 추가 CLI 인자 목록

    Returns:
        True: 성공, False: 실패
    """
    # 이미 완료된 실험인지 확인 (resume 지원)
    results_file = os.path.join(output_dir, 'benchmark_results.json')
    if os.path.exists(results_file):
        logger.info(f"벤치마크 결과 이미 존재, 건너뜀: {results_file}")
        return True

    script_dir = Path(__file__).resolve().parent
    benchmark_script = script_dir / 'run_benchmark.py'
    if not benchmark_script.exists():
        logger.error(f"run_benchmark.py를 찾을 수 없습니다: {benchmark_script}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, str(benchmark_script),
        '--config', config_path,
        '--split-csv', split_csv_path,
        '--output-dir', output_dir,
        '--models', 'deeplabv3plus',
        '--groups', 'baseline_raw',
        '--epochs', str(epochs),
        '--seed', str(seed),
        '--no-fid',  # FID는 split 실험에 불필요
    ]
    if data_dir:
        cmd.extend(['--data-dir', data_dir])
    if annotation_csv:
        cmd.extend(['--csv', annotation_csv])
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"벤치마크 실행 시작: epochs={epochs}, output={output_dir}")
    logger.info(f"  명령: {' '.join(cmd)}")

    start_time = time.time()
    try:
        # stdout/stderr를 실시간 출력 (Colab에서 진행 상황 확인 가능)
        result = subprocess.run(
            cmd,
            timeout=3600 * 12,  # 최대 12시간 (DeepLabV3+ 300ep 기준 넉넉히)
        )
        elapsed = time.time() - start_time
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))

        if result.returncode != 0:
            logger.error(
                f"벤치마크 실패 (exit code {result.returncode}), "
                f"소요시간: {elapsed_str}"
            )
            return False

        logger.info(f"벤치마크 완료, 소요시간: {elapsed_str}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("벤치마크 타임아웃 (12시간)")
        return False
    except Exception as e:
        logger.error(f"벤치마크 실행 중 예외: {e}")
        return False


# ============================================================================
# Step 3: 결과 수집
# ============================================================================

def collect_results(output_dir: str, label: str) -> Optional[Dict]:
    """
    벤치마크 결과 디렉토리에서 메트릭을 수집.

    Args:
        output_dir: 실험 출력 디렉토리
        label: split 라벨 (예: 'split_70_15_15')

    Returns:
        결과 딕셔너리 또는 None (결과 파일 없음)
    """
    results_file = os.path.join(output_dir, 'benchmark_results.json')
    if not os.path.exists(results_file):
        logger.warning(f"결과 파일 없음: {results_file}")
        return None

    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"결과 파일 읽기 실패: {results_file} — {e}")
        return None

    # DeepLabV3+ + baseline_raw 결과 추출
    experiments = data.get('experiments', [])
    seg_result = None

    for exp in experiments:
        model = exp.get('model', '')
        dataset_group = exp.get('dataset_group', '')
        if 'deeplabv3' in model.lower() and 'baseline' in dataset_group.lower():
            seg_result = exp
            break

    if seg_result is None:
        logger.warning(f"DeepLabV3+ baseline 결과를 찾을 수 없음: {results_file}")
        return None

    metrics = seg_result.get('test_metrics', seg_result.get('metrics', {}))

    # split 통계 (dataset_split.json에서 더 정확한 정보 수집)
    split_info_file = os.path.join(output_dir, 'dataset_split.json')
    split_counts = {}
    if os.path.exists(split_info_file):
        try:
            with open(split_info_file, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
                split_counts = {
                    'num_train': split_data.get('num_train', 0),
                    'num_val': split_data.get('num_val', 0),
                    'num_test': split_data.get('num_test', 0),
                }
        except (json.JSONDecodeError, IOError):
            pass

    return {
        'label': label,
        'dice_mean': metrics.get('dice_mean', 0.0),
        'class_dice': metrics.get('class_dice', {}),
        'iou_mean': metrics.get('iou_mean', 0.0),
        'class_iou': metrics.get('class_iou', {}),
        'loss': metrics.get('loss', 0.0),
        'best_epoch': seg_result.get('best_epoch', -1),
        'total_epochs': seg_result.get('total_epochs', -1),
        'training_time': seg_result.get('training_time', ''),
        'split_counts': split_counts,
        'output_dir': output_dir,
    }


# ============================================================================
# Step 4: 비교 리포트 생성
# ============================================================================

def generate_comparison_report(
    results: List[Dict],
    output_dir: str,
) -> str:
    """
    모든 split 비율의 결과를 비교하는 리포트 생성.

    Returns:
        비교 테이블 문자열 (콘솔 출력용)
    """
    # JSON 리포트 저장
    report = {
        'experiment': 'split_ratio_comparison',
        'created_at': datetime.now().isoformat(),
        'num_ratios': len(results),
        'results': results,
    }

    # 최고 Dice 찾기
    if results:
        best = max(results, key=lambda r: r.get('dice_mean', 0.0))
        report['best_split'] = best['label']
        report['best_dice'] = best['dice_mean']

    report_path = os.path.join(output_dir, 'split_comparison.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"비교 리포트 저장: {report_path}")

    # 콘솔 테이블 생성
    lines = []
    lines.append("")
    lines.append("=" * 100)
    lines.append("  Split 비율 실험 결과 비교 (DeepLabV3+ / baseline_raw)")
    lines.append("=" * 100)
    lines.append("")

    # 헤더
    header = (
        f"{'Split':<12} | {'Train':>5} | {'Val':>5} | {'Test':>5} | "
        f"{'Dice':>6} | {'C1':>6} | {'C2':>6} | {'C3':>6} | {'C4':>6} | "
        f"{'IoU':>6} | {'Epoch':>5} | {'Time':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        sc = r.get('split_counts', {})
        cd = r.get('class_dice', {})

        # class_dice 키: '0', '1', '2', '3' (0-indexed) 또는 'class_1' 등
        # 유연하게 처리
        c1 = cd.get('0', cd.get('class_1', cd.get(0, 0.0)))
        c2 = cd.get('1', cd.get('class_2', cd.get(1, 0.0)))
        c3 = cd.get('2', cd.get('class_3', cd.get(2, 0.0)))
        c4 = cd.get('3', cd.get('class_4', cd.get(3, 0.0)))

        row = (
            f"{r['label']:<12} | "
            f"{sc.get('num_train', '?'):>5} | "
            f"{sc.get('num_val', '?'):>5} | "
            f"{sc.get('num_test', '?'):>5} | "
            f"{r['dice_mean']:>6.4f} | "
            f"{c1:>6.4f} | {c2:>6.4f} | {c3:>6.4f} | {c4:>6.4f} | "
            f"{r['iou_mean']:>6.4f} | "
            f"{r.get('best_epoch', '?'):>5} | "
            f"{r.get('training_time', 'N/A'):>10}"
        )
        lines.append(row)

    lines.append("-" * len(header))

    # 최고 split 표시
    if results:
        best = max(results, key=lambda r: r.get('dice_mean', 0.0))
        lines.append(f"\n  ★ 최고 Dice: {best['label']} → {best['dice_mean']:.4f}")

    lines.append("")
    lines.append("=" * 100)

    table_str = "\n".join(lines)
    return table_str


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Split 비율별 DeepLabV3+ 성능 비교 실험',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시 (Colab):
  python scripts/run_split_experiment.py \\
      --csv /content/drive/MyDrive/data/Severstal/train.csv \\
      --data-dir /content/drive/MyDrive/data/Severstal/train_images \\
      --config configs/benchmark_experiment.yaml \\
      --output-dir /content/drive/MyDrive/data/Severstal/casda/outputs/split_experiment \\
      --ratios "50/25/25,60/20/20,70/15/15,80/10/10" \\
      --epochs 100 --seed 42

비율 형식:
  - 백분율: '70/15/15' (합=100, 자동 정규화)
  - 소수: '0.7/0.15/0.15' (합=1.0)
  - 여러 비율은 쉼표로 구분: '50/25/25,60/20/20,70/15/15,80/10/10'
        """,
    )
    parser.add_argument(
        '--csv', required=True,
        help='Severstal train.csv 경로 (annotation CSV). '
             'run_benchmark.py에 --csv로도 전달됩니다.',
    )
    parser.add_argument(
        '--data-dir', required=True,
        help='학습 이미지 디렉토리 경로 (예: /content/drive/.../train_images). '
             'run_benchmark.py에 --data-dir로 전달됩니다.',
    )
    parser.add_argument(
        '--config', required=True,
        help='benchmark_experiment.yaml 경로',
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='실험 결과 루트 디렉토리. 각 비율별 하위 디렉토리가 생성됨.',
    )
    parser.add_argument(
        '--ratios', type=str,
        default='50/25/25,60/20/20,70/15/15,80/10/10',
        help='쉼표로 구분된 split 비율 목록 (기본: "50/25/25,60/20/20,70/15/15,80/10/10")',
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='모든 비율에 적용할 학습 에폭 수 (기본: 100)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='랜덤 시드 (기본: 42). split 생성과 학습 모두에 적용.',
    )
    parser.add_argument(
        '--splits-dir', type=str, default=None,
        help='split CSV 저장 디렉토리 (기본: {output-dir}/splits/)',
    )
    parser.add_argument(
        '--skip-existing', action='store_true', default=True,
        help='이미 완료된 실험은 건너뜀 (기본: True)',
    )
    parser.add_argument(
        '--no-skip-existing', action='store_true',
        help='기존 결과가 있어도 재실행',
    )
    parser.add_argument(
        '--benchmark-args', type=str, nargs='*', default=None,
        help='run_benchmark.py에 전달할 추가 인자 '
             '(예: --benchmark-args --yolo-dir /path/to/yolo)',
    )
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # skip_existing 처리
    skip_existing = args.skip_existing and not args.no_skip_existing

    # 입력 검증
    if not os.path.exists(args.csv):
        logger.error(f"Annotation CSV를 찾을 수 없습니다: {args.csv}")
        sys.exit(1)
    if not os.path.isdir(args.data_dir):
        logger.error(f"이미지 디렉토리를 찾을 수 없습니다: {args.data_dir}")
        sys.exit(1)
    if not os.path.exists(args.config):
        logger.error(f"Config 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)

    # 비율 파싱
    ratio_strings = [r.strip() for r in args.ratios.split(',') if r.strip()]
    ratios = []
    for rs in ratio_strings:
        try:
            ratios.append(parse_ratio_string(rs))
        except ValueError as e:
            logger.error(f"비율 파싱 실패: {e}")
            sys.exit(1)

    if not ratios:
        logger.error("유효한 비율이 없습니다.")
        sys.exit(1)

    # 디렉토리 설정
    output_dir = os.path.abspath(args.output_dir)
    splits_dir = args.splits_dir or os.path.join(output_dir, 'splits')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  Split 비율 실험 시작")
    logger.info("=" * 70)
    logger.info(f"  Annotation CSV : {args.csv}")
    logger.info(f"  Data dir       : {args.data_dir}")
    logger.info(f"  Config         : {args.config}")
    logger.info(f"  Output dir     : {output_dir}")
    logger.info(f"  Splits dir     : {splits_dir}")
    logger.info(f"  Epochs         : {args.epochs}")
    logger.info(f"  Seed           : {args.seed}")
    logger.info(f"  비율 목록 ({len(ratios)}개):")
    for train_r, val_r, test_r in ratios:
        label = ratio_to_label(train_r, val_r, test_r)
        logger.info(f"    {label}: train={train_r}, val={val_r}, test={test_r}")
    logger.info("=" * 70)

    # 실험 메타데이터 저장
    experiment_meta = {
        'experiment': 'split_ratio_comparison',
        'started_at': datetime.now().isoformat(),
        'annotation_csv': args.csv,
        'config': args.config,
        'epochs': args.epochs,
        'seed': args.seed,
        'ratios': [
            {'train': t, 'val': v, 'test': te, 'label': ratio_to_label(t, v, te)}
            for t, v, te in ratios
        ],
    }
    meta_path = os.path.join(output_dir, 'experiment_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_meta, f, indent=2, ensure_ascii=False)

    # ====================================================================
    # 실험 루프
    # ====================================================================
    all_results = []
    failed_ratios = []
    total_start = time.time()

    for idx, (train_r, val_r, test_r) in enumerate(ratios, 1):
        label = ratio_to_label(train_r, val_r, test_r)
        logger.info("")
        logger.info(f"{'='*60}")
        logger.info(f"  [{idx}/{len(ratios)}] {label} "
                     f"(train={train_r}, val={val_r}, test={test_r})")
        logger.info(f"{'='*60}")

        # 경로 설정
        split_csv_path = os.path.join(
            splits_dir,
            f"{label}_seed{args.seed}.csv",
        )
        exp_output_dir = os.path.join(output_dir, label)

        # -- Step 1: Split CSV 생성 --
        logger.info(f"  [Step 1/3] Split CSV 생성 ...")
        if not create_split_csv(
            annotation_csv=args.csv,
            output_path=split_csv_path,
            train_ratio=train_r,
            val_ratio=val_r,
            test_ratio=test_r,
            seed=args.seed,
        ):
            logger.error(f"  ✗ Split CSV 생성 실패 — {label} 건너뜀")
            failed_ratios.append(label)
            continue

        # -- Step 2: 벤치마크 실행 --
        logger.info(f"  [Step 2/3] DeepLabV3+ 벤치마크 실행 ...")

        # 기존 결과가 있고 skip_existing이면 건너뜀
        results_file = os.path.join(exp_output_dir, 'benchmark_results.json')
        if skip_existing and os.path.exists(results_file):
            logger.info(f"  이미 완료된 실험, 결과 수집만 진행: {results_file}")
        else:
            if not run_benchmark(
                config_path=args.config,
                split_csv_path=split_csv_path,
                output_dir=exp_output_dir,
                epochs=args.epochs,
                seed=args.seed,
                data_dir=args.data_dir,
                annotation_csv=args.csv,
                extra_args=args.benchmark_args,
            ):
                logger.error(f"  ✗ 벤치마크 실패 — {label}")
                failed_ratios.append(label)
                continue

        # -- Step 3: 결과 수집 --
        logger.info(f"  [Step 3/3] 결과 수집 ...")
        result = collect_results(exp_output_dir, label)
        if result is not None:
            all_results.append(result)
            logger.info(f"  ✓ {label}: Dice={result['dice_mean']:.4f}")
        else:
            logger.warning(f"  결과 수집 실패 — {label}")
            failed_ratios.append(label)

    # ====================================================================
    # 비교 리포트
    # ====================================================================
    total_elapsed = time.time() - total_start
    total_str = time.strftime('%H:%M:%S', time.gmtime(total_elapsed))

    logger.info("")
    logger.info(f"{'='*70}")
    logger.info(f"  전체 실험 완료 — 총 소요시간: {total_str}")
    logger.info(f"  성공: {len(all_results)}/{len(ratios)}, "
                f"실패: {len(failed_ratios)}/{len(ratios)}")
    logger.info(f"{'='*70}")

    if all_results:
        table_str = generate_comparison_report(all_results, output_dir)
        print(table_str)
    else:
        logger.warning("수집된 결과가 없습니다. 비교 리포트를 생성할 수 없습니다.")

    if failed_ratios:
        logger.warning(f"실패한 비율: {', '.join(failed_ratios)}")

    # 메타데이터 업데이트
    experiment_meta['completed_at'] = datetime.now().isoformat()
    experiment_meta['total_elapsed'] = total_str
    experiment_meta['successful'] = len(all_results)
    experiment_meta['failed'] = failed_ratios
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_meta, f, indent=2, ensure_ascii=False)

    sys.exit(0 if not failed_ratios else 1)


if __name__ == '__main__':
    main()
