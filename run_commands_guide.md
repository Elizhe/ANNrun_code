# GPU 실험 실행 가이드

## 🎯 실험 개요
- **GPU**: RTX 2060 SUPER (8GB VRAM)
- **조합 수**: 216개 (12 아키텍처 × 9 정규화 × 2 Adam 옵티마이저)
- **Features**: himawari_srad, himawari_dtr, himawari_prec, himawari_tmax, agera5_srad, agera5_dtr, agera5_prec, agera5_tmax
- **Preprocessing**: minmax, clearness, weather_flags

## 📁 파일 구성

1. **gpu_experiment_plan.csv** - 실험 설정 파일 (1개 기본 설정)
2. **gpu_config_override.py** - GPU 설정 관리자 (Adam만 사용)
3. **gpu_memory_monitor.py** - GPU 메모리 모니터링
4. **gpu_experiment_runner.py** - 실험 실행 스크립트

## 🚀 실행 순서

### 1단계: GPU 환경 체크
```bash
# GPU 상태 확인
python gpu_memory_monitor.py

# 또는 시스템 체크
python scripts/system_check.py
```

### 2단계: GPU 설정 생성
```bash
# GPU 최적화 설정 생성
python gpu_config_override.py
```

### 3단계: 실험 실행

#### 방법 A: 직접 main.py 사용
```bash
# 기본 실행 (권장)
python main.py parallel gpu_experiment_plan.csv 1 --gpu-ids 0 --parallel-backend joblib --n-jobs 1

# 멀티프로세싱 사용
python main.py parallel gpu_experiment_plan.csv 1 --gpu-ids 0 --parallel-backend multiprocessing --n-jobs 1
```

#### 방법 B: GPU 실험 러너 사용 (추천)
```bash
# 간단 실행
python gpu_experiment_runner.py

# 백그라운드 모니터링과 함께
python gpu_experiment_runner.py --monitor-only &
python gpu_experiment_runner.py

# 특정 설정으로 실행
python gpu_experiment_runner.py --backend joblib --n-jobs 1 --gpu-id 0
```

## 📊 모니터링

### GPU 사용량 실시간 모니터링
```bash
# nvidia-smi로 실시간 모니터링
nvidia-smi -l 1

# 내장 모니터 사용
python gpu_experiment_runner.py --monitor-only
```

### 메모리 사용량 체크
```bash
# 간단한 메모리 체크
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 상세 모니터링
python gpu_memory_monitor.py
```

## ⚙️ 최적화 설정

### RTX 2060 SUPER 최적 배치 사이즈
```
single_32: 1024
single_64: 512
single_128: 256
double_64_32: 256
triple_128_64_32: 128
deep_256_128_64_32: 64
```

### GPU 메모리 설정
- **메모리 한계**: 7.5GB (시스템용 0.5GB 예약)
- **Mixed Precision**: FP16 사용 (메모리 효율성)
- **Memory Growth**: 동적 할당 활성화

## 🔧 문제 해결

### GPU가 인식되지 않는 경우
```bash
# CUDA 드라이버 확인
nvidia-smi

# CUDA 설치 확인
nvcc --version

# TensorFlow GPU 확인
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# PyTorch GPU 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 메모리 부족 오류 발생시
```bash
# 배치 사이즈 줄이기 (gpu_config_override.py에서 수정)
# 또는 단일 작업으로 실행
python main.py single gpu_experiment_plan.csv 1 --gpu-ids 0
```

### 실험 중단 후 재시작
```bash
# 결과 폴더 확인
ls results_gpu/

# 특정 실험부터 재시작 (main.py에서 --resume 옵션 확인)
python main.py parallel gpu_experiment_plan.csv 1 --gpu-ids 0 --resume
```

## 📈 결과 확인

### 실험 완료 후
```bash
# 결과 폴더 구조 확인
ls -la results_gpu/

# 로그 파일 확인
tail -f results_gpu/gpu_experiment_*.log

# 결과 분석 (분석 스크립트가 있는 경우)
python analyze_results.py results_gpu/
```

## 💡 추천 설정

### 최적 성능을 위한 실행
```bash
# 1. GPU 체크
python gpu_memory_monitor.py

# 2. 실험 실행 (백그라운드 모니터링)
python gpu_experiment_runner.py &

# 3. 모니터링
nvidia-smi -l 5
```

### 안정성을 위한 실행
```bash
# 단일 작업으로 안전하게
python main.py single gpu_experiment_plan.csv 1 --gpu-ids 0 --log-level DEBUG
```

## 📝 예상 소요 시간

- **전체 216개 조합**: 약 8-12시간 (실험 복잡도에 따라)
- **단일 실험**: 약 2-5분
- **메모리 최적화**: 각 아키텍처별 자동 배치 사이즈 조정

## 🎯 성능 팁

1. **메모리 정리**: 각 실험 후 GPU 캐시 정리
2. **온도 관리**: GPU 온도 80°C 이하 유지
3. **전력 관리**: 고성능 모드로 설정
4. **배경 프로그램**: 다른 GPU 사용 프로그램 종료