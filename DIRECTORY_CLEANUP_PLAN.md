# Model Directory Cleanup Plan

## 🚨 현재 문제

프로젝트에 `model/`과 `models/` 두 디렉토리가 공존하여 혼란이 있습니다.

### 현재 구조
```
./model/          # 원래 통합 버전
├── backbone.py
├── dmrgcn_gpgraph.py      # DMRGCNGPGraph 클래스
├── gpgraph_adapter.py  
└── utils.py

./models/         # Shape Refactor 버전  
└── dmrgcn_gpgraph.py      # DMRGCN_GPGraph_Model 클래스
```

### 임포트 현황
- **기존 스크립트**: `from model.dmrgcn_gpgraph import DMRGCNGPGraph`
- **새 스크립트**: `from models.dmrgcn_gpgraph import DMRGCN_GPGraph_Model`

## 🎯 추천 해결책: models/ → model/ 통합

### Step 1: 디렉토리 정리
```bash
# 1. 기존 model을 백업
mv model/ model_legacy/

# 2. models를 메인 model로 이동
mv models/ model/

# 3. 필요한 파일들을 model/로 복사
cp model_legacy/backbone.py model/
cp model_legacy/gpgraph_adapter.py model/
cp model_legacy/utils.py model/
```

### Step 2: 임포트 경로 통일
모든 스크립트에서 다음과 같이 수정:
```python
# Before
from models.dmrgcn_gpgraph import DMRGCN_GPGraph_Model

# After  
from model.dmrgcn_gpgraph import DMRGCN_GPGraph_Model
```

### Step 3: 중복 파일 정리
`model/dmrgcn_gpgraph.py`에서 두 버전을 통합:
- **메인**: Shape Refactor 버전 (DMRGCN_GPGraph_Model)
- **레거시**: 기존 버전은 별도 클래스로 유지 (호환성)

## 🏆 최종 목표 구조

```
./model/                    # 통합된 메인 모델 디렉토리
├── __init__.py
├── backbone.py             # DMRGCN 백본
├── dmrgcn_gpgraph.py      # 통합 모델 (두 클래스 모두)
├── gpgraph_adapter.py     # GP-Graph 어댑터
└── utils.py               # 유틸리티

./model_legacy/            # 백업 (필요시 참조)
└── (원본 파일들)
```

## 🔧 실행 방법

### 자동 실행
```bash
./cleanup_directories.sh
```

### 수동 실행
1. 백업 생성
2. 디렉토리 재배치  
3. 임포트 경로 수정
4. 테스트 실행

## ✅ 완료 후 확인사항

- [ ] 모든 스크립트가 정상 임포트
- [ ] demo_final.py 실행 성공
- [ ] train_unified.py 실행 성공  
- [ ] 기존 train.py, test.py 호환성 유지

## 📊 예상 효과

✅ **명확한 구조**: 단일 model/ 디렉토리  
✅ **일관된 임포트**: from model.* 패턴 통일  
✅ **하위 호환성**: 기존 코드도 동작  
✅ **최신 기능**: Shape Refactor 버전이 메인  

이 정리를 통해 프로젝트 구조가 깔끔해지고 혼란이 해소될 것입니다! 🚀
