# DMRGCN 프로젝트 문서 인덱스

이 문서는 DMRGCN 프로젝트의 모든 문서를 한눈에 볼 수 있도록 정리한 인덱스입니다.

## 📚 문서 구조

```
DMRGCN/
├── README.md                          # 메인 문서 (시작하기)
├── README_INDEX.md                    # 이 파일 (문서 인덱스)
├── README_4REL_SDD.md                 # 4-relation extension 상세
├── README_THREAT_VISUALIZATION.md     # Threat score 시각화 가이드
└── VISUALIZE_BOOKSTORE_V0.md         # Bookstore video0 시각화
```

## 🚀 빠른 시작

### 처음 시작하는 경우

1. **[README.md](./README.md)** 읽기
   - 프로젝트 개요
   - 빠른 시작 가이드
   - 기본 사용법

2. **Threat Score 시각화 시도**
   ```bash
   ./visualize_bookstore_video0.sh 20
   ```

3. 필요시 상세 문서 참조

## 📖 문서별 내용

### 1. README.md (메인 문서)

**대상**: 모든 사용자

**내용**:
- 프로젝트 개요
- 주요 확장 사항 (4-relation, threat score)
- 빠른 시작 가이드
- 데이터셋 설정
- 모델 학습
- Threat score 시각화
- 프로젝트 구조
- 문제 해결

**언제 읽나요?**
- 프로젝트를 처음 시작할 때
- 전체적인 구조를 파악하고 싶을 때
- 빠른 참조가 필요할 때

---

### 2. README_4REL_SDD.md (4-Relation Extension)

**대상**: 모델 개발자, 연구자

**내용**:
- 4-relation extension 상세 설명
- Threat score 계산 방법
- 그래프 구성 방법
- 모델 구조 및 학습 방법
- 데이터 형식 및 전처리
- Ablation 실험 방법

**언제 읽나요?**
- 4-relation extension의 상세한 구현을 이해하고 싶을 때
- Threat score 계산 방법을 수정하고 싶을 때
- 모델 구조를 변경하고 싶을 때
- 새로운 relation을 추가하고 싶을 때

**주요 섹션**:
- Key Goals
- What Changed (새로운 파일, 수정된 파일)
- Graph Construction
- Model Flow
- Training
- Design Choices & Tunables

---

### 3. README_THREAT_VISUALIZATION.md (Threat Score 시각화)

**대상**: 모든 사용자 (특히 시각화를 사용하는 경우)

**내용**:
- Threat score 시각화 전체 가이드
- 시각화 모드 설명 (arrows, circles, heatmap, all)
- 다양한 사용 예시
- 시각화 요소 설명
- 문제 해결 방법

**언제 읽나요?**
- Threat score를 시각화하고 싶을 때
- 시각화 모드를 변경하고 싶을 때
- 시각화 결과를 해석하고 싶을 때
- 시각화 관련 문제가 발생했을 때

**주요 섹션**:
- 사용 방법
- 시각화 모드 예시
- Threat Score 색상 매핑
- 시각화 요소 설명
- 문제 해결

---

### 4. VISUALIZE_BOOKSTORE_V0.md (Bookstore Video0 시각화)

**대상**: Bookstore video0를 사용하는 사용자

**내용**:
- Bookstore video0 전용 시각화 가이드
- 빠른 시작 명령어
- 사용 가능한 track ID 확인 방법
- 예시 명령어 모음
- 문제 해결

**언제 읽나요?**
- Bookstore video0를 시각화하고 싶을 때
- 빠른 참조가 필요할 때
- 특정 track ID를 찾고 싶을 때

**주요 섹션**:
- 빠른 시작
- 사용 가능한 Track IDs
- 파라미터 설명
- 예시 명령어

---

## 🎯 사용 시나리오별 가이드

### 시나리오 1: Threat Score 시각화만 하고 싶은 경우

1. **[README.md](./README.md)** → "빠른 시작" 섹션
2. **[VISUALIZE_BOOKSTORE_V0.md](./VISUALIZE_BOOKSTORE_V0.md)** → 빠른 시작 명령어
3. 실행: `./visualize_bookstore_video0.sh 20`

**필요한 문서**:
- README.md (빠른 시작)
- VISUALIZE_BOOKSTORE_V0.md (상세 명령어)

---

### 시나리오 2: 모델을 학습하고 싶은 경우

1. **[README.md](./README.md)** → "데이터셋 설정", "모델 학습" 섹션
2. **[README_4REL_SDD.md](./README_4REL_SDD.md)** → "Training" 섹션
3. 학습 명령어 실행

**필요한 문서**:
- README.md (전체 개요)
- README_4REL_SDD.md (상세 설정)

---

### 시나리오 3: Threat Score 계산 방법을 수정하고 싶은 경우

1. **[README_4REL_SDD.md](./README_4REL_SDD.md)** → "Graph Construction", "Design Choices & Tunables" 섹션
2. `utils/affordance.py` 파일 수정
3. 테스트 및 검증

**필요한 문서**:
- README_4REL_SDD.md (전체)
- 코드: `utils/affordance.py`

---

### 시나리오 4: 새로운 Relation을 추가하고 싶은 경우

1. **[README_4REL_SDD.md](./README_4REL_SDD.md)** → 전체 문서
2. `utils/sdd_dataloader.py` 수정 (새 relation 추가)
3. `model/predictor.py` 수정 (relation 수 및 split 설정)
4. 테스트 및 검증

**필요한 문서**:
- README_4REL_SDD.md (전체)
- 코드: `utils/sdd_dataloader.py`, `model/predictor.py`

---

### 시나리오 5: 문제가 발생한 경우

1. **[README.md](./README.md)** → "문제 해결" 섹션
2. 해당 기능의 상세 문서 확인
3. 문제 해결 섹션 참조

**필요한 문서**:
- README.md (문제 해결)
- 해당 기능의 상세 문서

---

## 📝 문서 작성 가이드

새로운 문서를 추가할 때:

1. **명확한 제목**: 문서의 목적을 명확히
2. **대상 독자**: 누구를 위한 문서인지 명시
3. **목차**: 주요 섹션 나열
4. **예시**: 실제 사용 예시 포함
5. **문제 해결**: 자주 발생하는 문제와 해결 방법
6. **링크**: 관련 문서로의 링크

---

## 🔗 관련 링크

- **원본 DMRGCN**: https://github.com/InhwanBae/DMRGCN
- **논문**: AAAI 2021
- **데이터셋**: 
  - SDD: `/raid/guest/SDD_2beon/`
  - 비디오: `/raid/guest/OATMeal_Queens/newbie/yoonhee/modeling/DMRGCN/sdd_datasets/video/`

---

## 📧 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

**마지막 업데이트**: 2024년 11월

