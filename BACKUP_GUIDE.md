# 🚀 체크포인트 백업 가이드

깃에 올리지 않는 대용량 파일들(체크포인트, 캐시 등)을 SCP로 쉽게 백업/복원하는 방법입니다.

## 📦 포함되는 파일들

### 자동 백업 대상
- `checkpoints_unified/` - 학습된 모델 체크포인트 (28MB)
- `data_cache/` - 전처리된 데이터 캐시 (334KB)  
- `test_results_comprehensive/` - 테스트 결과 (20KB)
- `*.pth` - 개별 모델 파일들
- `*.pkl` - 피클 데이터 파일들

### 제외되는 파일
- `copy_dmrgcn/`, `copy_gpgraph/` 내부의 원본 체크포인트들 (이미 깃에 포함)
- 소스 코드 파일들 (깃으로 관리)

## 🛠️ 사용 방법

### 1. Bash 스크립트 (간단한 사용)

```bash
# 실행 권한 확인
chmod +x backup_checkpoints.sh

# 백업할 파일 목록 확인
./backup_checkpoints.sh list

# 로컬에 압축 파일 생성
./backup_checkpoints.sh package

# 원격 서버로 업로드
./backup_checkpoints.sh upload user@server.com:/backup/

# 원격에서 다운로드
./backup_checkpoints.sh download user@server.com:/backup/comjonsul_backup_20251215_123456.tar.gz

# SSH 연결 테스트
./backup_checkpoints.sh test user@server.com
```

### 2. Python 스크립트 (고급 기능)

```bash
# 실행 권한 확인  
chmod +x scp_helper.py

# 백업할 파일 목록 및 크기 확인
python scp_helper.py list

# 로컬에 압축 파일 생성 (커스텀 이름)
python scp_helper.py package --output my_backup.tar.gz

# 원격 서버로 업로드
python scp_helper.py upload user@server.com:/backup/

# 원격에서 다운로드
python scp_helper.py download user@server.com:/backup/file.tar.gz

# SSH 연결 테스트
python scp_helper.py test user@server.com
```

## 📋 상세 사용 예시

### 백업 workflow 
```bash
# 1. 현재 상태 확인
./backup_checkpoints.sh list
# 또는
python scp_helper.py list

# 2. 원격 서버로 백업
./backup_checkpoints.sh upload username@backup-server.com:/home/username/backups/

# 3. 백업 완료 확인 (원격 서버에서)
ssh username@backup-server.com "ls -la /home/username/backups/"
```

### 복원 workflow
```bash
# 1. 원격에서 최신 백업 다운로드
./backup_checkpoints.sh download username@backup-server.com:/home/username/backups/comjonsul_backup_20251215_143022.tar.gz

# 2. 압축 해제 (자동으로 물어봄)
# 스크립트가 자동으로 압축 해제 여부를 확인합니다

# 3. 복원 완료 확인
ls -la checkpoints_unified/
```

## 🔧 고급 옵션

### SSH 키 기반 인증 설정
```bash
# SSH 키 생성 (없는 경우)
ssh-keygen -t rsa -b 4096

# 공개키를 원격 서버에 복사
ssh-copy-id username@server.com

# 이후 비밀번호 없이 백업 가능
```

### 자동화된 정기 백업
```bash
# crontab에 추가 (매일 자정 백업)
0 0 * * * cd /root/Comjonsul && ./backup_checkpoints.sh upload user@backup:/backups/ > /var/log/backup.log 2>&1
```

### 선택적 백업
```bash
# 특정 체크포인트만 수동으로 백업
tar -czf manual_backup.tar.gz checkpoints_unified/quick_test_5epochs-eth/
scp manual_backup.tar.gz user@server:/backups/
```

## 🚨 주의사항

### 보안
- SSH 키 기반 인증 사용 권장
- 백업 서버의 접근 권한 제한
- 중요한 모델의 경우 암호화 백업 고려

### 네트워크
- 대용량 파일 전송시 안정적인 네트워크 환경 확인
- 전송 중단시 `rsync` 사용 고려:
  ```bash
  rsync -avz --progress backup.tar.gz user@server:/path/
  ```

### 저장 공간
- 원격 서버의 디스크 용량 확인
- 정기적인 오래된 백업 정리:
  ```bash
  # 30일 이상 된 백업 파일 삭제
  find /backup/path -name "comjonsul_backup_*.tar.gz" -mtime +30 -delete
  ```

## 📊 백업 용량 예상

| 구성요소 | 크기 | 설명 |
|---------|------|------|
| checkpoints_unified | ~28MB | 학습된 모델들 |
| data_cache | ~334KB | 전처리 캐시 |
| test_results | ~20KB | 테스트 결과 |
| **총 압축 크기** | **~25MB** | gzip 압축 적용 |

## 🔄 복원 체크리스트

백업에서 복원 후 다음을 확인하세요:

```bash
# 1. 체크포인트 파일 존재 확인
ls -la checkpoints_unified/quick_test_5epochs-eth/

# 2. 모델 로딩 테스트
python -c "
import torch
checkpoint = torch.load('checkpoints_unified/quick_test_5epochs-eth/eth_best.pth', map_location='cpu')
print('✅ 체크포인트 로딩 성공')
print(f'Epoch: {checkpoint.get(\"epoch\", \"N/A\")}')
print(f'Loss: {checkpoint.get(\"loss\", \"N/A\")}')
"

# 3. 간단한 추론 테스트
python simple_test.py --model_path ./checkpoints_unified/quick_test_5epochs-eth/eth_best.pth --num_samples 5
```

---

이제 체크포인트를 안전하게 백업하고 언제든지 복원할 수 있습니다! 🚀
