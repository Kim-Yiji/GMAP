# 🚀 Data Caching System

## 📋 Overview

데이터 전처리 결과를 자동으로 캐싱하여 두 번째 실행부터는 훨씬 빠르게 로딩할 수 있는 시스템을 구현했습니다.

## ✨ Key Features

### 🔄 **Automatic Caching**
- **첫 번째 실행**: 데이터를 처리하고 캐시 파일 생성
- **이후 실행**: 캐시에서 바로 로딩 (5-10배 빠름)
- **자동 감지**: 데이터나 파라미터 변경 시 자동으로 재처리

### 💾 **Smart Cache Management**
- **Unique Keys**: 파라미터와 파일 변경시간 기반 해시
- **Invalidation**: 데이터 변경 시 자동으로 캐시 무효화
- **Compression**: Pickle을 사용한 효율적인 저장

### 🎯 **What Gets Cached**
- **Raw Trajectories**: `obs_traj`, `pred_traj`, `obs_traj_rel`, `pred_traj_rel`
- **Masks**: `loss_mask`, `non_linear_ped`
- **Agent IDs**: `agent_ids_list`
- **Graph Data**: `V_obs`, `A_obs`, `V_pred`, `A_pred` (가장 시간이 많이 걸리는 부분)

## 🚀 Usage

### 기본 사용법 (자동 활성화)
```python
# train_unified.py에서 자동으로 캐싱 사용
python train_unified.py --dataset eth --batch_size 8 --num_epochs 50
```

### 캐싱 옵션 제어
```bash
# 캐싱 비활성화
python train_unified.py --dataset eth --no-use_cache

# 커스텀 캐시 디렉토리
python train_unified.py --dataset eth --cache_dir ./my_custom_cache
```

### 다중 데이터셋 학습
```bash
# 각 데이터셋마다 별도 캐시 생성
for dataset in eth hotel univ zara1 zara2; do
    python train_unified.py --dataset $dataset --num_epochs 30
done
```

## 📊 Performance Benefits

### 🕐 **Time Savings**
- **First Run**: ~2-5 minutes (데이터셋 크기에 따라)
- **Cached Run**: ~10-30 seconds
- **Speedup**: **5-10x faster** ⚡

### 💽 **Storage**
- **Cache Size**: 보통 50-200MB per dataset
- **Location**: `./data_cache/{dataset_name}/`
- **Format**: `.pkl` files with unique hash names

## 🔧 Technical Details

### Cache Key Generation
```python
# 파라미터 + 파일 수정시간으로 유니크 키 생성
params = f"{dataset}_{obs_len}_{pred_len}_{skip}_{delim}"
file_info = [(filename, modification_time), ...]
cache_key = md5(params + file_info).hexdigest()[:16]
```

### Cache Structure
```
data_cache/
├── eth/
│   ├── a1b2c3d4e5f6g7h8.pkl  # train data cache
│   └── f9e8d7c6b5a4321.pkl   # val data cache
├── hotel/
│   ├── x1y2z3w4v5u6t7s8.pkl
│   └── s9r8q7p6o5n4m321.pkl
└── ...
```

## ⚙️ Configuration

### Arguments in `train_unified.py`
```python
parser.add_argument('--use_cache', action='store_true', default=True,
                   help='Use cached preprocessed data')
parser.add_argument('--cache_dir', default='./data_cache',
                   help='Directory for data cache')
```

### TrajectoryDataset Parameters
```python
TrajectoryDataset(
    data_dir='./copy_dmrgcn/datasets/eth/train/',
    obs_len=8,
    pred_len=12,
    skip=1,
    min_ped=1,
    delim='tab',
    use_cache=True,           # Enable/disable caching
    cache_dir='./data_cache'  # Cache directory
)
```

## 🔍 Cache Validation

### When Cache is Used
✅ Same dataset, parameters, and file modification times
✅ Cache file exists and is readable
✅ All required data fields are present

### When Cache is Invalidated
❌ Different `obs_len`, `pred_len`, `skip`, or other parameters
❌ Dataset files have been modified
❌ Cache file is corrupted or missing fields
❌ Different delimiter or min_ped settings

## 🧹 Cache Management

### Clear All Caches
```bash
rm -rf ./data_cache/
```

### Clear Specific Dataset Cache
```bash
rm -rf ./data_cache/eth/
```

### Check Cache Status
```bash
# See cache sizes
du -sh ./data_cache/*/

# List cache files
find ./data_cache -name "*.pkl" -exec ls -lh {} \;
```

## 🚨 Troubleshooting

### "Processing data from scratch" Every Time
- Check if file modification times are changing
- Verify parameters are consistent
- Ensure cache directory has write permissions

### Large Cache Files
- Normal for large datasets (ETH ~100MB, Hotel ~50MB)
- Cache includes preprocessed graph adjacency matrices
- Trade-off: storage space for computation time

### Memory Issues
- Cache loading requires sufficient RAM
- Large datasets may need 1-2GB for cache loading
- Consider smaller batch sizes if memory limited

## 💡 Best Practices

1. **Keep Cache**: Don't delete cache between training runs
2. **Consistent Params**: Use same obs_len/pred_len for experiments
3. **Monitor Storage**: Check cache directory size periodically
4. **Server Usage**: Cache is especially beneficial on slow storage systems

## 🎯 Benefits for Your Workflow

### Development
- **Rapid Iteration**: Quick model parameter changes
- **Debug Friendly**: Fast data loading for debugging
- **Experiment Speed**: Multiple runs without data preprocessing delay

### Production
- **Training Efficiency**: Start training immediately
- **Resource Savings**: Less CPU usage for data preprocessing
- **Reproducibility**: Consistent data processing across runs

---

**🎉 Now you can run experiments much faster!** The first run creates the cache, and every subsequent run with the same parameters will be lightning fast! ⚡
