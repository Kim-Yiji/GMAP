#!/usr/bin/env python3
"""
Quick start script for the Unified Trajectory Prediction Model
This script provides a simple way to test the model setup and basic functionality.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import UnifiedTrajectoryPredictor
    from utils import TrajectoryDataset
    print("✅ Model imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def check_dataset():
    """Check if datasets are available"""
    dataset_path = './dataset/eth/'
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please ensure the dataset is properly copied.")
        return False
    
    train_path = dataset_path + 'train/'
    if not os.path.exists(train_path) or not os.listdir(train_path):
        print(f"❌ Training data not found at {train_path}")
        return False
    
    print("✅ Dataset check passed!")
    return True


def test_model_creation():
    """Test model creation and basic forward pass"""
    print("Testing model creation...")
    
    try:
        model = UnifiedTrajectoryPredictor(
            n_stgcn=1,
            n_tpcnn=2,  # Reduced for quick test
            input_feat=2,
            output_feat=5,
            seq_len=8,
            pred_seq_len=12,
            kernel_size=3
        )
        print("✅ Model created successfully!")
        
        # Test with dummy data
        batch_size, channels, time_steps, num_peds = 1, 2, 8, 3
        dummy_input = torch.randn(batch_size, channels, time_steps, num_peds)
        dummy_adj = torch.randn(batch_size, 3, time_steps, num_peds, num_peds)
        
        # Test forward pass
        with torch.no_grad():
            output, groups, aux_info = model(dummy_input, dummy_adj)
            print(f"✅ Forward pass successful! Output shape: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_dataloader():
    """Test data loading"""
    print("Testing data loader...")
    
    try:
        # Create a small dataset for testing
        dataset = TrajectoryDataset(
            './dataset/eth/train/',
            obs_len=8,
            pred_len=12,
            skip=1,
            min_ped=1
        )
        
        if len(dataset) == 0:
            print("❌ Dataset is empty!")
            return False
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Test loading one batch
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:  # Only test first batch
                break
                
            print(f"✅ Data loading successful! Batch shapes:")
            for i, tensor in enumerate(batch):
                if torch.is_tensor(tensor):
                    print(f"  Tensor {i}: {tensor.shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False


def main():
    """Main function for quick start testing"""
    print("🚀 Unified Trajectory Prediction Model - Quick Start Test")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available! Device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Run tests
    tests = [
        ("Dataset Check", check_dataset),
        ("Model Creation", test_model_creation),
        ("Data Loader", test_dataloader)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed!")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The model is ready to use.")
        print("\nNext steps:")
        print("1. Train the model: ./run_train.sh eth unified-model 50")
        print("2. Test the model: ./run_test.sh eth unified-model 20")
        print("3. Check results in checkpoints/unified-model-eth/")
    else:
        print("⚠️  Some tests failed. Please check the setup.")
        
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
