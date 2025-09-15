# Simple test script to verify basic functionality without pytest dependency

import torch
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from utils.shapes import (
    assert_shape, log_shape, ensure_time_axis, mask_features, mask_adj
)

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_shapes():
    """Test basic shape utilities"""
    print("🧪 Testing basic shape utilities...")
    
    # Test assert_shape
    x = torch.randn(2, 8, 5, 64)
    try:
        assert_shape(x, 'b t n d', 'features')
        print("✅ assert_shape working")
    except Exception as e:
        print(f"❌ assert_shape failed: {e}")
        return False
    
    # Test ensure_time_axis
    try:
        A = torch.randn(2, 5, 5)
        A_expanded = ensure_time_axis(A, T=8)
        assert A_expanded.shape == (2, 8, 5, 5)
        print("✅ ensure_time_axis working")
    except Exception as e:
        print(f"❌ ensure_time_axis failed: {e}")
        return False
    
    # Test mask_features
    try:
        H = torch.randn(2, 8, 5, 64)
        M = torch.ones(2, 8, 5)
        H_masked = mask_features(H, M)
        assert H_masked.shape == H.shape
        print("✅ mask_features working")
    except Exception as e:
        print(f"❌ mask_features failed: {e}")
        return False
    
    return True

def test_simple_model():
    """Test a simple version of the unified model"""
    print("🧪 Testing simple unified model...")
    
    try:
        from model.dmrgcn_gpgraph import DMRGCN_GPGraph_Model
        
        # Create simple model
        model = DMRGCN_GPGraph_Model(
            d_in=2,
            d_h=32,
            d_gp_in=32,
            T_pred=8,
            dmrgcn_hidden_dims=[16, 32],
            use_simple_head=True  # Use simple head to avoid complex dependencies
        )
        
        # Create test data
        B, T_obs, N, d_in = 1, 6, 3, 2
        T_pred = 8
        
        X = torch.randn(B, T_obs, N, d_in)
        A_obs = torch.randn(B, T_obs, N, N)
        M_obs = torch.ones(B, T_obs, N)
        M_pred = torch.ones(B, T_pred, N)
        
        # Forward pass
        delta_Y = model(X, A_obs, M_obs, M_pred=M_pred)
        
        # Check output shape
        expected_shape = (B, T_pred, N, 2)
        if delta_Y.shape == expected_shape:
            print(f"✅ Model forward pass successful: {delta_Y.shape}")
            return True
        else:
            print(f"❌ Wrong output shape: got {delta_Y.shape}, expected {expected_shape}")
            return False
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """Test loss computation"""
    print("🧪 Testing loss computation...")
    
    try:
        from model.dmrgcn_gpgraph import DMRGCN_GPGraph_Model
        
        model = DMRGCN_GPGraph_Model(
            d_in=2, d_h=32, T_pred=8, 
            dmrgcn_hidden_dims=[16, 32],
            use_simple_head=True
        )
        
        B, T_pred, N = 1, 8, 3
        
        delta_Y_pred = torch.randn(B, T_pred, N, 2)
        delta_Y_true = torch.randn(B, T_pred, N, 2)
        M_pred = torch.ones(B, T_pred, N)
        
        loss = model.compute_loss(delta_Y_pred, delta_Y_true, M_pred)
        
        if loss.item() >= 0:
            print(f"✅ Loss computation successful: {loss.item():.6f}")
            return True
        else:
            print(f"❌ Loss computation failed: {loss.item()}")
            return False
            
    except Exception as e:
        print(f"❌ Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple tests"""
    print("🚀 Starting simple validation tests...")
    
    success = True
    
    # Test 1: Basic shape utilities
    if not test_basic_shapes():
        success = False
    
    # Test 2: Simple model
    if not test_simple_model():
        success = False
    
    # Test 3: Loss computation
    if not test_loss_computation():
        success = False
    
    if success:
        print("\n🎉 All simple tests passed!")
        print("📏 Shape validation system working")
        print("🔧 Adapters handle dimension correctly")
        print("🚀 Unified model produces correct output: [B, T_pred, N, 2]")
    else:
        print("\n❌ Some tests failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
