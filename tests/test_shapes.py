# Unit tests for shape utilities and adapters
# Validates strict [B, T, N, d] shape conventions

import torch
import pytest
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.shapes import (
    assert_shape, log_shape, ensure_time_axis, ensure_relation_axis,
    mask_features, mask_adj, aggregate_time_features, pad_sequence_batch
)
from adapters.dmrgcn_adapter import DMRGCNEncoderAdapter
from adapters.gpgraph_head import GPGraphHead, SimpleRegressionHead
from model.dmrgcn_gpgraph import DMRGCN_GPGraph_Model

# Setup logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestShapeUtilities:
    """Test shape utility functions"""
    
    def test_assert_shape_valid(self):
        """Test assert_shape with valid inputs"""
        x = torch.randn(2, 8, 5, 64)
        assert_shape(x, 'b t n d', 'features')  # Should pass
        
        A = torch.randn(2, 8, 5, 5)
        assert_shape(A, 'b t n n', 'adjacency')  # Should pass
    
    def test_assert_shape_invalid(self):
        """Test assert_shape with invalid inputs"""
        x = torch.randn(2, 8, 5, 64)
        
        with pytest.raises(AssertionError):
            assert_shape(x, 'b t n n', 'wrong_spec')
        
        with pytest.raises(AssertionError):
            assert_shape(x, 'b t n', 'too_few_dims')
    
    def test_ensure_time_axis(self):
        """Test time axis expansion"""
        # Test [B, N, N] -> [B, T, N, N]
        A = torch.randn(2, 5, 5)
        A_expanded = ensure_time_axis(A, T=8)
        
        assert A_expanded.shape == (2, 8, 5, 5)
        assert torch.allclose(A_expanded[:, 0], A)
        assert torch.allclose(A_expanded[:, 7], A)
        
        # Test [B, T, N, N] unchanged
        A_time = torch.randn(2, 8, 5, 5)
        A_unchanged = ensure_time_axis(A_time, T=8)
        assert torch.allclose(A_unchanged, A_time)
    
    def test_ensure_relation_axis(self):
        """Test relation axis expansion"""
        # Test [B, T, N, N] -> [B, T, R, N, N]
        A = torch.randn(2, 8, 5, 5)
        A_rel = ensure_relation_axis(A, R_expected=3)
        
        assert A_rel.shape == (2, 8, 3, 5, 5)
        for r in range(3):
            assert torch.allclose(A_rel[:, :, r], A)
    
    def test_mask_features(self):
        """Test feature masking"""
        H = torch.randn(2, 8, 5, 64)
        M = torch.randint(0, 2, (2, 8, 5)).float()
        
        H_masked = mask_features(H, M)
        
        assert H_masked.shape == H.shape
        
        # Check masking is applied correctly
        for b in range(2):
            for t in range(8):
                for n in range(5):
                    if M[b, t, n] == 0:
                        assert torch.allclose(H_masked[b, t, n], torch.zeros(64))
                    else:
                        assert torch.allclose(H_masked[b, t, n], H[b, t, n])
    
    def test_mask_adj(self):
        """Test adjacency masking"""
        A = torch.randn(2, 8, 5, 5)
        M = torch.randint(0, 2, (2, 8, 5)).float()
        
        A_masked = mask_adj(A, M)
        
        assert A_masked.shape == A.shape
        
        # Check edge masking: edge exists only if both nodes are valid
        for b in range(2):
            for t in range(8):
                for i in range(5):
                    for j in range(5):
                        if M[b, t, i] == 0 or M[b, t, j] == 0:
                            assert A_masked[b, t, i, j] == 0
    
    def test_aggregate_time_features(self):
        """Test temporal aggregation"""
        H = torch.randn(2, 8, 5, 64)
        M = torch.ones(2, 8, 5)
        M[:, 6:, :] = 0  # Last 2 timesteps invalid
        
        # Test 'last' aggregation
        H_last = aggregate_time_features(H, M, method='last')
        assert H_last.shape == (2, 5, 64)
        
        # Should use timestep 5 (last valid) for all agents
        assert torch.allclose(H_last, H[:, 5])
        
        # Test 'mean' aggregation
        H_mean = aggregate_time_features(H, M, method='mean')
        assert H_mean.shape == (2, 5, 64)
        
        # Check mean is computed correctly
        expected_mean = H[:, :6].mean(dim=1)  # Average over valid timesteps
        assert torch.allclose(H_mean, expected_mean, atol=1e-6)
    
    def test_pad_sequence_batch(self):
        """Test sequence padding"""
        # Create sequences of different lengths
        seq1 = torch.randn(5, 3, 64)   # T=5
        seq2 = torch.randn(8, 3, 64)   # T=8
        seq3 = torch.randn(3, 3, 64)   # T=3
        
        sequences = [seq1, seq2, seq3]
        padded, mask = pad_sequence_batch(sequences)
        
        assert padded.shape == (3, 8, 3, 64)  # (B, T_max, N, d)
        assert mask.shape == (3, 8, 3)
        
        # Check padding correctness
        assert torch.allclose(padded[0, :5], seq1)
        assert torch.allclose(padded[1, :8], seq2)
        assert torch.allclose(padded[2, :3], seq3)
        
        # Check mask correctness
        assert mask[0, :5].all() and not mask[0, 5:].any()
        assert mask[1, :8].all()
        assert mask[2, :3].all() and not mask[2, 3:].any()


class TestDMRGCNAdapter:
    """Test DMRGCN encoder adapter"""
    
    def test_dmrgcn_adapter_forward(self):
        """Test DMRGCN adapter forward pass"""
        # Model parameters
        B, T_obs, N, d_in = 2, 8, 5, 2
        d_h = 128
        
        # Create adapter
        adapter = DMRGCNEncoderAdapter(
            d_in=d_in,
            d_h=d_h,
            hidden_dims=[32, 32, 64, 128],
            distance_scales=[0.5, 1.0, 2.0]
        )
        
        # Create inputs
        X = torch.randn(B, T_obs, N, d_in)
        A = torch.randn(B, T_obs, N, N)
        M = torch.ones(B, T_obs, N)
        
        # Forward pass
        H = adapter(X, A, M)
        
        # Check output shape
        assert H.shape == (B, T_obs, N, d_h)
        
        # Check masking: if we zero out some agents, their features should be zero
        M_partial = M.clone()
        M_partial[:, :, 2] = 0  # Mask out agent 2
        
        H_masked = adapter(X, A, M_partial)
        assert torch.allclose(H_masked[:, :, 2], torch.zeros(B, T_obs, d_h))
    
    def test_dmrgcn_adapter_shapes(self):
        """Test various input shapes for DMRGCN adapter"""
        adapter = DMRGCNEncoderAdapter(d_in=2, d_h=64)
        
        # Test different batch sizes and sequence lengths
        test_cases = [
            (1, 4, 3, 2),   # Small case
            (4, 12, 10, 2), # Larger case
            (2, 1, 1, 2),   # Minimal case
        ]
        
        for B, T, N, d_in in test_cases:
            X = torch.randn(B, T, N, d_in)
            A = torch.randn(B, T, N, N)
            M = torch.ones(B, T, N)
            
            H = adapter(X, A, M)
            assert H.shape == (B, T, N, 64)


class TestGPGraphHead:
    """Test GP-Graph head adapter"""
    
    def test_gpgraph_head_forward(self):
        """Test GP-Graph head forward pass"""
        # Model parameters
        B, T_obs, N, d_h = 2, 8, 5, 128
        T_pred = 12
        
        # Create head
        head = GPGraphHead(
            d_h=d_h,
            d_gp_in=128,
            T_pred=T_pred,
            group_type='euclidean',
            enable_paths={'agent': True, 'intra': True, 'inter': False}
        )
        
        # Create inputs
        H = torch.randn(B, T_obs, N, d_h)
        A_obs = torch.randn(B, T_obs, N, N)
        M_obs = torch.ones(B, T_obs, N)
        M_pred = torch.ones(B, T_pred, N)
        
        # Forward pass
        delta_Y = head(H, A_obs, M_obs, M_pred)
        
        # Check output shape
        assert delta_Y.shape == (B, T_pred, N, 2)
        
        # Test with group returns
        delta_Y, groups = head(H, A_obs, M_obs, M_pred, return_groups=True)
        assert delta_Y.shape == (B, T_pred, N, 2)
        assert groups.shape == (N,)  # Group indices for batch size 1
    
    def test_simple_regression_head(self):
        """Test simple regression head"""
        B, T_obs, N, d_h = 2, 8, 5, 128
        T_pred = 12
        
        head = SimpleRegressionHead(d_h=d_h, T_pred=T_pred)
        
        H = torch.randn(B, T_obs, N, d_h)
        M_obs = torch.ones(B, T_obs, N)
        M_pred = torch.ones(B, T_pred, N)
        
        delta_Y = head(H, M_obs, M_pred)
        assert delta_Y.shape == (B, T_pred, N, 2)


class TestUnifiedModel:
    """Test complete unified model"""
    
    def test_unified_model_forward(self):
        """Test complete model forward pass"""
        # Model parameters
        B, T_obs, N, d_in = 2, 8, 5, 2
        T_pred = 12
        
        # Create model
        model = DMRGCN_GPGraph_Model(
            d_in=d_in,
            d_h=64,
            d_gp_in=64,
            T_pred=T_pred,
            dmrgcn_hidden_dims=[32, 32, 64],
            use_simple_head=True  # Use simple head for faster testing
        )
        
        # Create inputs
        X = torch.randn(B, T_obs, N, d_in)
        A_obs = torch.randn(B, T_obs, N, N)
        M_obs = torch.ones(B, T_obs, N)
        M_pred = torch.ones(B, T_pred, N)
        
        # Forward pass
        delta_Y = model(X, A_obs, M_obs, M_pred=M_pred)
        
        # Check output shape
        assert delta_Y.shape == (B, T_pred, N, 2)
        
        print(f"✅ Unified model forward pass successful: {delta_Y.shape}")
    
    def test_unified_model_with_adjacency_expansion(self):
        """Test model with adjacency expansion from [B, N, N] to [B, T, N, N]"""
        B, T_obs, N, d_in = 1, 6, 4, 2
        T_pred = 8
        
        model = DMRGCN_GPGraph_Model(
            d_in=d_in,
            d_h=32,
            T_pred=T_pred,
            dmrgcn_hidden_dims=[16, 32],
            use_simple_head=True
        )
        
        # Create inputs with static adjacency
        X = torch.randn(B, T_obs, N, d_in)
        A_static = torch.randn(B, N, N)  # Static adjacency
        M_obs = torch.ones(B, T_obs, N)
        
        # Forward pass should automatically expand adjacency
        delta_Y = model(X, A_static, M_obs)
        
        assert delta_Y.shape == (B, T_pred, N, 2)
        print(f"✅ Static adjacency expansion successful: {delta_Y.shape}")
    
    def test_unified_model_loss_computation(self):
        """Test loss computation"""
        B, T_pred, N = 2, 12, 5
        
        model = DMRGCN_GPGraph_Model(T_pred=T_pred, use_simple_head=True)
        
        # Create prediction and ground truth
        delta_Y_pred = torch.randn(B, T_pred, N, 2)
        delta_Y_true = torch.randn(B, T_pred, N, 2)
        M_pred = torch.ones(B, T_pred, N)
        M_pred[:, 8:, 2] = 0  # Mask out some predictions
        
        # Compute loss
        loss = model.compute_loss(delta_Y_pred, delta_Y_true, M_pred)
        
        assert loss.item() >= 0
        assert loss.requires_grad
        print(f"✅ Loss computation successful: {loss.item():.6f}")
    
    def test_unified_model_trajectory_prediction(self):
        """Test trajectory prediction (absolute coordinates)"""
        B, T_obs, N, d_in = 1, 8, 3, 2
        T_pred = 12
        
        model = DMRGCN_GPGraph_Model(
            d_in=d_in,
            T_pred=T_pred,
            use_simple_head=True
        )
        
        # Create inputs with position data
        X = torch.cumsum(torch.randn(B, T_obs, N, d_in) * 0.1, dim=1)  # Smooth trajectories
        A_obs = torch.ones(B, T_obs, N, N) * 0.1  # Weak connections
        M_obs = torch.ones(B, T_obs, N)
        M_pred = torch.ones(B, T_pred, N)
        
        # Get trajectory predictions
        trajectories = model.predict_trajectories(X, A_obs, M_obs, M_pred, return_absolute=True)
        
        assert trajectories.shape == (B, T_pred, N, 2)
        
        # Check that trajectories start from last observed position
        last_obs_pos = X[:, -1, :, :]
        first_pred_pos = trajectories[:, 0, :, :]
        
        # Should be approximately continuous (within reasonable range)
        position_jump = torch.norm(first_pred_pos - last_obs_pos, dim=-1)
        assert position_jump.max() < 5.0  # Reasonable continuity
        
        print(f"✅ Trajectory prediction successful: {trajectories.shape}")
    
    def test_model_info(self):
        """Test model info retrieval"""
        model = DMRGCN_GPGraph_Model(
            d_in=4,
            d_h=128,
            T_pred=12,
            use_multimodal=True
        )
        
        info = model.get_model_info()
        
        assert info['model_type'] == 'DMRGCN+GPGraph'
        assert info['input_dim'] == 4
        assert info['hidden_dim'] == 128
        assert info['prediction_length'] == 12
        assert info['encoder_type'] == 'multimodal'
        assert 'total_parameters' in info
        
        print(f"✅ Model info: {info}")


def run_all_tests():
    """Run all tests manually"""
    print("🧪 Running shape utilities tests...")
    
    # Shape utilities tests
    test_shapes = TestShapeUtilities()
    test_shapes.test_assert_shape_valid()
    test_shapes.test_ensure_time_axis()
    test_shapes.test_ensure_relation_axis()
    test_shapes.test_mask_features()
    test_shapes.test_mask_adj()
    test_shapes.test_aggregate_time_features()
    test_shapes.test_pad_sequence_batch()
    print("✅ Shape utilities tests passed")
    
    # DMRGCN adapter tests
    print("\n🧪 Running DMRGCN adapter tests...")
    test_dmrgcn = TestDMRGCNAdapter()
    test_dmrgcn.test_dmrgcn_adapter_forward()
    test_dmrgcn.test_dmrgcn_adapter_shapes()
    print("✅ DMRGCN adapter tests passed")
    
    # GP-Graph head tests
    print("\n🧪 Running GP-Graph head tests...")
    test_gpgraph = TestGPGraphHead()
    test_gpgraph.test_gpgraph_head_forward()
    test_gpgraph.test_simple_regression_head()
    print("✅ GP-Graph head tests passed")
    
    # Unified model tests
    print("\n🧪 Running unified model tests...")
    test_unified = TestUnifiedModel()
    test_unified.test_unified_model_forward()
    test_unified.test_unified_model_with_adjacency_expansion()
    test_unified.test_unified_model_loss_computation()
    test_unified.test_unified_model_trajectory_prediction()
    test_unified.test_model_info()
    print("✅ Unified model tests passed")
    
    print("\n🎉 All tests passed successfully!")
    print("📏 Shape validation system working correctly")
    print("🔧 Adapters handle dimension mismatches properly")
    print("🚀 Unified model produces correct output shapes")


if __name__ == "__main__":
    run_all_tests()
