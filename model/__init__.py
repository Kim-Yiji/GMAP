# Unified Model Package
# Contains both legacy and refactored model implementations

# Main Shape-Refactored Model (Recommended)
from .dmrgcn_gpgraph import DMRGCN_GPGraph_Model

# Components
from .backbone import DMRGCNBackbone
from .gpgraph_adapter import GroupAssignment, GroupIntegration
from .utils import *

# Legacy compatibility (if needed)
try:
    from .dmrgcn_gpgraph import DMRGCNGPGraph  # Legacy class name
except ImportError:
    pass  # Legacy class might not exist in refactored version

__all__ = [
    'DMRGCN_GPGraph_Model',  # Main model
    'DMRGCNBackbone',        # Backbone
    'GroupAssignment',       # Group processing
    'GroupIntegration',      # Feature fusion
]
