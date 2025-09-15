#!/bin/bash

# Model Directory Cleanup Script
# Consolidates model/ and models/ into a single, clean structure

echo "🧹 Starting Model Directory Cleanup..."

# Check if directories exist
if [ ! -d "model" ] || [ ! -d "models" ]; then
    echo "❌ Error: model/ or models/ directory not found"
    exit 1
fi

# Step 1: Backup existing model directory
echo "📦 Step 1: Backing up existing model/ directory..."
if [ -d "model_legacy" ]; then
    echo "⚠️  model_legacy already exists, removing..."
    rm -rf model_legacy
fi
mv model model_legacy
echo "✅ Backed up model/ → model_legacy/"

# Step 2: Move models to model
echo "🔄 Step 2: Moving models/ → model/..."
mv models model
echo "✅ Moved models/ → model/"

# Step 3: Copy essential files from legacy
echo "📋 Step 3: Copying essential files from model_legacy/..."
cp model_legacy/backbone.py model/
cp model_legacy/gpgraph_adapter.py model/
cp model_legacy/utils.py model/
cp model_legacy/__init__.py model/
echo "✅ Copied essential files to model/"

# Step 4: Update import statements in Python files
echo "🔧 Step 4: Updating import statements..."

# Find all Python files and update imports
find . -name "*.py" -not -path "./model_legacy/*" -not -path "./__pycache__/*" -not -path "./model/__pycache__/*" | while read file; do
    if grep -q "from models\." "$file"; then
        echo "   Updating imports in: $file"
        sed -i.bak 's/from models\./from model\./g' "$file"
        rm "$file.bak"  # Remove backup file
    fi
done

echo "✅ Updated import statements"

# Step 5: Create unified __init__.py
echo "📝 Step 5: Creating unified __init__.py..."
cat > model/__init__.py << 'EOF'
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
EOF
echo "✅ Created unified __init__.py"

# Step 6: Validation
echo "🔍 Step 6: Validation..."

# Check if critical files exist
critical_files=("model/dmrgcn_gpgraph.py" "model/backbone.py" "model/__init__.py")
for file in "${critical_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Critical file missing: $file"
        exit 1
    fi
done

echo "✅ All critical files present"

# Try to import the main model (basic syntax check)
python3 -c "
try:
    from model.dmrgcn_gpgraph import DMRGCN_GPGraph_Model
    print('✅ Import test successful: DMRGCN_GPGraph_Model')
except Exception as e:
    print(f'❌ Import test failed: {e}')
    exit(1)
" || exit 1

echo ""
echo "🎉 Cleanup Complete!"
echo ""
echo "📋 Summary:"
echo "   ✅ model_legacy/  - Backup of original model/"
echo "   ✅ model/         - Unified model directory"
echo "   ✅ Updated imports in all Python files"
echo "   ✅ Created unified __init__.py"
echo ""
echo "🚀 You can now use:"
echo "   from model.dmrgcn_gpgraph import DMRGCN_GPGraph_Model"
echo ""
echo "🧪 Test with:"
echo "   python demo_final.py"
echo "   python test_simple.py"
