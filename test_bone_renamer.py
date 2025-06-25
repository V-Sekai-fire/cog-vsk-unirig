#!/usr/bin/env python3
"""
Test script for the bone renaming functionality.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_bone_renamer():
    """Test the bone renaming functionality."""
    try:
        from src.inference.bone_renamer import BoneRenamer
        
        # Test with Mixamo configuration
        print("Testing Mixamo bone renaming...")
        mixamo_renamer = BoneRenamer("configs/skeleton/mixamo.yaml")
        mixamo_renamer.print_mapping()
        
        # Test with VRoid configuration
        print("\nTesting VRoid bone renaming...")
        vroid_renamer = BoneRenamer("configs/skeleton/vroid.yaml")
        vroid_renamer.print_mapping()
        
        print("\n✅ Bone renaming test completed successfully!")
        print("The bone renamer is ready to use.")
        
    except Exception as e:
        print(f"❌ Error testing bone renamer: {e}")
        return False
    
    return True

def test_config_files():
    """Test that configuration files exist and are valid."""
    config_files = [
        "configs/skeleton/mixamo.yaml",
        "configs/skeleton/vroid.yaml"
    ]
    
    for config_file in config_files:
        if not Path(config_file).exists():
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required fields
            if 'parts_order' not in config:
                print(f"❌ Missing 'parts_order' in {config_file}")
                return False
            
            if 'parts' not in config:
                print(f"❌ Missing 'parts' in {config_file}")
                return False
            
            print(f"✅ Configuration file {config_file} is valid")
            
        except Exception as e:
            print(f"❌ Error reading configuration file {config_file}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Bone Renaming Functionality")
    print("=" * 50)
    
    # Test configuration files
    if not test_config_files():
        print("❌ Configuration file test failed")
        sys.exit(1)
    
    # Test bone renamer
    if not test_bone_renamer():
        print("❌ Bone renamer test failed")
        sys.exit(1)
    
    print("\n🎉 All tests passed! The bone renaming functionality is ready to use.")
    print("\nUsage:")
    print("1. In the Gradio app, select 'mixamo' or 'vroid' from the 'Bone Renaming' dropdown")
    print("2. Or use the command line: python src/inference/bone_renamer.py input.fbx --config configs/skeleton/mixamo.yaml") 