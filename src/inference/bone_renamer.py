"""
Bone Renamer for UniRig

This script renames bones in FBX files to match standard naming conventions
like Mixamo or VRoid. It can be used as a post-processing step after inference.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
import bpy
import numpy as np


class BoneRenamer:
    """Renames bones in FBX files based on configuration files."""
    
    def __init__(self, config_path: str):
        """
        Initialize the bone renamer with a configuration file.
        
        Args:
            config_path: Path to the skeleton configuration file (e.g., mixamo.yaml, vroid.yaml)
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.bone_mapping = self._create_bone_mapping()
    
    def _load_config(self) -> Dict:
        """Load the skeleton configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_bone_mapping(self) -> Dict[str, str]:
        """
        Create a mapping from generic bone names (bone_0, bone_1, etc.) to standard names.
        
        Returns:
            Dictionary mapping generic names to standard names
        """
        mapping = {}
        bone_index = 0
        
        # Get the order of parts
        parts_order = self.config.get('parts_order', [])
        
        for part in parts_order:
            if part in self.config.get('parts', {}):
                part_bones = self.config['parts'][part]
                for bone_name in part_bones:
                    mapping[f'bone_{bone_index}'] = bone_name
                    bone_index += 1
        
        return mapping
    
    def rename_bones_in_fbx(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Rename bones in an FBX file according to the configuration.
        
        Args:
            input_path: Path to the input FBX file
            output_path: Path for the output FBX file (if None, overwrites input)
            
        Returns:
            Path to the output file
        """
        if output_path is None:
            output_path = input_path
        
        # Clean Blender scene
        self._clean_bpy()
        
        try:
            # Load the FBX file
            bpy.ops.import_scene.fbx(filepath=input_path, ignore_leaf_bones=False, use_image_search=False)
            
            # Find the armature
            armature = None
            for obj in bpy.data.objects:
                if obj.type == 'ARMATURE':
                    armature = obj
                    break
            
            if armature is None:
                raise ValueError("No armature found in the FBX file")
            
            # Rename bones
            self._rename_armature_bones(armature)
            
            # Export the modified file
            bpy.ops.export_scene.fbx(
                filepath=output_path,
                check_existing=False,
                add_leaf_bones=True
            )
            
            print(f"Successfully renamed bones and exported to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error processing file {input_path}: {e}")
            raise
        finally:
            # Clean up
            self._clean_bpy()
    
    def _rename_armature_bones(self, armature):
        """Rename bones in the armature according to the mapping."""
        # Get all bones in the armature
        bones = list(armature.data.bones)
        
        # Create a mapping from current bone names to new names
        current_to_new = {}
        
        for i, bone in enumerate(bones):
            generic_name = f'bone_{i}'
            if generic_name in self.bone_mapping:
                new_name = self.bone_mapping[generic_name]
                current_to_new[bone.name] = new_name
                print(f"Mapping {bone.name} -> {new_name}")
            else:
                print(f"No mapping found for {bone.name}, keeping original name")
        
        # Rename bones
        for old_name, new_name in current_to_new.items():
            if old_name in armature.data.bones:
                armature.data.bones[old_name].name = new_name
        
        # Update vertex groups to match new bone names
        self._update_vertex_groups(current_to_new)
    
    def _update_vertex_groups(self, name_mapping: Dict[str, str]):
        """Update vertex group names to match the new bone names."""
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.vertex_groups:
                for old_name, new_name in name_mapping.items():
                    if old_name in obj.vertex_groups:
                        obj.vertex_groups[old_name].name = new_name
    
    def _clean_bpy(self):
        """Clean up Blender scene."""
        # Remove all objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Remove all data
        for collection in bpy.data.collections:
            bpy.data.collections.remove(collection)
        
        # Clear unused data
        bpy.ops.outliner.orphans_purge(do_recursive=True)
    
    def get_bone_mapping(self) -> Dict[str, str]:
        """Get the current bone mapping."""
        return self.bone_mapping.copy()
    
    def print_mapping(self):
        """Print the current bone mapping for debugging."""
        print(f"Bone mapping from {self.config_path}:")
        for generic_name, standard_name in self.bone_mapping.items():
            print(f"  {generic_name} -> {standard_name}")


def rename_bones(
    input_file: str,
    output_file: Optional[str] = None,
    config_file: str = "configs/skeleton/mixamo.yaml"
) -> str:
    """
    Convenience function to rename bones in an FBX file.
    
    Args:
        input_file: Path to the input FBX file
        output_file: Path for the output FBX file (if None, overwrites input)
        config_file: Path to the skeleton configuration file
        
    Returns:
        Path to the output file
    """
    renamer = BoneRenamer(config_file)
    return renamer.rename_bones_in_fbx(input_file, output_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rename bones in FBX files")
    parser.add_argument("input_file", help="Input FBX file path")
    parser.add_argument("--output", "-o", help="Output FBX file path (optional)")
    parser.add_argument("--config", "-c", default="configs/skeleton/mixamo.yaml", 
                       help="Skeleton configuration file path")
    
    args = parser.parse_args()
    
    try:
        output_path = rename_bones(args.input_file, args.output, args.config)
        print(f"Successfully processed: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1) 