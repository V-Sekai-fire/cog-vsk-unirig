#!/usr/bin/env python3
"""
Simple script to rename bones in GLB/GLTF files by directly manipulating the JSON structure.
This approach doesn't require Blender or complex 3D libraries.
"""

import json
import yaml
import gzip
import struct
from pathlib import Path
import logging
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GLBProcessor:
    def __init__(self, config_path: str = "configs/skeleton/mixamo.yaml"):
        """Initialize with skeleton configuration."""
        self.bone_names = self._load_bone_config(config_path)
        
    def _load_bone_config(self, config_path: str) -> List[str]:
        """Load bone names from configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        bone_names = []
        parts_order = config.get('parts_order', [])
        
        for part in parts_order:
            if part in config.get('parts', {}):
                bone_names.extend(config['parts'][part])
        
        logger.info(f"Loaded {len(bone_names)} bone names from {config_path}")
        return bone_names
    
    def read_glb(self, filepath: str) -> tuple:
        """Read GLB file and return JSON and binary data."""
        with open(filepath, 'rb') as f:
            # Read GLB header
            magic = f.read(4)
            if magic != b'glTF':
                raise ValueError("Not a valid GLB file")
            
            version = struct.unpack('<I', f.read(4))[0]
            length = struct.unpack('<I', f.read(4))[0]
            
            # Read JSON chunk
            json_length = struct.unpack('<I', f.read(4))[0]
            json_type = f.read(4)
            if json_type != b'JSON':
                raise ValueError("Invalid JSON chunk")
            
            json_data = f.read(json_length)
            gltf_json = json.loads(json_data.decode('utf-8'))
            
            # Read binary chunk if present
            binary_data = None
            if f.tell() < length:
                binary_length = struct.unpack('<I', f.read(4))[0]
                binary_type = f.read(4)
                if binary_type == b'BIN\x00':
                    binary_data = f.read(binary_length)
            
            return gltf_json, binary_data
    
    def write_glb(self, filepath: str, gltf_json: dict, binary_data: Optional[bytes] = None):
        """Write GLB file with JSON and binary data."""
        json_str = json.dumps(gltf_json, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')
        
        # Pad JSON to 4-byte boundary
        json_padding = (4 - (len(json_bytes) % 4)) % 4
        json_bytes += b' ' * json_padding
        
        # Calculate total length
        total_length = 12 + 8 + len(json_bytes)  # header + JSON chunk header + JSON data
        if binary_data:
            binary_padding = (4 - (len(binary_data) % 4)) % 4
            binary_data += b'\x00' * binary_padding
            total_length += 8 + len(binary_data)  # binary chunk header + binary data
        
        with open(filepath, 'wb') as f:
            # Write GLB header
            f.write(b'glTF')
            f.write(struct.pack('<I', 2))  # version 2
            f.write(struct.pack('<I', total_length))
            
            # Write JSON chunk
            f.write(struct.pack('<I', len(json_bytes)))
            f.write(b'JSON')
            f.write(json_bytes)
            
            # Write binary chunk if present
            if binary_data:
                f.write(struct.pack('<I', len(binary_data)))
                f.write(b'BIN\x00')
                f.write(binary_data)
    
    def rename_bones(self, gltf_json: dict, bone_mapping: Dict[str, str]) -> dict:
        """Rename bones in GLTF JSON structure."""
        # Create a copy to avoid modifying the original
        new_gltf = json.loads(json.dumps(gltf_json))
        
        # Rename bones in nodes
        if 'nodes' in new_gltf:
            for node in new_gltf['nodes']:
                if 'name' in node and node['name'] in bone_mapping:
                    old_name = node['name']
                    new_name = bone_mapping[old_name]
                    node['name'] = new_name
                    logger.debug(f"Renamed node: {old_name} -> {new_name}")
        
        # Rename bones in skins
        if 'skins' in new_gltf:
            for skin in new_gltf['skins']:
                if 'joints' in skin:
                    for i, joint_index in enumerate(skin['joints']):
                        if joint_index < len(new_gltf.get('nodes', [])):
                            node = new_gltf['nodes'][joint_index]
                            if 'name' in node and node['name'] in bone_mapping:
                                old_name = node['name']
                                new_name = bone_mapping[old_name]
                                node['name'] = new_name
                                logger.debug(f"Renamed skin joint: {old_name} -> {new_name}")
        
        # Rename bones in animations
        if 'animations' in new_gltf:
            for anim in new_gltf['animations']:
                if 'channels' in anim:
                    for channel in anim['channels']:
                        if 'target' in channel and 'node' in channel['target']:
                            node_index = channel['target']['node']
                            if node_index < len(new_gltf.get('nodes', [])):
                                node = new_gltf['nodes'][node_index]
                                if 'name' in node and node['name'] in bone_mapping:
                                    old_name = node['name']
                                    new_name = bone_mapping[old_name]
                                    node['name'] = new_name
                                    logger.debug(f"Renamed animation target: {old_name} -> {new_name}")
        
        return new_gltf
    
    def process_glb(self, input_path: str, output_path: str, bone_mapping: Optional[Dict[str, str]] = None) -> bool:
        """Process GLB file to rename bones."""
        try:
            # Read GLB file
            logger.info(f"Reading GLB file: {input_path}")
            gltf_json, binary_data = self.read_glb(input_path)
            
            # Extract current bone names
            current_bones = []
            if 'nodes' in gltf_json:
                for node in gltf_json['nodes']:
                    if 'name' in node:
                        current_bones.append(node['name'])
            
            logger.info(f"Found {len(current_bones)} nodes in GLB file")
            logger.info(f"Current bone order: {current_bones}")
            logger.info(f"Expected bone order: {self.bone_names}")
            
            # Create bone mapping if not provided
            if bone_mapping is None:
                bone_mapping = self._create_bone_mapping(current_bones)
            
            # Log the mapping for debugging
            logger.info("Bone mapping:")
            for old_name, new_name in bone_mapping.items():
                logger.info(f"  {old_name} -> {new_name}")
            
            # Rename bones
            logger.info("Renaming bones...")
            new_gltf = self.rename_bones(gltf_json, bone_mapping)
            
            # Write updated GLB file
            logger.info(f"Writing updated GLB file: {output_path}")
            self.write_glb(output_path, new_gltf, binary_data)
            
            logger.info("Bone renaming completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error processing GLB file: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _create_bone_mapping(self, current_bones: List[str]) -> Dict[str, str]:
        """Create mapping from current bone names to standard names."""
        bone_mapping = {}
        
        # Preserve the original order of bones from the GLB file
        # Don't sort them alphabetically as this destroys the hierarchy
        for i, current_name in enumerate(current_bones):
            if i < len(self.bone_names):
                standard_name = self.bone_names[i]
                bone_mapping[current_name] = standard_name
                logger.info(f"Mapping {current_name} -> {standard_name}")
            else:
                bone_mapping[current_name] = current_name
                logger.warning(f"No standard name for {current_name}, keeping original")
        
        return bone_mapping


def rename_bones_in_glb(input_file: str, output_file: str, config_file: str = "configs/skeleton/mixamo.yaml") -> bool:
    """
    Convenience function to rename bones in a GLB file.
    
    Args:
        input_file: Path to input GLB file
        output_file: Path to output GLB file
        config_file: Path to skeleton configuration file
        
    Returns:
        True if successful, False otherwise
    """
    processor = GLBProcessor(config_file)
    return processor.process_glb(input_file, output_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rename bones in GLB file to standard names")
    parser.add_argument("input", help="Input GLB file path")
    parser.add_argument("output", help="Output GLB file path")
    parser.add_argument("--config", default="configs/skeleton/mixamo.yaml", 
                       help="Path to skeleton configuration file")
    parser.add_argument("--mapping", help="JSON file with custom bone name mapping")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize processor
    processor = GLBProcessor(args.config)
    
    # Load custom mapping if provided
    bone_mapping = None
    if args.mapping:
        try:
            with open(args.mapping, 'r') as f:
                bone_mapping = json.load(f)
            logger.info(f"Loaded custom bone mapping from: {args.mapping}")
        except Exception as e:
            logger.error(f"Failed to load custom mapping: {e}")
            exit(1)
    
    # Process the GLB file
    success = processor.process_glb(args.input, args.output, bone_mapping)
    
    if success:
        exit(0)
    else:
        exit(1) 