# UniRig Replicate Cog

This repository contains a Replicate cog for UniRig, an automated 3D model rigging system that generates skeletons and skinning weights for 3D models using deep learning.

## About UniRig

UniRig is a research project from Tsinghua University & Tripo that automatically generates rigging (skeletons and skinning weights) for 3D models. It uses deep learning to understand 3D geometry and create appropriate bone structures and weight assignments.

**Original Paper**: [UniRig: Towards Unified 3D Object Rigging](https://arxiv.org/abs/2504.12451)

**Project Page**: [https://zjp-shadow.github.io/works/UniRig/](https://zjp-shadow.github.io/works/UniRig/)

**Original Repository**: [https://huggingface.co/VAST-AI/UniRig](https://huggingface.co/VAST-AI/UniRig)

## Usage

### Input Parameters

- **input_file**: Input 3D model file (required)
  - Supported formats: `.obj`, `.fbx`, `.glb`, `.dae`, `.gltf`, `.vrm`
- **skeleton_type**: Type of skeleton to generate (default: "articulationxl")
  - `articulationxl`: Generic bone names (bone_0, bone_1, etc.)
  - `vroid`: Descriptive names (J_Bip_C_Hips, J_Bip_C_Spine, etc.)
- **seed**: Random seed for reproducible results (default: 12345, range: 0-100000)
- **output_format**: Output format for the final rigged model (default: "full_rigging")
  - `skeleton_only`: Returns model with skeleton only
  - `full_rigging`: Returns model with skeleton and skinning weights

### Output

The model returns a 3D file in the same format as the input, containing the rigged model with skeleton and/or skinning weights.

## What UniRig Does

1. **Skeleton Generation**: Analyzes the 3D model geometry and generates an appropriate bone structure
2. **Skinning Weight Assignment**: Calculates how each vertex should be influenced by each bone
3. **Rigging Integration**: Combines the skeleton and skinning weights with the original model

## Model Details

- **Model Type**: Deep Learning-based 3D Rigging System
- **Input**: 3D Model Files (various formats)
- **Output**: Rigged 3D Model with skeleton and skinning weights
- **GPU Memory**: ~8-16GB VRAM required
- **Processing Time**: ~2-5 minutes depending on model complexity

## Key Features

- **Automatic Rigging**: No manual bone placement or weight painting required
- **Multiple Skeleton Types**: Support for different naming conventions
- **Format Preservation**: Output maintains the same format as input
- **Reproducible Results**: Seed-based generation for consistent outputs
- **Flexible Output**: Choose between skeleton-only or full rigging

## Technical Implementation

The cog implementation:

1. Validates input 3D model format and structure
2. Extracts mesh data and preprocesses geometry
3. Runs skeleton generation using trained neural networks
4. Generates skinning weights based on skeleton and geometry
5. Merges results back into the original model format
6. Returns the fully rigged 3D model

## Supported File Formats

### Input Formats
- `.obj` - Wavefront OBJ
- `.fbx` - Autodesk FBX
- `.glb` - Binary glTF
- `.dae` - COLLADA
- `.gltf` - glTF
- `.vrm` - VRM (VRoid format)

### Output Formats
The output will be in the same format as the input file.

## Use Cases

- **Game Development**: Automatically rig character models for animation
- **3D Animation**: Prepare models for character animation workflows
- **VR/AR Applications**: Rig avatars and characters for immersive experiences
- **3D Printing**: Add articulation data for poseable printed models
- **Research**: Study automated rigging techniques and bone structure generation

## Limitations

- **GPU Requirements**: Requires significant GPU memory (8GB+ recommended)
- **Model Complexity**: Very high-poly models may require preprocessing
- **Geometry Requirements**: Works best with manifold, clean geometry
- **Processing Time**: Complex models may take several minutes to process
- **Format Constraints**: Some advanced material properties may not be preserved

## Performance Tips

**For faster processing:**
- Use models with reasonable polygon counts (under 100K faces)
- Ensure clean, manifold geometry
- Use simpler skeleton types when appropriate

**For best quality:**
- Provide high-quality input geometry
- Use appropriate skeleton type for your use case
- Ensure models are properly scaled and oriented

## Example Usage

```python
import replicate

# Basic rigging with default settings
output = replicate.run(
    "your-username/unirig:latest",
    input={
        "input_file": open("character.fbx", "rb"),
        "skeleton_type": "articulationxl",
        "output_format": "full_rigging"
    }
)

# Skeleton-only generation with VRoid naming
output = replicate.run(
    "your-username/unirig:latest",
    input={
        "input_file": open("avatar.vrm", "rb"),
        "skeleton_type": "vroid",
        "output_format": "skeleton_only",
        "seed": 42
    }
)
```

## Local Development

### Prerequisites

- [Cog](https://github.com/replicate/cog) installed
- NVIDIA GPU with CUDA support
- At least 16GB GPU memory recommended
- Docker installed and running

### Building the Cog

```bash
cog build
```

### Running Predictions

```bash
cog predict -i input_file=@path/to/model.fbx -i skeleton_type=articulationxl
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce model complexity or polygon count
   - Ensure no other GPU processes are running
   - Try with simpler models first

2. **Unsupported File Format**
   - Check that your file format is in the supported list
   - Ensure the file is not corrupted
   - Try converting to a different supported format

3. **Poor Rigging Quality**
   - Ensure input geometry is clean and manifold
   - Check model scale and orientation
   - Try different skeleton types

4. **Long Processing Times**
   - Complex models naturally take longer
   - GPU performance affects processing speed
   - Consider simplifying geometry for faster results

## Citation

If you use this model, please cite the original paper:

```bibtex
@article{liu2024unirig,
  title={UniRig: Towards Unified 3D Object Rigging},
  author={Liu, Zhenyu and others},
  journal={arXiv preprint arXiv:2504.12451},
  year={2024}
}
```

## License

This project follows the license terms from the original UniRig repository. Please refer to the original project for detailed licensing information.

## Support

For issues related to:
- **Cog implementation**: Open an issue in this repository
- **UniRig algorithm**: Refer to the original project page and paper
- **Replicate platform**: Check Replicate's documentation

## Related Projects

- **Original UniRig**: [https://huggingface.co/VAST-AI/UniRig](https://huggingface.co/VAST-AI/UniRig)
- **Project Page**: [https://zjp-shadow.github.io/works/UniRig/](https://zjp-shadow.github.io/works/UniRig/)
- **Research Paper**: [https://arxiv.org/abs/2504.12451](https://arxiv.org/abs/2504.12451)
