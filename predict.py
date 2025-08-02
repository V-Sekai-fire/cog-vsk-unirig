import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Tuple

import lightning as L
import torch
import yaml
from box import Box
from cog import BasePredictor, Input, Path as CogPath

def setup_python_environment():
    """Setup hybrid Python environment for both system and Blender operations"""
    import sys
    
    # Detect if we're already in Blender
    try:
        import bpy
        print("Running in Blender Python environment")
        return "blender"
    except ImportError:
        print("Running in system Python environment")
        
        # Set up Blender paths for when we need them
        blender_path = "/opt/blender-4.5.1-linux-x64"
        if os.path.exists(blender_path):
            os.environ["BLENDER_PATH"] = blender_path
            os.environ["PATH"] = f"{blender_path}:{os.environ.get('PATH', '')}"
            
            # Ensure system packages are available to Blender's Python
            blender_python = f"{blender_path}/4.5/python/bin/python3.11"
            if os.path.exists(blender_python):
                # Install critical packages in Blender's Python if needed
                try:
                    subprocess.run([blender_python, "-m", "pip", "install", "numpy==1.26.4", "torch>=2.5.1"], 
                                 check=False, capture_output=True, timeout=60)
                    print("Installed critical packages in Blender Python")
                except Exception as e:
                    print(f"Warning: Could not install packages in Blender Python: {e}")
            
            print(f"Blender environment set up at: {blender_path}")
        else:
            print("Warning: Blender not found at expected location")
        
        return "system"

def execute_blender_operation(script_content, *args):
    """Execute Python code that requires bpy through Blender subprocess"""
    blender_path = os.environ.get("BLENDER_PATH", "/opt/blender-4.5.1-linux-x64")
    blender_executable = f"{blender_path}/blender"
    
    if not os.path.exists(blender_executable):
        raise RuntimeError(f"Blender executable not found at: {blender_executable}")
    
    # Create temporary script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run Blender in background with the script
        cmd = [blender_executable, "--background", "--python", script_path] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        return result.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError("Blender operation timed out")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Blender operation failed: {e.stderr}")
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass

def install_runtime_dependencies():
    """Install packages that need to be installed at runtime"""
    try:
        env_type = setup_python_environment()
        print(f"Python environment type: {env_type}")
        print("Runtime dependencies installed successfully")
    except Exception as e:
        print(f"Warning: Failed to install some runtime dependencies: {e}")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Setting up UniRig predictor...")
        
        # Install runtime dependencies and detect environment
        install_runtime_dependencies()
        self.python_env = setup_python_environment()
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Create temp directory
        self.temp_dir = Path("/tmp/unirig")
        self.temp_dir.mkdir(exist_ok=True)
        
        print("UniRig setup complete!")

    def execute_blender_if_needed(self, operation_name: str, script_content: str = None, *args):
        """Execute Blender operations if needed, with fallback handling"""
        if self.python_env == "blender":
            # Already in Blender, can import bpy directly
            try:
                import bpy
                print(f"Executing {operation_name} in Blender environment")
                return True
            except ImportError:
                print(f"Warning: bpy not available even in Blender environment for {operation_name}")
                return False
        else:
            # In system Python, use subprocess if script provided
            if script_content:
                try:
                    print(f"Executing {operation_name} via Blender subprocess")
                    result = execute_blender_operation(script_content, *args)
                    print(f"Blender operation {operation_name} completed successfully")
                    return result
                except Exception as e:
                    print(f"Warning: Blender operation {operation_name} failed: {e}")
                    return False
            else:
                print(f"Skipping {operation_name} - no Blender script provided")
                return False

    def validate_input_file(self, file_path: str) -> bool:
        """Validate if the input file format is supported."""
        supported_formats = ['.obj', '.fbx', '.glb', '.dae', '.gltf', '.vrm']
        if not file_path or not Path(file_path).exists():
            return False
        
        file_ext = Path(file_path).suffix.lower()
        return file_ext in supported_formats

    def extract_mesh_python(self, input_file: str, output_dir: str) -> str:
        """Extract mesh data from 3D model using Python"""
        from src.data.extract import extract_builtin, get_files
        
        # Create extraction parameters
        files = get_files(
            data_name="raw_data.npz",
            inputs=str(input_file),
            input_dataset_dir=None,
            output_dataset_dir=output_dir,
            force_override=True,
            warning=False,
        )
        
        if not files:
            raise RuntimeError("No files to extract")
        
        # Run the actual extraction
        timestamp = str(int(time.time()))
        extract_builtin(
            output_folder=output_dir,
            target_count=50000,
            num_runs=1,
            id=0,
            time=timestamp,
            files=files,
        )
        
        # Return the directory path where raw_data.npz was created
        expected_npz_dir = files[0][1]
        expected_npz_file = Path(expected_npz_dir) / "raw_data.npz"
        
        if not expected_npz_file.exists():
            raise RuntimeError(f"Extraction failed: {expected_npz_file} not found")
        
        return expected_npz_dir

    def run_inference_python(
        self,
        input_model_path: str, 
        inference_type: str, 
        seed: int = 12345, 
        skeleton_type: str = "articulationxl"
    ) -> str:
        """
        Unified inference function for both skeleton and skin inference.
        
        Args:
            input_model_path: Path to input 3D model file (GLB, FBX, GLTF, VRM, etc.)
            inference_type: Type of inference ("skeleton" or "skin")
            seed: Random seed for reproducible results
            skeleton_type: Type of skeleton to generate for skeleton inference
            
        Returns:
            Path to temporary output file containing the result
        """
        from src.data.datapath import Datapath
        from src.data.dataset import DatasetConfig, UniRigDatasetModule
        from src.data.transform import TransformConfig
        from src.inference.download import download
        from src.model.parse import get_model
        from src.system.parse import get_system, get_writer
        from src.tokenizer.parse import get_tokenizer
        from src.tokenizer.spec import TokenizerConfig

        # Set random seed for skeleton inference
        if inference_type == "skeleton":
            L.seed_everything(seed, workers=True)
        
        # Load task and model configurations based on inference type
        if inference_type == "skeleton":
            task_config_path = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
            transform_config_path = "configs/transform/inference_ar_transform.yaml"
            model_config_path = "configs/model/unirig_ar_350m_1024_81920_float32.yaml"
            system_config_path = "configs/system/ar_inference_articulationxl.yaml"
            tokenizer_config_path = "configs/tokenizer/tokenizer_parts_articulationxl_256.yaml"
            data_name = "raw_data.npz"
        else:  # skin
            task_config_path = "configs/task/quick_inference_unirig_skin.yaml"
            transform_config_path = "configs/transform/inference_skin_transform.yaml"
            model_config_path = "configs/model/unirig_skin.yaml"
            system_config_path = "configs/system/skin.yaml"
            tokenizer_config_path = None
            data_name = "predict_skeleton.npz"
        
        # Load task configuration
        if not Path(task_config_path).exists():
            raise FileNotFoundError(f"Task configuration file not found: {task_config_path}")
        
        with open(task_config_path, 'r') as f:
            task = Box(yaml.safe_load(f))
        
        # Create temporary output file
        temp_output = tempfile.mktemp(suffix=".fbx")
        
        # Setup data directory and datapath
        if inference_type == "skeleton":
            npz_dir = Path(temp_output).parent / "npz"
            npz_dir.mkdir(exist_ok=True)
            npz_data_dir = self.extract_mesh_python(input_model_path, npz_dir)
            datapath = Datapath(files=[npz_data_dir], cls=None)
        else:  # skin
            skeleton_work_dir = Path(input_model_path).parent
            all_npz_files = list(skeleton_work_dir.rglob("**/*.npz"))
            if not all_npz_files:
                raise RuntimeError(f"No NPZ files found for skin inference in {skeleton_work_dir}")
            skeleton_npz_dir = all_npz_files[0].parent
            datapath = Datapath(files=[str(skeleton_npz_dir)], cls=None)
        
        # Load common configurations
        data_config = Box(yaml.safe_load(open("configs/data/quick_inference.yaml", 'r')))
        transform_config = Box(yaml.safe_load(open(transform_config_path, 'r')))
        
        # Setup tokenizer and model
        if inference_type == "skeleton":
            tokenizer_config = TokenizerConfig.parse(config=Box(yaml.safe_load(open(tokenizer_config_path, 'r'))))
            tokenizer = get_tokenizer(config=tokenizer_config)
            model_config = Box(yaml.safe_load(open(model_config_path, 'r')))
            model = get_model(tokenizer=tokenizer, **model_config)
        else:  # skin
            tokenizer_config = None
            tokenizer = None
            model_config = Box(yaml.safe_load(open(model_config_path, 'r')))
            model = get_model(tokenizer=None, **model_config)
        
        # Setup datasets and transforms
        predict_dataset_config = DatasetConfig.parse(config=data_config.predict_dataset_config).split_by_cls()
        predict_transform_config = TransformConfig.parse(config=transform_config.predict_transform_config)
        
        # Create data module
        data = UniRigDatasetModule(
            process_fn=model._process_fn,
            predict_dataset_config=predict_dataset_config,
            predict_transform_config=predict_transform_config,
            tokenizer_config=tokenizer_config,
            debug=False,
            data_name=data_name,
            datapath=datapath,
            cls=None,
        )
        
        # Setup callbacks and writer
        callbacks = []
        writer_config = task.writer.copy()
        
        if inference_type == "skeleton":
            writer_config['npz_dir'] = str(npz_dir)
            writer_config['output_dir'] = str(Path(temp_output).parent)
            writer_config['output_name'] = Path(temp_output).name
            writer_config['user_mode'] = False
        else:  # skin
            writer_config['npz_dir'] = str(skeleton_npz_dir)
            writer_config['output_name'] = str(temp_output)
            writer_config['user_mode'] = True
            writer_config['export_fbx'] = True
        
        callbacks.append(get_writer(**writer_config, order_config=predict_transform_config.order_config))
        
        # Get system
        system_config = Box(yaml.safe_load(open(system_config_path, 'r')))
        
        # Dynamically set skeleton type for skeleton inference
        if inference_type == "skeleton":
            system_config.generate_kwargs.assign_cls = skeleton_type
        
        system = get_system(**system_config, model=model, steps_per_epoch=1)
        
        # Setup trainer
        trainer_config = task.trainer
        resume_from_checkpoint = download(task.resume_from_checkpoint)
        
        trainer = L.Trainer(callbacks=callbacks, logger=None, **trainer_config)
        
        # Run prediction
        trainer.predict(system, datamodule=data, ckpt_path=resume_from_checkpoint, return_predictions=False)
        
        # Handle output file location and validation
        if inference_type == "skeleton":
            input_name_stem = Path(input_model_path).stem
            actual_output_dir = Path(temp_output).parent / input_name_stem
            actual_output_file = actual_output_dir / "skeleton.fbx"
            
            if not actual_output_file.exists():
                alt_files = list(Path(temp_output).parent.rglob("skeleton.fbx"))
                if alt_files:
                    actual_output_file = alt_files[0]
                else:
                    raise RuntimeError(f"Skeleton FBX file not found. Expected at: {actual_output_file}")
            
            if actual_output_file != Path(temp_output):
                shutil.copy2(actual_output_file, temp_output)
        
        else:  # skin
            if not Path(temp_output).exists():
                skin_files = list(Path(temp_output).parent.rglob("*skin*.fbx"))
                if skin_files:
                    actual_output_file = skin_files[0]
                    shutil.copy2(actual_output_file, temp_output)
                else:
                    raise RuntimeError(f"Skin FBX file not found. Expected at: {temp_output}")
        
        return str(temp_output)

    def merge_results_python(self, source_file: str, target_file: str, output_file: str) -> str:
        """Merge results using Python"""
        from src.inference.merge import transfer
        
        # Validate input paths
        if not Path(source_file).exists():
            raise ValueError(f"Source file does not exist: {source_file}")
        if not Path(target_file).exists():
            raise ValueError(f"Target file does not exist: {target_file}")
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use the transfer function directly
        transfer(source=str(source_file), target=str(target_file), output=str(output_path), add_root=False)
        
        # Validate that the output file was created
        if not output_path.exists():
            raise RuntimeError(f"Merge failed: Output file not created at {output_path}")
        
        return str(output_path.resolve())

    def predict(
        self,
        input_file: CogPath = Input(description="Input 3D model file (.obj, .fbx, .glb, .dae, .gltf, .vrm)"),
        skeleton_type: str = Input(
            description="Skeleton type to generate",
            choices=["articulationxl", "vroid"],
            default="articulationxl"
        ),
        seed: int = Input(
            description="Random seed for reproducible results",
            default=12345,
            ge=0,
            le=100000
        ),
        output_format: str = Input(
            description="Output format for the final rigged model",
            choices=["skeleton_only", "full_rigging"],
            default="full_rigging"
        )
    ) -> CogPath:
        """
        Generate skeleton and skinning weights for a 3D model.
        
        Args:
            input_file: 3D model file to rig
            skeleton_type: Type of skeleton to generate (articulationxl or vroid)
            seed: Random seed for reproducible results
            output_format: Whether to output skeleton only or full rigging
            
        Returns:
            Path to the rigged 3D model
        """
        
        # Validate input file
        if not self.validate_input_file(str(input_file)):
            raise ValueError("Invalid or unsupported file format. Supported formats: .obj, .fbx, .glb, .dae, .gltf, .vrm")
        
        # Create working directory
        file_stem = Path(input_file).stem
        work_dir = self.temp_dir / f"{file_stem}_{seed}_{int(time.time())}"
        work_dir.mkdir(exist_ok=True)
        
        # Copy input file to working directory
        input_file_path = work_dir / Path(input_file).name
        shutil.copy2(input_file, input_file_path)
        
        print(f"Processing {input_file_path} with skeleton type: {skeleton_type}")
        
        try:
            # Step 1: Generate skeleton
            intermediate_skeleton_file = work_dir / f"{file_stem}_skeleton.fbx"
            final_skeleton_file = work_dir / f"{file_stem}_skeleton_only{Path(input_file).suffix}"
            
            print("Generating skeleton...")
            skeleton_result = self.run_inference_python(
                str(input_file_path), 
                "skeleton", 
                seed, 
                skeleton_type=skeleton_type
            )
            # Copy the result to our expected location
            shutil.copy2(skeleton_result, intermediate_skeleton_file)
            
            print("Merging skeleton with original model...")
            self.merge_results_python(
                str(intermediate_skeleton_file), 
                str(input_file_path), 
                str(final_skeleton_file)
            )
            
            # If only skeleton is requested, return it
            if output_format == "skeleton_only":
                final_output = Path(tempfile.mktemp(suffix=Path(input_file).suffix))
                shutil.copy2(final_skeleton_file, final_output)
                return CogPath(final_output)
            
            # Step 2: Generate skinning weights
            print("Generating skinning weights...")
            intermediate_skin_file = work_dir / f"{file_stem}_skin.fbx"
            final_skin_file = work_dir / f"{file_stem}_full_rigging{Path(input_file).suffix}"
            
            skin_result = self.run_inference_python(
                str(intermediate_skeleton_file), 
                "skin"
            )
            # Copy the result to our expected location
            shutil.copy2(skin_result, intermediate_skin_file)
            
            print("Merging skinning with original model...")
            self.merge_results_python(
                str(intermediate_skin_file), 
                str(input_file_path), 
                str(final_skin_file)
            )
            
            # Copy final result to output location
            final_output = Path(tempfile.mktemp(suffix=Path(input_file).suffix))
            shutil.copy2(final_skin_file, final_output)
            
            print(f"Rigging complete! Output saved to: {final_output}")
            return CogPath(final_output)
            
        except Exception as e:
            print(f"Error during processing: {e}")
            raise e
        finally:
            # Clean up working directory
            try:
                shutil.rmtree(work_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up working directory: {e}")
