import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Tuple

import gradio as gr
import lightning as L
import spaces
import torch
import yaml

subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)


# Get the PyTorch and CUDA versions
torch_version = torch.__version__.split("+")[0]  # Strips any "+cuXXX" suffix
cuda_version = torch.version.cuda
spconv_version = "-cu121" if cuda_version else "" 

# Format CUDA version to match the URL convention (e.g., "cu118" for CUDA 11.8)
if cuda_version:
    cuda_version = f"cu{cuda_version.replace('.', '')}"
else:
    cuda_version = "cpu"  # Fallback in case CUDA is not available


subprocess.run(f'pip install spconv{spconv_version}', shell=True)
subprocess.run(f'pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html --no-cache-dir', shell=True)


class UniRigDemo:
    """Main class for the UniRig Gradio demo application."""
    
    def __init__(self):
        # Create temp directory in current directory instead of system temp
        base_dir = Path(__file__).parent
        self.temp_dir = base_dir / "tmp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Supported file formats
        self.supported_formats = ['.obj', '.fbx', '.glb']
        
    def validate_input_file(self, file_path: str) -> bool:
        """Validate if the input file format is supported."""
        if not file_path or not Path(file_path).exists():
            return False
        
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_formats
    
    def generate_skeleton(self, input_file: str, seed: int = 12345) -> Tuple[str, str, str]:
        """
        OPERATION 1: Generate skeleton for the input 3D model using Python
        
        Args:
            input_file: Path to the input 3D model file
            seed: Random seed for reproducible results
            
        Returns:
            Tuple of (status_message, output_file_path, preview_info)
        """
        # Validate input
        if not self.validate_input_file(input_file):
            return "Error: Invalid or unsupported file format. Supported: " + ", ".join(self.supported_formats), "", ""
        
        # Create working directory
        file_stem = Path(input_file).stem
        input_model_dir = self.temp_dir / f"{file_stem}_{seed}"
        input_model_dir.mkdir(exist_ok=True)

        # Copy input file to working directory
        input_file = Path(input_file)
        shutil.copy2(input_file, input_model_dir / input_file.name)
        input_file = input_model_dir / input_file.name
        print(f"New input file path: {input_file}")
        
        # Generate skeleton using Python (replaces bash script)
        output_file = input_model_dir / f"{file_stem}_skeleton.fbx"
        
        self.run_skeleton_inference_python(input_file, output_file, seed)

        if not output_file.exists():
            return "Error: Skeleton file was not generated", "", ""
        
        print(f"Generated skeleton at: {output_file}")
        return str(output_file)

    def merge_results(self, original_file: str, rigged_file: str, output_file) -> str:
        """
        OPERATION 3: Merge the rigged skeleton/skin with the original model using Python functions.
        
        Args:
            original_file: Path to the original 3D model
            rigged_file: Path to the rigged file (skeleton or skin)
            
        Returns:
            Tuple of (status_message, output_file_path, preview_info)
        """
        if not original_file or not Path(original_file).exists():
            return "Error: Original file not provided or doesn't exist", "", ""
        
        if not rigged_file or not Path(rigged_file).exists():
            return "Error: Rigged file not provided or doesn't exist", "", ""
        
        # Create output file
        work_dir = Path(rigged_file).parent
        output_file = work_dir / f"{Path(original_file).stem}_rigged.glb"
        
        # Run merge using Python function
        try:
            self.merge_results_python(rigged_file, original_file, str(output_file))
        except Exception as e:
            error_msg = f"Error: Merge failed: {str(e)}"
            traceback.print_exc()
            return error_msg, "", ""
        
        # Validate that the output file exists and is a file (not a directory)
        output_file_abs = output_file.resolve()
        if not output_file_abs.exists():
            return "Error: Merged file was not generated", "", ""
        
        if not output_file_abs.is_file():
            return f"Error: Output path is not a valid file: {output_file_abs}", "", ""
        
        # Generate preview information
        preview_info = self.generate_model_preview(str(output_file_abs))
        
        return "✅ Model rigging completed successfully!", str(output_file_abs), preview_info

    @spaces.GPU()
    def complete_pipeline(self, input_file: str, seed: int = 12345) -> Tuple[str, str, str, str, str]:
        """
        Run the complete rigging pipeline: skeleton generation → skinning → merge.
        
        Args:
            input_file: Path to the input 3D model file
            seed: Random seed for reproducible results
            
        Returns:
            Tuple of status messages and file paths for each step
        """
        # Validate input file
        if not self.validate_input_file(input_file):
            raise gr.Error(f"Error: Invalid or unsupported file format. Supported formats: {', '.join(self.supported_formats)}")
        
        # Create working directory
        file_stem = Path(input_file).stem
        input_model_dir = self.temp_dir / f"{file_stem}_{seed}"
        input_model_dir.mkdir(exist_ok=True)

        # Copy input file to working directory
        input_file = Path(input_file)
        shutil.copy2(input_file, input_model_dir / input_file.name)
        input_file = input_model_dir / input_file.name
        print(f"New input file path: {input_file}")
        
        # Step 1: Generate skeleton        
        output_skeleton_file = input_model_dir / f"{file_stem}_skeleton.fbx"
        self.run_skeleton_inference_python(input_file, output_skeleton_file, seed)        

        # Step 2: Generate skinning
        output_skin_file = input_model_dir / f"{file_stem}_skin.fbx"
        self.run_skin_inference_python(output_skeleton_file, output_skin_file)
        
        # Step 3: Merge results
        final_file = input_model_dir / f"{file_stem}_rigged.glb"
        self.merge_results_python(output_skin_file, input_file, final_file)

        return str(final_file)
        
    def extract_mesh_python(self, input_file: str, output_dir: str) -> str:
        """
        Extract mesh data from 3D model using Python (replaces extract.sh)
        Returns path to generated .npz file
        """
        # Import required modules
        from src.data.extract import get_files, extract_builtin
        
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
        # The dataset expects to find raw_data.npz in this directory
        expected_npz_dir = files[0][1]  # This is the output directory
        expected_npz_file = Path(expected_npz_dir) / "raw_data.npz"
        
        if not expected_npz_file.exists():
            raise RuntimeError(f"Extraction failed: {expected_npz_file} not found")
        
        return expected_npz_dir  # Return the directory containing raw_data.npz
    
    def run_skeleton_inference_python(self, input_file: str, output_file: str, seed: int = 12345) -> str:
        """
        Run skeleton inference using Python (replaces skeleton part of generate_skeleton.sh)
        Returns path to skeleton FBX file
        """
        from box import Box

        from src.data.datapath import Datapath
        from src.data.dataset import DatasetConfig, UniRigDatasetModule
        from src.data.transform import TransformConfig
        from src.inference.download import download
        from src.model.parse import get_model
        from src.system.parse import get_system, get_writer
        from src.tokenizer.parse import get_tokenizer
        from src.tokenizer.spec import TokenizerConfig
        
        # Set random seed
        L.seed_everything(seed, workers=True)
        
        # Load task configuration
        task_config_path = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
        if not Path(task_config_path).exists():
            raise FileNotFoundError(f"Task configuration file not found: {task_config_path}")
        
        # Load the task configuration
        with open(task_config_path, 'r') as f:
            task = Box(yaml.safe_load(f))
        
        # Create temporary npz directory
        npz_dir = Path(output_file).parent / "npz"
        npz_dir.mkdir(exist_ok=True)
        
        # Extract mesh data
        npz_data_dir = self.extract_mesh_python(input_file, npz_dir)
        
        # Setup datapath with the directory containing raw_data.npz
        datapath = Datapath(files=[npz_data_dir], cls=None)
        
        # Load configurations
        data_config = Box(yaml.safe_load(open("configs/data/quick_inference.yaml", 'r')))
        transform_config = Box(yaml.safe_load(open("configs/transform/inference_ar_transform.yaml", 'r')))
        
        # Get tokenizer
        tokenizer_config = TokenizerConfig.parse(config=Box(yaml.safe_load(open("configs/tokenizer/tokenizer_parts_articulationxl_256.yaml", 'r'))))
        tokenizer = get_tokenizer(config=tokenizer_config)
        
        # Get model
        model_config = Box(yaml.safe_load(open("configs/model/unirig_ar_350m_1024_81920_float32.yaml", 'r')))
        model = get_model(tokenizer=tokenizer, **model_config)
        
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
            data_name="raw_data.npz",
            datapath=datapath,
            cls=None,
        )
        
        # Setup callbacks and writer
        callbacks = []
        writer_config = task.writer.copy()
        writer_config['npz_dir'] = str(npz_dir)
        writer_config['output_dir'] = str(Path(output_file).parent)
        writer_config['output_name'] = Path(output_file).name
        writer_config['user_mode'] = False  # Set to False to enable NPZ export
        print(f"Writer config: {writer_config}")
        # But we want the FBX to go to our specified location when in user mode for FBX
        callbacks.append(get_writer(**writer_config, order_config=predict_transform_config.order_config))
        
        # Get system
        system_config = Box(yaml.safe_load(open("configs/system/ar_inference_articulationxl.yaml", 'r')))
        system = get_system(**system_config, model=model, steps_per_epoch=1)
        
        # Setup trainer
        trainer_config = task.trainer
        resume_from_checkpoint = download(task.resume_from_checkpoint)
        
        trainer = L.Trainer(callbacks=callbacks, logger=None, **trainer_config)
        
        # Run prediction
        trainer.predict(system, datamodule=data, ckpt_path=resume_from_checkpoint, return_predictions=False)
        
        # The actual output file will be in a subdirectory named after the input file
        # Look for the generated skeleton.fbx file
        input_name_stem = Path(input_file).stem
        actual_output_dir = Path(output_file).parent / input_name_stem
        actual_output_file = actual_output_dir / "skeleton.fbx"
        
        if not actual_output_file.exists():
            # Try alternative locations - look for any skeleton.fbx file in the output directory
            alt_files = list(Path(output_file).parent.rglob("skeleton.fbx"))
            if alt_files:
                actual_output_file = alt_files[0]
                print(f"Found skeleton at alternative location: {actual_output_file}")
            else:
                # List all files for debugging
                all_files = list(Path(output_file).parent.rglob("*"))
                print(f"Available files: {[str(f) for f in all_files]}")
                raise RuntimeError(f"Skeleton FBX file not found. Expected at: {actual_output_file}")
        
        # Copy to the expected output location
        if actual_output_file != Path(output_file):
            shutil.copy2(actual_output_file, output_file)
            print(f"Copied skeleton from {actual_output_file} to {output_file}")
        
        print(f"Generated skeleton at: {output_file}")
        return str(output_file)
    
    def run_skin_inference_python(self, skeleton_file: str, output_file: str) -> str:
        """
        Run skin inference using Python (replaces skin part of generate_skin.sh)
        Returns path to skin FBX file
        """
        from box import Box

        from src.data.datapath import Datapath
        from src.data.dataset import DatasetConfig, UniRigDatasetModule
        from src.data.transform import TransformConfig
        from src.inference.download import download
        from src.model.parse import get_model
        from src.system.parse import get_system, get_writer
        
        # Load task configuration
        task_config_path = "configs/task/quick_inference_unirig_skin.yaml"
        with open(task_config_path, 'r') as f:
            task = Box(yaml.safe_load(f))
                
        # Look for files matching predict_skeleton.npz pattern recursively
        skeleton_work_dir = Path(skeleton_file).parent
        all_npz_files = list(skeleton_work_dir.rglob("**/*.npz"))
        
        # Setup datapath - need to pass the directory containing the NPZ file
        skeleton_npz_dir = all_npz_files[0].parent
        datapath = Datapath(files=[str(skeleton_npz_dir)], cls=None)
        
        # Load configurations
        data_config = Box(yaml.safe_load(open("configs/data/quick_inference.yaml", 'r')))
        transform_config = Box(yaml.safe_load(open("configs/transform/inference_skin_transform.yaml", 'r')))
        
        # Get model
        model_config = Box(yaml.safe_load(open("configs/model/unirig_skin.yaml", 'r')))
        model = get_model(tokenizer=None, **model_config)
        
        # Setup datasets and transforms
        predict_dataset_config = DatasetConfig.parse(config=data_config.predict_dataset_config).split_by_cls()
        predict_transform_config = TransformConfig.parse(config=transform_config.predict_transform_config)
        
        # Create data module
        data = UniRigDatasetModule(
            process_fn=model._process_fn,
            predict_dataset_config=predict_dataset_config,
            predict_transform_config=predict_transform_config,
            tokenizer_config=None,
            debug=False,
            data_name="predict_skeleton.npz",
            datapath=datapath,
            cls=None,
        )
        
        # Setup callbacks and writer
        callbacks = []
        writer_config = task.writer.copy()
        writer_config['npz_dir'] = str(skeleton_npz_dir)
        writer_config['output_name'] = str(output_file)
        writer_config['user_mode'] = True
        writer_config['export_fbx'] = True  # Enable FBX export
        callbacks.append(get_writer(**writer_config, order_config=predict_transform_config.order_config))
        
        # Get system
        system_config = Box(yaml.safe_load(open("configs/system/skin.yaml", 'r')))
        system = get_system(**system_config, model=model, steps_per_epoch=1)
        
        # Setup trainer
        trainer_config = task.trainer
        resume_from_checkpoint = download(task.resume_from_checkpoint)
        
        trainer = L.Trainer(callbacks=callbacks, logger=None, **trainer_config)
        
        # Run prediction
        trainer.predict(system, datamodule=data, ckpt_path=resume_from_checkpoint, return_predictions=False)
        
        # The skin FBX file should be generated with the specified output name
        # Since user_mode is True and export_fbx is True, it should create the file directly
        if not Path(output_file).exists():
            # Look for generated skin FBX files in the output directory
            skin_files = list(Path(output_file).parent.rglob("*skin*.fbx"))
            if skin_files:
                actual_output_file = skin_files[0]
                # Copy/move to the expected location
                shutil.copy2(actual_output_file, output_file)
            else:
                raise RuntimeError(f"Skin FBX file not found. Expected at: {output_file}")
        
        return str(output_file)
    
    def merge_results_python(self, source_file: str, target_file: str, output_file: str) -> str:
        """
        Merge results using Python (replaces merge.sh)
        Returns path to merged file
        """
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
        
        # Validate that the output file was created and is a valid file
        if not output_path.exists():
            raise RuntimeError(f"Merge failed: Output file not created at {output_path}")
        
        if not output_path.is_file():
            raise RuntimeError(f"Merge failed: Output path is not a valid file: {output_path}")
        
        return str(output_path.resolve())


def create_app():
    """Create and configure the Gradio interface."""
    
    demo_instance = UniRigDemo()
    
    with gr.Blocks(title="UniRig - 3D Model Rigging Demo") as interface:
        
        # Header
        gr.HTML("""
        <div class="title" style="text-align: center">
            <h1>🎯 UniRig: Automated 3D Model Rigging</h1>
            <p style="font-size: 1.1em; color: #6b7280;">
                Leverage deep learning to automatically generate skeletons and skinning weights for your 3D models
            </p>
        </div>
        """)
        
        # Information Section
        gr.HTML("""
        <h3>📚 About UniRig</h3>
        <p>UniRig is a state-of-the-art framework that automates the complex process of 3D model rigging:</p>
        <ul>
            <li><strong>Skeleton Generation:</strong> AI predicts optimal bone structures</li>
            <li><strong>Skinning Weights:</strong> Automatic vertex-to-bone weight assignment</li>
            <li><strong>Universal Support:</strong> Works with humans, animals, and objects</li>
        </ul>
        <p><strong>Supported formats:</strong> .obj, .fbx, .glb</p>
        """)
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_3d_model = gr.File(
                    label="Upload 3D Model",
                    file_types=[".obj", ".fbx", ".glb"],
                    type="filepath",
                )
                
                with gr.Row(equal_height=True):
                    seed = gr.Number(
                        value=12345,
                        label="Random Seed (for reproducible results)",
                        scale=4,
                    )
                    random_btn = gr.Button("🔄 Random Seed", variant="secondary", scale=1)
                pipeline_btn = gr.Button("🎯 Start Complete Pipeline", variant="primary", size="lg")
            
            pipeline_skeleton_out = gr.File(label="Final Rigged Model")
        
        random_btn.click(
            fn=lambda: int(torch.randint(0, 100000, (1,)).item()),
            outputs=seed
        )
        
        pipeline_btn.click(
            fn=demo_instance.complete_pipeline,
            inputs=[input_3d_model, seed],
            outputs=[pipeline_skeleton_out]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2em; padding: 1em; border-radius: 8px;">
            <p style="color: #6b7280;">
                🔬 <strong>UniRig</strong> - Research by Tsinghua University & Tripo<br>
                📄 <a href="https://arxiv.org/abs/2504.12451" target="_blank">Paper</a> | 
                🏠 <a href="https://zjp-shadow.github.io/works/UniRig/" target="_blank">Project Page</a> | 
                🤗 <a href="https://huggingface.co/VAST-AI/UniRig" target="_blank">Models</a>
            </p>
        </div>
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    app = create_app()
    
    # Launch configuration
    app.queue().launch()
