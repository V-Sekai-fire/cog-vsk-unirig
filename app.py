import gradio as gr
import tempfile
import os
import sys
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Optional, Tuple, List
import spaces

import subprocess
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

# Add the current directory to the path so we can import UniRig modules
sys.path.insert(0, str(Path(__file__).parent))

import trimesh
import yaml

class UniRigDemo:
    """Main class for the UniRig Gradio demo application."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.temp_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Supported file formats
        self.supported_formats = ['.obj', '.fbx', '.glb', '.gltf', '.vrm']
        
        # Initialize models (will be loaded on demand)
        self.skeleton_model = None
        self.skin_model = None
        
    def load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config {config_path}: {str(e)}")
    
    def validate_input_file(self, file_path: str) -> bool:
        """Validate if the input file format is supported."""
        if not file_path or not os.path.exists(file_path):
            return False
        
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_formats
    
    def preprocess_model(self, input_file: str, output_dir: str) -> str:
        """
        Preprocess the 3D model for inference.
        This extracts mesh data and saves it as .npz format.
        """
        try:
            # Create extraction command
            extract_cmd = [
                'python', '-m', 'src.data.extract',
                '--config', 'configs/data/quick_inference.yaml',
                '--input', input_file,
                '--output_dir', output_dir,
                '--force_override', 'true',
                '--faces_target_count', '50000'
            ]
            
            # Run extraction
            result = subprocess.run(
                extract_cmd, 
                cwd=str(Path(__file__).parent),
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Extraction failed: {result.stderr}")
            
            # Find the generated .npz file
            npz_files = list(Path(output_dir).glob("*.npz"))
            if not npz_files:
                raise RuntimeError("No .npz file generated during preprocessing")
            
            return str(npz_files[0])
            
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {str(e)}")
    
    def generate_skeleton(self, input_file: str, seed: int = 12345) -> Tuple[str, str, str]:
        """
        Generate skeleton for the input 3D model.
        
        Args:
            input_file: Path to the input 3D model file
            seed: Random seed for reproducible results
            
        Returns:
            Tuple of (status_message, output_file_path, preview_info)
        """
        try:
            # Validate input
            if not self.validate_input_file(input_file):
                return "Error: Invalid or unsupported file format. Supported: " + ", ".join(self.supported_formats), "", ""
            
            # Create working directory
            work_dir = os.path.join(self.temp_dir, f"skeleton_{seed}")
            os.makedirs(work_dir, exist_ok=True)
            
            # Copy input file to work directory
            input_name = Path(input_file).name
            work_input = os.path.join(work_dir, input_name)
            shutil.copy2(input_file, work_input)
            
            # Generate skeleton using the launch script
            output_file = os.path.join(work_dir, f"{Path(input_name).stem}_skeleton.fbx")
            
            skeleton_cmd = [
                'bash', 'launch/inference/generate_skeleton.sh',
                '--input', work_input,
                '--output', output_file,
                '--seed', str(seed)
            ]
            
            # Run skeleton generation
            result = subprocess.run(
                skeleton_cmd,
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return f"Error: Skeleton generation failed: {result.stderr}", "", ""
            
            if not os.path.exists(output_file):
                return "Error: Skeleton file was not generated", "", ""
            
            # Generate preview information
            preview_info = self.generate_model_preview(output_file)
            
            return "✅ Skeleton generated successfully!", output_file, preview_info
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            traceback.print_exc()
            return error_msg, "", ""
    
    def generate_skinning(self, skeleton_file: str) -> Tuple[str, str, str]:
        """
        Generate skinning weights for the skeleton.
        
        Args:
            skeleton_file: Path to the skeleton file (from skeleton generation step)
            
        Returns:
            Tuple of (status_message, output_file_path, preview_info)
        """
        try:
            if not skeleton_file or not os.path.exists(skeleton_file):
                return "Error: No skeleton file provided or file doesn't exist", "", ""
            
            # Create output directory
            work_dir = Path(skeleton_file).parent
            output_file = os.path.join(work_dir, f"{Path(skeleton_file).stem}_skin.fbx")
            
            # Generate skinning using the launch script
            skin_cmd = [
                'bash', 'launch/inference/generate_skin.sh',
                '--input', skeleton_file,
                '--output', output_file
            ]
            
            # Run skinning generation
            result = subprocess.run(
                skin_cmd,
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return f"Error: Skinning generation failed: {result.stderr}", "", ""
            
            if not os.path.exists(output_file):
                return "Error: Skinning file was not generated", "", ""
            
            # Generate preview information
            preview_info = self.generate_model_preview(output_file)
            
            return "✅ Skinning weights generated successfully!", output_file, preview_info
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            traceback.print_exc()
            return error_msg, "", ""
    
    def merge_results(self, original_file: str, rigged_file: str) -> Tuple[str, str, str]:
        """
        Merge the rigged skeleton/skin with the original model.
        
        Args:
            original_file: Path to the original 3D model
            rigged_file: Path to the rigged file (skeleton or skin)
            
        Returns:
            Tuple of (status_message, output_file_path, preview_info)
        """
        try:
            if not original_file or not os.path.exists(original_file):
                return "Error: Original file not provided or doesn't exist", "", ""
            
            if not rigged_file or not os.path.exists(rigged_file):
                return "Error: Rigged file not provided or doesn't exist", "", ""
            
            # Create output file
            work_dir = Path(rigged_file).parent
            output_file = os.path.join(work_dir, f"{Path(original_file).stem}_rigged.glb")
            
            # Merge using the launch script
            merge_cmd = [
                'bash', 'launch/inference/merge.sh',
                '--source', rigged_file,
                '--target', original_file,
                '--output', output_file
            ]
            
            # Run merge
            result = subprocess.run(
                merge_cmd,
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return f"Error: Merge failed: {result.stderr}", "", ""
            
            if not os.path.exists(output_file):
                return "Error: Merged file was not generated", "", ""
            
            # Generate preview information
            preview_info = self.generate_model_preview(output_file)
            
            return "✅ Model rigging completed successfully!", output_file, preview_info
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            traceback.print_exc()
            return error_msg, "", ""
    
    def generate_model_preview(self, model_path: str) -> str:
        """
        Generate preview information for a 3D model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            HTML string with model information
        """
        try:
            if not os.path.exists(model_path):
                return "Model file not found"
            
            # Try to load with trimesh for basic info
            try:
                mesh = trimesh.load(model_path)
                if hasattr(mesh, 'vertices'):
                    vertices_count = len(mesh.vertices)
                    faces_count = len(mesh.faces) if hasattr(mesh, 'faces') else 0
                else:
                    vertices_count = 0
                    faces_count = 0
            except Exception:
                vertices_count = 0
                faces_count = 0
            
            file_size = os.path.getsize(model_path)
            file_size_mb = file_size / (1024 * 1024)
            
            preview_html = f"""
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
                <h4>📊 Model Information</h4>
                <p><strong>File:</strong> {Path(model_path).name}</p>
                <p><strong>Size:</strong> {file_size_mb:.2f} MB</p>
                <p><strong>Vertices:</strong> {vertices_count:,}</p>
                <p><strong>Faces:</strong> {faces_count:,}</p>
                <p><strong>Format:</strong> {Path(model_path).suffix.upper()}</p>
            </div>
            """
            
            return preview_html
            
        except Exception as e:
            return f"Error generating preview: {str(e)}"
    
    def complete_pipeline(self, input_file: str, seed: int = 12345) -> Tuple[str, str, str, str, str]:
        """
        Run the complete rigging pipeline: skeleton generation → skinning → merge.
        
        Args:
            input_file: Path to the input 3D model file
            seed: Random seed for reproducible results
            
        Returns:
            Tuple of status messages and file paths for each step
        """
        try:
            # Step 1: Generate skeleton
            skeleton_status, skeleton_file, skeleton_preview = self.generate_skeleton(input_file, seed)
            if not skeleton_file:
                return skeleton_status, "", "", "", ""
            
            # Step 2: Generate skinning
            skin_status, skin_file, skin_preview = self.generate_skinning(skeleton_file)
            if not skin_file:
                return f"{skeleton_status}\n{skin_status}", skeleton_file, "", "", ""
            
            # Step 3: Merge results
            merge_status, final_file, final_preview = self.merge_results(input_file, skin_file)
            
            # Combine all status messages
            combined_status = f"""
            🏗️ **Pipeline Complete!**
            
            **Step 1 - Skeleton Generation:** ✅ Complete
            **Step 2 - Skinning Weights:** ✅ Complete  
            **Step 3 - Final Merge:** ✅ Complete
            
            {merge_status}
            """
            
            return combined_status, skeleton_file, skin_file, final_file, final_preview
            
        except Exception as e:
            error_msg = f"Pipeline Error: {str(e)}"
            traceback.print_exc()
            return error_msg, "", "", "", ""


def create_demo_interface():
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
        <p><strong>Supported formats:</strong> .obj, .fbx, .glb, .gltf, .vrm</p>
        """)
        
        # Main Interface Tabs
        with gr.Tabs():
            
            # Complete Pipeline Tab
            with gr.Tab("🚀 Complete Pipeline", elem_id="pipeline-tab"):                
                with gr.Row():
                    with gr.Column(scale=1):
                        pipeline_input = gr.Model3D(
                            label="Upload 3D Model",
                            display_mode="solid",
                        )
                        pipeline_seed = gr.Slider(
                            minimum=1,
                            maximum=99999,
                            value=12345,
                            step=1,
                            label="Random Seed (for reproducible results)"
                        )
                        pipeline_btn = gr.Button("🎯 Start Complete Pipeline", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        pipeline_status = gr.Markdown("Ready to process your 3D model...")
                        pipeline_preview = gr.HTML("")
                
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h4>📥 Download Results</h4>")
                        pipeline_skeleton_out = gr.File(label="Skeleton (.fbx)", visible=False)
                        pipeline_skin_out = gr.File(label="Skinning Weights (.fbx)", visible=False)
                        pipeline_final_out = gr.File(label="Final Rigged Model (.glb)", visible=False)
                
                pipeline_btn.click(
                    fn=demo_instance.complete_pipeline,
                    inputs=[pipeline_input, pipeline_seed],
                    outputs=[pipeline_status, pipeline_skeleton_out, pipeline_skin_out, 
                            pipeline_final_out, pipeline_preview]
                )
            
            # Step-by-Step Tab
            with gr.Tab("🔧 Step-by-Step Process", elem_id="stepwise-tab"):
                gr.HTML("<h3>Manual Step-by-Step Rigging Process</h3>")
                gr.HTML("<p>Process your model step by step with full control over each stage.</p>")
                
                # Step 1: Skeleton Generation
                with gr.Group():
                    gr.HTML("<h4>Step 1: Skeleton Generation</h4>")
                    with gr.Row():
                        with gr.Column():
                            step1_input = gr.File(
                                label="Upload 3D Model",
                                file_types=[".obj", ".fbx", ".glb", ".gltf", ".vrm"],
                                type="filepath"
                            )
                            step1_seed = gr.Slider(
                                minimum=1,
                                maximum=99999,
                                value=12345,
                                step=1,
                                label="Random Seed"
                            )
                            step1_btn = gr.Button("Generate Skeleton", variant="secondary")
                        
                        with gr.Column():
                            step1_status = gr.Markdown("Upload a model to start...")
                            step1_preview = gr.HTML("")
                            step1_output = gr.File(label="Skeleton File (.fbx)", visible=False)
                
                # Step 2: Skinning Generation
                with gr.Group():
                    gr.HTML("<h4>Step 2: Skinning Weight Generation</h4>")
                    with gr.Row():
                        with gr.Column():
                            step2_input = gr.File(
                                label="Skeleton File (from Step 1)",
                                file_types=[".fbx"],
                                type="filepath"
                            )
                            step2_btn = gr.Button("Generate Skinning Weights", variant="secondary")
                        
                        with gr.Column():
                            step2_status = gr.Markdown("Complete Step 1 first...")
                            step2_preview = gr.HTML("")
                            step2_output = gr.File(label="Skinning File (.fbx)", visible=False)
                
                # Step 3: Merge Results
                with gr.Group():
                    gr.HTML("<h4>Step 3: Merge with Original Model</h4>")
                    with gr.Row():
                        with gr.Column():
                            step3_original = gr.File(
                                label="Original Model",
                                file_types=[".obj", ".fbx", ".glb", ".gltf", ".vrm"],
                                type="filepath"
                            )
                            step3_rigged = gr.File(
                                label="Rigged File (from Step 2)",
                                file_types=[".fbx"],
                                type="filepath"
                            )
                            step3_btn = gr.Button("Merge Results", variant="secondary")
                        
                        with gr.Column():
                            step3_status = gr.Markdown("Complete previous steps first...")
                            step3_preview = gr.HTML("")
                            step3_output = gr.File(label="Final Rigged Model (.glb)", visible=False)
                
                # Event handlers for step-by-step
                step1_btn.click(
                    fn=demo_instance.generate_skeleton,
                    inputs=[step1_input, step1_seed],
                    outputs=[step1_status, step1_output, step1_preview]
                )
                
                step2_btn.click(
                    fn=demo_instance.generate_skinning,
                    inputs=[step2_input],
                    outputs=[step2_status, step2_output, step2_preview]
                )
                
                step3_btn.click(
                    fn=demo_instance.merge_results,
                    inputs=[step3_original, step3_rigged],
                    outputs=[step3_status, step3_output, step3_preview]
                )
                
                # Auto-populate step 2 input when step 1 completes
                step1_output.change(
                    fn=lambda x: x,
                    inputs=[step1_output],
                    outputs=[step2_input]
                )
                
                # Auto-populate step 3 rigged input when step 2 completes
                step2_output.change(
                    fn=lambda x: x,
                    inputs=[step2_output],
                    outputs=[step3_rigged]
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
            <p style="color: #9ca3af; font-size: 0.9em;">
                ⚡ Powered by PyTorch & Gradio | 🎯 GPU recommended for optimal performance
            </p>
        </div>
        """)
    
    return interface


def main():
    """Main function to launch the Gradio demo."""
    
    # Create and launch the interface
    demo = create_demo_interface()
    
    # Launch configuration
    demo.queue().launch()


if __name__ == "__main__":
    main()