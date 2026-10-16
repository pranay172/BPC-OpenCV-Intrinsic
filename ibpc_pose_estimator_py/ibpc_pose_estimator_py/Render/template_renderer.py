import subprocess
import os
import logging
import time

class TemplateRenderer:
    def __init__(self, blender_path, models_dir, output_dir):
        self.blender_path = os.path.expanduser(blender_path)
        self.models_dir = os.path.expanduser(models_dir)
        self.output_dir = os.path.expanduser(output_dir)
        
        # Make sure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Render templates during initialization
        self.render_templates()
    
    def render_templates(self):
        """Run BlenderProc to render templates"""
        self.logger.info(f"Rendering templates")
        
        # Get the path to the render_custom_templates.py script
        # Assuming it's in the same directory as your current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        render_script = os.path.join(script_dir, "render_bop_templates.py")
        
        # Build the command
        cmd = [
            "blenderproc", "run",
            "--custom-blender-path", self.blender_path,
            render_script,
            "--models_dir", self.models_dir, 
            "--output_dir", self.output_dir
        ]
        
        # Log the command
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        # Execute the command
        try:
            # For real-time output (useful for debugging)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Process output in real-time
            for line in iter(process.stdout.readline, ''):
                self.logger.info(line.strip())
                
            # Wait for process to complete
            process.wait()
            
            if process.returncode != 0:
                self.logger.error(f"BlenderProc rendering failed with code {process.returncode}")
                raise RuntimeError("Template rendering failed")
            
            self.logger.info("Template rendering completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error running BlenderProc: {str(e)}")
            raise
    
    def verify_templates(self):
        """Verify that templates were created successfully"""
        template_dir = os.path.join(self.output_dir, "templates")
        if not os.path.exists(template_dir):
            self.logger.error(f"Template directory not found: {template_dir}")
            return False
        
        # Check if at least some template files exist
        rgb_files = [f for f in os.listdir(template_dir) if f.startswith("rgb_")]
        if not rgb_files:
            self.logger.error("No template RGB files found")
            return False
            
        return True

# Usage example
if __name__ == "__main__":
    renderer = TemplateRenderer(
        blender_path="./blender/",
        models_dir="../../../ipd/models/",
        output_dir="../../../../output",
    )
    
    if renderer.verify_templates():
        print("Templates generated successfully!")