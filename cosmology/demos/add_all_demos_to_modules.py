#!/usr/bin/env python3
"""
Script to add all demos and visualization files to every module in the TSAMS repository.
"""

import os
import subprocess
import shutil
import glob

def run_command(command, cwd=None):
    """Run a shell command and return the output"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, 
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                           text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout.strip()

def main():
    # Main TSAMS repository path
    tsams_repo = "TSAMS"
    
    # Change to TSAMS repository
    os.chdir(tsams_repo)
    
    # Get all visualization and demo files from the workspace
    workspace_dir = ".."
    all_visualization_files = []
    
    # Find all visualization files
    for pattern in ["*visualizer*.py", "*visualization*.py", "*visualizer*.mp4", "*visualization*.mp4", 
                   "*visualizer*.png", "*visualization*.png", "*prime_generator*.mp4", "*prime_generator*.png"]:
        all_visualization_files.extend(glob.glob(os.path.join(workspace_dir, pattern)))
    
    # Find all demo files
    demo_files = []
    for pattern in ["*demo*.py", "*example*.py", "*test*.py"]:
        demo_files.extend(glob.glob(os.path.join(workspace_dir, pattern)))
    
    # List of all modules in the repository
    modules = [
        "benchmarks",
        "biology",
        "chemistry",
        "classical",
        "core",
        "cosmology",
        "cryptography",
        "hybrid",
        "integration",
        "physics",
        "prime_generator",
        "quantum",
        "visualization"
    ]
    
    # Add visualization and demo directories to each module
    for module in modules:
        # Create visualization directory if it doesn't exist
        viz_dir = os.path.join(module, "visualization")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create demos directory if it doesn't exist
        demos_dir = os.path.join(module, "demos")
        os.makedirs(demos_dir, exist_ok=True)
        
        # Copy all visualization files to the module's visualization directory
        for file_path in all_visualization_files:
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                target_path = os.path.join(viz_dir, file_name)
                print(f"Copying {file_path} to {target_path}")
                shutil.copy2(file_path, target_path)
        
        # Copy all demo files to the module's demos directory
        for file_path in demo_files:
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                target_path = os.path.join(demos_dir, file_name)
                print(f"Copying {file_path} to {target_path}")
                shutil.copy2(file_path, target_path)
    
    # Check if we have any changes to commit
    status = run_command("git status --porcelain")
    if status:
        # Add all changes
        run_command("git add .")
        
        # Commit the changes
        run_command('git commit -m "Add all demos and visualization files to every module"')
        
        # Push to GitHub
        run_command("git push origin master")
        print("Successfully added all demos and visualization files to every module and pushed to GitHub")
    else:
        print("No changes to commit")
    
    # Return to workspace directory
    os.chdir("..")
    
    print("All demos and visualization files added to every module successfully")

if __name__ == "__main__":
    main()