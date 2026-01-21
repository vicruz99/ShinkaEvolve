#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Delete a deployed ShinkaEvolve task.")
    parser.add_argument("task_name", help="The name of the deployed task (e.g., 'autocorrelation_ineq')")
    args = parser.parse_args()

    task_name = args.task_name

    # === 1. Path Setup ===
    # Current script location: auxiliary/new_task/delete_task.py
    script_dir = Path(__file__).parent.resolve()
    
    # Repo root: auxiliary/new_task/ -> auxiliary/ -> root/
    repo_root = script_dir.parent.parent

    # Define paths to potential artifacts based on standard deployment structure
    paths_to_remove = [
        # 1. The source code directory
        repo_root / "examples" / task_name,
        
        # 2. The task configuration file
        repo_root / "configs" / "task" / f"{task_name}.yaml",
        
        # 3. The variant configuration directory (if it follows the folder structure)
        repo_root / "configs" / "variant" / task_name,
        
        # 4. The variant configuration file (legacy or flat structure fallback)
        repo_root / "configs" / "variant" / f"{task_name}.yaml",
    ]

    # === 2. Scan for Existence ===
    found_paths = []
    for p in paths_to_remove:
        if p.exists():
            found_paths.append(p)

    if not found_paths:
        print(f"[INFO] No files or directories found for task '{task_name}'.")
        print("Nothing to delete.")
        sys.exit(0)

    # === 3. User Permission ===
    print("\n" + "="*50)
    print(f"DELETION PLAN FOR TASK: '{task_name}'")
    print("="*50)
    print("The following files and directories will be PERMANENTLY DELETED:\n")

    for p in found_paths:
        ptype = "DIR " if p.is_dir() else "FILE"
        print(f"    [{ptype}] {p}")

    print("\n" + "-"*50)
    print("WARNING: This action cannot be undone.")
    response = input(f">>> Are you sure you want to delete task '{task_name}'? [y/N]: ").strip().lower()

    if response != 'y':
        print("Aborted by user.")
        sys.exit(0)

    # === 4. Execution ===
    print("\nDeleting...")
    
    for p in found_paths:
        try:
            if p.is_dir():
                shutil.rmtree(p)
                print(f"Deleted directory: {p.name}/")
            else:
                p.unlink()
                print(f"Deleted file:      {p.name}")
        except Exception as e:
            print(f"Error deleting {p}: {e}")

    print("\n[SUCCESS] Task deletion complete.")

if __name__ == "__main__":
    main()