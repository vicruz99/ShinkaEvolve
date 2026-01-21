#!/usr/bin/env python3
import os
import sys
import re
import shutil
import argparse
from pathlib import Path

# Use omegaconf since hydra-core is installed
try:
    from omegaconf import OmegaConf
except ImportError:
    print("Error: OmegaConf is missing. Ensure hydra-core is installed.")
    sys.exit(1)

def flatten_dict(d, parent_key='', sep='.'):
    """Flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def extract_placeholders(text):
    """Finds unique strings inside curly braces."""
    pattern = r'(?<!\{)\{([a-zA-Z0-9_.]+)\}(?!\})'
    return set(re.findall(pattern, text))

def indent_text(text, spaces=4):
    """Indents multi-line strings for YAML safety."""
    if not isinstance(text, str) or '\n' not in text:
        return text
    lines = text.strip().split('\n')
    indentation = " " * spaces
    return f"\n{indentation}".join(lines)

def render_template(tpl_path, flat_config):
    """Reads a template, checks for missing keys, and returns rendered content."""
    if not tpl_path.exists():
        print(f"Error: Template not found at {tpl_path}")
        sys.exit(1)
        
    content = tpl_path.read_text()
    placeholders = extract_placeholders(content)
    
    missing_keys = []
    used_keys = set()
    
    for key in placeholders:
        if key in flat_config:
            content = content.replace(f"{{{key}}}", str(flat_config[key]))
            used_keys.add(key)
        else:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"ERROR: Template {tpl_path.name} requires keys missing from config.yaml:")
        for k in missing_keys:
            print(f"  - {k}")
        sys.exit(1)
        
    return content, used_keys

def main():
    parser = argparse.ArgumentParser(description="Deploy a ShinkaEvolve task.")
    parser.add_argument("task_name", help="The name of the folder inside 'tasks/'")
    args = parser.parse_args()

    # === 1. Path Setup ===
    script_dir = Path(__file__).parent.resolve()
    task_source_dir = script_dir / "tasks" / args.task_name
    config_file = task_source_dir / "config.yaml"
    repo_root = script_dir.parent.parent
    template_root = script_dir / "template"

    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        sys.exit(1)

    # === 2. Load & Process Config ===
    print(f"--> Reading config: {config_file}")
    try:
        conf_obj = OmegaConf.load(config_file)
        config = OmegaConf.to_container(conf_obj, resolve=True)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    if "task_name" not in config:
        print("Error: 'task_name' is required in config.yaml")
        sys.exit(1)

    deploy_name = config["task_name"]
    flat_config = flatten_dict(config)
    
    # Indent multi-line strings
    for k, v in flat_config.items():
        if isinstance(v, str) and '\n' in v:
            flat_config[k] = indent_text(v, spaces=4)

    # === 3. Template Preparation (Dry Run) ===
    tpl_conf = config.get("template", {})
    task_tpl_path = template_root / "configs" / "task" / tpl_conf.get("task", "task.yaml")
    variant_tpl_path = template_root / "configs" / "variant" / tpl_conf.get("variant", "variant.yaml")

    task_content, used_task = render_template(task_tpl_path, flat_config)
    variant_content, used_variant = render_template(variant_tpl_path, flat_config)

    # Identify ALL items to copy (excluding config.yaml and junk)
    items_to_copy = []
    if task_source_dir.exists():
        for item in task_source_dir.iterdir():
            if item.name == "config.yaml":
                continue
            if item.name == "__pycache__" or item.name.startswith("."):
                continue
            items_to_copy.append(item)

    # Define Destinations
    dest_examples_dir = repo_root / "examples" / deploy_name
    dest_task_config = repo_root / "configs" / "task" / f"{deploy_name}.yaml"
    dest_variant_dir = repo_root / "configs" / "variant" / deploy_name
    dest_variant_config = dest_variant_dir / "variant.yaml"

    # === 4. User Permission ===
    print("\n" + "="*50)
    print(f"DEPLOYMENT PLAN FOR TASK: '{deploy_name}'")
    print("="*50)
    
    print(f"\n[1] Create Directories:")
    print(f"    + {dest_examples_dir}")
    print(f"    + {dest_variant_dir}")
    
    print(f"\n[2] Copy Task Files:")
    if not items_to_copy:
        print("    (No files found to copy other than config.yaml)")
    for item in items_to_copy:
        ftype = "DIR " if item.is_dir() else "FILE"
        print(f"    [{ftype}] {item.name} -> {dest_examples_dir / item.name}")
        
    print(f"\n[3] Generate Config Files:")
    print(f"    + {dest_task_config}")
    print(f"    + {dest_variant_config}")
    
    print("\n" + "-"*50)
    
    response = input("\n>>> Do you want to proceed with these changes? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted by user.")
        sys.exit(0)

    # === 5. Execution ===
    print("\nExecuting...")
    
    # A. Create Directories
    if not dest_examples_dir.exists():
        os.makedirs(dest_examples_dir, exist_ok=True)
        print(f"Created {dest_examples_dir}")
        
    if not dest_variant_dir.exists():
        os.makedirs(dest_variant_dir, exist_ok=True)
        print(f"Created {dest_variant_dir}")

    # B. Copy Files/Dirs
    for item in items_to_copy:
        src = item
        dst = dest_examples_dir / item.name
        
        if item.is_dir():
            # dirs_exist_ok=True allows overwriting if dir exists
            if dst.exists():
                 shutil.rmtree(dst) # Clean overwrite for folders
            shutil.copytree(src, dst)
            print(f"Copied directory {item.name}/")
        else:
            shutil.copy(src, dst)
            print(f"Copied file {item.name}")

    # C. Write Configs
    dest_task_config.parent.mkdir(parents=True, exist_ok=True)
    dest_task_config.write_text(task_content)
    print(f"Written {dest_task_config.name}")

    dest_variant_config.write_text(variant_content)
    print(f"Written {dest_variant_config.name} in {dest_variant_dir.name}/")

    print("\n[SUCCESS] Task deployed successfully.")
    print(f"Run command: python shinka/shinka_launch task={deploy_name} variant={deploy_name}/variant")

if __name__ == "__main__":
    main()