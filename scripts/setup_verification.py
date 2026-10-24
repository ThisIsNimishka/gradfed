#!/usr/bin/env python3
"""
Setup Verification Script for GradFed
This script verifies that GradFed and its components are properly configured.

Usage:
    python scripts/setup_verification.py
"""

import os
import sys
import importlib.util

# Determine repository root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(SCRIPT_DIR, '..', 'src')):
    REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
elif os.path.isdir(os.path.join(os.getcwd(), 'src')):
    REPO_ROOT = os.getcwd()
else:
    REPO_ROOT = os.getcwd()

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_file_exists(rel_path, min_lines=50):
    """Check if file exists and has sufficient content"""
    filepath = os.path.join(REPO_ROOT, rel_path)
    if not os.path.exists(filepath):
        return False, "❌ File not found", 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        num_lines = len(lines)
        
        if num_lines < min_lines:
            return False, f"⚠️ File too short ({num_lines} lines, expected {min_lines}+)", num_lines
        
        return True, f"✅ OK ({num_lines} lines)", num_lines
    except Exception as e:
        return False, f"❌ Error reading: {e}", 0

def check_directory_structure():
    """Verify semantic directory structure"""
    print_header("1. Checking Directory Structure")
    
    required_dirs = [
        "src",
        "src/ppo_server",
        "baseline",
        "data",
        "docs",
        "scripts",
        "results",
        "logs",
    ]
    
    all_good = True
    for dir_rel in required_dirs:
        dir_path = os.path.join(REPO_ROOT, dir_rel)
        exists = os.path.isdir(dir_path)
        status = "✅" if exists else "❌"
        print(f"{status} {dir_rel}/")
        if not exists:
            all_good = False
    
    return all_good

def check_original_files():
    """Check original FedPPO files"""
    print_header("2. Checking Original FedPPO Foundation Files")
    
    original_files = [
        ("src/options.py", 50),
        ("src/utils.py", 100),
        ("src/models.py", 100),
        ("src/sampling.py", 50),
        ("src/ppo_server/ppo.py", 100),
        ("baseline/fedavg_main.py", 50),
    ]
    
    all_good = True
    for rel_path, min_lines in original_files:
        exists, msg, lines = check_file_exists(rel_path, min_lines)
        print(f"{msg:60s} {rel_path}")
        if not exists:
            all_good = False
    
    return all_good

def check_gradfed_files():
    """Check GradFed files"""
    print_header("3. Checking GradFed Core Files")
    
    gradfed_files = [
        ("src/gradient_monitor.py", 200),
        ("src/ppo_server/gradfed_ppo_update.py", 230),
        ("src/gradfed_main.py", 350),
        ("src/visualize_results.py", 350),
        ("scripts/run_comparison.sh", 80),
        ("docs/setup_guide.md", 80),
    ]
    
    all_good = True
    missing_files = []
    
    for rel_path, min_lines in gradfed_files:
        exists, msg, lines = check_file_exists(rel_path, min_lines)
        print(f"{msg:60s} {rel_path}")
        if not exists:
            all_good = False
            missing_files.append(rel_path)
    
    return all_good, missing_files

def test_imports():
    """Test if Python can import the modules"""
    print_header("4. Testing Python Imports")
    
    # Add root and src to path
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    src_dir = os.path.join(REPO_ROOT, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    imports_to_test = [
        ("src.options", "args_parser"),
        ("src.utils", "get_dataset"),
        ("src.models", "ResNet10Cifar"),
        ("src.sampling", "cifar_noniid"),
        ("src.gradient_monitor", "GradientMonitor"),
        ("src.ppo_server.gradfed_ppo_update", "GradFedPPO"),
    ]
    
    all_good = True
    for module_name, class_name in imports_to_test:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print(f"✅ {module_name}.{class_name}")
            else:
                print(f"⚠️ {module_name} imported but {class_name} not found")
                all_good = False
        except Exception as e:
            print(f"❌ {module_name}.{class_name} - Error: {e}")
            all_good = False
    
    return all_good

def check_permissions():
    """Check file permissions"""
    print_header("5. Checking File Permissions")
    
    comparison_script = os.path.join(REPO_ROOT, "scripts", "run_comparison.sh")
    if os.path.exists(comparison_script):
        # On POSIX check executable bit; on Windows check read/write
        if os.name == 'nt':
            print(f"✅ scripts/run_comparison.sh exists (Windows host)")
            return True
        else:
            is_executable = os.access(comparison_script, os.X_OK)
            if is_executable:
                print(f"✅ scripts/run_comparison.sh is executable")
                return True
            else:
                print(f"⚠️ scripts/run_comparison.sh exists but not executable")
                print(f"   Fix with: chmod +x scripts/run_comparison.sh")
                return False
    else:
        print(f"❌ scripts/run_comparison.sh not found")
        return False

def check_disk_space():
    """Check available disk space"""
    print_header("6. Checking Disk Space")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(REPO_ROOT)
        
        print(f"Total Space: {total // (2**30)} GB")
        print(f"Used Space:  {used // (2**30)} GB")
        print(f"Free Space:  {free // (2**30)} GB")
        
        if free > 5 * (2**30):
            print(f"✅ Sufficient disk space available")
            return True
        else:
            print(f"⚠️ Low disk space (< 5GB free)")
            return False
    except Exception as e:
        print(f"⚠️ Could not check disk space: {e}")
        return True

def print_quick_start_guide(missing_files):
    """Print guide on what to do next"""
    print_header("Next Steps")
    
    if missing_files:
        print("\n⚠️ Missing Files Detected!")
        print("\nPlease check that the following files exist:")
        for filepath in missing_files:
            print(f"  - {filepath}")
        print("\nPlease check that all repository files are present according to README.md.")
    else:
        print("\n✅ All files and semantic directories are present!")
        print("\nYou can now:")
        print("\n1. Run a quick test (10 rounds):")
        print("   python src/gradfed_main.py --dataset cifar10 --model resnet10 \\")
        print("       --num_users 20 --frac 0.2 --local_ep 5 --epochs 10 --gpu 0")
        print("\n2. Monitor live training:")
        print("   tail -f logs/gradfed_training.log")
        print("\n3. Run full benchmark experiment (30 rounds):")
        print("   python src/gradfed_main.py --dataset cifar10 --model resnet10 \\")
        print("       --num_users 50 --frac 0.2 --local_ep 10 --epochs 30 --gpu 0")
        print("\n4. Compare with all baselines:")
        print("   chmod +x scripts/run_comparison.sh && ./scripts/run_comparison.sh")

def main():
    """Main verification function"""
    print("\n" + "="*70)
    print("  GradFed System & Semantic Setup Verification")
    print("="*70)
    print(f"\nRepository Root: {REPO_ROOT}")
    
    results = {}
    results['structure'] = check_directory_structure()
    results['original'] = check_original_files()
    results['gradfed'], missing_files = check_gradfed_files()
    results['imports'] = test_imports()
    results['permissions'] = check_permissions()
    results['disk'] = check_disk_space()
    
    print_header("Summary")
    
    checks = [
        ("Directory Structure", results['structure']),
        ("Original FedPPO Files", results['original']),
        ("GradFed Files", results['gradfed']),
        ("Python Imports", results['imports']),
        ("File Permissions", results['permissions']),
        ("Disk Space", results['disk']),
    ]
    
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10s} {check_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("  🎉 ALL CHECKS PASSED! GradFed setup is complete and healthy.")
        print("="*70)
        print_quick_start_guide([])
    else:
        print("  ⚠️ SOME CHECKS FAILED! Please inspect the diagnostics above.")
        print("="*70)
        print_quick_start_guide(missing_files if not results['gradfed'] else [])
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
