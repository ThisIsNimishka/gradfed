#!/usr/bin/env python3
"""
Setup Verification Script for fedppov2
This script checks if GradFed is properly integrated into your fedppov2 folder

Usage:
    cd /path/to/fedppov2
    python setup_verification.py
"""

import os
import sys
import importlib.util

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_file_exists(filepath, min_lines=50):
    """Check if file exists and has sufficient content"""
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
    """Verify directory structure"""
    print_header("1. Checking Directory Structure")
    
    required_dirs = [
        "src",
        "src/ppo_server",
        "baseline",
        "data",
        "save",
    ]
    
    all_good = True
    for dir_path in required_dirs:
        exists = os.path.isdir(dir_path)
        status = "✅" if exists else "❌"
        print(f"{status} {dir_path}/")
        if not exists:
            all_good = False
    
    return all_good

def check_original_files():
    """Check original FedPPO files"""
    print_header("2. Checking Original FedPPO Files")
    
    original_files = [
        ("src/options.py", 50),
        ("src/utils.py", 100),
        ("src/models.py", 100),
        ("src/sampling.py", 50),
        ("src/ppo_server/ppo.py", 100),
        ("baselines/fedavg_main.py", 50),
    ]
    
    all_good = True
    for filepath, min_lines in original_files:
        exists, msg, lines = check_file_exists(filepath, min_lines)
        print(f"{msg:60s} {filepath}")
        if not exists:
            all_good = False
    
    return all_good

def check_gradfed_files():
    """Check new GradFed files"""
    print_header("3. Checking GradFed Files")
    
    gradfed_files = [
        ("src/gradient_monitor.py", 200),
        ("src/ppo_server/gradfed_ppo_update.py", 230),
        ("src/gradfed_main.py", 350),
        ("src/visualize_results.py", 350),
        ("run_comparison.sh", 100),
    ]
    
    all_good = True
    missing_files = []
    
    for filepath, min_lines in gradfed_files:
        exists, msg, lines = check_file_exists(filepath, min_lines)
        print(f"{msg:60s} {filepath}")
        if not exists:
            all_good = False
            missing_files.append(filepath)
    
    return all_good, missing_files

def test_imports():
    """Test if Python can import the modules"""
    print_header("4. Testing Python Imports")
    
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    
    imports_to_test = [
        ("src.options", "args_parser"),
        ("src.utils", "get_dataset"),
        ("src.models", "ResNet10"),
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
    
    # Check if run_comparison.sh is executable
    comparison_script = "run_comparison.sh"
    if os.path.exists(comparison_script):
        is_executable = os.access(comparison_script, os.X_OK)
        if is_executable:
            print(f"✅ {comparison_script} is executable")
            return True
        else:
            print(f"⚠️ {comparison_script} exists but not executable")
            print(f"   Fix with: chmod +x {comparison_script}")
            return False
    else:
        print(f"❌ {comparison_script} not found")
        return False

def check_disk_space():
    """Check available disk space"""
    print_header("6. Checking Disk Space")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        print(f"Total Space: {total // (2**30)} GB")
        print(f"Used Space:  {used // (2**30)} GB")
        print(f"Free Space:  {free // (2**30)} GB")
        
        # Check if we have at least 10GB free
        if free > 10 * (2**30):
            print(f"✅ Sufficient disk space available")
            return True
        else:
            print(f"⚠️ Low disk space (< 10GB free)")
            return False
    except Exception as e:
        print(f"⚠️ Could not check disk space: {e}")
        return True

def print_quick_start_guide(missing_files):
    """Print guide on what to do next"""
    print_header("Next Steps")
    
    if missing_files:
        print("\n⚠️ Missing Files Detected!")
        print("\nPlease add the following files:")
        for filepath in missing_files:
            print(f"  - {filepath}")
        print("\nRefer to the artifacts provided by Claude to get the file contents.")
    else:
        print("\n✅ All files are present!")
        print("\nYou can now:")
        print("\n1. Run a quick test:")
        print("   python src/gradfed_main.py --dataset cifar10 --model resnet10 \\")
        print("       --num_users 10 --frac 0.3 --local_ep 2 --epochs 3 --gpu 0")
        print("\n2. Check if it's working:")
        print("   tail -f gradfed_training.log")
        print("\n3. Run full experiment:")
        print("   python src/gradfed_main.py --dataset cifar10 --model resnet10 \\")
        print("       --num_users 100 --frac 0.1 --local_ep 10 --epochs 100 --gpu 0")
        print("\n4. Compare with baselines:")
        print("   ./run_comparison.sh")

def main():
    """Main verification function"""
    print("\n" + "="*70)
    print("  GradFed Setup Verification for fedppov2")
    print("="*70)
    print(f"\nCurrent Directory: {os.getcwd()}")
    
    # Run all checks
    results = {}
    
    results['structure'] = check_directory_structure()
    results['original'] = check_original_files()
    results['gradfed'], missing_files = check_gradfed_files()
    results['imports'] = test_imports()
    results['permissions'] = check_permissions()
    results['disk'] = check_disk_space()
    
    # Summary
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
        print("  🎉 ALL CHECKS PASSED! Setup is complete.")
        print("="*70)
        print_quick_start_guide([])
    else:
        print("  ⚠️ SOME CHECKS FAILED! Please fix the issues above.")
        print("="*70)
        print_quick_start_guide(missing_files if not results['gradfed'] else [])
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
