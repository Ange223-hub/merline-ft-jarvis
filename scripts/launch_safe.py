#!/usr/bin/env python3
"""
MERLINE - Safe Launch Script
Lance MERLINE de manière sûre avec vérifications complètes
"""

import sys
import os
import subprocess

def print_banner():
    """Print MERLINE banner"""
    print("\n" + "="*70)
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "MERLINE - Safe Launch" + " "*32 + "║")
    print("║" + "  Modular Ethical Responsive Local Intelligent Neural Entity" + " "*5 + "║")
    print("╚" + "="*68 + "╝")
    print("="*70 + "\n")

def run_verification():
    """Run all verification tests"""
    print("[LAUNCH] Running verification checks...")
    print()
    
    script = os.path.join(os.path.dirname(__file__), "test_safe_launch.py")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    
    print(result.stdout)
    
    if result.returncode != 0:
        print("\n[ERROR] Verification failed!")
        print(result.stderr)
        return False
    
    return True

def launch_merline():
    """Launch MERLINE"""
    print("[LAUNCH] Starting MERLINE...")
    print()
    
    main_file = os.path.join(os.path.dirname(__file__), "..", "main.py")
    
    try:
        subprocess.run([sys.executable, main_file])
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] MERLINE shutting down...")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to launch: {e}")
        return False
    
    return True

def main():
    """Main entry point"""
    print_banner()
    
    print("[LAUNCH] MERLINE Safe Launch Script")
    print("[LAUNCH] Python", sys.version.split()[0])
    print()
    
    # Run verification
    if not run_verification():
        print("[ABORT] Cannot launch due to verification errors")
        return 1
    
    print("[LAUNCH] All checks passed! Starting MERLINE...")
    print()
    
    # Launch MERLINE
    if not launch_merline():
        print("[ABORT] Failed to launch MERLINE")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
