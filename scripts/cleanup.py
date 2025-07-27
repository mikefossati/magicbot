#!/usr/bin/env python3
"""
MagicBot Project Cleanup Script
Cleans up temporary files, debug data, and organizes project structure
"""

import os
import shutil
import glob
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_python_cache(project_root: Path):
    """Remove Python cache files and directories"""
    logger.info("Cleaning Python cache files...")
    
    # Remove __pycache__ directories
    for pycache_dir in project_root.rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir)
            logger.info(f"Removed {pycache_dir}")
    
    # Remove .pyc files
    for pyc_file in project_root.rglob("*.pyc"):
        pyc_file.unlink()
        logger.info(f"Removed {pyc_file}")

def organize_debug_files(project_root: Path):
    """Move debug and temporary files to debug directory"""
    debug_dir = project_root / "debug"
    debug_dir.mkdir(exist_ok=True)
    
    logger.info("Organizing debug files...")
    
    # Move JSON files from root to debug
    json_files = list(project_root.glob("*.json"))
    for json_file in json_files:
        if json_file.parent == project_root:  # Only root level JSON files
            dest = debug_dir / json_file.name
            shutil.move(str(json_file), str(dest))
            logger.info(f"Moved {json_file.name} to debug/")
    
    # Move backtest results if they exist in root
    backtest_patterns = [
        "*backtest*.json",
        "*optimization*.json", 
        "*results*.json",
        "*config*.json"
    ]
    
    for pattern in backtest_patterns:
        for file in project_root.glob(pattern):
            if file.parent == project_root:
                dest = debug_dir / file.name
                if not dest.exists():
                    shutil.move(str(file), str(dest))
                    logger.info(f"Moved {file.name} to debug/")

def cleanup_logs(project_root: Path):
    """Clean up old log files (keep last 7 days)"""
    logs_dir = project_root / "logs"
    if not logs_dir.exists():
        return
    
    logger.info("Cleaning old log files...")
    
    # Keep only recent log files
    log_files = list(logs_dir.glob("*.log"))
    if len(log_files) > 7:
        # Sort by modification time and keep only the newest 7
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        for old_log in log_files[7:]:
            old_log.unlink()
            logger.info(f"Removed old log file: {old_log.name}")

def cleanup_temp_files(project_root: Path):
    """Remove temporary and backup files"""
    logger.info("Cleaning temporary files...")
    
    temp_patterns = [
        "*.tmp",
        "*.temp", 
        "*~",
        "*.bak",
        "*.swp",
        "*.swo",
        ".DS_Store"
    ]
    
    for pattern in temp_patterns:
        for temp_file in project_root.rglob(pattern):
            temp_file.unlink()
            logger.info(f"Removed temp file: {temp_file}")

def organize_results_directories(project_root: Path):
    """Organize results directories"""
    logger.info("Organizing results directories...")
    
    # Create results directory structure if it doesn't exist
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (results_dir / "backtests").mkdir(exist_ok=True)
    (results_dir / "optimizations").mkdir(exist_ok=True)
    (results_dir / "reports").mkdir(exist_ok=True)
    
    # Move any results directories from root
    for item in project_root.iterdir():
        if item.is_dir() and "result" in item.name.lower():
            if item.parent == project_root and item.name != "results":
                dest = results_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
                    logger.info(f"Moved {item.name} to results/")

def validate_project_structure(project_root: Path):
    """Validate that essential project structure is intact"""
    logger.info("Validating project structure...")
    
    essential_dirs = [
        "src",
        "tests", 
        "config",
        "scripts",
        "docs"
    ]
    
    essential_files = [
        "requirements.txt",
        "README.md",
        ".gitignore"
    ]
    
    for dir_name in essential_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            logger.warning(f"Essential directory missing: {dir_name}")
        else:
            logger.info(f"✓ {dir_name} directory exists")
    
    for file_name in essential_files:
        file_path = project_root / file_name
        if not file_path.exists():
            logger.warning(f"Essential file missing: {file_name}")
        else:
            logger.info(f"✓ {file_name} exists")

def main():
    parser = argparse.ArgumentParser(description="Clean up MagicBot project")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                       help="Project root directory (default: current directory)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be cleaned without actually doing it")
    parser.add_argument("--skip-logs", action="store_true",
                       help="Skip cleaning log files")
    
    args = parser.parse_args()
    
    project_root = args.project_root.resolve()
    logger.info(f"Cleaning project at: {project_root}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")
        return
    
    try:
        # Perform cleanup operations
        cleanup_python_cache(project_root)
        organize_debug_files(project_root)
        cleanup_temp_files(project_root)
        organize_results_directories(project_root)
        
        if not args.skip_logs:
            cleanup_logs(project_root)
        
        validate_project_structure(project_root)
        
        logger.info("✅ Cleanup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
