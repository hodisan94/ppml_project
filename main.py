#!/usr/bin/env python3
"""
PPML Experiment Pipeline - Main Orchestrator
Runs the complete Privacy-Preserving Machine Learning experiment pipeline
including training models and executing specialized privacy attacks.
"""

import os
import sys
import shutil
import subprocess
import argparse
import logging
import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


# Global timing dictionary
timing_data = {}

# Configure logging
def setup_logging():
    """Set up logging configuration"""
    log_dir = Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)-5s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "main.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


@contextmanager
def change_directory(path):
    """Context manager for safely changing directories"""
    original = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original)


def safe_execute(cmd, cwd=None, description=""):
    """Execute a command safely with error handling and timing"""
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    try:
        logger.info(f"Executing: {description}")
        logger.debug(f"Command: {' '.join(cmd)}")
        logger.debug(f"Working directory: {cwd or os.getcwd()}")
        
        result = subprocess.run(
            cmd, 
            cwd=cwd,
            check=True, 
            capture_output=True, 
            text=True
        )
        
        if result.stdout:
            logger.debug(f"STDOUT: {result.stdout}")
        if result.stderr:
            logger.debug(f"STDERR: {result.stderr}")
        
        execution_time = time.time() - start_time
        timing_data[description] = execution_time
        
        logger.info(f"SUCCESS: {description} completed successfully in {execution_time:.2f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        timing_data[description] = execution_time
        
        logger.error(f"FAILED: {description} failed!")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        logger.error(f"Time taken: {execution_time:.2f}s")
        return False
    except Exception as e:
        execution_time = time.time() - start_time
        timing_data[description] = execution_time
        
        logger.error(f"FAILED: {description} failed with error: {e}")
        logger.error(f"Time taken: {execution_time:.2f}s")
        return False


def create_output_structure():
    """Create the organized output directory structure"""
    logger = logging.getLogger(__name__)
    
    directories = [
        "output/models/naive",
        "output/models/federated", 
        "output/models/federated_dp",
        "output/results/naive",
        "output/results/federated",
        "output/results/federated_dp",
        "output/logs",
        "output/summary"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Created output directory structure")


def run_naive_training():
    """Run naive RF model training"""
    logger = logging.getLogger(__name__)
    logger.info("[1/8] Running naive RF training...")
    
    return safe_execute(
        [sys.executable, "train_rf_and_save.py"],
        cwd="models/RF/Naive",
        description="Naive RF training"
    )


def run_federated_training():
    """Run federated RF model training"""
    logger = logging.getLogger(__name__)
    logger.info("[2/8] Running federated RF training...")
    
    return safe_execute(
        [sys.executable, "fl.py"],
        cwd="models/RF/FL", 
        description="Federated RF training"
    )


def run_federated_dp_training():
    """Run federated RF with DP training"""
    logger = logging.getLogger(__name__)
    logger.info("[3/8] Running federated RF + DP training...")
    
    return safe_execute(
        [sys.executable, "fl+dp.py"],
        cwd="models/RF/FL+DP",
        description="Federated RF + DP training"
    )


def run_mia_attacks():
    """Run all specialized MIA attacks"""
    logger = logging.getLogger(__name__)
    logger.info("[4/8] Running MIA attacks...")
    
    success = True
    
    # MIA on Naive model (uses specialized script)
    logger.info("  Running MIA on naive model...")
    if not safe_execute(
        [sys.executable, "mia_attack_rf_naive.py"],
        cwd="models/RF/Naive",
        description="MIA attack on naive model"
    ):
        success = False
    
    # MIA on Federated model (uses specialized script)
    logger.info("  Running MIA on federated model...")
    if not safe_execute(
        [sys.executable, "mia_attack_federated.py"],
        cwd="models/RF/FL",
        description="MIA attack on federated model"
    ):
        success = False
    
    # MIA on Federated + DP model (uses specialized script)
    logger.info("  Running MIA on federated + DP model...")
    if not safe_execute(
        [sys.executable, "mia_attack_federated+dp.py"],
        cwd="models/RF/FL+DP",
        description="MIA attack on federated + DP model"
    ):
        success = False
    
    return success


def run_inversion_attacks():
    """Run model inversion attacks on all models"""
    logger = logging.getLogger(__name__)
    logger.info("[5/8] Running inversion attacks...")
    
    return safe_execute(
        [sys.executable, "attacks/inversion_attack.py"],
        description="Model inversion attacks"
    )


def run_aia_attacks():
    """Run attribute inference attacks on all models"""
    logger = logging.getLogger(__name__)
    logger.info("[6/8] Running attribute inference attacks...")
    
    return safe_execute(
        [sys.executable, "attacks/aia.py"],
        description="Attribute inference attacks"
    )


def organize_outputs():
    """Organize all outputs into the structured format"""
    logger = logging.getLogger(__name__)
    logger.info("[7/8] Organizing outputs...")
    
    try:
        # Organize model files
        organize_model_files()
        
        # Organize result files
        organize_result_files()
        
        logger.info("SUCCESS: Output organization completed")
        return True
        
    except Exception as e:
        logger.error(f"FAILED: Output organization failed: {e}")
        return False


def organize_model_files():
    """Move model files to organized structure"""
    logger = logging.getLogger(__name__)
    
    # Organize Naive model files
    naive_files = [
        ("models/RF/Naive/rf_naive_model.pkl", "output/models/naive/rf_naive_model.pkl"),
        ("models/RF/Naive/X_member.npy", "output/models/naive/X_member.npy"),
        ("models/RF/Naive/y_member.npy", "output/models/naive/y_member.npy"),
        ("models/RF/Naive/X_nonmember.npy", "output/models/naive/X_nonmember.npy"),
        ("models/RF/Naive/y_nonmember.npy", "output/models/naive/y_nonmember.npy"),
    ]
    
    # Also move centralized baseline files if they exist
    centralized_files = [
        ("models/RF/Naive/centralized_model.pkl", "output/models/naive/centralized_model.pkl"),
        ("models/RF/Naive/centralized_X_train.npy", "output/models/naive/centralized_X_train.npy"),
        ("models/RF/Naive/centralized_y_train.npy", "output/models/naive/centralized_y_train.npy"),
        ("models/RF/Naive/centralized_X_test.npy", "output/models/naive/centralized_X_test.npy"),
        ("models/RF/Naive/centralized_y_test.npy", "output/models/naive/centralized_y_test.npy"),
        ("models/RF/Naive/centralized_metrics.json", "output/results/naive/centralized_metrics.json"),
    ]
    
    # Organize Federated model files
    federated_files = [
        ("models/RF/FL/federated_model.pkl", "output/models/federated/federated_model.pkl"),
        ("models/RF/FL/federated_X_train.npy", "output/models/federated/federated_X_train.npy"),
        ("models/RF/FL/federated_y_train.npy", "output/models/federated/federated_y_train.npy"),
        ("models/RF/FL/federated_X_test.npy", "output/models/federated/federated_X_test.npy"),
        ("models/RF/FL/federated_y_test.npy", "output/models/federated/federated_y_test.npy"),
        ("models/RF/FL/fl_metrics.json", "output/results/federated/fl_metrics.json"),
        ("models/RF/FL/model_comparison.png", "output/results/federated/model_comparison.png"),
    ]
    
    # Organize Federated DP model files
    federated_dp_files = [
        ("models/RF/FL+DP/federated_model_dp.pkl", "output/models/federated_dp/federated_model_dp.pkl"),
        ("models/RF/FL+DP/federated_X_train.npy", "output/models/federated_dp/federated_X_train.npy"),
        ("models/RF/FL+DP/federated_y_train.npy", "output/models/federated_dp/federated_y_train.npy"),
        ("models/RF/FL+DP/federated_X_test.npy", "output/models/federated_dp/federated_X_test.npy"),
        ("models/RF/FL+DP/federated_y_test.npy", "output/models/federated_dp/federated_y_test.npy"),
        ("models/RF/FL+DP/fl_dp_metrics.json", "output/results/federated_dp/fl_dp_metrics.json"),
    ]
    
    # Move all files
    all_files = naive_files + centralized_files + federated_files + federated_dp_files
    
    for src, dst in all_files:
        if os.path.exists(src):
            shutil.move(src, dst)
            logger.debug(f"Moved {src} -> {dst}")


def organize_result_files():
    """Move attack result files to organized structure"""
    logger = logging.getLogger(__name__)
    
    # Organize MIA attack results (from specialized scripts)
    mia_mappings = [
        # Naive model MIA results
        ("models/RF/Naive/attack_results", "output/results/naive", "naive"),
        # Federated model MIA results  
        ("models/RF/FL/attack_results", "output/results/federated", "federated"),
        # Federated DP model MIA results
        ("models/RF/FL+DP/attack_results", "output/results/federated_dp", "federated_dp"),
    ]
    
    for source_dir, target_dir, model_name in mia_mappings:
        if os.path.exists(source_dir):
            organize_attack_results(source_dir, target_dir, model_name)
    
    # Organize inversion attack results
    if os.path.exists("results/inversion_attack"):
        organize_inversion_results()
    
    # Organize AIA attack results
    if os.path.exists("results/attribute_inference"):
        organize_aia_results()


def organize_attack_results(source_dir, target_dir, model_name):
    """Organize MIA attack results with proper naming"""
    logger = logging.getLogger(__name__)
    
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        
        if filename.endswith(".json"):
            target_path = os.path.join(target_dir, f"mia_results_{model_name}.json")
        elif filename.endswith(".png"):
            target_path = os.path.join(target_dir, f"mia_roc_{model_name}.png")
        else:
            target_path = os.path.join(target_dir, filename)
        
        shutil.move(source_path, target_path)
        logger.debug(f"Moved {source_path} -> {target_path}")


def organize_inversion_results():
    """Organize inversion attack results"""
    logger = logging.getLogger(__name__)
    
    source_dir = "results/inversion_attack"
    mappings = {
        "naive": "output/results/naive",
        "federated_+_dp": "output/results/federated_dp",  # Handle + in filename
        "federated": "output/results/federated"
    }
    
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        
        # Determine target based on filename content
        if "naive" in filename:
            target_dir = mappings["naive"]
            target_filename = filename
        elif "federated_+_dp" in filename:
            target_dir = mappings["federated_+_dp"] 
            target_filename = filename.replace("federated_+_dp", "federated_dp")
        elif "federated" in filename:
            target_dir = mappings["federated"]
            target_filename = filename
        else:
            continue  # Skip unknown files
        
        target_path = os.path.join(target_dir, target_filename)
        shutil.move(source_path, target_path)
        logger.debug(f"Moved {source_path} -> {target_path}")


def organize_aia_results():
    """Organize attribute inference attack results"""
    logger = logging.getLogger(__name__)
    
    source_dir = "results/attribute_inference"
    mappings = {
        "naive": "output/results/naive",
        "federated_+_dp": "output/results/federated_dp",  # Handle + in filename  
        "federated": "output/results/federated"
    }
    
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        
        # Determine target based on filename content
        if "naive" in filename:
            target_dir = mappings["naive"]
            target_filename = filename
        elif "federated_+_dp" in filename:
            target_dir = mappings["federated_+_dp"]
            target_filename = filename.replace("federated_+_dp", "federated_dp")  
        elif "federated" in filename:
            target_dir = mappings["federated"]
            target_filename = filename
        else:
            continue  # Skip unknown files
        
        target_path = os.path.join(target_dir, target_filename)
        shutil.move(source_path, target_path)
        logger.debug(f"Moved {source_path} -> {target_path}")


def generate_summary():
    """Generate experiment summary"""
    logger = logging.getLogger(__name__)
    logger.info("[8/8] Generating experiment summary...")
    
    try:
        summary = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "pipeline_version": "1.0",
                "models_trained": ["naive", "federated", "federated_dp"],
                "attacks_executed": ["mia", "inversion", "aia"]
            },
            "models": {},
            "attacks": {}
        }
        
        # Collect model metrics
        collect_model_metrics(summary)
        
        # Collect attack metrics  
        collect_attack_metrics(summary)
        
        # Save summary
        with open("output/summary/experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("SUCCESS: Experiment summary generated")
        return True
        
    except Exception as e:
        logger.error(f"FAILED: Summary generation failed: {e}")
        return False


def collect_model_metrics(summary):
    """Collect model performance metrics"""
    model_metrics_files = [
        ("output/results/naive/centralized_metrics.json", "naive"),
        ("output/results/federated/fl_metrics.json", "federated"),
        ("output/results/federated_dp/fl_dp_metrics.json", "federated_dp")
    ]
    
    for metrics_file, model_name in model_metrics_files:
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                summary["models"][model_name] = json.load(f)


def collect_attack_metrics(summary):
    """Collect attack performance metrics"""
    attack_files = [
        ("output/results/naive/mia_results_naive.json", "naive", "mia"),
        ("output/results/federated/mia_results_federated.json", "federated", "mia"),
        ("output/results/federated_dp/mia_results_federated_dp.json", "federated_dp", "mia")
    ]
    
    for attack_file, model_name, attack_type in attack_files:
        if os.path.exists(attack_file):
            with open(attack_file, "r") as f:
                if model_name not in summary["attacks"]:
                    summary["attacks"][model_name] = {}
                summary["attacks"][model_name][attack_type] = json.load(f)


def save_timing_report():
    """Save timing report to file"""
    logger = logging.getLogger(__name__)
    
    timing_report = {
        "timestamp": datetime.now().isoformat(),
        "total_time": sum(timing_data.values()),
        "steps": timing_data
    }
    
    with open("output/summary/timing_report.json", "w") as f:
        json.dump(timing_report, f, indent=2)
    
    # Also print timing summary
    logger.info("\n" + "=" * 60)
    logger.info("TIMING SUMMARY")
    logger.info("=" * 60)
    total_time = sum(timing_data.values())
    logger.info(f"Total execution time: {total_time:.2f}s")
    for step, time_taken in timing_data.items():
        logger.info(f"{step}: {time_taken:.2f}s")
    logger.info("=" * 60)


def run_full_pipeline():
    """Run the complete PPML experiment pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("Starting PPML Experiment Pipeline")
    logger.info("=" * 60)
    
    # Create output structure
    create_output_structure()
    
    steps = [
        ("Training", [
            run_naive_training,
            run_federated_training, 
            run_federated_dp_training
        ]),
        ("Attacks", [
            run_mia_attacks,
            run_inversion_attacks,
            run_aia_attacks
        ]),
        ("Organization", [
            organize_outputs,
            generate_summary
        ])
    ]
    
    total_success = True
    
    for phase_name, phase_functions in steps:
        logger.info(f"\n--- {phase_name} Phase ---")
        for func in phase_functions:
            success = func()
            if not success:
                logger.warning(f"Step {func.__name__} failed, but continuing...")
                total_success = False
    
    logger.info("\n" + "=" * 60)
    if total_success:
        logger.info("SUCCESS: Pipeline completed successfully!")
    else:
        logger.warning("WARNING: Pipeline completed with some errors - check logs")
    
    logger.info("Results available in: output/")
    logger.info("Summary available in: output/summary/experiment_summary.json")
    
    # Save timing report
    save_timing_report()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PPML Experiment Pipeline")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--train-models", action="store_true", help="Train models only")
    parser.add_argument("--run-attacks", action="store_true", help="Run attacks only")
    parser.add_argument("--organize-only", action="store_true", help="Organize outputs only")
    parser.add_argument("--clean", action="store_true", help="Clean output directory first")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Clean outputs if requested
    if args.clean:
        if os.path.exists("output"):
            shutil.rmtree("output")
            logger.info("Cleaned output directory")
    
    # Execute based on arguments
    if args.all or (not any([args.train_models, args.run_attacks, args.organize_only])):
        run_full_pipeline()
    elif args.train_models:
        create_output_structure()
        run_naive_training()
        run_federated_training()
        run_federated_dp_training()
    elif args.run_attacks:
        create_output_structure()
        run_mia_attacks()
        run_inversion_attacks()
        run_aia_attacks()
    elif args.organize_only:
        create_output_structure()
        organize_outputs()
        generate_summary()


if __name__ == "__main__":
    main()