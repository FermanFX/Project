"""
Logging utilities for experiment tracking.

Saves experiment runs to the starter_pack/results/ folder structure:
- logs/      : Lightweight experiment logs
- metrics/   : Per-run metrics and performance scores
- statistics/: Aggregated statistics across seeds
- tables/    : Summary comparison tables
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class ExperimentLogger:
    """
    Centralized logging for experiments with structured output.

    Tracks:
    - Experiment metadata and configuration
    - Per-epoch training metrics
    - Final run results
    """

    def __init__(self, experiment_name: str, run_id: Optional[str] = None,
                 results_dir: str = 'starter_pack/results'):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment (e.g., 'linear_gaussian', 'digits')
            run_id: Unique run identifier. If None, auto-generated from timestamp.
            results_dir: Base results directory
        """
        self.experiment_name = experiment_name
        self.run_id = run_id or self._generate_run_id()
        self.results_dir = results_dir

        # Create directory structure
        self.logs_dir = os.path.join(results_dir, 'logs')
        self.metrics_dir = os.path.join(results_dir, 'metrics')
        self.statistics_dir = os.path.join(results_dir, 'statistics')
        self.tables_dir = os.path.join(results_dir, 'tables')

        for d in [self.logs_dir, self.metrics_dir, self.statistics_dir, self.tables_dir]:
            os.makedirs(d, exist_ok=True)

        # Initialize log data storage
        self.config = {}
        self.metrics_history = []  # List of epoch metrics
        self.final_metrics = {}

        # Setup file logging
        self.log_file = os.path.join(self.logs_dir, f'exp_{self.run_id}.log')
        self._setup_file_logger()

    @staticmethod
    def _generate_run_id() -> str:
        """Generate unique run ID from timestamp."""
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def _setup_file_logger(self) -> None:
        """Configure Python logger to write to file."""
        self.logger = logging.getLogger(self.run_id)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log experiment configuration.

        Args:
            config: Dictionary of hyperparameters and settings
        """
        self.config = config
        self.logger.info("=" * 60)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info("=" * 60)
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_acc: float, val_acc: float) -> None:
        """
        Log per-epoch training metrics.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_acc: Training accuracy
            val_acc: Validation accuracy
        """
        metrics = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'train_acc': float(train_acc),
            'val_acc': float(val_acc)
        }
        self.metrics_history.append(metrics)

        self.logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

    def log_result(self, model_name: str, test_loss: float, test_acc: float,
                   seed: int, additional_metrics: Optional[Dict] = None) -> None:
        """
        Log final results for a model.

        Args:
            model_name: Name of the model (e.g., 'softmax', 'nn')
            test_loss: Test loss
            test_acc: Test accuracy
            seed: Random seed used
            additional_metrics: Optional dict of additional metrics
        """
        result = {
            'model': model_name,
            'seed': seed,
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'timestamp': datetime.now().isoformat()
        }

        if additional_metrics:
            result.update(additional_metrics)

        self.final_metrics[model_name] = result

        self.logger.info(f"Final Results - {model_name.upper()}")
        self.logger.info(f"  Seed: {seed}")
        self.logger.info(f"  Test Loss: {test_loss:.4f}")
        self.logger.info(f"  Test Accuracy: {test_acc:.4f}")
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.logger.info(f"  {key}: {value}")

    def save_metrics(self) -> str:
        """
        Save metrics to metrics/ folder.

        Returns:
            Path to saved metrics file
        """
        metrics_file = os.path.join(
            self.metrics_dir,
            f'{self.experiment_name}_{self.run_id}_metrics.json'
        )

        data = {
            'run_id': self.run_id,
            'experiment': self.experiment_name,
            'config': self.config,
            'history': self.metrics_history,
            'final_metrics': self.final_metrics,
            'timestamp': datetime.now().isoformat()
        }

        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Metrics saved to {metrics_file}")
        return metrics_file

    def log_message(self, message: str, level: str = 'info') -> None:
        """
        Log arbitrary message.

        Args:
            message: Message to log
            level: Log level ('info', 'warning', 'error')
        """
        getattr(self.logger, level)(message)


def _json_safe(obj: Any) -> Any:
    """Convert NumPy and other non-JSON scalars to JSON-safe types."""
    try:
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_json_safe(v) for v in obj)
        if isinstance(obj, np.ndarray):
            return [_json_safe(v) for v in obj.tolist()]
        if hasattr(obj, 'item') and not isinstance(obj, (str, bytes, bytearray)):
            value = obj.item()
            if isinstance(value, (np.generic, )):
                return _json_safe(value)
            return value
        if isinstance(obj, (np.generic,)):
            return obj.item()
    except Exception:
        pass
    return obj


def save_comparison_table(results: Dict[str, Dict],
                          filename: str = 'comparison_results.json',
                          results_dir: str = 'starter_pack/results') -> str:
    """
    Save comparison table of multiple models/experiments.

    Args:
        results: Dictionary of results by model/experiment
        filename: Output filename
        results_dir: Base results directory

    Returns:
        Path to saved file
    """
    output_path = os.path.join(results_dir, 'tables', filename)

    with open(output_path, 'w') as f:
        json.dump(_json_safe(results), f, indent=2)

    print(f"Comparison table saved to {output_path}")
    return output_path


def save_statistics(stats: Dict[str, Any],
                    filename: str = 'seed_statistics.json',
                    results_dir: str = 'starter_pack/results') -> str:
    """
    Save aggregated statistics across seeds.

    Args:
        stats: Dictionary of statistics
        filename: Output filename
        results_dir: Base results directory

    Returns:
        Path to saved file
    """
    output_path = os.path.join(results_dir, 'statistics', filename)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved to {output_path}")
    return output_path
