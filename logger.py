#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified logging system for LSNs experiments.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class LSNsLogger:
    """Unified logger for LSNs experiments with file and console output."""
    
    def __init__(
        self,
        name: str = "LSNs",
        log_level: str = "INFO",
        log_dir: Optional[Path] = None,
        save_logs: bool = True
    ):
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = log_dir
        self.save_logs = save_logs
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if self.save_logs and self.log_dir:
            self.log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
    
    def log_experiment_start(self, config: dict) -> None:
        """Log experiment start with configuration."""
        self.info("=" * 60)
        self.info("LSNs EXPERIMENT STARTED")
        self.info("=" * 60)
        self.info(f"Configuration: {config}")
        self.info("=" * 60)
    
    def log_experiment_end(self, results: dict) -> None:
        """Log experiment end with results summary."""
        self.info("=" * 60)
        self.info("LSNs EXPERIMENT COMPLETED")
        self.info("=" * 60)
        self.info(f"Results summary: {results}")
        self.info("=" * 60)
    
    def log_progress(self, current: int, total: int, description: str = "Progress") -> None:
        """Log progress with percentage."""
        percentage = (current / total) * 100
        self.info(f"{description}: {current}/{total} ({percentage:.1f}%)")


# Global logger instance
_logger = None


def get_logger(
    name: str = "LSNs",
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    save_logs: bool = True
) -> LSNsLogger:
    """Get or create a global logger instance."""
    global _logger
    if _logger is None:
        _logger = LSNsLogger(name, log_level, log_dir, save_logs)
    return _logger


def setup_logging(
    name: str = "LSNs",
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    save_logs: bool = True
) -> LSNsLogger:
    """Setup and return a new logger instance."""
    return LSNsLogger(name, log_level, log_dir, save_logs) 