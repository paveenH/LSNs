#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for LSNs experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Configuration for model loading and setup."""
    name: str
    path: str
    device: Optional[str] = None
    torch_dtype: str = "float32"
    device_map: Optional[str] = None
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""
    network_type: Literal["language", "theory-of-mind", "multiple-demand"] = "language"
    batch_size: int = 8
    max_length: Optional[int] = None
    num_workers: int = 0


@dataclass
class AnalysisConfig:
    """Configuration for analysis methods."""
    method: Literal["ttest", "nmd", "mean"] = "ttest"
    percentage: float = 5.0
    localize_range: str = "100-100"
    pooling: Literal["last", "mean", "sum", "orig"] = "last"
    seed: int = 42
    fdr_correction: bool = True


@dataclass
class ExperimentConfig:
    """Main configuration for LSNs experiments."""
    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("results"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    overwrite: bool = False
    
    # Logging configuration
    log_level: str = "INFO"
    save_logs: bool = True
    
    def __post_init__(self):
        """Ensure directories exist and convert strings to Paths."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to dataclass instances
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        if 'data' in config_dict:
            config_dict['data'] = DataConfig(**config_dict['data'])
        if 'analysis' in config_dict:
            config_dict['analysis'] = AnalysisConfig(**config_dict['analysis'])
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': {
                'name': self.model.name,
                'path': self.model.path,
                'device': self.model.device,
                'torch_dtype': self.model.torch_dtype,
                'device_map': self.model.device_map,
                'trust_remote_code': self.model.trust_remote_code
            },
            'data': {
                'network_type': self.data.network_type,
                'batch_size': self.data.batch_size,
                'max_length': self.data.max_length,
                'num_workers': self.data.num_workers
            },
            'analysis': {
                'method': self.analysis.method,
                'percentage': self.analysis.percentage,
                'localize_range': self.analysis.localize_range,
                'pooling': self.analysis.pooling,
                'seed': self.analysis.seed,
                'fdr_correction': self.analysis.fdr_correction
            },
            'output_dir': str(self.output_dir),
            'cache_dir': str(self.cache_dir),
            'overwrite': self.overwrite,
            'log_level': self.log_level,
            'save_logs': self.save_logs
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_experiment_name(self) -> str:
        """Generate a unique experiment name based on configuration."""
        return f"{self.model.name}_{self.data.network_type}_{self.analysis.method}_{self.analysis.pooling}"
    
    def get_cache_path(self, file_type: str) -> Path:
        """Generate cache file path for different file types."""
        experiment_name = self.get_experiment_name()
        return self.cache_dir / f"{experiment_name}_{file_type}.npy" 