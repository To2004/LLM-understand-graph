"""
Configuration file loader
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    TODO: Team Member Assignment - [UTILS TEAM]
    
    TODO [UTILS-006]:
        - Read YAML configuration file
        - Validate required fields
        - Merge with default config
        - Support environment variable substitution
        - Handle missing files gracefully
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # TODO: Implement config loading
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # TODO: Add validation and defaults
    return config
