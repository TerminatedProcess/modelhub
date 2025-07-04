#!/usr/bin/env python3
"""
Configuration management for ModelHub CLI
Handles YAML config loading and default creation
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration data structure"""
    model_hub_path: str
    page_size: int
    file_extensions: List[str]
    default_deploy_targets: List[Dict[str, Any]]

class ConfigManager:
    """Manages configuration loading and creation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = None
        
    def load_config(self) -> Config:
        """Load configuration from file, create default if missing"""
        if not self.config_path.exists():
            self.create_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            self.config = Config(
                model_hub_path=config_data.get('model_hub', {}).get('path', './model-hub'),
                page_size=config_data.get('model_hub', {}).get('page_size', 20),
                file_extensions=config_data.get('file_extensions', ['.safetensors', '.ckpt', '.pth', '.pt']),
                default_deploy_targets=config_data.get('default_deploy_targets', [])
            )
            
            # Store raw config data for classifier
            self.raw_config = config_data
            
            return self.config
            
        except Exception as e:
            raise Exception(f"Error loading config: {e}")
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get raw config dictionary for classifier"""
        if not hasattr(self, 'raw_config'):
            self.load_config()
        return self.raw_config
    
    def create_default_config(self):
        """Create default configuration file"""
        default_config = {
            'model_hub': {
                'path': './model-hub',
                'page_size': 1000
            },
            'file_extensions': [
                '.safetensors',
                '.ckpt', 
                '.pth',
                '.pt',
                '.gguf'
            ],
            'classification': {
                'confidence_threshold': 0.5,
                'enable_external_apis': True,
                'reclassify_civitai_models': True,
                'manual_overrides': {
                    # Example: "model_filename.safetensors": "primary_type/sub_type"
                },
                'size_rules': {
                    'checkpoint': {'min': 1_000_000_000, 'max': 50_000_000_000},
                    'lora': {'min': 1_000_000, 'max': 2_000_000_000},
                    'vae': {'min': 50_000_000, 'max': 2_000_000_000},
                    'controlnet': {'min': 100_000_000, 'max': 10_000_000_000},
                    'clip': {'min': 10_000_000, 'max': 2_000_000_000},
                    'text_encoder': {'min': 10_000_000, 'max': 25_000_000_000},
                    'unet': {'min': 500_000_000, 'max': 30_000_000_000},
                    'gguf': {'min': 100_000_000, 'max': 200_000_000_000},
                    'upscaler': {'min': 1_000_000, 'max': 500_000_000},
                    'embedding': {'min': 1_000, 'max': 50_000_000},
                    'video_model': {'min': 100_000_000, 'max': 100_000_000_000}
                }
            },
            'default_deploy_targets': [
                {
                    'name': 'comfyui',
                    'display_name': 'ComfyUI',
                    'base_path': '~/comfy/ComfyUI/models',
                    'enabled': False
                },
                {
                    'name': 'wan',
                    'display_name': 'WAN Video',
                    'base_path': '~/pinokio/api/wan.git/app',
                    'enabled': False
                },
                {
                    'name': 'forge',
                    'display_name': 'Forge WebUI',
                    'base_path': '~/pinokio/api/stable-diffusion-webui-forge.git/app/models',
                    'enabled': False
                },
                {
                    'name': 'image_upscale',
                    'display_name': 'Image Upscale',
                    'base_path': '~/pinokio/api/Image-Upscale.git/models',
                    'enabled': False
                }
            ]
        }
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
            print(f"Created default config file: {self.config_path}")
        except Exception as e:
            raise Exception(f"Error creating default config: {e}")
    
    def get_model_hub_path(self) -> Path:
        """Get the model hub path as a Path object"""
        if not self.config:
            self.load_config()
        return Path(self.config.model_hub_path).expanduser().resolve()
    
    def get_db_path(self) -> Path:
        """Get the database path"""
        return self.get_model_hub_path() / "modelhub.db"