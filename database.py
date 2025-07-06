#!/usr/bin/env python3
"""
Database operations for ModelHub CLI
Handles all database interactions for models and deployment
"""

import sqlite3
import os
import hashlib
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Model:
    """Model data structure"""
    id: int
    file_hash: str
    filename: str
    file_size: int
    file_extension: str
    primary_type: str
    sub_type: str
    confidence: float
    classification_method: str
    tensor_count: Optional[int]
    architecture: Optional[str]
    precision: Optional[str]
    quantization: Optional[str]
    triggers: Optional[str]
    filename_score: float
    size_score: float
    metadata_score: float
    tensor_score: float
    classified_at: str
    created_at: str
    updated_at: str
    reclassify: str
    deleted: bool
    debug_info: Optional[str]

@dataclass
class DeployTarget:
    """Deploy target data structure"""
    id: int
    name: str
    display_name: str
    base_path: str
    enabled: bool
    created_at: str
    updated_at: str

@dataclass
class DeployMapping:
    """Deploy mapping data structure"""
    id: int
    target_id: int
    model_type: str
    folder_path: str

@dataclass
class DeployLink:
    """Deploy link data structure"""
    id: int
    model_id: int
    target_id: int
    source_path: str
    deploy_path: str
    is_deployed: bool
    deployed_at: Optional[str]
    created_at: str

class ModelHubDB:
    """Database interface for ModelHub with dual database support"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path  # modelhub.db path
        self.classification_db_path = db_path.parent / "classification.db"
        self.conn = None  # modelhub.db connection
        self.class_conn = None  # classification.db connection
        self.ensure_databases_exist()
        
    def ensure_databases_exist(self):
        """Ensure both databases and model-hub directory exist, create if missing"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create modelhub.db if missing
        if not self.db_path.exists():
            self.create_modelhub_database()
            
        # Create classification.db if missing
        if not self.classification_db_path.exists():
            self.create_classification_database()
    
    def connect(self):
        """Connect to both databases"""
        try:
            # Connect to modelhub.db (model data)
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Connect to classification.db (rules and config)
            self.class_conn = sqlite3.connect(self.classification_db_path)
            self.class_conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise Exception(f"Database connection failed: {e}")
    
    def disconnect(self):
        """Disconnect from both databases"""
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.class_conn:
            self.class_conn.close()
            self.class_conn = None
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def create_modelhub_database(self):
        """Create modelhub.db with model data tables only"""
        print(f"Creating modelhub database at {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create original models table
            cursor.execute("""
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT UNIQUE NOT NULL,
                                            
                    -- Reclassification flag
                    reclassify TEXT DEFAULT '',
                    
                    -- Soft delete flag
                    deleted BOOLEAN DEFAULT FALSE,
                    
                    -- Debug information for classification troubleshooting
                    debug_info TEXT,
                    
                    filename TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_extension TEXT NOT NULL,
                    
                    -- Classification results
                    primary_type TEXT NOT NULL,
                    sub_type TEXT NOT NULL,
                    triggers TEXT,  -- Trigger words (denormalized for easy browsing)
                    confidence REAL NOT NULL,
                    classification_method TEXT NOT NULL,
                    
                    -- Technical metadata
                    tensor_count INTEGER,
                    architecture TEXT,
                    precision TEXT,
                    quantization TEXT,
                    
                    -- Classification scores
                    filename_score REAL DEFAULT 0.0,
                    size_score REAL DEFAULT 0.0,
                    metadata_score REAL DEFAULT 0.0,
                    tensor_score REAL DEFAULT 0.0,
                    
                    -- Timestamps
                    classified_at TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP 
                )
            """)
            
            
            cursor.execute("""
                CREATE TABLE model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    metadata_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE
                )
            """)
            
            # Create deploy tables
            cursor.execute("""
                CREATE TABLE deploy_targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,
                    base_path TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE deploy_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id INTEGER NOT NULL,
                    model_type TEXT NOT NULL,
                    folder_path TEXT NOT NULL,
                    FOREIGN KEY (target_id) REFERENCES deploy_targets (id) ON DELETE CASCADE,
                    UNIQUE(target_id, model_type)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE deploy_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    source_path TEXT NOT NULL,
                    deploy_path TEXT NOT NULL,
                    is_deployed BOOLEAN DEFAULT FALSE,
                    deployed_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES deploy_targets (id) ON DELETE CASCADE,
                    UNIQUE(model_id, target_id)
                )
            """)
            
            
            # Create indexes for modelhub tables
            cursor.execute("CREATE INDEX idx_models_hash ON models (file_hash)")
            cursor.execute("CREATE INDEX idx_models_type ON models (primary_type, sub_type)")
            cursor.execute("CREATE INDEX idx_models_triggers ON models (triggers)")
            cursor.execute("CREATE INDEX idx_metadata_model ON model_metadata (model_id)")
            cursor.execute("CREATE INDEX idx_metadata_key ON model_metadata (key)")
            cursor.execute("CREATE INDEX idx_deploy_targets_name ON deploy_targets (name)")
            cursor.execute("CREATE INDEX idx_deploy_mappings_target ON deploy_mappings (target_id)")
            cursor.execute("CREATE INDEX idx_deploy_mappings_type ON deploy_mappings (model_type)")
            cursor.execute("CREATE INDEX idx_deploy_links_model ON deploy_links (model_id)")
            cursor.execute("CREATE INDEX idx_deploy_links_target ON deploy_links (target_id)")
            cursor.execute("CREATE INDEX idx_deploy_links_deployed ON deploy_links (is_deployed)")
            
            conn.commit()
            print("✓ Created modelhub database tables and indexes")
            
            # Create default deploy data
            self._create_default_deploy_data(cursor)
            
            conn.commit()
            print("✓ Created default deployment configuration")
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error creating database: {e}")
        finally:
            conn.close()
    
    def create_classification_database(self):
        """Create classification.db with classification rules and configuration"""
        print(f"Creating classification database at {self.classification_db_path}")
        
        conn = sqlite3.connect(self.classification_db_path)
        cursor = conn.cursor()
        
        try:
            # Create classification rules tables
            cursor.execute("""
                CREATE TABLE model_types (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,
                    description TEXT,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE size_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    min_size INTEGER NOT NULL,
                    max_size INTEGER NOT NULL,
                    confidence_weight REAL DEFAULT 0.3,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_type)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE sub_type_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    primary_type TEXT NOT NULL,
                    sub_type TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    pattern_type TEXT DEFAULT 'filename',
                    confidence_weight REAL DEFAULT 0.8,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE architecture_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    architecture TEXT NOT NULL,
                    tensor_pattern TEXT NOT NULL,
                    confidence_weight REAL DEFAULT 0.9,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE external_apis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    priority INTEGER DEFAULT 1,
                    rate_limit_delay REAL DEFAULT 1.0,
                    timeout_seconds INTEGER DEFAULT 10,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # New data-driven classification tables
            cursor.execute("""
                CREATE TABLE detection_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT NOT NULL,
                    rule_type TEXT NOT NULL,           -- 'base_model', 'primary_type', 'sub_type', 'modifier'
                    source_type TEXT NOT NULL,         -- 'civitai_field', 'filename', 'metadata_field', 'tensor_pattern', 'file_size'
                    source_field TEXT,                 -- 'baseModel', 'ss_base_model_version', 'name'
                    pattern TEXT NOT NULL,             -- Regex or exact match pattern
                    output_value TEXT NOT NULL,        -- What this rule produces: 'flux', 'checkpoint', 'quantized'
                    confidence REAL DEFAULT 0.5,      -- How confident this rule is
                    priority INTEGER DEFAULT 1,       -- Rule execution order
                    fallback_rule_id INTEGER,         -- Chain to another rule if this fails
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (fallback_rule_id) REFERENCES detection_rules(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE classification_workflows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_name TEXT NOT NULL,      -- 'standard_classification', 'fast_classification'
                    step_order INTEGER NOT NULL,     -- 1, 2, 3...
                    rule_group TEXT NOT NULL,        -- 'base_model_detection', 'primary_type_detection'
                    required BOOLEAN DEFAULT FALSE,  -- Must this step succeed?
                    weight REAL DEFAULT 1.0,         -- How much this step contributes to final confidence
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE pattern_modifiers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    modifier_name TEXT NOT NULL,      -- 'quantization_detector', 'variant_detector'
                    pattern TEXT NOT NULL,            -- 'quantiz|fp8|gguf|int8|int4'
                    modifier_suffix TEXT NOT NULL,    -- '-quantized', '-dev', '-schnell'
                    applies_to_types TEXT,            -- 'checkpoint,lora' or 'all'
                    applies_to_subtypes TEXT,         -- 'flux,sdxl' or 'all'
                    priority INTEGER DEFAULT 1,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE external_api_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_name TEXT NOT NULL,           -- 'civitai', 'huggingface'
                    api_field TEXT NOT NULL,          -- 'type', 'baseModel', 'tags'
                    api_value TEXT NOT NULL,          -- 'checkpoint', 'SDXL', 'LoRA'
                    modelhub_field TEXT NOT NULL,     -- 'primary_type', 'base_model', 'sub_type'
                    modelhub_value TEXT NOT NULL,     -- 'checkpoint', 'sdxl', 'lora'
                    confidence REAL DEFAULT 0.9,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE confidence_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,        -- 'civitai', 'tensor_analysis', 'filename', 'file_size', 'metadata'
                    field_type TEXT NOT NULL,         -- 'primary_type', 'sub_type', 'base_model'
                    base_weight REAL NOT NULL,        -- Base confidence contribution
                    quality_multiplier REAL DEFAULT 1.0, -- Adjust based on data quality
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for classification tables
            cursor.execute("CREATE INDEX idx_model_types_name ON model_types (name)")
            cursor.execute("CREATE INDEX idx_size_rules_type ON size_rules (model_type)")
            cursor.execute("CREATE INDEX idx_sub_type_rules_primary ON sub_type_rules (primary_type)")
            cursor.execute("CREATE INDEX idx_architecture_patterns_arch ON architecture_patterns (architecture)")
            cursor.execute("CREATE INDEX idx_external_apis_priority ON external_apis (priority)")
            
            # Create indexes for new data-driven tables
            cursor.execute("CREATE INDEX idx_detection_rules_type ON detection_rules (rule_type)")
            cursor.execute("CREATE INDEX idx_detection_rules_source ON detection_rules (source_type)")
            cursor.execute("CREATE INDEX idx_detection_rules_priority ON detection_rules (priority)")
            cursor.execute("CREATE INDEX idx_workflows_order ON classification_workflows (step_order)")
            cursor.execute("CREATE INDEX idx_pattern_modifiers_priority ON pattern_modifiers (priority)")
            cursor.execute("CREATE INDEX idx_api_mappings_api ON external_api_mappings (api_name)")
            cursor.execute("CREATE INDEX idx_confidence_weights_source ON confidence_weights (source_type)")
            
            conn.commit()
            print("✓ Created classification database tables and indexes")
            
            # Create default classification data
            self._create_default_classification_data(cursor)
            
            conn.commit()
            print("✓ Created default classification configuration")
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error creating classification database: {e}")
        finally:
            conn.close()
    
    def _create_default_deploy_data(self, cursor):
        """Create default deploy targets and mappings"""
        
        # Default deploy targets
        deploy_targets = [
            ('comfyui', 'ComfyUI', '~/comfy/ComfyUI/models', False),
            ('wan', 'WAN Video', '~/pinokio/api/wan.git/app', False),
            ('forge', 'Forge WebUI', '~/pinokio/api/stable-diffusion-webui-forge.git/app/models', False),
            ('image_upscale', 'Image Upscale', '~/pinokio/api/Image-Upscale.git/models', False)
        ]
        
        cursor.executemany("""
            INSERT INTO deploy_targets (name, display_name, base_path, enabled)
            VALUES (?, ?, ?, ?)
        """, deploy_targets)
        
        # Get target IDs for mappings
        cursor.execute("SELECT id, name FROM deploy_targets")
        target_ids = {name: id for id, name in cursor.fetchall()}
        
        # Default deploy mappings
        deploy_mappings = []
        
        # ComfyUI mappings - comprehensive coverage of all model types
        comfyui_mappings = [
            # Core diffusion models
            ('checkpoint', 'checkpoints'),
            ('lora', 'loras'),
            ('vae', 'vae'),
            ('controlnet', 'controlnet'),
            ('unet', 'unet'),
            
            # Text and vision encoders
            ('clip', 'clip'),
            ('clip_vision', 'clip_vision'),
            ('text_encoder', 'text_encoders'),
            
            # Specialized models
            ('embedding', 'embeddings'),
            ('upscaler', 'upscale_models'),
            ('hypernetwork', 'hypernetworks'),
            
            # Face and image processing
            ('facerestore', 'facerestore_models'),
            ('insightface', 'insightface'),
            ('photomaker', 'photomaker'),
            ('style_model', 'style_models'),
            
            # Advanced AI models
            ('diffuser', 'diffusers'),
            ('gligen', 'gligen'),
            ('grounding_dino', 'grounding-dino'),
            ('ultralytics', 'ultralytics'),
            ('sam', 'sams'),
            ('rmbg', 'RMBG'),
            
            # Lightweight models
            ('vae_approx', 'vae_approx'),
            
            # GGUF models - deploy to appropriate sub-folders based on content
            ('gguf', 'checkpoints'),  # Default, but should be determined by base model
            
            # Fallback
            ('unknown', 'unknown')
        ]
        for model_type, folder_path in comfyui_mappings:
            deploy_mappings.append((target_ids['comfyui'], model_type, folder_path))
        
        # WAN mappings
        wan_mappings = [
            ('checkpoint', 'ckpts'), ('lora', 'loras'), ('video_lora', 'loras_i2v'),
            ('hunyuan_lora', 'loras_hunyuan'), ('ltxv_lora', 'loras_ltxv'), ('unknown', 'ckpts')
        ]
        for model_type, folder_path in wan_mappings:
            deploy_mappings.append((target_ids['wan'], model_type, folder_path))
        
        # Forge mappings
        forge_mappings = [
            ('checkpoint', 'Stable-diffusion'), ('lora', 'Lora'), ('vae', 'VAE'),
            ('controlnet', 'ControlNet'), ('embedding', 'embeddings'),
            ('upscaler', 'ESRGAN'), ('hypernetwork', 'hypernetworks'),
            ('text_encoder', 'text_encoder'), ('unknown', 'Stable-diffusion')
        ]
        for model_type, folder_path in forge_mappings:
            deploy_mappings.append((target_ids['forge'], model_type, folder_path))
        
        # Image Upscale mappings
        image_upscale_mappings = [
            ('checkpoint', 'models/Stable-diffusion'), ('lora', 'Lora'), ('vae', 'VAE'),
            ('controlnet', 'ControlNet'), ('embedding', 'embeddings'),
            ('upscaler', 'upscalers'), ('unknown', 'models/Stable-diffusion')
        ]
        for model_type, folder_path in image_upscale_mappings:
            deploy_mappings.append((target_ids['image_upscale'], model_type, folder_path))
        
        cursor.executemany("""
            INSERT INTO deploy_mappings (target_id, model_type, folder_path)
            VALUES (?, ?, ?)
        """, deploy_mappings)
    
    def _create_default_classification_data(self, cursor):
        """Create default classification rules and data"""
        
        # Default model types - aligned with ComfyUI structure
        model_types = [
            # Core diffusion model types
            ('checkpoint', 'Checkpoint', 'Large diffusion model checkpoints'),
            ('lora', 'LoRA', 'Low-Rank Adaptation fine-tuning models'),
            ('vae', 'VAE', 'Variational Autoencoder models'),
            ('controlnet', 'ControlNet', 'Conditional control models'),
            ('unet', 'UNet', 'U-Net diffusion models'),
            ('gguf', 'GGUF', 'GGML Universal File format models'),
            
            # Text and vision encoders
            ('clip', 'CLIP', 'Text-image embedding models'),
            ('clip_vision', 'CLIP Vision', 'CLIP vision models'),
            ('text_encoder', 'Text Encoder', 'Text encoding models'),
            
            # Specialized model types
            ('embedding', 'Embedding', 'Textual inversion embeddings'),
            ('upscaler', 'Upscaler', 'Image upscaling models'),
            ('hypernetwork', 'Hypernetwork', 'Hypernetwork models'),
            
            # Face and image processing models
            ('facerestore', 'Face Restore', 'Face restoration models'),
            ('insightface', 'InsightFace', 'Face analysis and swapping models'),
            ('photomaker', 'PhotoMaker', 'PhotoMaker models'),
            ('style_model', 'Style Model', 'T2I style transfer models'),
            
            # Advanced AI models
            ('diffuser', 'Diffusers', 'Diffusers format models'),
            ('gligen', 'GLIGEN', 'Grounded language-image generation models'),
            ('grounding_dino', 'Grounding DINO', 'Object detection models'),
            ('ultralytics', 'Ultralytics', 'YOLO detection/segmentation models'),
            ('sam', 'SAM', 'Segment Anything Models'),
            ('rmbg', 'RMBG', 'Background removal models'),
            
            # Lightweight and approximation models
            ('vae_approx', 'VAE Approx', 'Lightweight VAE approximation models'),
            
            # Legacy and unknown
            ('unknown', 'Unknown', 'Unclassified models')
        ]
        
        cursor.executemany("""
            INSERT INTO model_types (name, display_name, description)
            VALUES (?, ?, ?)
        """, model_types)
        
        # Default size rules (in bytes) - comprehensive ComfyUI model coverage
        size_rules = [
            # Core diffusion models
            ('checkpoint', 1_000_000_000, 50_000_000_000, 0.3),     # 1GB-50GB
            ('lora', 1_000_000, 2_000_000_000, 0.4),                # 1MB-2GB
            ('vae', 50_000_000, 2_000_000_000, 0.4),                # 50MB-2GB
            ('controlnet', 100_000_000, 10_000_000_000, 0.3),       # 100MB-10GB
            ('unet', 500_000_000, 30_000_000_000, 0.3),             # 500MB-30GB
            ('gguf', 100_000_000, 200_000_000_000, 0.2),            # 100MB-200GB
            
            # Text and vision encoders
            ('clip', 10_000_000, 2_000_000_000, 0.3),               # 10MB-2GB
            ('clip_vision', 10_000_000, 1_000_000_000, 0.3),        # 10MB-1GB
            ('text_encoder', 10_000_000, 25_000_000_000, 0.3),      # 10MB-25GB
            
            # Specialized models
            ('embedding', 1_000, 50_000_000, 0.5),                  # 1KB-50MB
            ('upscaler', 1_000_000, 500_000_000, 0.4),              # 1MB-500MB
            ('hypernetwork', 1_000_000, 200_000_000, 0.4),          # 1MB-200MB
            
            # Face and image processing
            ('facerestore', 10_000_000, 1_000_000_000, 0.4),        # 10MB-1GB
            ('insightface', 100_000_000, 1_000_000_000, 0.4),       # 100MB-1GB
            ('photomaker', 50_000_000, 2_000_000_000, 0.4),         # 50MB-2GB
            ('style_model', 10_000_000, 500_000_000, 0.4),          # 10MB-500MB
            
            # Advanced AI models
            ('diffuser', 500_000_000, 50_000_000_000, 0.3),         # 500MB-50GB
            ('gligen', 100_000_000, 5_000_000_000, 0.3),            # 100MB-5GB
            ('grounding_dino', 100_000_000, 2_000_000_000, 0.4),    # 100MB-2GB
            ('ultralytics', 1_000_000, 500_000_000, 0.4),           # 1MB-500MB
            ('sam', 100_000_000, 10_000_000_000, 0.4),              # 100MB-10GB
            ('rmbg', 1_000_000, 500_000_000, 0.4),                  # 1MB-500MB
            
            # Lightweight models
            ('vae_approx', 1_000_000, 100_000_000, 0.5),            # 1MB-100MB
        ]
        
        cursor.executemany("""
            INSERT INTO size_rules (model_type, min_size, max_size, confidence_weight)
            VALUES (?, ?, ?, ?)
        """, size_rules)
        
        # Default sub-type rules (filename patterns) - aligned with ComfyUI base models
        sub_type_rules = [
            # Checkpoint sub-types - ordered by specificity
            ('checkpoint', 'flux', 'flux', 'filename', 0.9),
            ('checkpoint', 'sd3', 'sd3', 'filename', 0.8),
            ('checkpoint', 'sdxl', 'xl|sdxl', 'filename', 0.8),
            ('checkpoint', 'wan', 'wan|wanvideo', 'filename', 0.8),
            ('checkpoint', 'pony', 'pony', 'filename', 0.8),
            ('checkpoint', 'sd15', '1\\.5|v1-5|v15', 'filename', 0.7),
            ('checkpoint', 'upscale', 'upscal|realesrgan|esrgan', 'filename', 0.8),
            ('checkpoint', 'unknown', '.*', 'filename', 0.1),  # Default fallback
            
            # LoRA sub-types - ordered by specificity  
            ('lora', 'flux', 'flux', 'filename', 0.9),
            ('lora', 'sd3', 'sd3', 'filename', 0.8),
            ('lora', 'sdxl', 'xl|sdxl', 'filename', 0.8),
            ('lora', 'pony', 'pony', 'filename', 0.8),
            ('lora', 'wan', 'wan|wanvideo', 'filename', 0.8),
            ('lora', 'sd15', '1\\.5|v1-5|v15', 'filename', 0.7),
            ('lora', 'unknown', '.*', 'filename', 0.1),  # Default fallback
            
            # ControlNet sub-types
            ('controlnet', 'flux', 'flux', 'filename', 0.9),
            ('controlnet', 'sd3', 'sd3', 'filename', 0.8),
            ('controlnet', 'sdxl', 'xl|sdxl', 'filename', 0.8),
            ('controlnet', 'wan', 'wan|wanvideo', 'filename', 0.8),
            ('controlnet', 'sd15', '1\\.5|v1-5|v15', 'filename', 0.7),
            ('controlnet', 'unknown', '.*', 'filename', 0.1),  # Default fallback
            
            # VAE sub-types
            ('vae', 'video', 'video', 'filename', 0.9),
            ('vae', 'flux', 'flux', 'filename', 0.8),
            ('vae', 'sd3', 'sd3', 'filename', 0.8),
            ('vae', 'sdxl', 'xl|sdxl', 'filename', 0.8),
            ('vae', 'sd15', '1\\.5|v1-5|v15', 'filename', 0.7),
            ('vae', 'unknown', '.*', 'filename', 0.1),  # Default fallback
            
            # GGUF sub-types - map to base model for deployment
            ('gguf', 'quantized_model', '.*', 'filename', 0.9),  # All GGUF are quantized
            
            # Ultralytics sub-types for YOLO models
            ('ultralytics', 'bbox', 'detect|bbox|yolo.*detect', 'filename', 0.8),
            ('ultralytics', 'segm', 'segment|segm|yolo.*seg', 'filename', 0.8),
            ('ultralytics', 'unknown', '.*', 'filename', 0.1),  # Default fallback
            
            # Generic sub-types for specialized models (most don't need sub-categorization)
            ('facerestore', 'standard', '.*', 'filename', 0.5),
            ('insightface', 'standard', '.*', 'filename', 0.5),
            ('photomaker', 'standard', '.*', 'filename', 0.5),
            ('style_model', 'standard', '.*', 'filename', 0.5),
            ('diffuser', 'standard', '.*', 'filename', 0.5),
            ('gligen', 'standard', '.*', 'filename', 0.5),
            ('grounding_dino', 'standard', '.*', 'filename', 0.5),
            ('sam', 'standard', '.*', 'filename', 0.5),
            ('rmbg', 'standard', '.*', 'filename', 0.5),
            ('vae_approx', 'standard', '.*', 'filename', 0.5),
            ('clip', 'standard', '.*', 'filename', 0.5),
            ('clip_vision', 'standard', '.*', 'filename', 0.5),
            ('text_encoder', 'standard', '.*', 'filename', 0.5),
            ('unet', 'standard', '.*', 'filename', 0.5),
            ('embedding', 'standard', '.*', 'filename', 0.5),
            ('upscaler', 'standard', '.*', 'filename', 0.5),
            ('hypernetwork', 'standard', '.*', 'filename', 0.5),
        ]
        
        cursor.executemany("""
            INSERT INTO sub_type_rules (primary_type, sub_type, pattern, pattern_type, confidence_weight)
            VALUES (?, ?, ?, ?, ?)
        """, sub_type_rules)
        
        # Default architecture patterns (tensor analysis) - comprehensive detection
        architecture_patterns = [
            # Core model architecture patterns
            ('lora', '\\.lora_up\\.|lora_down\\.|alpha$|hada_w1_|hada_w2_', 0.95),
            ('controlnet', 'control_model|controlnet|input_hint_block', 0.95),
            ('vae', 'encoder\\.|decoder\\.|autoencoder|quant_conv|post_quant_conv', 0.9),
            ('unet', 'diffusion_model|unet|input_blocks|middle_block|output_blocks', 0.9),
            
            # Advanced architecture patterns
            ('flux', 'double_blocks|single_blocks|img_attn|txt_attn|guidance_in', 0.95),
            ('video_vae', 'downsamples.*residual|upsamples.*residual|time_embedding', 0.95),
            ('clip', 'text_model|text_projection|visual|transformer\\.resblocks', 0.85),
            ('clip_vision', 'vision_model|visual\\.transformer|patch_embedding', 0.9),
            ('text_encoder', 'encoder\\.block|encoder\\.embed_tokens|shared\\.weight', 0.85),
            
            # Specialized model patterns
            ('sam', 'image_encoder|mask_decoder|prompt_encoder', 0.95),
            ('grounding_dino', 'bbox_embed|class_embed|query_embed', 0.9),
            ('ultralytics', 'model\\.\\d+|detect|segment|classify', 0.85),
            ('insightface', 'fc1\\.weight|features\\.|landmark', 0.9),
            ('facerestore', 'generator|discriminator|face_', 0.85),
            
            # Upscaler and restoration patterns
            ('upscaler', 'upsampler|conv_up|pixel_shuffle|esrgan', 0.85),
            ('rmbg', 'backbone|decode_head|auxiliary_head', 0.85),
            
            # Embedding and style patterns
            ('embedding', 'string_to_param|emb_params|.*\\.bin', 0.9),
            ('hypernetwork', 'linear_\\d+|mlp\\.|hypernetwork', 0.85),
            ('style_model', 'style_proj|content_proj|adapter', 0.85),
            
            # Diffuser format patterns
            ('diffuser', 'scheduler|tokenizer|feature_extractor', 0.8),
            
            # GGUF specific patterns (handled separately but included for completeness)
            ('gguf', 'blk\\.|attn_|ffn_|norm', 0.7),
        ]
        
        cursor.executemany("""
            INSERT INTO architecture_patterns (architecture, tensor_pattern, confidence_weight)
            VALUES (?, ?, ?)
        """, architecture_patterns)
        
        # Default external APIs
        external_apis = [
            ('civitai', True, 1, 1.0, 10)
        ]
        
        cursor.executemany("""
            INSERT INTO external_apis (name, enabled, priority, rate_limit_delay, timeout_seconds)
            VALUES (?, ?, ?, ?, ?)
        """, external_apis)
        
        # Default detection rules - migrate hard-coded logic to data-driven rules
        detection_rules = [
            # Base model detection rules from CivitAI
            (1, 'flux_civitai_base', 'base_model', 'civitai_field', 'baseModel', '(?i)flux', 'flux', 0.95, 1, None, True),
            (2, 'sdxl_civitai_base', 'base_model', 'civitai_field', 'baseModel', '(?i)sdxl|xl', 'sdxl', 0.95, 1, None, True),
            (3, 'sd3_civitai_base', 'base_model', 'civitai_field', 'baseModel', '(?i)sd\\s*3|sd3', 'sd3', 0.95, 1, None, True),
            (4, 'sd15_civitai_base', 'base_model', 'civitai_field', 'baseModel', '(?i)sd\\s*1\\.5|sd15', 'sd15', 0.95, 1, None, True),
            (5, 'sd2_civitai_base', 'base_model', 'civitai_field', 'baseModel', '(?i)sd\\s*2|sd2', 'sd2', 0.95, 1, None, True),
            (6, 'pony_civitai_base', 'base_model', 'civitai_field', 'baseModel', '(?i)pony', 'pony', 0.95, 1, None, True),
            (7, 'wan_civitai_base', 'base_model', 'civitai_field', 'baseModel', '(?i)wan|video', 'wan', 0.9, 1, None, True),
            
            # Base model detection from filename (fallback)
            (8, 'flux_filename_base', 'base_model', 'filename', None, '(?i)flux', 'flux', 0.8, 2, None, True),
            (9, 'sdxl_filename_base', 'base_model', 'filename', None, '(?i)xl|sdxl', 'sdxl', 0.8, 2, None, True),
            (10, 'sd3_filename_base', 'base_model', 'filename', None, '(?i)sd3', 'sd3', 0.8, 2, None, True),
            (11, 'pony_filename_base', 'base_model', 'filename', None, '(?i)pony', 'pony', 0.8, 2, None, True),
            (12, 'wan_filename_base', 'base_model', 'filename', None, '(?i)wan|wanvideo', 'wan', 0.8, 2, None, True),
            
            # Primary type fallback rules for unknown CivitAI types
            (13, 'flux_unknown_to_checkpoint', 'primary_type', 'base_model_conditional', 'flux', 'unknown', 'checkpoint', 0.7, 1, None, True),
            (14, 'pony_unknown_to_checkpoint', 'primary_type', 'base_model_conditional', 'pony', 'unknown', 'checkpoint', 0.8, 1, None, True),
            (15, 'sdxl_unknown_to_checkpoint', 'primary_type', 'base_model_conditional', 'sdxl', 'unknown', 'checkpoint', 0.7, 1, None, True),
        ]
        
        cursor.executemany("""
            INSERT INTO detection_rules (id, rule_name, rule_type, source_type, source_field, pattern, output_value, confidence, priority, fallback_rule_id, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, detection_rules)
        
        # Pattern modifiers for variants and quantization
        pattern_modifiers = [
            (1, 'flux_dev_variant', '(?i)dev(?!ice)', '-dev', 'checkpoint,lora,controlnet,vae', 'flux', 1, True),
            (2, 'flux_schnell_variant', '(?i)schnell', '-schnell', 'checkpoint,lora,controlnet,vae', 'flux', 1, True),
            (3, 'quantization_global', '(?i)quantiz|fp8|gguf|int8|int4', '-quantized', 'all', 'all', 1, True),
            (4, 'anime_style', '(?i)anime|manga|cartoon', '-anime', 'all', 'all', 2, True),
            (5, 'realism_style', '(?i)realism|realistic|photorealistic', '-realism', 'all', 'all', 2, True),
            (6, 'portrait_style', '(?i)portrait|headshot|face', '-portrait', 'all', 'all', 3, True),
            (7, 'video_variant', '(?i)i2v|text2video|t2v|motion', '-i2v', 'all', 'all', 1, True),
        ]
        
        cursor.executemany("""
            INSERT INTO pattern_modifiers (id, modifier_name, pattern, modifier_suffix, applies_to_types, applies_to_subtypes, priority, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, pattern_modifiers)
        
        # External API mappings (migrate hard-coded CivitAI type mapping)
        external_api_mappings = [
            (1, 'civitai', 'type', 'checkpoint', 'primary_type', 'checkpoint', 0.9, True),
            (2, 'civitai', 'type', 'textualinversion', 'primary_type', 'embedding', 0.9, True),
            (3, 'civitai', 'type', 'lora', 'primary_type', 'lora', 0.9, True),
            (4, 'civitai', 'type', 'lycoris', 'primary_type', 'lora', 0.9, True),
            (5, 'civitai', 'type', 'hypernetwork', 'primary_type', 'hypernetwork', 0.9, True),
            (6, 'civitai', 'type', 'controlnet', 'primary_type', 'controlnet', 0.9, True),
            (7, 'civitai', 'type', 'vae', 'primary_type', 'vae', 0.9, True),
            (8, 'civitai', 'type', 'poses', 'primary_type', 'pose', 0.9, True),
            (9, 'civitai', 'type', 'wildcards', 'primary_type', 'wildcard', 0.9, True),
            (10, 'civitai', 'type', 'workflows', 'primary_type', 'workflow', 0.9, True),
            (11, 'civitai', 'type', 'other', 'primary_type', 'unknown', 0.5, True),
        ]
        
        cursor.executemany("""
            INSERT INTO external_api_mappings (id, api_name, api_field, api_value, modelhub_field, modelhub_value, confidence, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, external_api_mappings)
        
        # Confidence weights (migrate hard-coded scoring weights)
        confidence_weights = [
            (1, 'civitai', 'primary_type', 1.0, 1.0, True),
            (2, 'civitai', 'base_model', 1.0, 1.0, True),
            (3, 'tensor_analysis', 'primary_type', 0.5, 1.0, True),
            (4, 'filename', 'primary_type', 0.2, 1.0, True),
            (5, 'filename', 'sub_type', 0.2, 1.0, True),
            (6, 'file_size', 'primary_type', 0.2, 1.0, True),
            (7, 'metadata', 'primary_type', 0.1, 1.0, True),
            (8, 'metadata', 'base_model', 0.3, 1.0, True),
        ]
        
        cursor.executemany("""
            INSERT INTO confidence_weights (id, source_type, field_type, base_weight, quality_multiplier, enabled)
            VALUES (?, ?, ?, ?, ?, ?)
        """, confidence_weights)
        
        # Classification workflow (define the standard classification process)
        classification_workflows = [
            (1, 'standard_classification', 1, 'external_api_lookup', False, 1.0, True),
            (2, 'standard_classification', 2, 'base_model_detection', False, 0.9, True),
            (3, 'standard_classification', 3, 'primary_type_detection', True, 1.0, True),
            (4, 'standard_classification', 4, 'sub_type_detection', False, 0.8, True),
            (5, 'standard_classification', 5, 'pattern_modifiers', False, 0.5, True),
            (6, 'standard_classification', 6, 'tensor_analysis', False, 0.5, True),
            (7, 'standard_classification', 7, 'file_size_fallback', False, 0.2, True),
        ]
        
        cursor.executemany("""
            INSERT INTO classification_workflows (id, workflow_name, step_order, rule_group, required, weight, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, classification_workflows)
    
    # Model operations
    def get_models(self, limit: int = 100, offset: int = 0, 
                   filter_type: Optional[str] = None, 
                   search_term: Optional[str] = None,
                   sort_by: str = "classified_at",
                   sort_order: str = "DESC") -> List[Model]:
        """Get models with filtering and pagination"""
        
        query = """
        SELECT id, file_hash, filename, file_size, file_extension,
               primary_type, sub_type, confidence, classification_method,
               tensor_count, architecture, precision, quantization,
               triggers, filename_score, size_score, metadata_score,
               tensor_score, classified_at, created_at, updated_at, reclassify, deleted, debug_info
        FROM models
        """
        
        params = []
        conditions = ["deleted = 0"]  # Always exclude deleted records
        
        if filter_type:
            conditions.append("primary_type = ?")
            params.append(filter_type)
        
        if search_term:
            conditions.append("(filename LIKE ? OR triggers LIKE ?)")
            params.extend([f"%{search_term}%", f"%{search_term}%"])
        
        query += " WHERE " + " AND ".join(conditions)
        
        query += f" ORDER BY {sort_by} {sort_order} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = self.conn.execute(query, params)
        models = []
        
        for row in cursor.fetchall():
            models.append(Model(**dict(row)))
        
        return models
    
    def get_model_count(self, filter_type: Optional[str] = None, 
                       search_term: Optional[str] = None) -> int:
        """Get total count of models with filters"""
        query = "SELECT COUNT(*) FROM models"
        params = []
        conditions = ["deleted = 0"]  # Always exclude deleted records
        
        if filter_type:
            conditions.append("primary_type = ?")
            params.append(filter_type)
        
        if search_term:
            conditions.append("(filename LIKE ? OR triggers LIKE ?)")
            params.extend([f"%{search_term}%", f"%{search_term}%"])
        
        query += " WHERE " + " AND ".join(conditions)
        
        cursor = self.conn.execute(query, params)
        return cursor.fetchone()[0]
    
    def get_model_types(self) -> List[Tuple[str, int]]:
        """Get model types with counts"""
        cursor = self.conn.execute("""
            SELECT primary_type, COUNT(*) as count 
            FROM models 
            WHERE deleted = 0
            GROUP BY primary_type 
            ORDER BY count DESC
        """)
        return cursor.fetchall()
    
    def get_model_by_id(self, model_id: int) -> Optional[Model]:
        """Get a specific model by ID"""
        cursor = self.conn.execute("""
            SELECT id, file_hash, filename, file_size, file_extension,
                   primary_type, sub_type, confidence, classification_method,
                   tensor_count, architecture, precision, quantization,
                   triggers, filename_score, size_score, metadata_score,
                   tensor_score, classified_at, created_at, updated_at, reclassify, deleted, debug_info
            FROM models WHERE id = ? AND deleted = 0
        """, (model_id,))
        
        row = cursor.fetchone()
        if row:
            return Model(**dict(row))
        return None
    
    def get_model_metadata(self, model_id: int) -> List[Tuple[str, str]]:
        """Get metadata for a specific model"""
        cursor = self.conn.execute("""
            SELECT key, value 
            FROM model_metadata 
            WHERE model_id = ?
            ORDER BY key
        """, (model_id,))
        return cursor.fetchall()
    
    def store_model_metadata(self, model_id: int, metadata_dict: Dict[str, Any]):
        """Store metadata key-value pairs for a model"""
        # Clear existing metadata for this model
        self.conn.execute("DELETE FROM model_metadata WHERE model_id = ?", (model_id,))
        
        # Insert new metadata
        for key, value in metadata_dict.items():
            # Convert value to JSON string if it's not already a string
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value)
            else:
                value_str = str(value)
            
            # Determine metadata type from key
            metadata_type = 'extracted'  # Default type
            if 'civitai' in key.lower():
                metadata_type = 'civitai'
            elif 'safetensors' in key.lower():
                metadata_type = 'safetensors'
            elif 'gguf' in key.lower():
                metadata_type = 'gguf'
            elif 'tensor' in key.lower():
                metadata_type = 'tensor_analysis'
            elif 'file' in key.lower():
                metadata_type = 'file_info'
            
            self.conn.execute("""
                INSERT INTO model_metadata (model_id, metadata_type, key, value)
                VALUES (?, ?, ?, ?)
            """, (model_id, metadata_type, key, value_str))
        
        self.conn.commit()
    
    def get_model_metadata_dict(self, model_id: int) -> Dict[str, str]:
        """Get metadata as a dictionary for a specific model"""
        metadata_pairs = self.get_model_metadata(model_id)
        return {key: value for key, value in metadata_pairs}
    
    # Deploy operations
    def get_deploy_targets(self) -> List[DeployTarget]:
        """Get all deploy targets"""
        cursor = self.conn.execute("""
            SELECT id, name, display_name, base_path, enabled, created_at, updated_at
            FROM deploy_targets
            ORDER BY name
        """)
        
        targets = []
        for row in cursor.fetchall():
            targets.append(DeployTarget(**dict(row)))
        return targets
    
    def get_deploy_mappings(self, target_id: Optional[int] = None) -> List[DeployMapping]:
        """Get deploy mappings, optionally filtered by target"""
        query = """
            SELECT id, target_id, model_type, folder_path
            FROM deploy_mappings
        """
        params = []
        
        if target_id:
            query += " WHERE target_id = ?"
            params.append(target_id)
        
        query += " ORDER BY target_id, model_type"
        
        cursor = self.conn.execute(query, params)
        mappings = []
        
        for row in cursor.fetchall():
            mappings.append(DeployMapping(**dict(row)))
        return mappings
    
    def get_deploy_links(self, model_id: Optional[int] = None) -> List[DeployLink]:
        """Get deploy links, optionally filtered by model"""
        query = """
            SELECT id, model_id, target_id, source_path, deploy_path,
                   is_deployed, deployed_at, created_at
            FROM deploy_links
        """
        params = []
        
        if model_id:
            query += " WHERE model_id = ?"
            params.append(model_id)
        
        query += " ORDER BY created_at DESC"
        
        cursor = self.conn.execute(query, params)
        links = []
        
        for row in cursor.fetchall():
            links.append(DeployLink(**dict(row)))
        return links
    
    # Data-driven classification methods
    def get_detection_rules(self, rule_type: Optional[str] = None, source_type: Optional[str] = None) -> List[Dict]:
        """Get detection rules from database"""
        query = """
            SELECT id, rule_name, rule_type, source_type, source_field, pattern, output_value, 
                   confidence, priority, fallback_rule_id, enabled
            FROM detection_rules
            WHERE enabled = 1
        """
        params = []
        
        if rule_type:
            query += " AND rule_type = ?"
            params.append(rule_type)
        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)
            
        query += " ORDER BY priority ASC, confidence DESC"
        
        cursor = self.class_conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_pattern_modifiers(self, applies_to_type: Optional[str] = None, applies_to_subtype: Optional[str] = None) -> List[Dict]:
        """Get pattern modifiers from database"""
        query = """
            SELECT id, modifier_name, pattern, modifier_suffix, applies_to_types, applies_to_subtypes, priority, enabled
            FROM pattern_modifiers
            WHERE enabled = 1
        """
        params = []
        
        if applies_to_type:
            query += " AND (applies_to_types = 'all' OR applies_to_types LIKE ?)"
            params.append(f'%{applies_to_type}%')
        if applies_to_subtype:
            query += " AND (applies_to_subtypes = 'all' OR applies_to_subtypes LIKE ?)"
            params.append(f'%{applies_to_subtype}%')
            
        query += " ORDER BY priority ASC"
        
        cursor = self.class_conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_external_api_mappings(self, api_name: str) -> List[Dict]:
        """Get external API mappings from database"""
        cursor = self.class_conn.execute("""
            SELECT id, api_name, api_field, api_value, modelhub_field, modelhub_value, confidence, enabled
            FROM external_api_mappings
            WHERE enabled = 1 AND api_name = ?
            ORDER BY confidence DESC
        """, (api_name,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_confidence_weights(self, source_type: Optional[str] = None, field_type: Optional[str] = None) -> List[Dict]:
        """Get confidence weights from database"""
        query = """
            SELECT id, source_type, field_type, base_weight, quality_multiplier, enabled
            FROM confidence_weights
            WHERE enabled = 1
        """
        params = []
        
        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)
        if field_type:
            query += " AND field_type = ?"
            params.append(field_type)
            
        cursor = self.class_conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_classification_workflow(self, workflow_name: str = 'standard_classification') -> List[Dict]:
        """Get classification workflow steps from database"""
        cursor = self.class_conn.execute("""
            SELECT id, workflow_name, step_order, rule_group, required, weight, enabled
            FROM classification_workflows
            WHERE enabled = 1 AND workflow_name = ?
            ORDER BY step_order ASC
        """, (workflow_name,))
        return [dict(row) for row in cursor.fetchall()]

    # Classification rule methods
    def get_size_rules(self) -> Dict[str, Dict[str, int]]:
        """Get size rules for model classification"""
        cursor = self.class_conn.execute("""
            SELECT model_type, min_size, max_size, confidence_weight
            FROM size_rules WHERE enabled = 1
        """)
        
        size_rules = {}
        for row in cursor.fetchall():
            model_type, min_size, max_size, confidence_weight = row
            size_rules[model_type] = {
                'min': min_size,
                'max': max_size,
                'confidence_weight': confidence_weight
            }
        
        return size_rules
    
    def get_sub_type_rules(self) -> List[Tuple[str, str, str, str, float]]:
        """Get sub-type classification rules"""
        cursor = self.class_conn.execute("""
            SELECT primary_type, sub_type, pattern, pattern_type, confidence_weight
            FROM sub_type_rules WHERE enabled = 1
            ORDER BY confidence_weight DESC
        """)
        
        return cursor.fetchall()
    
    def get_architecture_patterns(self) -> List[Tuple[str, str, float]]:
        """Get architecture detection patterns"""
        cursor = self.class_conn.execute("""
            SELECT architecture, tensor_pattern, confidence_weight
            FROM architecture_patterns WHERE enabled = 1
            ORDER BY confidence_weight DESC
        """)
        
        return cursor.fetchall()
    
    def get_external_apis(self) -> List[Tuple[str, bool, int, float, int]]:
        """Get external API configuration"""
        cursor = self.class_conn.execute("""
            SELECT name, enabled, priority, rate_limit_delay, timeout_seconds
            FROM external_apis WHERE enabled = 1
            ORDER BY priority ASC
        """)
        
        return cursor.fetchall()
    
    def get_model_types(self) -> List[Tuple[str, str]]:
        """Get supported model types"""
        cursor = self.class_conn.execute("""
            SELECT name, display_name
            FROM model_types WHERE enabled = 1
            ORDER BY name
        """)
        
        return cursor.fetchall()
    
    # Stub methods for future implementation
    def scan_directory(self, directory: Path, extensions: List[str]) -> List[Path]:
        """Scan directory recursively for model files"""
        model_files = []
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Recursively find all files with matching extensions
        for ext in extensions:
            # Use glob to find files with this extension
            pattern = f"**/*{ext}"
            files = directory.glob(pattern)
            
            for file_path in files:
                if file_path.is_file() and not file_path.is_symlink():
                    model_files.append(file_path)
        
        return sorted(model_files)
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file, using cached hash file if available"""
        try:
            # Check for cached hash file in source location
            source_hash_file = file_path.parent / (file_path.stem + '.hash')
            if source_hash_file.exists() and source_hash_file.stat().st_size > 0:
                try:
                    with open(source_hash_file, 'r', encoding='utf-8') as f:
                        cached_hash = f.read().strip()
                        if cached_hash and len(cached_hash) == 64:  # SHA-256 is 64 hex chars
                            return cached_hash
                except Exception as e:
                    print(f"Error reading cached hash file {source_hash_file}: {e}")
                    # Fall through to compute hash
            
            # Compute hash if no cached version or cached version is invalid
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            computed_hash = hash_sha256.hexdigest()
            
            # Note: Hash file will be created in destination during import_model
            return computed_hash
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _is_python_package_file(self, file_path: Path) -> bool:
        """Check if file is a Python package file that should be excluded"""
        filename_lower = file_path.name.lower()
        
        # Python package indicators
        python_package_patterns = [
            'distutils',
            'setuptools', 
            'pip',
            'wheel',
            'pkg_resources',
            'google_generativeai',
            'coloredlogs',
            '__pycache__',
            '.pyc',
            '.pyo'
        ]
        
        # Check for .pth files with package names or small size
        if file_path.suffix.lower() == '.pth':
            # Check if filename contains package indicators
            if any(pattern in filename_lower for pattern in python_package_patterns):
                return True
            # Very small .pth files are likely package files
            try:
                if file_path.stat().st_size < 1000:  # Less than 1KB
                    return True
            except:
                pass
        
        return False
    
    def import_model(self, file_path: Path, model_hub_path: Path, quiet: bool = False, config_manager=None, preserve_originals: bool = False) -> Optional[Model]:
        """Import a model file into the hub"""
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Skip LFS pointer files entirely - they're not actual models
        from classifier import SafeTensorsExtractor
        if SafeTensorsExtractor.is_lfs_pointer_file(file_path):
            if not quiet:
                print(f"Skipping LFS pointer file (not downloaded): {file_path.name}")
            return None
        
        # Skip Python package files - they're not models
        if self._is_python_package_file(file_path):
            if not quiet:
                print(f"Skipping Python package file: {file_path.name}")
            return None
        
        # Calculate file hash
        file_hash = self.calculate_file_hash(file_path)
        
        # Check if model already exists (including deleted ones)
        existing_model = self.get_model_by_hash(file_hash)
        if existing_model:
            # Verify that the storage file actually exists
            storage_dir = model_hub_path / "models" / file_hash
            storage_path = storage_dir / existing_model.filename
            
            if not storage_path.exists():
                # Model exists in DB but file is missing - treat as new import
                if not quiet:
                    print(f"Model exists in DB but file missing - reimporting: {existing_model.filename}")
                # Delete the stale database record
                self.conn.execute("DELETE FROM models WHERE file_hash = ?", (file_hash,))
                self.conn.commit()
                existing_model = None
        
        if existing_model:
            # If model was deleted, undelete it
            if existing_model.deleted:
                if not quiet:
                    print(f"Undeleting model: {existing_model.filename}")
                self.conn.execute(
                    "UPDATE models SET deleted = 0 WHERE file_hash = ?",
                    (file_hash,)
                )
                self.conn.commit()
                # Get updated model record
                existing_model = self.get_model_by_hash(file_hash)
            else:
                if not quiet:
                    print(f"Model already exists: {existing_model.filename}")
            
            # Even for existing models, we still need to handle symlink conversion
            # Use passed preserve_originals parameter
            
            # If not preserving originals, convert this duplicate to symlink
            if not preserve_originals:
                # Get storage path for the existing model
                storage_dir = model_hub_path / "models" / file_hash
                storage_path = storage_dir / existing_model.filename
                
                if storage_path.exists():
                    try:
                        # Replace current file with symlink to existing storage
                        file_path.unlink()  # Remove current file
                        file_path.symlink_to(storage_path)  # Create symlink
                        if not quiet:
                            print(f"Converted duplicate to symlink: {file_path.name}")
                    except Exception as e:
                        if not quiet:
                            print(f"Warning: Failed to create symlink for duplicate: {e}")
                else:
                    if not quiet:
                        print(f"Warning: Storage file not found for existing model: {storage_path}")
            
            # Create hash file in destination if it doesn't exist (for existing models)
            dest_hash_file = storage_path.parent / (storage_path.stem + '.hash')
            
            if not dest_hash_file.exists() or dest_hash_file.stat().st_size == 0:
                # First check if source has a hash file we can copy
                source_hash_file = file_path.parent / (file_path.stem + '.hash')
                hash_to_write = file_hash  # Default to known hash from DB
                source_method = "database"
                
                if source_hash_file.exists() and source_hash_file.stat().st_size > 0:
                    try:
                        with open(source_hash_file, 'r', encoding='utf-8') as f:
                            source_hash = f.read().strip()
                            if source_hash and len(source_hash) == 64:  # Valid SHA-256
                                hash_to_write = source_hash
                                source_method = "source file"
                    except Exception as e:
                        if not quiet:
                            print(f"Warning: Could not read source hash file {source_hash_file}: {e}")
                
                # Write hash file to destination
                try:
                    with open(dest_hash_file, 'w', encoding='utf-8') as f:
                        f.write(hash_to_write)
                    if not quiet:
                        if source_method == "source file":
                            print(f"📄 Reused hash file from source directory: {dest_hash_file.name}")
                        else:
                            print(f"🔢 Generated new hash file: {dest_hash_file.name}")
                except Exception as e:
                    if not quiet:
                        print(f"Warning: Could not create hash file {dest_hash_file}: {e}")
            
            return existing_model
        
        # Get file info
        file_size = file_path.stat().st_size
        filename = file_path.name
        file_extension = file_path.suffix
        
        # Create model-hub storage directory based on hash
        storage_dir = model_hub_path / "models" / file_hash
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_path = storage_dir / filename
        
        # Move or copy file to model-hub based on volume and symlink settings
        try:
            # Load config for classification
            if config_manager is None:
                from config import ConfigManager
                config_manager = ConfigManager()
                config_manager.load_config()
            
            raw_config = config_manager.get_raw_config()
            
            # Check if source and destination are on the same volume
            source_stat = file_path.stat()
            dest_stat = model_hub_path.stat()
            same_volume = source_stat.st_dev == dest_stat.st_dev
            
            # Only move if originals will be replaced with symlinks
            if same_volume and not preserve_originals:
                # Same volume AND not preserving originals: move file (much faster)
                shutil.move(str(file_path), str(storage_path))
                if not quiet:
                    print(f"Moved {filename} to model-hub (same volume)")
                moved_file = True
            else:
                # Different volume OR preserving originals: copy file
                shutil.copy2(file_path, storage_path)
                reason = "different volume" if not same_volume else "preserving originals"
                if not quiet:
                    print(f"Copied {filename} to model-hub ({reason})")
                moved_file = False
                
        except Exception as e:
            raise Exception(f"Failed to move/copy file: {e}")
        
        # Create hash file in destination directory
        dest_hash_file = storage_path.parent / (storage_path.stem + '.hash')
        
        # Check if we can reuse a hash file from source directory
        source_hash_file = file_path.parent / (file_path.stem + '.hash')
        source_method = "generated"
        
        if source_hash_file.exists() and source_hash_file.stat().st_size > 0:
            try:
                with open(source_hash_file, 'r', encoding='utf-8') as f:
                    source_hash = f.read().strip()
                    if source_hash and len(source_hash) == 64 and source_hash == file_hash:  # Valid SHA-256 and matches
                        source_method = "reused from source"
            except Exception:
                pass  # Fall back to generated method
        
        try:
            with open(dest_hash_file, 'w', encoding='utf-8') as f:
                f.write(file_hash)
            if not quiet:
                if source_method == "reused from source":
                    print(f"📄 Reused hash file from source directory: {dest_hash_file.name}")
                else:
                    print(f"🔢 Generated new hash file: {dest_hash_file.name}")
        except Exception as e:
            if not quiet:
                print(f"Warning: Could not create hash file {dest_hash_file}: {e}")
        
        # Comprehensive classification using new system
        from classifier import ModelClassifier
        classifier = ModelClassifier(raw_config, database=self)
        
        # Classify the model
        classification = classifier.classify_model(storage_path, file_hash, quiet)
        
        # Prepare trigger words for database storage
        triggers_str = ", ".join(classification.triggers) if classification.triggers else None
        
        # Debug output for trigger extraction
        if not quiet and classification.triggers and 'lora' in classification.primary_type.lower():
            print(f"    🎯 Extracted triggers: {', '.join(classification.triggers)}")
        
        classified_at = datetime.now().isoformat()
        
        # Insert into database with comprehensive classification data including enhanced scoring
        cursor = self.conn.execute("""
            INSERT INTO models (
                file_hash, filename, file_size, file_extension,
                primary_type, sub_type, confidence, classification_method,
                tensor_count, architecture, precision, quantization, triggers, 
                filename_score, size_score, metadata_score, tensor_score, classified_at, debug_info
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_hash, filename, file_size, file_extension,
            classification.primary_type, classification.sub_type, 
            classification.confidence, classification.method,
            classification.tensor_count, classification.architecture,
            getattr(classification, 'precision', None), getattr(classification, 'quantization', None),
            triggers_str, 
            getattr(classification, 'filename_score', 0.0), getattr(classification, 'size_score', 0.0),
            getattr(classification, 'metadata_score', 0.0), getattr(classification, 'tensor_score', 0.0),
            classified_at, getattr(classification, 'debug_info', None)
        ))
        
        model_id = cursor.lastrowid
        self.conn.commit()
        
        # Store raw metadata if available
        if hasattr(classification, 'raw_metadata') and classification.raw_metadata:
            try:
                self.store_model_metadata(model_id, classification.raw_metadata)
                if not quiet:
                    print(f"    📊 Stored metadata ({len(classification.raw_metadata)} entries)")
            except Exception as e:
                if not quiet:
                    print(f"    ⚠️  Warning: Failed to store metadata: {e}")
        
        # Replace original file with symlink (only if not preserving originals)
        if not preserve_originals:
            try:
                if not moved_file:
                    # File was copied, so original still exists - replace with symlink
                    file_path.unlink()  # Remove original file
                    file_path.symlink_to(storage_path)  # Create symlink
                    if not quiet:
                        print(f"Created symlink for {filename}")
                else:
                    # File was moved, so create symlink at original location
                    file_path.symlink_to(storage_path)  # Create symlink
                    if not quiet:
                        print(f"Created symlink for {filename}")
            except Exception as e:
                print(f"Warning: Failed to create symlink for {filename}: {e}")
        else:
            if not quiet:
                print(f"Preserved original file (preserve_originals=True)")
        
        # Return the created model
        return self.get_model_by_id(model_id)
    
    def get_model_by_hash(self, file_hash: str) -> Optional[Model]:
        """Get a model by its file hash"""
        cursor = self.conn.execute("""
            SELECT id, file_hash, filename, file_size, file_extension,
                   primary_type, sub_type, confidence, classification_method,
                   tensor_count, architecture, precision, quantization,
                   triggers, filename_score, size_score, metadata_score,
                   tensor_score, classified_at, created_at, updated_at, reclassify, deleted, debug_info
            FROM models WHERE file_hash = ?
        """, (file_hash,))
        
        row = cursor.fetchone()
        if row:
            return Model(**dict(row))
        return None
    
    def classify_model(self, model: Model) -> Model:
        """Classify or reclassify a model (STUB)"""
        # TODO: Implement model classification
        pass
    
    def delete_model(self, model_id: int) -> bool:
        """Delete model and associated files (STUB)"""
        # TODO: Implement model deletion
        pass
    
    def create_deploy_link(self, model_id: int, target_id: int) -> Optional[DeployLink]:
        """Create deployment symlink (STUB)"""
        # TODO: Implement deploy link creation
        pass
    
    def cleanup_orphaned_files(self) -> List[Path]:
        """Find and move orphaned files (STUB)"""
        # TODO: Implement orphaned file cleanup
        pass
    
    def find_duplicates(self) -> List[List[Model]]:
        """Find potential duplicate models (STUB)"""
        # TODO: Implement duplicate detection
        pass
    
    def export_symlinks(self, model_ids: List[int]) -> str:
        """Export symlink commands for clipboard (STUB)"""
        # TODO: Implement symlink export
        pass
    
    def generate_missing_hash_files(self, model_hub_path: Path, quiet: bool = False) -> Tuple[int, int]:
        """Generate hash files for models that don't have them
        
        Args:
            model_hub_path: Path to model hub directory
            quiet: If True, suppress output messages
            
        Returns:
            Tuple of (files_processed, files_created)
        """
        files_processed = 0
        files_created = 0
        
        if not quiet:
            print("Scanning for models without hash files...")
        
        # Get all models from database
        models = self.get_models(limit=999999, offset=0)
        
        for model in models:
            if model.deleted:
                continue
                
            # Construct expected model file path
            model_dir = model_hub_path / "models" / model.file_hash
            model_file = model_dir / model.filename
            
            if not model_file.exists():
                if not quiet:
                    print(f"Warning: Model file not found: {model_file}")
                continue
                
            files_processed += 1
            hash_file = model_file.parent / (model_file.stem + '.hash')
            
            # Check if hash file needs to be created or recreated
            needs_hash_file = (
                not hash_file.exists() or 
                hash_file.stat().st_size == 0  # Zero-byte file
            )
            
            if needs_hash_file:
                try:
                    # Write the hash to the file
                    with open(hash_file, 'w', encoding='utf-8') as f:
                        f.write(model.file_hash)
                    files_created += 1
                    
                    if not quiet:
                        print(f"Created hash file: {hash_file.name}")
                        
                except Exception as e:
                    if not quiet:
                        print(f"Error creating hash file for {model.filename}: {e}")
            else:
                # Verify existing hash file matches database
                try:
                    with open(hash_file, 'r', encoding='utf-8') as f:
                        file_hash = f.read().strip()
                        if file_hash != model.file_hash:
                            if not quiet:
                                print(f"Warning: Hash file mismatch for {model.filename}")
                                print(f"  File hash: {file_hash}")
                                print(f"  DB hash:   {model.file_hash}")
                except Exception as e:
                    if not quiet:
                        print(f"Error reading hash file for {model.filename}: {e}")
        
        if not quiet:
            print(f"Processed {files_processed} models, created {files_created} hash files")
        
        return files_processed, files_created
    
    def cleanup_models(self, model_hub_path: Path) -> Dict[str, int]:
        """
        Comprehensive cleanup of model database and files
        Returns dict with cleanup statistics
        """
        results = {
            'deleted_records_removed': 0,
            'orphaned_records_removed': 0,
            'reverse_orphans_moved': 0
        }
        
        models_path = model_hub_path / 'models'
        review_path = model_hub_path / 'review'
        
        # Create review directory if it doesn't exist
        review_path.mkdir(exist_ok=True)
        
        # 1. Remove deleted records and move their hash folders
        cursor = self.conn.cursor()
        cursor.execute("SELECT file_hash FROM models WHERE deleted = 1")
        deleted_hashes = [row[0] for row in cursor.fetchall()]
        
        for file_hash in deleted_hashes:
            hash_dir = models_path / file_hash
            if hash_dir.exists():
                # Move to review folder
                review_hash_dir = review_path / file_hash
                try:
                    if review_hash_dir.exists():
                        shutil.rmtree(review_hash_dir)
                    shutil.move(str(hash_dir), str(review_hash_dir))
                except Exception as e:
                    print(f"Error moving deleted model {file_hash}: {e}")
                    continue
            
            # Remove from database
            cursor.execute("DELETE FROM models WHERE file_hash = ?", (file_hash,))
            results['deleted_records_removed'] += 1
        
        # 2. Find and remove orphaned records (DB records without files)
        cursor.execute("SELECT file_hash FROM models WHERE deleted = 0")
        active_hashes = [row[0] for row in cursor.fetchall()]
        
        for file_hash in active_hashes:
            hash_dir = models_path / file_hash
            if not hash_dir.exists():
                # No corresponding file, remove from database
                cursor.execute("DELETE FROM models WHERE file_hash = ?", (file_hash,))
                results['orphaned_records_removed'] += 1
        
        # 3. Find reverse orphans (hash folders without DB records)
        if models_path.exists():
            existing_hash_dirs = [d.name for d in models_path.iterdir() if d.is_dir()]
            cursor.execute("SELECT file_hash FROM models")
            db_hashes = set(row[0] for row in cursor.fetchall())
            
            for hash_dir_name in existing_hash_dirs:
                if hash_dir_name not in db_hashes:
                    # This is a reverse orphan - move to review
                    hash_dir = models_path / hash_dir_name
                    review_hash_dir = review_path / hash_dir_name
                    try:
                        if review_hash_dir.exists():
                            shutil.rmtree(review_hash_dir)
                        shutil.move(str(hash_dir), str(review_hash_dir))
                        results['reverse_orphans_moved'] += 1
                    except Exception as e:
                        print(f"Error moving reverse orphan {hash_dir_name}: {e}")
        
        # Commit all changes
        self.conn.commit()
        
        return results