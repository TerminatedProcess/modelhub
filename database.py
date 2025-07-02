#!/usr/bin/env python3
"""
Database operations for ModelHub CLI
Handles all database interactions for models and deployment
"""

import sqlite3
import os
import hashlib
import shutil
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
    """Database interface for ModelHub"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self.ensure_database_exists()
        
    def ensure_database_exists(self):
        """Ensure database and model-hub directory exist, create if missing"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.db_path.exists():
            self.create_database()
    
    def connect(self):
        """Connect to database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise Exception(f"Database connection failed: {e}")
    
    def disconnect(self):
        """Disconnect from database"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def create_database(self):
        """Create new database with all tables and default data"""
        print(f"Creating new database at {self.db_path}")
        
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
            
            # Create indexes
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
            
            # Create indexes for classification tables
            cursor.execute("CREATE INDEX idx_model_types_name ON model_types (name)")
            cursor.execute("CREATE INDEX idx_size_rules_type ON size_rules (model_type)")
            cursor.execute("CREATE INDEX idx_sub_type_rules_primary ON sub_type_rules (primary_type)")
            cursor.execute("CREATE INDEX idx_architecture_patterns_arch ON architecture_patterns (architecture)")
            cursor.execute("CREATE INDEX idx_external_apis_priority ON external_apis (priority)")
            
            # Create view for active (non-deleted) models
            cursor.execute("""
                CREATE VIEW active_models AS
                SELECT id, file_hash, filename, file_size, file_extension,
                       primary_type, sub_type, confidence, classification_method,
                       tensor_count, architecture, precision, quantization,
                       triggers, filename_score, size_score, metadata_score,
                       tensor_score, classified_at, created_at, updated_at, reclassify, deleted
                FROM models 
                WHERE deleted = FALSE
            """)
            
            conn.commit()
            print("✓ Created database tables and indexes")
            
            # Create default deploy data
            self._create_default_deploy_data(cursor)
            
            # Create default classification data
            self._create_default_classification_data(cursor)
            
            conn.commit()
            print("✓ Created default deploy and classification configuration")
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error creating database: {e}")
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
        
        # ComfyUI mappings
        comfyui_mappings = [
            ('checkpoint', 'checkpoints'), ('lora', 'loras'), ('vae', 'vae'),
            ('controlnet', 'controlnet'), ('embedding', 'embeddings'),
            ('upscaler', 'upscale_models'), ('text_encoder', 'text_encoders'),
            ('clip', 'clip'), ('unet', 'unet'), ('unknown', 'unknown')
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
        
        # Default model types
        model_types = [
            ('checkpoint', 'Checkpoint', 'Large diffusion model checkpoints'),
            ('lora', 'LoRA', 'Low-Rank Adaptation fine-tuning models'),
            ('vae', 'VAE', 'Variational Autoencoder models'),
            ('controlnet', 'ControlNet', 'Conditional control models'),
            ('clip', 'CLIP', 'Text-image embedding models'),
            ('text_encoder', 'Text Encoder', 'Text encoding models'),
            ('unet', 'UNet', 'U-Net diffusion models'),
            ('gguf', 'GGUF', 'GGML Universal File format models'),
            ('upscaler', 'Upscaler', 'Image upscaling models'),
            ('embedding', 'Embedding', 'Textual inversion embeddings'),
            ('video_model', 'Video Model', 'Video generation models'),
            ('mask_model', 'Mask Model', 'Segmentation and masking models'),
            ('audio_model', 'Audio Model', 'Audio processing models'),
            ('hypernetwork', 'Hypernetwork', 'Hypernetwork models'),
            ('unknown', 'Unknown', 'Unclassified models')
        ]
        
        cursor.executemany("""
            INSERT INTO model_types (name, display_name, description)
            VALUES (?, ?, ?)
        """, model_types)
        
        # Default size rules (in bytes)
        size_rules = [
            ('checkpoint', 1_000_000_000, 50_000_000_000, 0.3),  # 1GB-50GB
            ('lora', 1_000_000, 2_000_000_000, 0.4),             # 1MB-2GB
            ('vae', 50_000_000, 2_000_000_000, 0.4),             # 50MB-2GB
            ('controlnet', 100_000_000, 10_000_000_000, 0.3),    # 100MB-10GB
            ('clip', 10_000_000, 2_000_000_000, 0.3),            # 10MB-2GB
            ('text_encoder', 10_000_000, 25_000_000_000, 0.3),   # 10MB-25GB
            ('unet', 500_000_000, 30_000_000_000, 0.3),          # 500MB-30GB
            ('gguf', 100_000_000, 200_000_000_000, 0.2),         # 100MB-200GB
            ('upscaler', 1_000_000, 500_000_000, 0.4),           # 1MB-500MB
            ('embedding', 1_000, 50_000_000, 0.5),               # 1KB-50MB
            ('video_model', 100_000_000, 100_000_000_000, 0.3),  # 100MB-100GB
            ('mask_model', 100_000_000, 2_000_000_000, 0.3),     # 100MB-2GB
            ('audio_model', 50_000_000, 1_000_000_000, 0.3),     # 50MB-1GB
        ]
        
        cursor.executemany("""
            INSERT INTO size_rules (model_type, min_size, max_size, confidence_weight)
            VALUES (?, ?, ?, ?)
        """, size_rules)
        
        # Default sub-type rules (filename patterns)
        sub_type_rules = [
            # Checkpoint sub-types
            ('checkpoint', 'flux_checkpoint', 'flux', 'filename', 0.9),
            ('checkpoint', 'sdxl_checkpoint', 'xl|sdxl', 'filename', 0.8),
            ('checkpoint', 'sd3_checkpoint', 'sd3', 'filename', 0.8),
            ('checkpoint', 'sd15_checkpoint', '.*', 'filename', 0.1),  # Default
            
            # LoRA sub-types
            ('lora', 'flux_lora', 'flux', 'filename', 0.9),
            ('lora', 'sdxl_lora', 'xl|sdxl', 'filename', 0.8),
            ('lora', 'hunyuan_i2v_lora', 'hunyuan.*i2v', 'filename', 0.9),
            ('lora', 'hunyuan_lora', 'hunyuan', 'filename', 0.8),
            ('lora', 'i2v_lora', 'i2v', 'filename', 0.8),
            ('lora', 'ltxv_lora', 'ltxv|ltx.*video', 'filename', 0.8),
            ('lora', 'sd15_lora', '.*', 'filename', 0.1),  # Default
            
            # VAE sub-types
            ('vae', 'video_vae', 'video', 'filename', 0.9),
            ('vae', 'sdxl_vae', 'xl|sdxl', 'filename', 0.8),
            ('vae', 'sd15_vae', '.*', 'filename', 0.1),  # Default
            
            # GGUF sub-types
            ('gguf', 'ltx_video', 'ltxv|ltx.*video', 'filename', 0.9),
            ('gguf', 'text_to_video', 't2v|text2video', 'filename', 0.8),
            ('gguf', 'image_to_video', 'i2v|image2video', 'filename', 0.8),
            ('gguf', 'text_encoder', 'umt5|encoder', 'filename', 0.8),
            ('gguf', 'large_llm', '.*', 'filesize_large', 0.3),  # >10GB
            ('gguf', 'medium_llm', '.*', 'filesize_medium', 0.3),  # 1-10GB
            ('gguf', 'small_llm', '.*', 'filesize_small', 0.3),   # <1GB
        ]
        
        cursor.executemany("""
            INSERT INTO sub_type_rules (primary_type, sub_type, pattern, pattern_type, confidence_weight)
            VALUES (?, ?, ?, ?, ?)
        """, sub_type_rules)
        
        # Default architecture patterns (tensor analysis)
        architecture_patterns = [
            ('lora', '\\.lora_up\\.|lora_down\\.|alpha$', 0.95),
            ('flux', 'double_blocks|single_blocks|img_attn|txt_attn', 0.95),
            ('video_vae', 'downsamples.*residual|upsamples.*residual', 0.95),
            ('vae', 'encoder\\.|decoder\\.|autoencoder', 0.9),
            ('diffusion', 'diffusion_model|unet', 0.9),
            ('controlnet', 'control_model|controlnet', 0.9),
            ('clip', 'text_model|text_projection', 0.8),
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
               tensor_score, classified_at, created_at, updated_at, reclassify, deleted
        FROM active_models
        """
        
        params = []
        conditions = []
        
        if filter_type:
            conditions.append("primary_type = ?")
            params.append(filter_type)
        
        if search_term:
            conditions.append("(filename LIKE ? OR triggers LIKE ?)")
            params.extend([f"%{search_term}%", f"%{search_term}%"])
        
        if conditions:
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
        query = "SELECT COUNT(*) FROM active_models"
        params = []
        conditions = []
        
        if filter_type:
            conditions.append("primary_type = ?")
            params.append(filter_type)
        
        if search_term:
            conditions.append("(filename LIKE ? OR triggers LIKE ?)")
            params.extend([f"%{search_term}%", f"%{search_term}%"])
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        cursor = self.conn.execute(query, params)
        return cursor.fetchone()[0]
    
    def get_model_types(self) -> List[Tuple[str, int]]:
        """Get model types with counts"""
        cursor = self.conn.execute("""
            SELECT primary_type, COUNT(*) as count 
            FROM active_models 
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
                   tensor_score, classified_at, created_at, updated_at, reclassify, deleted
            FROM active_models WHERE id = ?
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
    
    # Classification rule methods
    def get_size_rules(self) -> Dict[str, Dict[str, int]]:
        """Get size rules for model classification"""
        cursor = self.conn.execute("""
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
        cursor = self.conn.execute("""
            SELECT primary_type, sub_type, pattern, pattern_type, confidence_weight
            FROM sub_type_rules WHERE enabled = 1
            ORDER BY confidence_weight DESC
        """)
        
        return cursor.fetchall()
    
    def get_architecture_patterns(self) -> List[Tuple[str, str, float]]:
        """Get architecture detection patterns"""
        cursor = self.conn.execute("""
            SELECT architecture, tensor_pattern, confidence_weight
            FROM architecture_patterns WHERE enabled = 1
            ORDER BY confidence_weight DESC
        """)
        
        return cursor.fetchall()
    
    def get_external_apis(self) -> List[Tuple[str, bool, int, float, int]]:
        """Get external API configuration"""
        cursor = self.conn.execute("""
            SELECT name, enabled, priority, rate_limit_delay, timeout_seconds
            FROM external_apis WHERE enabled = 1
            ORDER BY priority ASC
        """)
        
        return cursor.fetchall()
    
    def get_model_types(self) -> List[Tuple[str, str]]:
        """Get supported model types"""
        cursor = self.conn.execute("""
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
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def import_model(self, file_path: Path, model_hub_path: Path, quiet: bool = False, config_manager=None) -> Optional[Model]:
        """Import a model file into the hub"""
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Calculate file hash
        file_hash = self.calculate_file_hash(file_path)
        
        # Check if model already exists (including deleted ones)
        existing_model = self.get_model_by_hash(file_hash)
        if existing_model:
            # If model was deleted, undelete it
            if existing_model.deleted:
                if not quiet:
                    print(f"Undeleting model: {existing_model.filename}")
                self.conn.execute(
                    "UPDATE models SET deleted = FALSE WHERE file_hash = ?",
                    (file_hash,)
                )
                self.conn.commit()
                # Get updated model record
                existing_model = self.get_model_by_hash(file_hash)
            else:
                if not quiet:
                    print(f"Model already exists: {existing_model.filename}")
            
            # Even for existing models, we still need to handle symlink conversion
            # Load config for symlink settings
            if config_manager is None:
                from config import ConfigManager
                config_manager = ConfigManager()
                config_manager.load_config()
            
            raw_config = config_manager.get_raw_config()
            scanning_config = raw_config.get('scanning', {})
            preserve_originals = scanning_config.get('preserve_originals', False)
            
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
            # Load config for symlink settings
            if config_manager is None:
                from config import ConfigManager
                config_manager = ConfigManager()
                config_manager.load_config()
            
            raw_config = config_manager.get_raw_config()
            scanning_config = raw_config.get('scanning', {})
            preserve_originals = scanning_config.get('preserve_originals', False)
            
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
                filename_score, size_score, metadata_score, tensor_score, classified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_hash, filename, file_size, file_extension,
            classification.primary_type, classification.sub_type, 
            classification.confidence, classification.method,
            classification.tensor_count, classification.architecture,
            getattr(classification, 'precision', None), getattr(classification, 'quantization', None),
            triggers_str, 
            getattr(classification, 'filename_score', 0.0), getattr(classification, 'size_score', 0.0),
            getattr(classification, 'metadata_score', 0.0), getattr(classification, 'tensor_score', 0.0),
            classified_at
        ))
        
        model_id = cursor.lastrowid
        self.conn.commit()
        
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
                   tensor_score, classified_at, created_at, updated_at, reclassify, deleted
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