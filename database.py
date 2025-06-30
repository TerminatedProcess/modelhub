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
                    
                    filename TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_extension TEXT NOT NULL,
                    
                    -- Classification results
                    primary_type TEXT NOT NULL,
                    sub_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    classification_method TEXT NOT NULL,
                    
                    -- Technical metadata
                    tensor_count INTEGER,
                    architecture TEXT,
                    precision TEXT,
                    quantization TEXT,
                    
                    -- Trigger words (denormalized for easy browsing)
                    triggers TEXT,
                    
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
            
            conn.commit()
            print("✓ Created database tables and indexes")
            
            # Create default deploy data
            self._create_default_deploy_data(cursor)
            conn.commit()
            print("✓ Created default deploy configuration")
            
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
               tensor_score, classified_at, created_at, updated_at, reclassify
        FROM models
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
        query = "SELECT COUNT(*) FROM models"
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
            FROM models 
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
                   tensor_score, classified_at, created_at, updated_at, reclassify
            FROM models WHERE id = ?
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
    
    def import_model(self, file_path: Path, model_hub_path: Path, quiet: bool = False) -> Optional[Model]:
        """Import a model file into the hub"""
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Calculate file hash
        file_hash = self.calculate_file_hash(file_path)
        
        # Check if model already exists
        existing_model = self.get_model_by_hash(file_hash)
        if existing_model:
            if not quiet:
                print(f"Model already exists: {existing_model.filename}")
            return existing_model
        
        # Get file info
        file_size = file_path.stat().st_size
        filename = file_path.name
        file_extension = file_path.suffix
        
        # Create model-hub storage directory based on hash
        storage_dir = model_hub_path / "models" / file_hash
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_path = storage_dir / filename
        
        # Move or copy file to model-hub based on volume
        try:
            # Check if source and destination are on the same volume
            source_stat = file_path.stat()
            dest_stat = model_hub_path.stat()
            same_volume = source_stat.st_dev == dest_stat.st_dev
            
            if same_volume:
                # Same volume: move file (much faster)
                shutil.move(str(file_path), str(storage_path))
                if not quiet:
                    print(f"Moved {filename} to model-hub (same volume)")
                moved_file = True
            else:
                # Different volume: copy file
                shutil.copy2(file_path, storage_path)
                if not quiet:
                    print(f"Copied {filename} to model-hub (different volume)")
                moved_file = False
                
        except Exception as e:
            raise Exception(f"Failed to move/copy file: {e}")
        
        # Basic classification (as requested)
        primary_type = "checkpoint"
        sub_type = "wan"
        confidence = 1.0
        classification_method = "basic"
        classified_at = datetime.now().isoformat()
        
        # Insert into database
        cursor = self.conn.execute("""
            INSERT INTO models (
                file_hash, filename, file_size, file_extension,
                primary_type, sub_type, confidence, classification_method,
                classified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_hash, filename, file_size, file_extension,
            primary_type, sub_type, confidence, classification_method,
            classified_at
        ))
        
        model_id = cursor.lastrowid
        self.conn.commit()
        
        # Replace original file with symlink (only if we copied, not moved)
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
        
        # Return the created model
        return self.get_model_by_id(model_id)
    
    def get_model_by_hash(self, file_hash: str) -> Optional[Model]:
        """Get a model by its file hash"""
        cursor = self.conn.execute("""
            SELECT id, file_hash, filename, file_size, file_extension,
                   primary_type, sub_type, confidence, classification_method,
                   tensor_count, architecture, precision, quantization,
                   triggers, filename_score, size_score, metadata_score,
                   tensor_score, classified_at, created_at, updated_at, reclassify
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