#!/usr/bin/env python3
"""
Model Hub Classifier - Comprehensive AI model classification system
Provides accurate model type detection through multi-layer analysis
"""

import json
import hashlib
import struct
import os
import re
import requests
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager

@dataclass
class ModelMetadata:
    """Complete metadata structure for a model"""
    file_hash: str
    filename: str
    file_size: int
    file_extension: str
    
    # Classification results
    primary_type: str
    sub_type: str
    confidence: float
    classification_method: str
    
    # Technical metadata
    tensor_count: Optional[int] = None
    architecture: Optional[str] = None
    precision: Optional[str] = None
    quantization: Optional[str] = None
    
    # Extracted metadata
    safetensors_metadata: Optional[Dict] = None
    gguf_metadata: Optional[Dict] = None
    
    # Trigger words for LoRA models
    triggers: Optional[List[str]] = None
    
    # Classification details
    filename_score: float = 0.0
    size_score: float = 0.0
    metadata_score: float = 0.0
    tensor_score: float = 0.0
    
    classified_at: str = ""

class SafeTensorsExtractor:
    """Extract metadata from SafeTensors files"""
    
    @staticmethod
    def is_lfs_pointer_file(file_path: Path) -> bool:
        """Check if file is an undownloaded LFS pointer file"""
        try:
            # LFS pointer files are typically very small (< 200 bytes)
            file_size = file_path.stat().st_size
            if file_size > 500:  # Too large to be an LFS pointer
                return False
            
            # Read the first few lines to check for LFS markers
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(200)  # Read first 200 chars
                
            # Check for LFS pointer file markers
            lfs_markers = [
                'version https://git-lfs.github.com/spec/',
                'oid sha256:',
                'size '
            ]
            
            # Must contain all LFS markers to be considered an LFS pointer
            return all(marker in content for marker in lfs_markers)
            
        except Exception:
            # If we can't read it, assume it's not an LFS pointer
            return False
    
    @staticmethod
    def extract_metadata(file_path: Path) -> Dict:
        """Extract comprehensive metadata from SafeTensors file"""
        try:
            # Check if this is an LFS pointer file first
            if SafeTensorsExtractor.is_lfs_pointer_file(file_path):
                return {'error': 'LFS pointer file - not downloaded'}
            
            with open(file_path, 'rb') as f:
                # Read header length (first 8 bytes)
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    return {'error': 'File too small to be valid SafeTensors'}
                
                header_size = struct.unpack('<Q', header_size_bytes)[0]
                
                # Sanity check header size
                if header_size > 100_000_000:  # 100MB header limit
                    return {'error': 'Header size too large'}
                
                # Read header
                header_data = f.read(header_size)
                if len(header_data) < header_size:
                    return {'error': 'Incomplete header data'}
                
                header = json.loads(header_data.decode('utf-8'))
                
                metadata = header.get('__metadata__', {})
                tensors = {k: v for k, v in header.items() if k != '__metadata__'}
                
                # Calculate total tensor size
                total_size = 0
                for tensor_info in tensors.values():
                    if 'data_offsets' in tensor_info and len(tensor_info['data_offsets']) >= 2:
                        total_size += tensor_info['data_offsets'][1] - tensor_info['data_offsets'][0]
                
                return {
                    'metadata': metadata,
                    'tensors': tensors,
                    'tensor_count': len(tensors),
                    'tensor_names': list(tensors.keys()),
                    'total_tensor_size': total_size,
                    'header_size': header_size
                }
        except json.JSONDecodeError:
            return {'error': 'Invalid JSON in header'}
        except Exception as e:
            return {'error': f'Failed to extract metadata: {str(e)}'}

class GGUFExtractor:
    """Extract metadata from GGUF files"""
    
    @staticmethod
    def extract_metadata(file_path: Path) -> Dict:
        """Extract comprehensive metadata from GGUF file"""
        try:
            # Check if this is an LFS pointer file first
            if SafeTensorsExtractor.is_lfs_pointer_file(file_path):
                return {'error': 'LFS pointer file - not downloaded'}
            
            with open(file_path, 'rb') as f:
                # Read GGUF header
                magic = f.read(4)
                if magic != b'GGUF':
                    return {'error': 'Not a valid GGUF file'}
                
                version = struct.unpack('<I', f.read(4))[0]
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
                
                # Basic validation
                if tensor_count > 10000 or metadata_kv_count > 1000:
                    return {'error': 'Suspicious tensor or metadata counts'}
                
                metadata = {
                    'version': version,
                    'tensor_count': tensor_count,
                    'metadata_count': metadata_kv_count
                }
                
                # Try to read some basic metadata (simplified)
                # Full GGUF parsing would require complete specification
                try:
                    # Skip detailed parsing for now, just get basic info
                    file_size = file_path.stat().st_size
                    metadata['file_size'] = file_size
                    
                    # Estimate model type based on size and tensor count
                    if file_size > 10_000_000_000:  # > 10GB
                        metadata['estimated_type'] = 'large_llm'
                    elif file_size > 1_000_000_000:  # > 1GB
                        metadata['estimated_type'] = 'medium_llm'
                    else:
                        metadata['estimated_type'] = 'small_llm'
                        
                except Exception:
                    pass
                
                return metadata
                
        except Exception as e:
            return {'error': f'Failed to extract GGUF metadata: {str(e)}'}

class TensorAnalyzer:
    """Analyze tensor structures for classification"""
    
    def __init__(self, tensor_patterns: Dict):
        self.tensor_patterns = tensor_patterns
    
    def analyze_tensors(self, tensor_names: List[str]) -> Dict[str, float]:
        """Analyze tensor names against known patterns"""
        if not tensor_names:
            return {}
        
        scores = {}
        
        for model_type, patterns in self.tensor_patterns.items():
            matches = 0
            total_patterns = len(patterns)
            
            for pattern in patterns:
                pattern_matches = sum(
                    1 for name in tensor_names 
                    if re.search(pattern, name, re.IGNORECASE)
                )
                if pattern_matches > 0:
                    matches += 1
            
            scores[model_type] = matches / total_patterns if total_patterns > 0 else 0
        
        return scores
    
    def get_architecture_hints(self, tensor_names: List[str], metadata: Dict) -> str:
        """Try to determine model architecture from tensor names"""
        tensor_str = ' '.join(tensor_names).lower()
        
        # Check for quantization patterns first
        quantization_patterns = ['.absmax', '.quant_map', '.quantized', '.int8', '.int4']
        has_quantization = any(pattern in tensor_str for pattern in quantization_patterns)
        
        # Check for FLUX architecture patterns
        flux_patterns = ['double_blocks', 'single_blocks', 'img_attn', 'txt_attn', 'img_mlp', 'txt_mlp']
        has_flux = any(pattern in tensor_str for pattern in flux_patterns)
        
        # Check for specific architectures - order matters!
        if has_flux and has_quantization:
            return 'flux_quantized'
        elif has_flux:
            return 'flux'
        elif 'clip' in tensor_str:
            return 'clip'
        elif 'unet' in tensor_str or 'diffusion_model' in tensor_str:
            return 'diffusion'
        elif any('downsamples' in name and 'residual' in name for name in tensor_names):
            # Video VAE pattern detected
            return 'video_vae'
        elif 'vae' in tensor_str or 'autoencoder' in tensor_str:
            return 'vae'
        elif 'lora' in tensor_str:
            return 'lora'
        elif 'controlnet' in tensor_str or 'control_model' in tensor_str:
            return 'controlnet'
        elif has_quantization:
            return 'quantized'
        
        return 'unknown'


class CivitAILookup:
    """CivitAI API integration for model lookup by hash"""
    
    def __init__(self, enable_api: bool = True):
        self.enable_api = enable_api
        self.base_url = "https://civitai.com/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ModelHubClassifier/1.0'
        })
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
    
    def lookup_by_hash(self, file_hash: str) -> Dict:
        """Lookup model information by SHA-256 hash"""
        if not self.enable_api:
            return {'found': False, 'source': 'api_disabled'}
        
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
            
            # CivitAI API endpoint for hash lookup
            url = f"{self.base_url}/model-versions/by-hash/{file_hash}"
            
            print(f"    Querying CivitAI API...")
            
            response = self.session.get(url, timeout=10)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                return self.parse_civitai_response(data)
            elif response.status_code == 404:
                print(f"    Model not found on CivitAI")
                return {'found': False, 'source': 'not_found'}
            else:
                print(f"    CivitAI API error: {response.status_code}")
                return {'found': False, 'source': 'api_error'}
                
        except requests.exceptions.Timeout:
            print(f"    CivitAI API timeout")
            return {'found': False, 'source': 'timeout'}
        except requests.exceptions.RequestException as e:
            print(f"    CivitAI API error: {e}")
            return {'found': False, 'source': 'network_error'}
        except Exception as e:
            print(f"    CivitAI lookup failed: {e}")
            return {'found': False, 'source': 'unknown_error'}
    
    def parse_civitai_response(self, data: Dict) -> Dict:
        """Parse CivitAI API response to extract model classification"""
        try:
            model_info = data.get('model', {})
            version_info = data
            
            # Extract model type
            model_type = model_info.get('type', '').lower()
            
            # Map CivitAI types to our classification system
            type_mapping = {
                'checkpoint': 'checkpoint',
                'lora': 'lora',
                'lycoris': 'lora',  # LyCORIS is a type of LoRA
                'textualinversion': 'embedding',
                'hypernetwork': 'hypernetwork',
                'aestheticgradient': 'embedding',
                'controlnet': 'controlnet',
                'vae': 'vae',
                'upscaler': 'upscaler'
            }
            
            primary_type = type_mapping.get(model_type, 'unknown')
            
            # Extract base model information
            base_model = self.extract_base_model(model_info, version_info)
            
            # Extract additional metadata
            name = model_info.get('name', '')
            description = model_info.get('description', '')
            tags = [tag.get('name', '') for tag in model_info.get('tags', [])]
            
            print(f"    CivitAI: {name} ({primary_type})")
            print(f"    Base model: {base_model}")
            print(f"    Tags: {', '.join(tags[:3])}{'...' if len(tags) > 3 else ''}")
            
            return {
                'found': True,
                'source': 'civitai',
                'primary_type': primary_type,
                'base_model': base_model,
                'name': name,
                'description': description,
                'tags': tags,
                'confidence': 0.99,  # Very high confidence for CivitAI data
                'raw_data': data
            }
            
        except Exception as e:
            print(f"    Error parsing CivitAI response: {e}")
            return {'found': False, 'source': 'parse_error'}
    
    def extract_base_model(self, model_info: Dict, version_info: Dict) -> str:
        """Extract base model from CivitAI data"""
        # Check version base model first
        base_model = version_info.get('baseModel', '').lower()
        
        # Map common base models
        if 'flux' in base_model:
            return 'flux'
        elif 'sdxl' in base_model or 'xl' in base_model:
            return 'sdxl'
        elif 'sd 3' in base_model or 'sd3' in base_model:
            return 'sd3'
        elif 'sd 1.5' in base_model or 'sd15' in base_model:
            return 'sd15'
        elif 'sd 2' in base_model or 'sd2' in base_model:
            return 'sd2'
        
        # Fallback to model name analysis
        name = model_info.get('name', '').lower()
        if 'flux' in name:
            return 'flux'
        elif 'xl' in name or 'sdxl' in name:
            return 'sdxl'
        elif 'sd3' in name:
            return 'sd3'
        
        return base_model or 'unknown'

class DatabaseManager:
    """SQLite database manager for model classification cache"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with schema"""
        # Ensure the directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self.get_db_connection() as conn:
            # Create models table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
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
            ''')
            
            # Create metadata table for flexible key-value storage
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    metadata_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_models_hash ON models (file_hash)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_models_type ON models (primary_type, sub_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_models_triggers ON models (triggers)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_metadata_model ON model_metadata (model_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_metadata_key ON model_metadata (key)')
            
            conn.commit()
            
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def get_model_by_hash(self, file_hash: str) -> Optional[ModelMetadata]:
        """Get model metadata by file hash"""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM models WHERE file_hash = ?",
                (file_hash,)
            )
            row = cursor.fetchone()
            if row:
                # Convert row to dict and create ModelMetadata
                model_data = dict(row)
                
                # Get associated metadata
                metadata_cursor = conn.execute(
                    "SELECT metadata_type, key, value FROM model_metadata WHERE model_id = ?",
                    (model_data['id'],)
                )
                
                # Reconstruct metadata dictionaries
                safetensors_metadata = {}
                gguf_metadata = {}
                
                for meta_row in metadata_cursor:
                    meta_type, key, value = meta_row
                    try:
                        # Try to parse JSON values
                        parsed_value = json.loads(value) if value else None
                    except (json.JSONDecodeError, TypeError):
                        parsed_value = value
                    
                    if meta_type == 'safetensors':
                        safetensors_metadata[key] = parsed_value
                    elif meta_type == 'gguf':
                        gguf_metadata[key] = parsed_value
                
                # Add metadata to model data
                model_data['safetensors_metadata'] = safetensors_metadata if safetensors_metadata else None
                model_data['gguf_metadata'] = gguf_metadata if gguf_metadata else None
                
                # Parse triggers from database column
                triggers_str = model_data.get('triggers')
                if triggers_str:
                    model_data['triggers'] = [t.strip() for t in triggers_str.split(',') if t.strip()]
                else:
                    model_data['triggers'] = None
                
                # Remove database-specific fields
                model_data.pop('id', None)
                model_data.pop('created_at', None)
                model_data.pop('updated_at', None)
                model_data.pop('reclassify', None)  # Remove reclassify field if present
                
                return ModelMetadata(**model_data)
        return None
    
    def save_model(self, metadata: ModelMetadata):
        """Save or update model metadata"""
        with self.get_db_connection() as conn:
            # Convert triggers list to comma-separated string
            triggers_str = ", ".join(metadata.triggers) if metadata.triggers else None
            
            # Check if model exists
            cursor = conn.execute(
                "SELECT id FROM models WHERE file_hash = ?",
                (metadata.file_hash,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing model
                model_id = existing['id']
                conn.execute('''
                    UPDATE models SET
                        filename = ?, file_size = ?, file_extension = ?,
                        primary_type = ?, sub_type = ?, confidence = ?, classification_method = ?,
                        tensor_count = ?, architecture = ?, precision = ?, quantization = ?,
                        triggers = ?, filename_score = ?, size_score = ?, metadata_score = ?, tensor_score = ?,
                        classified_at = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE file_hash = ?
                ''', (
                    metadata.filename, metadata.file_size, metadata.file_extension,
                    metadata.primary_type, metadata.sub_type, metadata.confidence, metadata.classification_method,
                    metadata.tensor_count, metadata.architecture, metadata.precision, metadata.quantization,
                    triggers_str, metadata.filename_score, metadata.size_score, metadata.metadata_score, metadata.tensor_score,
                    metadata.classified_at, metadata.file_hash
                ))
                
                # Clear existing metadata
                conn.execute("DELETE FROM model_metadata WHERE model_id = ?", (model_id,))
            else:
                # Insert new model
                cursor = conn.execute('''
                    INSERT INTO models (
                        file_hash, filename, file_size, file_extension,
                        primary_type, sub_type, confidence, classification_method,
                        tensor_count, architecture, precision, quantization, triggers,
                        filename_score, size_score, metadata_score, tensor_score,
                        classified_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metadata.file_hash, metadata.filename, metadata.file_size, metadata.file_extension,
                    metadata.primary_type, metadata.sub_type, metadata.confidence, metadata.classification_method,
                    metadata.tensor_count, metadata.architecture, metadata.precision, metadata.quantization, triggers_str,
                    metadata.filename_score, metadata.size_score, metadata.metadata_score, metadata.tensor_score,
                    metadata.classified_at
                ))
                model_id = cursor.lastrowid
            
            # Save metadata
            self._save_metadata_dict(conn, model_id, 'safetensors', metadata.safetensors_metadata)
            self._save_metadata_dict(conn, model_id, 'gguf', metadata.gguf_metadata)
            
            conn.commit()
    
    def _save_metadata_dict(self, conn, model_id: int, metadata_type: str, metadata_dict: Optional[Dict]):
        """Save metadata dictionary to the metadata table"""
        if not metadata_dict:
            return
        
        for key, value in metadata_dict.items():
            # Convert complex values to JSON
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value)
            else:
                value_str = str(value) if value is not None else None
            
            conn.execute(
                "INSERT INTO model_metadata (model_id, metadata_type, key, value) VALUES (?, ?, ?, ?)",
                (model_id, metadata_type, key, value_str)
            )
    
    def clear_cache(self):
        """Clear all cached data"""
        with self.get_db_connection() as conn:
            conn.execute("DELETE FROM model_metadata")
            conn.execute("DELETE FROM models")
            conn.commit()
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        with self.get_db_connection() as conn:
            # Total models
            total = conn.execute("SELECT COUNT(*) as count FROM models").fetchone()['count']
            
            # By type
            type_stats = {}
            cursor = conn.execute(
                "SELECT primary_type, sub_type, COUNT(*) as count FROM models GROUP BY primary_type, sub_type"
            )
            for row in cursor:
                key = f"{row['primary_type']}_{row['sub_type']}"
                type_stats[key] = row['count']
            return {
                'total': total,
                'classifications': type_stats
            }
    


class ModelClassifier:
    """Comprehensive model classification system"""
    
    def __init__(self, hub_path: str, config: Dict):
        self.hub_path = Path(hub_path)
        self.models_path = self.hub_path / 'models'
        self.config = config
        
        # Initialize SQLite database
        self.db_file = self.hub_path / "classification_cache.db"
        self.db = DatabaseManager(self.db_file)
        
        # Get classification config with defaults
        self.class_config = config.get('classification', {})
        self.confidence_threshold = self.class_config.get('confidence_threshold', 0.35)
        self.weights = self.class_config.get('weights', {
            'filename': 0.2,  # Reduced - filenames can be misleading
            'size': 0.2,      # File size is a good indicator
            'tensor': 0.5,    # Highest - tensor analysis is most reliable
            'metadata': 0.1   # SafeTensors metadata when available
        })
        
        # Initialize external API lookups
        enable_api = self.class_config.get('enable_external_apis', True)
        self.civitai = CivitAILookup(enable_api=enable_api)
        
        # Initialize classification rules
        self.initialize_classification_rules()
    
    def get_hub_file_path(self, file_hash: str, filename: str) -> Path:
        """Get the hub file path for a model given its hash and filename"""
        return self.models_path / file_hash / filename
    
    def get_cached_model(self, file_hash: str) -> Optional[ModelMetadata]:
        """Get cached model metadata by file hash"""
        return self.db.get_model_by_hash(file_hash)
    
    def save_model_to_cache(self, metadata: ModelMetadata):
        """Save model metadata to cache"""
        self.db.save_model(metadata)
    
    def initialize_classification_rules(self):
        """Initialize comprehensive classification rules"""
        
        # Filename patterns with confidence weights
        self.filename_patterns = {
            'checkpoint': {
                'patterns': [
                    r'.*checkpoint.*\.safetensors$',
                    r'.*model.*\.safetensors$',
                    r'.*\.ckpt$',
                    r'.*diffusion.*\.safetensors$',
                    r'.*sd.*\.safetensors$',
                    r'.*xl.*\.safetensors$',
                    r'.*FLUX.*\.safetensors$',
                    r'.*flux.*\.safetensors$',
                    r'.*quantized.*\.safetensors$',
                    r'.*Quantized.*\.safetensors$'
                ],
                'negative_patterns': [
                    r'.*lora.*', r'.*vae.*', r'.*controlnet.*',
                    r'.*clip.*', r'.*unet.*', r'.*text.*encoder.*'
                ],
                'confidence': 0.8
            },
            'lora': {
                'patterns': [
                    r'.*lora.*\.safetensors$',
                    r'.*LoRA.*\.safetensors$',
                    r'.*adapter.*\.safetensors$',
                    r'.*\bLoRA\b.*\.safetensors$',
                    r'.*_epochs\.safetensors$',
                    r'.*epochs\.safetensors$',
                    r'.*Style.*FLUX.*LoRA.*\.safetensors$',  # Only FLUX LoRAs with explicit LoRA in name
                    r'.*FLUX.*LoRA.*\.safetensors$',         # Only FLUX LoRAs with explicit LoRA in name
                    r'.*_i2v_.*\.safetensors$',
                    r'.*i2v.*\.safetensors$',
                    r'.*Poses.*\.safetensors$',
                    r'.*_v\d+\.\d+\.safetensors$',
                    r'.*wan.*i2v.*\.safetensors$',
                    r'.*_pov_.*\.safetensors$'
                ],
                'confidence': 0.9
            },
            'vae': {
                'patterns': [
                    r'.*vae.*\.safetensors$',
                    r'.*VAE.*\.safetensors$',
                    r'.*autoencoder.*\.safetensors$',
                    r'.*_vae\.safetensors$',
                    r'.*\.vae\.safetensors$',
                    r'.*Wan.*VAE.*\.safetensors$',
                    r'.*video.*vae.*\.safetensors$'
                ],
                'confidence': 0.9
            },
            'controlnet': {
                'patterns': [
                    r'.*controlnet.*\.safetensors$',
                    r'.*ControlNet.*\.safetensors$',
                    r'.*control.*\.safetensors$'
                ],
                'confidence': 0.9
            },
            'clip': {
                'patterns': [
                    r'.*clip.*\.safetensors$',
                    r'.*CLIP.*\.safetensors$'
                ],
                'confidence': 0.8
            },
            'text_encoder': {
                'patterns': [
                    r'.*text_encoder.*\.safetensors$',
                    r'.*xlm.*roberta.*\.safetensors$',
                    r'.*t5.*\.safetensors$',
                    r'.*umt5.*\.safetensors$',
                    r'.*text.*model.*\.safetensors$'
                ],
                'confidence': 0.9
            },
            'unet': {
                'patterns': [
                    r'.*unet.*\.safetensors$',
                    r'.*UNet.*\.safetensors$',
                    r'.*diffusion_model.*\.safetensors$'
                ],
                'confidence': 0.8
            },
            'gguf': {
                'patterns': [r'.*\.gguf$'],
                'confidence': 0.95
            },
            'upscaler': {
                'patterns': [
                    r'.*esrgan.*\.safetensors$',
                    r'.*upscal.*\.safetensors$',
                    r'.*4x.*\.safetensors$',
                    r'.*2x.*\.safetensors$'
                ],
                'confidence': 0.8
            },
            'embedding': {
                'patterns': [
                    r'.*embedding.*\.safetensors$',
                    r'.*textual.*inversion.*\.safetensors$'
                ],
                'confidence': 0.8
            },
            'video_model': {
                'patterns': [
                    r'.*wan.*text2video.*\.safetensors$',  # Specific WAN text2video models
                    r'.*text2video.*\.safetensors$',
                    r'.*text2video.*\.gguf$',
                    r'.*_text2video_.*\.safetensors$',     # Pattern like wan2.1_text2video_1.3B
                    r'.*fusionx.*\.safetensors$',
                    r'.*fusionx.*\.gguf$',
                    r'.*vace.*\.safetensors$',
                    r'.*vace.*\.gguf$',
                    r'.*t2v.*\.safetensors$',
                    r'.*t2v.*\.gguf$',
                    r'.*i2v.*\.safetensors$',
                    r'.*i2v.*\.gguf$',
                    r'.*ltxv.*\.safetensors$',
                    r'.*hunyuan.*video.*\.safetensors$',
                    r'.*hunyuan.*i2v.*\.safetensors$',
                    r'.*ltx.*video.*\.safetensors$'
                ],
                'negative_patterns': [
                    r'.*vae.*', r'.*VAE.*'  # Exclude VAE files from video model classification
                ],
                'confidence': 0.95  # Increased confidence for video models
            },
            'mask_model': {
                'patterns': [
                    r'.*sam.*\.safetensors$',
                    r'.*mask.*\.safetensors$',
                    r'.*segment.*\.safetensors$'
                ],
                'confidence': 0.9
            },
            'audio_model': {
                'patterns': [
                    r'.*wav2vec.*\.safetensors$',
                    r'.*audio.*\.safetensors$',
                    r'.*speech.*\.safetensors$'
                ],
                'confidence': 0.9
            }
        }
        
        # Get size rules from config or use defaults - Updated for real-world models
        self.size_rules = self.class_config.get('size_rules', {
            'checkpoint': {'min': 1_000_000_000, 'max': 50_000_000_000},    # 1GB-50GB (FLUX models are huge)
            'lora': {'min': 1_000_000, 'max': 2_000_000_000},               # 1MB-2GB (some LoRAs are larger)
            'vae': {'min': 50_000_000, 'max': 2_000_000_000},               # 50MB-2GB (modern VAEs)
            'controlnet': {'min': 100_000_000, 'max': 10_000_000_000},      # 100MB-10GB
            'clip': {'min': 10_000_000, 'max': 2_000_000_000},              # 10MB-2GB (standard CLIP models)
            'text_encoder': {'min': 10_000_000, 'max': 25_000_000_000},     # 10MB-25GB (for large text encoders like UMT5-XXL)
            'unet': {'min': 500_000_000, 'max': 30_000_000_000},            # 500MB-30GB
            'llm': {'min': 100_000_000, 'max': 200_000_000_000},            # 100MB-200GB
            'gguf': {'min': 100_000_000, 'max': 200_000_000_000},           # 100MB-200GB (wide range for all GGUF types)
            'upscaler': {'min': 1_000_000, 'max': 500_000_000},             # 1MB-500MB
            'embedding': {'min': 1_000, 'max': 50_000_000},                 # 1KB-50MB
            'video_model': {'min': 100_000_000, 'max': 100_000_000_000},    # 100MB-100GB
            'mask_model': {'min': 100_000_000, 'max': 2_000_000_000},       # 100MB-2GB
            'audio_model': {'min': 50_000_000, 'max': 1_000_000_000}        # 50MB-1GB
        })
        
        # Tensor shape patterns for SafeTensors
        self.tensor_patterns = {
            'checkpoint': [
                r'model\.diffusion_model\..*',
                r'first_stage_model\..*',
                r'cond_stage_model\..*',
                r'double_blocks\..*',
                r'single_blocks\..*',
                r'.*\.img_attn\..*',
                r'.*\.txt_attn\..*',
                r'.*\.img_mlp\..*',
                r'.*\.txt_mlp\..*'
            ],
            'lora': [
                r'.*\.lora_up\..*',
                r'.*\.lora_down\..*',
                r'.*\.alpha$'
            ],
            'vae': [
                r'first_stage_model\.encoder\..*',
                r'first_stage_model\.decoder\..*',
                r'.*\.encoder\..*',
                r'.*\.decoder\..*'
            ],
            'video_vae': [
                r'encoder\.downsamples\..*\.residual\..*',
                r'decoder\.upsamples\..*\.residual\..*',
                r'encoder\.conv1\..*',
                r'decoder\.conv_out\..*',
                r'.*\.downsamples\..*',
                r'.*\.upsamples\..*'
            ],
            'controlnet': [
                r'control_model\..*',
                r'.*\.control\..*'
            ],
            'clip': [
                r'cond_stage_model\.transformer\..*',
                r'text_model\..*',
                r'.*\.text_projection$'
            ],
            'text_encoder': [
                r'.*\.encoder\..*',
                r'.*\.embeddings\..*',
                r'.*\.attention\..*',
                r'.*\.layer\..*',
                r'.*\.transformer\..*'
            ],
            'unet': [
                r'model\.diffusion_model\..*',
                r'.*\.time_embed\..*',
                r'.*\.input_blocks\..*',
                r'.*\.output_blocks\..*'
            ],
            'upscaler': [
                r'.*\.weight$',
                r'.*\.bias$',
                r'model\..*'
            ],
            'embedding': [
                r'.*\.weight$',
                r'string_to_param\..*'
            ],
            'video_model': [
                r'.*\.temporal\..*',
                r'.*\.video\..*',
                r'.*\.frame\..*',
                r'.*\.motion\..*'
            ]
        }
    
    def extract_hash_from_path(self, file_path: Path) -> str:
        """Extract hash from hub directory structure: /models/{hash}/filename"""
        try:
            # Check if file is in hub structure: .../models/{hash}/filename
            path_parts = file_path.parts
            models_index = None
            
            # Find 'models' directory in path
            for i, part in enumerate(path_parts):
                if part == 'models':
                    models_index = i
                    break
            
            if models_index is not None and len(path_parts) > models_index + 1:
                # Hash should be the directory immediately after 'models'
                hash_candidate = path_parts[models_index + 1]
                
                # Validate it looks like a SHA-256 hash (64 hex characters)
                if len(hash_candidate) == 64 and all(c in '0123456789abcdef' for c in hash_candidate.lower()):
                    return hash_candidate.lower()
            
            return ""
        except Exception:
            return ""
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Get file hash - first try extracting from path, fallback to calculation"""
        # Try to extract from hub directory structure first (much faster)
        extracted_hash = self.extract_hash_from_path(file_path)
        if extracted_hash:
            return extracted_hash
        
        # Fallback to calculation (for files not in hub structure)
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""
    
    def analyze_filename(self, filename: str) -> Dict:
        """Analyze filename for classification clues"""
        filename_lower = filename.lower()
        
        best_match = {'type': 'unknown', 'confidence': 0.0}
        
        for model_type, rules in self.filename_patterns.items():
            # Check positive patterns
            pattern_matches = any(
                re.search(pattern, filename, re.IGNORECASE) 
                for pattern in rules['patterns']
            )
            
            # Check negative patterns
            negative_matches = any(
                re.search(pattern, filename, re.IGNORECASE) 
                for pattern in rules.get('negative_patterns', [])
            )
            
            if pattern_matches and not negative_matches:
                confidence = rules['confidence']
                if confidence > best_match['confidence']:
                    best_match = {'type': model_type, 'confidence': confidence}
        
        return best_match
    
    def analyze_file_size(self, file_size: int) -> Dict:
        """Analyze file size for classification clues"""
        matches = []
        
        for model_type, size_range in self.size_rules.items():
            if size_range['min'] <= file_size <= size_range['max']:
                # Calculate confidence based on how well it fits the range
                range_size = size_range['max'] - size_range['min']
                distance_from_min = file_size - size_range['min']
                confidence = 0.5 + 0.3 * (distance_from_min / range_size)
                matches.append({'type': model_type, 'confidence': min(confidence, 0.8)})
        
        if matches:
            # Return the best match
            return max(matches, key=lambda x: x['confidence'])
        
        return {'type': 'unknown', 'confidence': 0.0}
    
    def determine_sub_type(self, primary_type: str, metadata: ModelMetadata) -> str:
        """Determine sub-type based on specific characteristics"""
        filename_lower = metadata.filename.lower()
        
        if primary_type == 'checkpoint':
            # Enhanced checkpoint sub-type detection for WAN
            wan_patterns = ['wan', 'wanvideo', 'wan2', 'multitalk', 'recammaster',
                           'minimax', 'fun-control', 'uni3c', 'fantasytalking']
            
            if any(pattern in filename_lower for pattern in wan_patterns):
                return 'wan_checkpoint'
            elif 'xl' in filename_lower or 'sdxl' in filename_lower:
                return 'sdxl_checkpoint'
            elif 'sd3' in filename_lower:
                return 'sd3_checkpoint'
            elif 'flux' in filename_lower:
                return 'flux_checkpoint'
            else:
                return 'sd15_checkpoint'
        
        elif primary_type == 'lora':
            # Enhanced LoRA sub-type detection for WAN
            wan_patterns = ['wan', 'wanvideo', 'wan2', 'multitalk', 'recammaster',
                           'minimax', 'fun-control', 'uni3c', 'fantasytalking']
            
            if 'hunyuan' in filename_lower and 'i2v' in filename_lower:
                return 'hunyuan_i2v_lora'
            elif 'hunyuan' in filename_lower:
                return 'hunyuan_lora'
            elif 'i2v' in filename_lower:
                return 'i2v_lora'
            elif 'ltxv' in filename_lower or ('ltx' in filename_lower and 'video' in filename_lower):
                return 'ltxv_lora'
            elif any(pattern in filename_lower for pattern in wan_patterns):
                return 'wan_lora'
            elif 'xl' in filename_lower or 'sdxl' in filename_lower:
                return 'sdxl_lora'
            elif 'flux' in filename_lower:
                return 'flux_lora'
            else:
                return 'sd15_lora'
        
        elif primary_type == 'vae':
            if 'wan' in filename_lower or 'video' in filename_lower:
                return 'video_vae'
            elif 'xl' in filename_lower or 'sdxl' in filename_lower:
                return 'sdxl_vae'
            else:
                return 'sd15_vae'
        
        elif primary_type == 'controlnet':
            # Check for WAN-specific patterns first
            wan_patterns = ['wan', 'wanvideo', 'wan2', 'multitalk', 'recammaster',
                           'minimax', 'fun-control', 'uni3c', 'fantasytalking']
            if any(pattern in filename_lower for pattern in wan_patterns):
                return 'wan_controlnet'
            elif 'xl' in filename_lower or 'sdxl' in filename_lower:
                return 'sdxl_controlnet'
            else:
                return 'sd15_controlnet'
        
        elif primary_type == 'llm':
            if metadata.file_size > 10_000_000_000:  # > 10GB
                return 'large_llm'
            elif metadata.file_size > 1_000_000_000:  # > 1GB
                return 'medium_llm'
            else:
                return 'small_llm'
        
        elif primary_type == 'gguf':
            # Enhanced WAN pattern detection for GGUF files
            wan_patterns = ['wan', 'wanvideo', 'wan2', 'multitalk', 'recammaster',
                           'minimax', 'fun-control', 'uni3c', 'fantasytalking']
            
            # Determine GGUF sub-type based on filename patterns
            if 'ltxv' in filename_lower or ('ltx' in filename_lower and 'video' in filename_lower):
                return 'ltx_video'
            elif any(pattern in filename_lower for pattern in wan_patterns) and ('t2v' in filename_lower or 'text2video' in filename_lower):
                return 'text_to_video'
            elif any(pattern in filename_lower for pattern in wan_patterns) and ('i2v' in filename_lower or 'image2video' in filename_lower):
                return 'image_to_video'
            elif any(pattern in filename_lower for pattern in wan_patterns) or 'phantom' in filename_lower:
                return 'wan_video'
            elif 'vace' in filename_lower or 'moviigen' in filename_lower:
                return 'text_to_video'
            elif 'umt5' in filename_lower or 'encoder' in filename_lower:
                return 'text_encoder'
            elif 'fusionx' in filename_lower:
                return 'wan_video'
            elif any(term in filename_lower for term in ['t2v', 'text2video', 'text_to_video']):
                return 'text_to_video'
            elif any(term in filename_lower for term in ['i2v', 'image2video', 'image_to_video']):
                return 'image_to_video'
            elif 'video' in filename_lower:
                return 'video_model'
            else:
                # Default to LLM for unrecognized GGUF files
                if metadata.file_size > 10_000_000_000:  # > 10GB
                    return 'large_llm'
                elif metadata.file_size > 1_000_000_000:  # > 1GB
                    return 'medium_llm'
                else:
                    return 'small_llm'
        
        elif primary_type == 'video_model':
            # Enhanced WAN pattern detection for video models
            wan_patterns = ['wan', 'wanvideo', 'wan2', 'multitalk', 'recammaster',
                           'minimax', 'fun-control', 'uni3c', 'fantasytalking']
            if 't2v' in filename_lower or 'text2video' in filename_lower:
                return 'text_to_video'
            elif 'i2v' in filename_lower:
                return 'image_to_video'
            elif any(pattern in filename_lower for pattern in wan_patterns):
                return 'wan_video'
            elif 'ltx' in filename_lower and 'video' in filename_lower:
                return 'ltx_video'
            elif 'hunyuan' in filename_lower and 'video' in filename_lower:
                return 'wan_video'  # Hunyuan video models go to ckpts
            else:
                return 'video_model'
        
        elif primary_type == 'clip':
            return 'clip'
        
        elif primary_type == 'text_encoder':
            if 'umt5' in filename_lower or 't5' in filename_lower:
                return 'umt5_text_encoder'
            elif 'xlm' in filename_lower or 'roberta' in filename_lower:
                return 'xlm_roberta_encoder'
            else:
                return 'text_encoder'
        
        elif primary_type == 'mask_model':
            if 'sam' in filename_lower:
                return 'mask_model'
            else:
                return 'mask_model'
        
        elif primary_type == 'audio_model':
            if 'wav2vec' in filename_lower:
                return 'audio_model'
            else:
                return 'audio_model'
        
        return primary_type
    
    def determine_sub_type_with_base(self, primary_type: str, base_model: str, metadata: ModelMetadata) -> str:
        """Enhanced sub-type determination using base model information from metadata"""
        
        # If we have base model info from metadata, use it
        if base_model:
            if primary_type == 'lora':
                return f"{base_model}_lora"
            elif primary_type == 'checkpoint':
                return f"{base_model}_checkpoint"
            elif primary_type == 'vae':
                return f"{base_model}_vae"
            elif primary_type == 'controlnet':
                return f"{base_model}_controlnet"
        
        # Fall back to filename-based detection
        return self.determine_sub_type(primary_type, metadata)
    
    def classify_model(self, model_path: Path, force_reclassify: bool = False, file_hash: str = None) -> ModelMetadata:
        """Comprehensive model classification - always reclassifies with latest logic"""
        
        # Check if this is an LFS pointer file - if so, skip classification entirely
        if SafeTensorsExtractor.is_lfs_pointer_file(model_path):
            return ModelMetadata(
                file_hash="",
                filename=model_path.name,
                file_size=model_path.stat().st_size,
                file_extension=model_path.suffix.lower(),
                primary_type="lfs_pointer",
                sub_type="undownloaded",
                confidence=1.0,
                classification_method="lfs_detection",
                classified_at=datetime.now().isoformat()
            )
        
        # Use provided hash or calculate if not provided
        if file_hash is None:
            file_hash = self.calculate_file_hash(model_path)
        
        if not file_hash:
            # Create minimal metadata for failed hash
            return ModelMetadata(
                file_hash="",
                filename=model_path.name,
                file_size=model_path.stat().st_size,
                file_extension=model_path.suffix.lower(),
                primary_type="unknown",
                sub_type="unknown",
                confidence=0.0,
                classification_method="failed",
                classified_at=datetime.now().isoformat()
            )
        
        # EARLY GGUF CLASSIFICATION: Skip expensive operations for GGUF files
        if model_path.suffix.lower() == '.gguf':
            # Check for cached metadata first
            cached_metadata = self.get_cached_model(file_hash)
            if cached_metadata and not force_reclassify:
                return cached_metadata
            
            # Create GGUF metadata structure
            metadata = ModelMetadata(
                file_hash=file_hash,
                filename=model_path.name,
                file_size=model_path.stat().st_size,
                file_extension=model_path.suffix.lower(),
                primary_type="gguf",
                sub_type="unknown",  # Will be determined below
                confidence=0.95,
                classification_method="gguf_auto_classification",
                classified_at=datetime.now().isoformat()
            )
            
            # Extract GGUF metadata if needed (reuse cached if available)
            if cached_metadata and cached_metadata.gguf_metadata:
                metadata.gguf_metadata = cached_metadata.gguf_metadata
                metadata.tensor_count = cached_metadata.tensor_count
            else:
                gguf_metadata = GGUFExtractor.extract_metadata(model_path)
                metadata.gguf_metadata = gguf_metadata
                metadata.tensor_count = gguf_metadata.get('tensor_count', 0)
            
            # Determine sub-type for GGUF
            metadata.sub_type = self.determine_sub_type('gguf', metadata)
            
            # Check manual overrides
            manual_overrides = self.class_config.get('manual_overrides', {})
            if metadata.filename in manual_overrides:
                override_classification = manual_overrides[metadata.filename]
                if '/' in override_classification:
                    primary_type, sub_type = override_classification.split('/', 1)
                    metadata.primary_type = primary_type
                    metadata.sub_type = sub_type
                    metadata.confidence = 1.0
                    metadata.classification_method = "manual_override"
            
            # Save to cache and return
            self.save_model_to_cache(metadata)
            return metadata
        
        # Check for cached metadata to reuse (but always reclassify)
        cached_metadata = self.get_cached_model(file_hash)
        
        # Initialize metadata structure
        metadata = ModelMetadata(
            file_hash=file_hash,
            filename=model_path.name,
            file_size=model_path.stat().st_size,
            file_extension=model_path.suffix.lower(),
            primary_type="unknown",
            sub_type="unknown",
            confidence=0.0,
            classification_method="multi_layer",
            classified_at=datetime.now().isoformat()
        )
        
        # Layer 1: Filename analysis
        filename_results = self.analyze_filename(model_path.name)
        metadata.filename_score = filename_results['confidence']
        
        # Layer 2: Size analysis
        size_results = self.analyze_file_size(model_path.stat().st_size)
        metadata.size_score = size_results['confidence']
        
        # Layer 3: Metadata extraction (reuse cached if available)
        if cached_metadata and cached_metadata.safetensors_metadata:
            # Reuse cached SafeTensors metadata
            metadata.safetensors_metadata = cached_metadata.safetensors_metadata
            metadata.tensor_count = cached_metadata.tensor_count
            metadata.architecture = cached_metadata.architecture
            
            # Recalculate tensor score with current patterns
            if 'tensor_names' in cached_metadata.safetensors_metadata:
                tensor_analyzer = TensorAnalyzer(self.tensor_patterns)
                tensor_results = tensor_analyzer.analyze_tensors(cached_metadata.safetensors_metadata['tensor_names'])
                metadata.tensor_score = max(tensor_results.values()) if tensor_results else 0
                metadata.architecture = tensor_analyzer.get_architecture_hints(
                    cached_metadata.safetensors_metadata['tensor_names'], cached_metadata.safetensors_metadata
                )
        elif cached_metadata and cached_metadata.gguf_metadata:
            # Reuse cached GGUF metadata
            metadata.gguf_metadata = cached_metadata.gguf_metadata
            metadata.tensor_count = cached_metadata.tensor_count
        elif model_path.suffix.lower() == '.safetensors':
            # Extract fresh SafeTensors metadata
            st_metadata = SafeTensorsExtractor.extract_metadata(model_path)
            metadata.safetensors_metadata = st_metadata
            
            if 'tensors' in st_metadata and 'tensor_names' in st_metadata:
                tensor_analyzer = TensorAnalyzer(self.tensor_patterns)
                tensor_results = tensor_analyzer.analyze_tensors(st_metadata['tensor_names'])
                metadata.tensor_score = max(tensor_results.values()) if tensor_results else 0
                metadata.tensor_count = st_metadata['tensor_count']
                metadata.architecture = tensor_analyzer.get_architecture_hints(
                    st_metadata['tensor_names'], st_metadata
                )
        elif model_path.suffix.lower() == '.gguf':
            # This should not be reached anymore since GGUF files are handled early
            # Extract fresh GGUF metadata
            gguf_metadata = GGUFExtractor.extract_metadata(model_path)
            metadata.gguf_metadata = gguf_metadata
            metadata.tensor_count = gguf_metadata.get('tensor_count', 0)
        
        # Layer 4: Check manual overrides first (highest priority)
        manual_overrides = self.class_config.get('manual_overrides', {})
        if metadata.filename in manual_overrides:
            override_classification = manual_overrides[metadata.filename]
            if '/' in override_classification:
                primary_type, sub_type = override_classification.split('/', 1)
                metadata.primary_type = primary_type
                metadata.sub_type = sub_type
                metadata.confidence = 1.0
                metadata.classification_method = "manual_override"
                
                # Save to cache and return
                self.save_model_to_cache(metadata)
                return metadata
        
        # Layer 5: External API lookups (after hash calculation)
        civitai_result = None
        if file_hash:  # Only lookup if we have a valid hash
            civitai_result = self.civitai.lookup_by_hash(file_hash)
        
        # Layer 6: Final classification decision
        classification = self.make_final_classification(
            filename_results, size_results, metadata, civitai_result
        )
        
        metadata.primary_type = classification['primary_type']
        metadata.sub_type = classification['sub_type']
        metadata.confidence = classification['confidence']
        
        # Extract trigger words for LoRA models
        if metadata.primary_type and 'lora' in metadata.primary_type.lower():
            if metadata.safetensors_metadata and 'metadata' in metadata.safetensors_metadata:
                triggers = self._extract_triggers_from_safetensors_metadata(
                    metadata.safetensors_metadata['metadata']
                )
                metadata.triggers = triggers
                if triggers:
                    print(f"    🎯 Extracted triggers: {', '.join(triggers)}")
        
        # Store classification source for display
        if hasattr(metadata, '__dict__'):
            metadata.source = classification.get('source', 'unknown')
        
        # Only cache known models - unknown models should be re-evaluated on each scan
        if metadata.primary_type != "unknown":
            self.save_model_to_cache(metadata)
        
        return metadata

    def _extract_triggers_from_safetensors_metadata(self, st_metadata: Dict) -> List[str]:
        """Extract triggers directly from SafeTensors metadata during classification"""
        triggers = []
        
        # Parse ss_tag_frequency
        if 'ss_tag_frequency' in st_metadata:
            triggers.extend(self._parse_tag_frequency(st_metadata['ss_tag_frequency']))
        
        # Parse ss_output_name
        if 'ss_output_name' in st_metadata:
            triggers.extend(self._parse_output_name(st_metadata['ss_output_name']))
        
        # Parse modelspec.title
        if 'modelspec.title' in st_metadata:
            triggers.extend(self._parse_model_title(st_metadata['modelspec.title']))
        
        # Look for explicit trigger fields
        for key, value in st_metadata.items():
            if any(pattern in key.lower() for pattern in ['trigger', 'activation', 'keyword']):
                triggers.extend(self._parse_trigger_field(str(value)))
        
        return self._clean_and_deduplicate_triggers(triggers)

    def _parse_tag_frequency(self, tag_frequency_str: str) -> List[str]:
        """Parse ss_tag_frequency JSON to extract trigger words"""
        try:
            import json
            tag_data = json.loads(tag_frequency_str)
            
            triggers = []
            # Handle nested structure: {"img": {"d4ndon": 511}}
            for dataset_name, tags in tag_data.items():
                if isinstance(tags, dict):
                    for tag, frequency in tags.items():
                        # Filter out common non-trigger tags
                        if self._is_likely_trigger_word(tag, frequency):
                            triggers.append(tag.upper())
            
            return triggers
        except (json.JSONDecodeError, TypeError, AttributeError):
            return []

    def _parse_output_name(self, output_name: str) -> List[str]:
        """Extract potential trigger words from ss_output_name"""
        if not output_name:
            return []
        
        # Common patterns: "ModelName_Style", "TriggerWord_LoRA", etc.
        # Split on common separators and filter
        potential_triggers = []
        for part in output_name.replace('_', ' ').replace('-', ' ').split():
            if self._is_likely_trigger_word(part):
                potential_triggers.append(part.upper())
        
        return potential_triggers

    def _parse_model_title(self, title: str) -> List[str]:
        """Extract potential trigger words from modelspec.title"""
        if not title:
            return []
        
        # Similar logic to output_name but more conservative
        potential_triggers = []
        for part in title.replace('_', ' ').replace('-', ' ').split():
            if self._is_likely_trigger_word(part) and len(part) > 2:
                potential_triggers.append(part.upper())
        
        return potential_triggers

    def _parse_trigger_field(self, value: str) -> List[str]:
        """Parse explicit trigger fields"""
        if not value:
            return []
        
        try:
            # Try to parse as JSON first
            import json
            parsed_value = json.loads(value)
            if isinstance(parsed_value, list):
                return [str(item).upper() for item in parsed_value if item]
            else:
                return [str(parsed_value).upper()]
        except (json.JSONDecodeError, TypeError):
            # Treat as string, split on common separators
            triggers = []
            for part in value.replace(',', ' ').replace(';', ' ').split():
                part = part.strip().strip('"\'')
                if part and len(part) > 1:
                    triggers.append(part.upper())
            return triggers

    def _is_likely_trigger_word(self, word: str, frequency: int = None) -> bool:
        """Determine if a word is likely a trigger word"""
        if not word or len(word) < 2:
            return False
        
        word_lower = word.lower()
        
        # Exclude common non-trigger words
        excluded_words = {
            'img', 'image', 'photo', 'picture', 'style', 'lora', 'model',
            'training', 'dataset', 'epoch', 'step', 'version', 'v1', 'v2',
            'base', 'fine', 'tuned', 'checkpoint', 'safetensors', 'the', 'and',
            'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        if word_lower in excluded_words:
            return False
        
        # If we have frequency data, high frequency suggests it's a trigger
        if frequency is not None and frequency > 100:
            return True
        
        # Look for patterns that suggest trigger words
        # - Mixed case/numbers (like D4NDON)
        # - Unique names
        # - Not common English words
        return True

    def _clean_and_deduplicate_triggers(self, triggers: List[str]) -> List[str]:
        """Clean and deduplicate trigger words"""
        if not triggers:
            return []
        
        # Remove duplicates while preserving order
        seen = set()
        cleaned = []
        
        for trigger in triggers:
            trigger_clean = trigger.strip().upper()
            if trigger_clean and trigger_clean not in seen and len(trigger_clean) > 1:
                seen.add(trigger_clean)
                cleaned.append(trigger_clean)
        
        return cleaned[:5]  # Limit to 5 trigger words
    
    def display_metadata_summary(self, st_metadata: Dict):
        """Display a clean summary of the most relevant metadata"""
        if not st_metadata:
            print("    No metadata found")
            return
        
        # Key metadata fields to display
        important_fields = [
            'model_type', 'type', 'architecture', 'class',
            'base_model', 'target_model', 'base', 'model',
            'description', 'tags', 'version',
            'epochs', 'steps', 'learning_rate', 'rank', 'alpha'
        ]
        
        displayed_count = 0
        max_display = 4
        
        for key, value in st_metadata.items():
            if displayed_count >= max_display:
                break
                
            key_lower = key.lower()
            
            # Check if this is an important field or contains useful info
            is_important = any(field in key_lower for field in important_fields)
            
            if is_important and value is not None:
                # Clean up the value for display
                value_str = str(value)
                if len(value_str) > 60:
                    value_str = value_str[:57] + "..."
                
                print(f"    {key}: {value_str}")
                displayed_count += 1
        
        # If we didn't find important fields, show first few fields
        if displayed_count == 0:
            for key, value in list(st_metadata.items())[:max_display]:
                if value is not None:
                    value_str = str(value)
                    if len(value_str) > 60:
                        value_str = value_str[:57] + "..."
                    print(f"    {key}: {value_str}")
                    displayed_count += 1
                    if displayed_count >= max_display:
                        break
        
        if displayed_count == 0:
            print("    Metadata present but no displayable fields")
    
    def extract_metadata_classification(self, metadata: ModelMetadata) -> Dict:
        """Extract classification from SafeTensors metadata (highest priority)"""
        if not metadata.safetensors_metadata:
            return {'type': None, 'confidence': 0.0, 'base_model': None}
        
        st_metadata = metadata.safetensors_metadata.get('metadata', {})
        if not st_metadata:
            return {'type': None, 'confidence': 0.0, 'base_model': None}
        
        # Metadata already displayed earlier in the process
        
        # Common metadata fields to check
        metadata_type = None
        base_model = None
        confidence = 0.95  # High confidence for direct metadata
        
        # Check for direct type indicators
        for key, value in st_metadata.items():
            key_lower = key.lower()
            value_str = str(value).lower() if value else ""
            
            # Model type detection - expanded patterns
            if any(pattern in key_lower for pattern in ['type', 'model_type', 'architecture', 'class']):
                if 'lora' in value_str:
                    metadata_type = 'lora'
                elif any(term in value_str for term in ['checkpoint', 'diffusion', 'stable_diffusion']):
                    metadata_type = 'checkpoint'
                elif 'vae' in value_str:
                    metadata_type = 'vae'
                elif 'controlnet' in value_str:
                    metadata_type = 'controlnet'
                elif 'unet' in value_str:
                    metadata_type = 'unet'
                elif 'clip' in value_str:
                    metadata_type = 'clip'
                elif any(term in value_str for term in ['text_encoder', 'encoder', 't5', 'umt5']):
                    metadata_type = 'text_encoder'
            
            # Base model detection - expanded patterns
            if any(pattern in key_lower for pattern in ['base', 'model', 'target', 'version']):
                if 'flux' in value_str:
                    base_model = 'flux'
                elif any(term in value_str for term in ['sdxl', 'xl', 'stable_diffusion_xl']):
                    base_model = 'sdxl'
                elif 'sd3' in value_str or 'stable_diffusion_3' in value_str:
                    base_model = 'sd3'
                elif any(term in value_str for term in ['sd15', 'sd1.5', 'stable_diffusion_1']):
                    base_model = 'sd15'
            
            # Check in description, tags, or other text fields
            if any(pattern in key_lower for pattern in ['description', 'info', 'tags', 'comment', 'notes']):
                if 'lora' in value_str and not metadata_type:
                    metadata_type = 'lora'
                if 'flux' in value_str and not base_model:
                    base_model = 'flux'
                elif 'sdxl' in value_str and not base_model:
                    base_model = 'sdxl'
                elif 'sd3' in value_str and not base_model:
                    base_model = 'sd3'
        
        # Special handling for training-related metadata
        if not metadata_type:
            # Check for training indicators that suggest LoRA
            training_indicators = ['epochs', 'steps', 'learning_rate', 'rank', 'alpha']
            if any(indicator in key_lower for key in st_metadata.keys() for indicator in training_indicators):
                metadata_type = 'lora'
        
        return {
            'type': metadata_type,
            'confidence': confidence if metadata_type else 0.0,
            'base_model': base_model
        }
    
    def make_final_classification(self, filename_results: Dict, size_results: Dict,
                                metadata: ModelMetadata, civitai_result: Dict = None) -> Dict:
        """Make final classification decision - CivitAI first, then metadata, then weighted scoring"""
        
        # PRIORITY 1: CivitAI API lookup (highest reliability for SD models)
        if civitai_result and civitai_result.get('found'):
            primary_type = civitai_result['primary_type']
            base_model = civitai_result['base_model']
            confidence = civitai_result['confidence']
            
            # Determine sub-type with CivitAI base model info
            sub_type = self.determine_sub_type_with_base(primary_type, base_model, metadata)
            
            return {
                'primary_type': primary_type,
                'sub_type': sub_type,
                'confidence': confidence,
                'source': 'civitai'
            }
        
        # PRIORITY 2: Check SafeTensors metadata
        metadata_classification = self.extract_metadata_classification(metadata)
        if metadata_classification['type'] and metadata_classification['confidence'] > 0.8:
            primary_type = metadata_classification['type']
            base_model = metadata_classification['base_model']
            confidence = metadata_classification['confidence']
            
            # Determine sub-type with base model info
            sub_type = self.determine_sub_type_with_base(primary_type, base_model, metadata)
            
            return {
                'primary_type': primary_type,
                'sub_type': sub_type,
                'confidence': confidence,
                'source': 'metadata'
            }
        
        # PRIORITY 3: Check for architecture-based detection
        if metadata.architecture == 'video_vae':
            return {
                'primary_type': 'vae',
                'sub_type': 'video_vae',
                'confidence': 0.95,  # High confidence for tensor-based detection
                'source': 'tensor_analysis'
            }
        elif metadata.architecture in ['flux', 'flux_quantized']:
            return {
                'primary_type': 'checkpoint',
                'sub_type': 'flux_checkpoint',
                'confidence': 0.95,  # High confidence for tensor-based detection
                'source': 'tensor_analysis'
            }
        
        # PRIORITY 4: Fall back to weighted scoring system
        type_scores = {}
        all_types = set(self.filename_patterns.keys())
        
        for model_type in all_types:
            score = 0.0
            
            # Filename contribution
            if filename_results.get('type') == model_type:
                score += self.weights['filename'] * filename_results['confidence']
            
            # Size contribution
            if size_results.get('type') == model_type:
                score += self.weights['size'] * size_results['confidence']
            
            # Tensor contribution (if available)
            if metadata.tensor_score > 0:
                score += self.weights['tensor'] * metadata.tensor_score
            
            # Metadata contribution (basic for now)
            if metadata.architecture and metadata.architecture in model_type:
                score += self.weights['metadata'] * 0.5
            
            type_scores[model_type] = score
        
        # Find best classification
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            best_score = type_scores[best_type]
        else:
            best_type = 'unknown'
            best_score = 0.0
        
        # Apply confidence threshold
        if best_score < self.confidence_threshold:
            best_type = 'unknown'
        
        # Determine sub-type
        sub_type = self.determine_sub_type(best_type, metadata)
        
        return {
            'primary_type': best_type,
            'sub_type': sub_type,
            'confidence': best_score,
            'source': 'local'
        }
    
    def classify_all_models(self, clear_cache: bool = False) -> Dict:
        """Classify all models in the hub"""
        
        if clear_cache:
            self.db.clear_cache()
            print("Classification cache cleared.")
        
        models_dir = self.hub_path / 'models'
        if not models_dir.exists():
            return {'error': 'Models directory not found'}
        
        results = {
            'total': 0,
            'new': 0,
            'updated': 0,
            'cached': 0,
            'failed': 0,
            'classifications': {}
        }
        
        # Find all model files
        model_files = []
        for hash_dir in models_dir.iterdir():
            if hash_dir.is_dir():
                for file_path in hash_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in ['.safetensors', '.gguf', '.ckpt', '.pth']:
                        model_files.append(file_path)
        
        results['total'] = len(model_files)
        
        if results['total'] == 0:
            print("No model files found in hub.")
            return results
        
        # Classify each model
        for i, model_path in enumerate(model_files, 1):
            try:
                print(f"[{i}/{len(model_files)}] Classifying {model_path.name}...")
                
                # Extract hash from path (fast) or calculate if needed
                file_hash = self.calculate_file_hash(model_path)
                if file_hash:
                    # Show if hash was extracted vs calculated
                    extracted_hash = self.extract_hash_from_path(model_path)
                    if extracted_hash:
                        print(f"    Hash: {file_hash[:16]}... (from path)")
                    else:
                        print(f"    Hash: {file_hash[:16]}... (calculated)")
                else:
                    print("    Hash calculation failed")
                
                was_cached = self.get_cached_model(file_hash) is not None if file_hash else False
                
                # Show metadata immediately (after fast hash extraction)
                if model_path.suffix.lower() == '.safetensors':
                    quick_metadata = SafeTensorsExtractor.extract_metadata(model_path)
                    if 'metadata' in quick_metadata:
                        self.display_metadata_summary(quick_metadata.get('metadata', {}))
                    else:
                        print("    No metadata found in SafeTensors header")
                elif model_path.suffix.lower() == '.gguf':
                    print("    GGUF file - metadata extraction during classification")
                else:
                    print("    Non-SafeTensors file - limited metadata available")
                
                # Always reclassify with latest logic (architectural requirement)
                metadata = self.classify_model(model_path, force_reclassify=True)
                
                if was_cached:
                    results['updated'] += 1  # Always updated since we always reclassify
                else:
                    results['new'] += 1
                
                # Track classification types
                class_key = f"{metadata.primary_type}_{metadata.sub_type}"
                results['classifications'][class_key] = results['classifications'].get(class_key, 0) + 1
                
                # Show classification source
                source_info = ""
                if hasattr(metadata, 'source'):
                    source_info = f" [{metadata.source}]"
                
                print(f"    → {metadata.primary_type}/{metadata.sub_type} (confidence: {metadata.confidence:.2f}){source_info}")
                
            except Exception as e:
                print(f"    → Failed: {e}")
                results['failed'] += 1
        
        # Get final statistics from database
        db_stats = self.db.get_statistics()
        results['classifications'] = db_stats['classifications']
        
        return results
    
    def reclassify_from_database(self) -> Dict:
        """Reclassify all models using cached database metadata with latest classification logic"""
        
        results = {
            'total': 0,
            'reclassified': 0,
            'failed': 0,
            'classifications': {}
        }
        
        try:
            with self.db.get_db_connection() as conn:
                # Get only models marked for reclassification
                cursor = conn.execute("""
                    SELECT id, file_hash, filename, file_size, file_extension,
                           primary_type, sub_type, confidence, classification_method,
                           tensor_count, architecture, precision, quantization, triggers
                    FROM models
                    WHERE primary_type != 'lfs_pointer' AND reclassify != ''
                    ORDER BY filename ASC
                """)
                models = cursor.fetchall()
                
                results['total'] = len(models)
                
                if results['total'] == 0:
                    print("No models found in database.")
                    return results
                
                print(f"Found {results['total']} models in database to reclassify...")
                
                for i, model_row in enumerate(models, 1):
                    try:
                        model_dict = dict(model_row)
                        print(f"[{i}/{len(models)}] Reclassifying {model_dict['filename']}...")
                        
                        # Reconstruct ModelMetadata from database row
                        metadata = ModelMetadata(
                            file_hash=model_dict['file_hash'],
                            filename=model_dict['filename'],
                            file_size=model_dict['file_size'],
                            file_extension=model_dict['file_extension'],
                            primary_type=model_dict['primary_type'],  # Will be updated
                            sub_type=model_dict['sub_type'],          # Will be updated
                            confidence=model_dict['confidence'],      # Will be updated
                            classification_method="reclassify_from_db",
                            tensor_count=model_dict.get('tensor_count'),
                            architecture=model_dict.get('architecture'),
                            precision=model_dict.get('precision'),
                            quantization=model_dict.get('quantization'),
                            classified_at=datetime.now().isoformat()
                        )
                        
                        # Get cached metadata from database
                        metadata_cursor = conn.execute(
                            "SELECT metadata_type, key, value FROM model_metadata WHERE model_id = ?",
                            (model_dict['id'],)
                        )
                        
                        # Reconstruct metadata dictionaries
                        safetensors_metadata = {}
                        gguf_metadata = {}
                        
                        for meta_row in metadata_cursor:
                            meta_type, key, value = meta_row
                            try:
                                # Try to parse JSON values
                                parsed_value = json.loads(value) if value else None
                            except (json.JSONDecodeError, TypeError):
                                parsed_value = value
                            
                            if meta_type == 'safetensors':
                                safetensors_metadata[key] = parsed_value
                            elif meta_type == 'gguf':
                                gguf_metadata[key] = parsed_value
                        
                        # Add metadata to ModelMetadata object
                        metadata.safetensors_metadata = safetensors_metadata if safetensors_metadata else None
                        metadata.gguf_metadata = gguf_metadata if gguf_metadata else None
                        
                        # Parse triggers from database column
                        triggers_str = model_dict.get('triggers')
                        if triggers_str:
                            metadata.triggers = [t.strip() for t in triggers_str.split(',') if t.strip()]
                        else:
                            metadata.triggers = None
                        
                        # Apply latest classification logic using cached metadata
                        old_classification = f"{model_dict['primary_type']}/{model_dict['sub_type']}"
                        
                        # PRIORITY 1: Check manual overrides first
                        manual_overrides = self.class_config.get('manual_overrides', {})
                        if metadata.filename in manual_overrides:
                            override_classification = manual_overrides[metadata.filename]
                            if '/' in override_classification:
                                primary_type, sub_type = override_classification.split('/', 1)
                                metadata.primary_type = primary_type
                                metadata.sub_type = sub_type
                                metadata.confidence = 1.0
                                metadata.classification_method = "manual_override"
                                
                                new_classification = f"{metadata.primary_type}/{metadata.sub_type}"
                                print(f"    → {old_classification} → {new_classification} (confidence: {metadata.confidence:.2f}) [MANUAL OVERRIDE]")
                                
                                # Save updated classification to database
                                self.save_model_to_cache(metadata)
                                
                                # Track classification types
                                class_key = f"{metadata.primary_type}_{metadata.sub_type}"
                                results['classifications'][class_key] = results['classifications'].get(class_key, 0) + 1
                                results['reclassified'] += 1
                                continue
                        
                        # Re-run classification logic with cached metadata
                        filename_results = self.analyze_filename(metadata.filename)
                        size_results = self.analyze_file_size(metadata.file_size)
                        
                        # Set scores from cached data
                        metadata.filename_score = filename_results['confidence']
                        metadata.size_score = size_results['confidence']
                        
                        # Handle GGUF files specifically - force GGUF classification
                        if metadata.file_extension.lower() == '.gguf':
                            # Force GGUF classification with high confidence
                            filename_results = {'type': 'gguf', 'confidence': 0.95}
                            size_results = {'type': 'gguf', 'confidence': 0.9}
                            metadata.filename_score = 0.95
                            metadata.size_score = 0.9
                            
                            # Force primary type to gguf
                            metadata.primary_type = 'gguf'
                            metadata.sub_type = self.determine_sub_type('gguf', metadata)
                            metadata.confidence = 0.95
                            
                            new_classification = f"{metadata.primary_type}/{metadata.sub_type}"
                            if old_classification != new_classification:
                                print(f"    → {old_classification} → {new_classification} (confidence: {metadata.confidence:.2f}) [GGUF AUTO-CLASSIFICATION]")
                            else:
                                print(f"    → {new_classification} (confidence: {metadata.confidence:.2f}) [GGUF AUTO-CLASSIFICATION]")
                            
                            # Save updated classification to database
                            self.save_model_to_cache(metadata)
                            
                            # Track classification types
                            class_key = f"{metadata.primary_type}_{metadata.sub_type}"
                            results['classifications'][class_key] = results['classifications'].get(class_key, 0) + 1
                            results['reclassified'] += 1
                            continue
                        
                        # Use cached tensor analysis if available (for SafeTensors)
                        if metadata.safetensors_metadata and 'tensor_names' in metadata.safetensors_metadata:
                            tensor_analyzer = TensorAnalyzer(self.tensor_patterns)
                            tensor_results = tensor_analyzer.analyze_tensors(metadata.safetensors_metadata['tensor_names'])
                            metadata.tensor_score = max(tensor_results.values()) if tensor_results else 0
                            metadata.architecture = tensor_analyzer.get_architecture_hints(
                                metadata.safetensors_metadata['tensor_names'], metadata.safetensors_metadata
                            )
                        
                        # Always perform CivitAI lookup during reclassification (architectural requirement)
                        civitai_result = None
                        if metadata.file_hash:
                            civitai_result = self.civitai.lookup_by_hash(metadata.file_hash)
                        
                        # Apply latest classification decision logic
                        classification = self.make_final_classification(
                            filename_results, size_results, metadata, civitai_result
                        )
                        
                        # Update classification
                        metadata.primary_type = classification['primary_type']
                        metadata.sub_type = classification['sub_type']
                        metadata.confidence = classification['confidence']
                        
                        new_classification = f"{metadata.primary_type}/{metadata.sub_type}"
                        
                        # Show what changed
                        if old_classification != new_classification:
                            print(f"    → {old_classification} → {new_classification} (confidence: {metadata.confidence:.2f}) [UPDATED]")
                        else:
                            print(f"    → {new_classification} (confidence: {metadata.confidence:.2f}) [UNCHANGED]")
                        
                        # Save updated classification to database
                        self.save_model_to_cache(metadata)
                        
                        # Track classification types
                        class_key = f"{metadata.primary_type}_{metadata.sub_type}"
                        results['classifications'][class_key] = results['classifications'].get(class_key, 0) + 1
                        
                        results['reclassified'] += 1
                        
                    except Exception as e:
                        print(f"    → Failed: {e}")
                        results['failed'] += 1
                
                # Clear reclassify flags for successfully processed models
                conn.execute("UPDATE models SET reclassify = '' WHERE reclassify != ''")
                conn.commit()
                
                # Get final statistics from database
                db_stats = self.db.get_statistics()
                results['classifications'] = db_stats['classifications']
                
                return results
                
        except Exception as e:
            return {'error': f'Database reclassification failed: {e}'}

def run_trigger_migration(hub_path: str = None):
    """Run the complete trigger word migration"""
    import yaml
    
    # Load config
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("Error: config.yaml not found. Please run from the modelhub directory.")
        return
    
    # Use provided hub_path or get from config
    if hub_path is None:
        hub_path = config.get('model_hub', {}).get('path', '/mnt/llm/model-hub')
    
    print("Starting trigger word migration...")
    print(f"Hub path: {hub_path}")
    
    # Initialize classifier
    classifier = ModelClassifier(hub_path, config)
    
    # Run database migration
    classifier.db.migrate_add_triggers_column()
    
    print("Migration completed successfully!")
    print("\nNext steps:")
    print("1. Run classification to extract trigger words for new models")
    print("2. Use 'Find Models' to see trigger words in search results")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "migrate-triggers":
            hub_path = sys.argv[2] if len(sys.argv) > 2 else None
            run_trigger_migration(hub_path)
        else:
            print("Usage:")
            print("  python hubclassify.py migrate-triggers [hub_path]")
    else:
        print("Model Hub Classifier")
        print("===================")
        print("Available commands:")
        print("  python hubclassify.py migrate-triggers [hub_path]  # Add triggers column and migrate existing data")