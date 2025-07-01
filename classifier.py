#!/usr/bin/env python3
"""
ModelHub Classification System
Comprehensive AI model classification with multi-layer analysis
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
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ClassificationResult:
    """Model classification result"""
    primary_type: str
    sub_type: str
    confidence: float
    method: str
    architecture: Optional[str] = None
    triggers: Optional[List[str]] = None
    base_model: Optional[str] = None
    tensor_count: Optional[int] = None

class SafeTensorsExtractor:
    """Extract metadata from SafeTensors files"""
    
    @staticmethod
    def is_lfs_pointer_file(file_path: Path) -> bool:
        """Check if file is an undownloaded LFS pointer file"""
        try:
            file_size = file_path.stat().st_size
            if file_size > 500:  # Too large to be an LFS pointer
                return False
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(200)
                
            lfs_markers = [
                'version https://git-lfs.github.com/spec/',
                'oid sha256:',
                'size '
            ]
            
            return all(marker in content for marker in lfs_markers)
            
        except Exception:
            return False
    
    @staticmethod
    def extract_metadata(file_path: Path) -> Dict:
        """Extract comprehensive metadata from SafeTensors file"""
        try:
            if SafeTensorsExtractor.is_lfs_pointer_file(file_path):
                return {'error': 'LFS pointer file - not downloaded'}
            
            with open(file_path, 'rb') as f:
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    return {'error': 'File too small to be valid SafeTensors'}
                
                header_size = struct.unpack('<Q', header_size_bytes)[0]
                
                if header_size > 100_000_000:  # 100MB header limit
                    return {'error': 'Header size too large'}
                
                header_data = f.read(header_size)
                if len(header_data) < header_size:
                    return {'error': 'Incomplete header data'}
                
                header = json.loads(header_data.decode('utf-8'))
                
                metadata = header.get('__metadata__', {})
                tensors = {k: v for k, v in header.items() if k != '__metadata__'}
                
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
            if SafeTensorsExtractor.is_lfs_pointer_file(file_path):
                return {'error': 'LFS pointer file - not downloaded'}
            
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic != b'GGUF':
                    return {'error': 'Not a valid GGUF file'}
                
                version = struct.unpack('<I', f.read(4))[0]
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
                
                if tensor_count > 10000 or metadata_kv_count > 1000:
                    return {'error': 'Suspicious tensor or metadata counts'}
                
                metadata = {
                    'version': version,
                    'tensor_count': tensor_count,
                    'metadata_count': metadata_kv_count
                }
                
                file_size = file_path.stat().st_size
                metadata['file_size'] = file_size
                
                if file_size > 10_000_000_000:  # > 10GB
                    metadata['estimated_type'] = 'large_llm'
                elif file_size > 1_000_000_000:  # > 1GB
                    metadata['estimated_type'] = 'medium_llm'
                else:
                    metadata['estimated_type'] = 'small_llm'
                
                return metadata
                
        except Exception as e:
            return {'error': f'Failed to extract GGUF metadata: {str(e)}'}

class CivitAILookup:
    """CivitAI API integration for model lookup by hash"""
    
    def __init__(self, enable_api: bool = True):
        self.enable_api = enable_api
        self.base_url = "https://civitai.com/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ModelHub/1.0'
        })
        self.rate_limit_delay = 1.0
        self.last_request_time = 0
    
    def lookup_by_hash(self, file_hash: str) -> Dict:
        """Lookup model information by SHA-256 hash"""
        if not self.enable_api:
            return {'found': False, 'source': 'api_disabled'}
        
        try:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
            
            url = f"{self.base_url}/model-versions/by-hash/{file_hash}"
            
            response = self.session.get(url, timeout=10)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                return self.parse_civitai_response(data)
            elif response.status_code == 404:
                return {'found': False, 'source': 'not_found'}
            else:
                return {'found': False, 'source': 'api_error'}
                
        except requests.exceptions.Timeout:
            return {'found': False, 'source': 'timeout'}
        except requests.exceptions.RequestException:
            return {'found': False, 'source': 'network_error'}
        except Exception:
            return {'found': False, 'source': 'unknown_error'}
    
    def parse_civitai_response(self, data: Dict) -> Dict:
        """Parse CivitAI API response to extract model classification"""
        try:
            model_info = data.get('model', {})
            version_info = data
            
            model_type = model_info.get('type', '').lower()
            
            type_mapping = {
                'checkpoint': 'checkpoint',
                'lora': 'lora',
                'lycoris': 'lora',
                'textualinversion': 'embedding',
                'hypernetwork': 'hypernetwork',
                'aestheticgradient': 'embedding',
                'controlnet': 'controlnet',
                'vae': 'vae',
                'upscaler': 'upscaler'
            }
            
            primary_type = type_mapping.get(model_type, 'unknown')
            base_model = self.extract_base_model(model_info, version_info)
            
            return {
                'found': True,
                'source': 'civitai',
                'primary_type': primary_type,
                'base_model': base_model,
                'name': model_info.get('name', ''),
                'confidence': 0.99,
                'raw_data': data
            }
            
        except Exception:
            return {'found': False, 'source': 'parse_error'}
    
    def extract_base_model(self, model_info: Dict, version_info: Dict) -> str:
        """Extract base model from CivitAI data"""
        base_model = version_info.get('baseModel', '').lower()
        
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
        
        name = model_info.get('name', '').lower()
        if 'flux' in name:
            return 'flux'
        elif 'xl' in name or 'sdxl' in name:
            return 'sdxl'
        elif 'sd3' in name:
            return 'sd3'
        
        return base_model or 'unknown'

class ModelClassifier:
    """Comprehensive model classification system"""
    
    def __init__(self, config: Dict, database=None):
        self.config = config
        self.class_config = config.get('classification', {})
        self.confidence_threshold = self.class_config.get('confidence_threshold', 0.5)
        self.database = database
        
        # Initialize external API lookups
        enable_api = self.class_config.get('enable_external_apis', True)
        self.civitai = CivitAILookup(enable_api=enable_api)
        
        # Load classification rules from database if available, fallback to config
        self.load_classification_rules()
    
    def load_classification_rules(self):
        """Load classification rules from database or config fallback"""
        if self.database:
            try:
                # Load from database
                self.size_rules = self.database.get_size_rules()
                self.sub_type_rules = self.database.get_sub_type_rules()
                self.architecture_patterns = self.database.get_architecture_patterns()
                self.external_apis = self.database.get_external_apis()
                self.model_types = self.database.get_model_types()
                
                # Convert architecture patterns to dict for easier lookup
                self.architecture_patterns_dict = {}
                for arch, pattern, confidence in self.architecture_patterns:
                    if arch not in self.architecture_patterns_dict:
                        self.architecture_patterns_dict[arch] = []
                    self.architecture_patterns_dict[arch].append((pattern, confidence))
                
            except Exception as e:
                print(f"Warning: Could not load classification rules from database: {e}")
                self.load_fallback_rules()
        else:
            self.load_fallback_rules()
    
    def load_fallback_rules(self):
        """Load fallback classification rules from config"""
        self.size_rules = self.class_config.get('size_rules', {
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
            'video_model': {'min': 100_000_000, 'max': 100_000_000_000},
        })
        
        # Default sub-type and architecture patterns
        self.sub_type_rules = []
        self.architecture_patterns = []
        self.external_apis = [('civitai', True, 1, 1.0, 10)]
        self.model_types = [('checkpoint', 'Checkpoint'), ('lora', 'LoRA'), ('vae', 'VAE')]
    
    def classify_model(self, file_path: Path, file_hash: str = None, quiet: bool = False) -> ClassificationResult:
        """Classify a model using multi-layer analysis"""
        
        # Check for LFS pointer files
        if SafeTensorsExtractor.is_lfs_pointer_file(file_path):
            return ClassificationResult(
                primary_type="lfs_pointer",
                sub_type="undownloaded",
                confidence=1.0,
                method="lfs_detection"
            )
        
        # Calculate hash if not provided
        if file_hash is None:
            file_hash = self.calculate_file_hash(file_path)
        
        if not file_hash:
            return ClassificationResult(
                primary_type="unknown",
                sub_type="unknown",
                confidence=0.0,
                method="failed"
            )
        
        # STEP 1: Check for GGUF files first
        if file_path.suffix.lower() == '.gguf':
            return self.classify_gguf(file_path, file_hash, quiet)
        
        # STEP 2: Check manual overrides
        manual_overrides = self.class_config.get('manual_overrides', {})
        if file_path.name in manual_overrides:
            override = manual_overrides[file_path.name]
            if '/' in override:
                primary_type, sub_type = override.split('/', 1)
                return ClassificationResult(
                    primary_type=primary_type,
                    sub_type=sub_type,
                    confidence=1.0,
                    method="manual_override"
                )
        
        # STEP 3: CivitAI API lookup
        if not quiet:
            print("    Querying CivitAI API...")
        
        civitai_result = self.civitai.lookup_by_hash(file_hash)
        if civitai_result.get('found'):
            primary_type = civitai_result['primary_type']
            base_model = civitai_result['base_model']
            sub_type = self.determine_sub_type(primary_type, base_model, file_path.name)
            
            if not quiet:
                print(f"    CivitAI: {civitai_result.get('name', 'Unknown')} ({primary_type})")
            
            return ClassificationResult(
                primary_type=primary_type,
                sub_type=sub_type,
                confidence=civitai_result['confidence'],
                method="civitai_api",
                base_model=base_model
            )
        
        # STEP 4: SafeTensors metadata analysis
        if file_path.suffix.lower() == '.safetensors':
            return self.classify_safetensors(file_path, file_hash, quiet)
        
        # STEP 5: File size analysis fallback
        return self.classify_by_size(file_path, quiet)
    
    def classify_gguf(self, file_path: Path, file_hash: str, quiet: bool = False) -> ClassificationResult:
        """Classify GGUF files"""
        gguf_metadata = GGUFExtractor.extract_metadata(file_path)
        
        sub_type = self.determine_gguf_subtype(file_path.name, file_path.stat().st_size)
        
        return ClassificationResult(
            primary_type="gguf",
            sub_type=sub_type,
            confidence=0.95,
            method="gguf_classification",
            tensor_count=gguf_metadata.get('tensor_count', 0)
        )
    
    def classify_safetensors(self, file_path: Path, file_hash: str, quiet: bool = False) -> ClassificationResult:
        """Classify SafeTensors files using metadata and tensor analysis"""
        st_metadata = SafeTensorsExtractor.extract_metadata(file_path)
        
        if 'error' in st_metadata:
            return self.classify_by_size(file_path, quiet)
        
        # Extract triggers for LoRA models
        triggers = None
        metadata = st_metadata.get('metadata', {})
        if metadata:
            triggers = self.extract_triggers_from_metadata(metadata)
        
        # Analyze tensor structure
        tensor_names = st_metadata.get('tensor_names', [])
        architecture = self.analyze_tensor_architecture(tensor_names)
        
        # Determine primary type from metadata or tensor analysis
        primary_type = self.determine_primary_type_from_metadata(metadata, architecture, file_path.stat().st_size)
        sub_type = self.determine_sub_type(primary_type, None, file_path.name)
        
        confidence = 0.9 if primary_type != 'unknown' else 0.3
        
        return ClassificationResult(
            primary_type=primary_type,
            sub_type=sub_type,
            confidence=confidence,
            method="safetensors_analysis",
            architecture=architecture,
            triggers=triggers,
            tensor_count=st_metadata.get('tensor_count', 0)
        )
    
    def classify_by_size(self, file_path: Path, quiet: bool = False) -> ClassificationResult:
        """Fallback classification using file size"""
        file_size = file_path.stat().st_size
        
        best_match = None
        best_confidence = 0.0
        
        for model_type, size_range in self.size_rules.items():
            if size_range['min'] <= file_size <= size_range['max']:
                range_size = size_range['max'] - size_range['min']
                distance_from_min = file_size - size_range['min']
                confidence = 0.3 + 0.2 * (distance_from_min / range_size)
                
                if confidence > best_confidence:
                    best_match = model_type
                    best_confidence = confidence
        
        if best_match:
            sub_type = self.determine_sub_type(best_match, None, file_path.name)
            return ClassificationResult(
                primary_type=best_match,
                sub_type=sub_type,
                confidence=best_confidence,
                method="size_analysis"
            )
        
        return ClassificationResult(
            primary_type="unknown",
            sub_type="unknown",
            confidence=0.0,
            method="size_analysis"
        )
    
    def determine_primary_type_from_metadata(self, metadata: Dict, architecture: str, file_size: int) -> str:
        """Determine primary type from SafeTensors metadata"""
        if not metadata:
            if architecture == 'lora':
                return 'lora'
            elif architecture in ['vae', 'video_vae']:
                return 'vae'
            elif architecture in ['flux', 'diffusion']:
                return 'checkpoint'
            return 'unknown'
        
        # Check metadata fields for type indicators
        for key, value in metadata.items():
            key_lower = key.lower()
            value_str = str(value).lower() if value else ""
            
            if any(pattern in key_lower for pattern in ['type', 'model_type', 'architecture']):
                if 'lora' in value_str:
                    return 'lora'
                elif any(term in value_str for term in ['checkpoint', 'diffusion']):
                    return 'checkpoint'
                elif 'vae' in value_str:
                    return 'vae'
                elif 'controlnet' in value_str:
                    return 'controlnet'
        
        # Check for training indicators (suggests LoRA)
        training_indicators = ['epochs', 'steps', 'learning_rate', 'rank', 'alpha']
        if any(indicator in key.lower() for key in metadata.keys() for indicator in training_indicators):
            return 'lora'
        
        # Fallback to architecture-based detection
        if architecture == 'lora':
            return 'lora'
        elif architecture in ['vae', 'video_vae']:
            return 'vae'
        elif architecture in ['flux', 'diffusion']:
            return 'checkpoint'
        
        return 'unknown'
    
    def analyze_tensor_architecture(self, tensor_names: List[str]) -> str:
        """Analyze tensor names to determine architecture using database patterns"""
        if not tensor_names:
            return 'unknown'
        
        tensor_str = ' '.join(tensor_names).lower()
        
        # Use database patterns if available
        if hasattr(self, 'architecture_patterns_dict') and self.architecture_patterns_dict:
            best_match = None
            best_confidence = 0.0
            
            for architecture, patterns in self.architecture_patterns_dict.items():
                for pattern, confidence in patterns:
                    import re
                    if re.search(pattern, tensor_str, re.IGNORECASE):
                        if confidence > best_confidence:
                            best_match = architecture
                            best_confidence = confidence
            
            if best_match:
                return best_match
        
        # Fallback patterns
        if any(pattern in tensor_str for pattern in ['.lora_up.', '.lora_down.', '.alpha']):
            return 'lora'
        elif any(pattern in tensor_str for pattern in ['double_blocks', 'single_blocks', 'img_attn', 'txt_attn']):
            return 'flux'
        elif any('downsamples' in name and 'residual' in name for name in tensor_names):
            return 'video_vae'
        elif any(pattern in tensor_str for pattern in ['encoder.', 'decoder.', 'autoencoder']):
            return 'vae'
        elif any(pattern in tensor_str for pattern in ['diffusion_model', 'unet']):
            return 'diffusion'
        elif 'control_model' in tensor_str or 'controlnet' in tensor_str:
            return 'controlnet'
        
        return 'unknown'
    
    def determine_sub_type(self, primary_type: str, base_model: Optional[str], filename: str) -> str:
        """Determine sub-type based on primary type and base model using database rules"""
        filename_lower = filename.lower()
        
        if base_model:
            return f"{base_model}_{primary_type}"
        
        # Use database rules if available
        if hasattr(self, 'sub_type_rules') and self.sub_type_rules:
            for rule_primary_type, sub_type, pattern, pattern_type, confidence in self.sub_type_rules:
                if rule_primary_type == primary_type:
                    if pattern_type == 'filename':
                        import re
                        if re.search(pattern, filename_lower, re.IGNORECASE):
                            return sub_type
                    elif pattern_type.startswith('filesize_'):
                        # Handle filesize-based rules for GGUF
                        if primary_type == 'gguf':
                            return sub_type
        
        # Fallback to basic logic
        return primary_type
    
    def determine_gguf_subtype(self, filename: str, file_size: int) -> str:
        """Determine GGUF sub-type from filename"""
        filename_lower = filename.lower()
        
        if 'ltxv' in filename_lower or ('ltx' in filename_lower and 'video' in filename_lower):
            return 'ltx_video'
        elif any(term in filename_lower for term in ['t2v', 'text2video']):
            return 'text_to_video'
        elif any(term in filename_lower for term in ['i2v', 'image2video']):
            return 'image_to_video'
        elif 'video' in filename_lower:
            return 'video_model'
        elif 'umt5' in filename_lower or 'encoder' in filename_lower:
            return 'text_encoder'
        else:
            # Default to LLM classification
            if file_size > 10_000_000_000:  # > 10GB
                return 'large_llm'
            elif file_size > 1_000_000_000:  # > 1GB
                return 'medium_llm'
            else:
                return 'small_llm'
    
    def extract_triggers_from_metadata(self, metadata: Dict) -> Optional[List[str]]:
        """Extract trigger words from SafeTensors metadata"""
        triggers = []
        
        # Parse ss_tag_frequency
        if 'ss_tag_frequency' in metadata:
            triggers.extend(self.parse_tag_frequency(metadata['ss_tag_frequency']))
        
        # Parse ss_output_name
        if 'ss_output_name' in metadata:
            triggers.extend(self.parse_output_name(metadata['ss_output_name']))
        
        # Look for explicit trigger fields
        for key, value in metadata.items():
            if any(pattern in key.lower() for pattern in ['trigger', 'activation', 'keyword']):
                triggers.extend(self.parse_trigger_field(str(value)))
        
        return self.clean_and_deduplicate_triggers(triggers) if triggers else None
    
    def parse_tag_frequency(self, tag_frequency_str: str) -> List[str]:
        """Parse ss_tag_frequency JSON to extract trigger words"""
        try:
            tag_data = json.loads(tag_frequency_str)
            triggers = []
            
            for dataset_name, tags in tag_data.items():
                if isinstance(tags, dict):
                    for tag, frequency in tags.items():
                        if self.is_likely_trigger_word(tag, frequency):
                            triggers.append(tag.upper())
            
            return triggers
        except (json.JSONDecodeError, TypeError, AttributeError):
            return []
    
    def parse_output_name(self, output_name: str) -> List[str]:
        """Extract potential trigger words from ss_output_name"""
        if not output_name:
            return []
        
        potential_triggers = []
        for part in output_name.replace('_', ' ').replace('-', ' ').split():
            if self.is_likely_trigger_word(part):
                potential_triggers.append(part.upper())
        
        return potential_triggers
    
    def parse_trigger_field(self, value: str) -> List[str]:
        """Parse explicit trigger fields"""
        if not value:
            return []
        
        try:
            parsed_value = json.loads(value)
            if isinstance(parsed_value, list):
                return [str(item).upper() for item in parsed_value if item]
            else:
                return [str(parsed_value).upper()]
        except (json.JSONDecodeError, TypeError):
            triggers = []
            for part in value.replace(',', ' ').replace(';', ' ').split():
                part = part.strip().strip('"\'')
                if part and len(part) > 1:
                    triggers.append(part.upper())
            return triggers
    
    def is_likely_trigger_word(self, word: str, frequency: int = None) -> bool:
        """Determine if a word is likely a trigger word"""
        if not word or len(word) < 2:
            return False
        
        word_lower = word.lower()
        
        excluded_words = {
            'img', 'image', 'photo', 'picture', 'style', 'lora', 'model',
            'training', 'dataset', 'epoch', 'step', 'version', 'v1', 'v2',
            'base', 'fine', 'tuned', 'checkpoint', 'safetensors', 'the', 'and',
            'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        if word_lower in excluded_words:
            return False
        
        if frequency is not None and frequency > 100:
            return True
        
        return True
    
    def clean_and_deduplicate_triggers(self, triggers: List[str]) -> List[str]:
        """Clean and deduplicate trigger words"""
        if not triggers:
            return []
        
        seen = set()
        cleaned = []
        
        for trigger in triggers:
            trigger_clean = trigger.strip().upper()
            if trigger_clean and trigger_clean not in seen and len(trigger_clean) > 1:
                seen.add(trigger_clean)
                cleaned.append(trigger_clean)
        
        return cleaned[:5]  # Limit to 5 trigger words
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""