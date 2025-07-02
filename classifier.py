#!/usr/bin/env python3
"""
Enhanced Model Hub Classifier - Comprehensive AI model classification system
Provides accurate model type detection through multi-layer analysis
Based on proven legacy classification engine with modern improvements
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
    """Model classification result - compatible with existing interface"""
    primary_type: str
    sub_type: str
    confidence: float
    method: str
    architecture: Optional[str] = None
    triggers: Optional[List[str]] = None
    base_model: Optional[str] = None
    tensor_count: Optional[int] = None
    
    # Additional scoring details for advanced analysis
    filename_score: float = 0.0
    size_score: float = 0.0
    metadata_score: float = 0.0
    tensor_score: float = 0.0

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
                # Read header length (first 8 bytes)
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    return {'error': 'File too small to be valid SafeTensors'}
                
                header_size = struct.unpack('<Q', header_size_bytes)[0]
                if header_size > 100_000_000:  # 100MB header limit
                    return {'error': 'Header too large, likely corrupt file'}
                
                # Read header JSON
                header_bytes = f.read(header_size)
                if len(header_bytes) < header_size:
                    return {'error': 'Incomplete header read'}
                
                header_json = json.loads(header_bytes.decode('utf-8'))
                
                # Extract tensor information
                tensors = {}
                metadata = {}
                
                for key, value in header_json.items():
                    if key == '__metadata__':
                        metadata = value
                    elif isinstance(value, dict) and 'shape' in value:
                        tensors[key] = value
                
                return {
                    'tensors': tensors,
                    'metadata': metadata,
                    'tensor_count': len(tensors),
                    'tensor_names': list(tensors.keys())
                }
                
        except Exception as e:
            return {'error': f'SafeTensors extraction failed: {e}'}

class GGUFExtractor:
    """Extract metadata from GGUF files"""
    
    @staticmethod
    def extract_metadata(file_path: Path) -> Dict:
        """Extract basic metadata from GGUF file"""
        try:
            with open(file_path, 'rb') as f:
                # Read GGUF header
                magic = f.read(4)
                if magic != b'GGUF':
                    return {'error': 'Not a valid GGUF file'}
                
                version = struct.unpack('<I', f.read(4))[0]
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
                
                return {
                    'version': version,
                    'tensor_count': tensor_count,
                    'metadata_count': metadata_kv_count,
                    'format': 'gguf'
                }
                
        except Exception as e:
            return {'error': f'GGUF extraction failed: {e}'}

class TensorAnalyzer:
    """Analyze tensor structures to determine model architecture"""
    
    def __init__(self):
        # Define tensor patterns for different model types
        self.tensor_patterns = {
            'lora': [
                r'.*\.lora_up\..*',
                r'.*\.lora_down\..*', 
                r'.*\.alpha$',
                r'.*lora_A\..*',
                r'.*lora_B\..*'
            ],
            'flux': [
                'double_blocks',
                'single_blocks', 
                'img_attn',
                'txt_attn',
                'guidance_in'
            ],
            'controlnet': [
                'control_model',
                'input_blocks',
                'middle_block',
                'output_blocks',
                'zero_convs'
            ],
            'vae': [
                'encoder.down',
                'decoder.up',
                'quant_conv',
                'post_quant_conv'
            ],
            'video_vae': [
                r'encoder\.downsamples\..*\.residual\..*',
                r'decoder\.upsamples\..*\.residual\..*',
                'temporal_down',
                'temporal_up'
            ],
            'clip': [
                'text_model.encoder',
                'text_projection',
                'visual.transformer'
            ],
            'unet': [
                'time_embed',
                'input_blocks',
                'middle_block', 
                'output_blocks'
            ]
        }
    
    def analyze_tensors(self, tensor_names: List[str]) -> Dict[str, float]:
        """Analyze tensor names against known patterns"""
        scores = {}
        
        for model_type, patterns in self.tensor_patterns.items():
            matches = 0
            total_patterns = len(patterns)
            
            for pattern in patterns:
                if any(re.search(pattern, name, re.IGNORECASE) for name in tensor_names):
                    matches += 1
            
            scores[model_type] = matches / total_patterns if total_patterns > 0 else 0.0
        
        return scores
    
    def get_architecture_hints(self, tensor_names: List[str], metadata: Dict) -> str:
        """Get architecture hints from tensor analysis"""
        tensor_str = ' '.join(tensor_names).lower()
        
        # Check for specific architectures in order of specificity
        if 'flux' in tensor_str and ('gguf' in tensor_str or 'quantized' in str(metadata)):
            return 'flux_quantized'
        elif any(flux_pattern in tensor_str for flux_pattern in ['double_blocks', 'single_blocks', 'img_attn']):
            return 'flux'
        elif 'downsamples' in tensor_str and 'residual' in tensor_str:
            return 'video_vae'
        elif 'time_embed' in tensor_str and 'middle_block' in tensor_str:
            return 'diffusion'
        elif 'lora' in tensor_str:
            return 'lora'
        else:
            return 'unknown'

class CivitAILookup:
    """CivitAI API integration for external model lookup"""
    
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
        """Look up model by SHA-256 hash"""
        if not self.enable_api:
            return {'found': False, 'source': 'api_disabled'}
        
        try:
            # Rate limiting
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
            
            url = f"{self.base_url}/model-versions/by-hash/{file_hash}"
            response = self.session.get(url, timeout=10)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                return self.parse_civitai_response(data)
            elif response.status_code == 404:
                return {'found': False, 'source': 'civitai_not_found'}
            else:
                return {'found': False, 'source': f'civitai_error_{response.status_code}'}
                
        except Exception as e:
            return {'found': False, 'source': f'civitai_exception_{type(e).__name__}'}
    
    def parse_civitai_response(self, data: Dict) -> Dict:
        """Parse CivitAI API response into our format"""
        try:
            model_info = data.get('model', {})
            version_info = data
            
            # Extract basic info
            name = model_info.get('name', 'Unknown')
            model_type = model_info.get('type', 'unknown').lower()
            
            # Map CivitAI types to our types
            type_mapping = {
                'checkpoint': 'checkpoint',
                'textualinversion': 'embedding',
                'lora': 'lora',
                'lycoris': 'lora',
                'hypernetwork': 'hypernetwork',
                'controlnet': 'controlnet',
                'vae': 'vae',
                'poses': 'pose',
                'wildcards': 'wildcard',
                'workflows': 'workflow',
                'other': 'unknown'
            }
            
            primary_type = type_mapping.get(model_type, 'unknown')
            base_model = self.extract_base_model(model_info, version_info)
            triggers = self.extract_civitai_triggers(version_info)
            
            return {
                'found': True,
                'source': 'civitai_api',
                'name': name,
                'primary_type': primary_type,
                'base_model': base_model,
                'triggers': triggers,
                'confidence': 0.95,  # High confidence for CivitAI data
                'civitai_id': model_info.get('id'),
                'version_id': version_info.get('id')
            }
            
        except Exception as e:
            return {'found': False, 'source': f'civitai_parse_error_{type(e).__name__}'}
    
    def extract_base_model(self, model_info: Dict, version_info: Dict) -> str:
        """Extract base model from CivitAI data"""
        base_model = version_info.get('baseModel', '').lower()
        
        # Check base model field first
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
        elif 'wan' in base_model or 'video' in base_model:
            return 'wan'
        
        # Check model name for additional patterns
        name = model_info.get('name', '').lower()
        if 'flux' in name:
            return 'flux'
        elif 'xl' in name or 'sdxl' in name:
            return 'sdxl'
        elif 'sd3' in name:
            return 'sd3'
        elif 'wan' in name or 'wanvideo' in name:
            return 'wan'
        elif 'hunyuan' in name:
            return 'hunyuan'
        elif 'ltx' in name or 'ltxv' in name:
            return 'ltxv'
        elif 'video' in name and any(v in name for v in ['lora', 'model']):
            return 'video'
        
        return base_model or 'unknown'
    
    def extract_civitai_triggers(self, version_info: Dict) -> List[str]:
        """Extract trigger words from CivitAI version info"""
        triggers = []
        
        # Check trainedWords field
        trained_words = version_info.get('trainedWords', [])
        if isinstance(trained_words, list):
            triggers.extend(trained_words)
        
        return triggers

class ModelClassifier:
    """Enhanced model classification system with multi-layer analysis"""
    
    def __init__(self, config: Dict, database=None):
        self.config = config
        self.class_config = config.get('classification', {})
        self.confidence_threshold = self.class_config.get('confidence_threshold', 0.5)
        self.database = database
        
        # Initialize components
        enable_api = self.class_config.get('enable_external_apis', True)
        self.civitai = CivitAILookup(enable_api=enable_api)
        self.tensor_analyzer = TensorAnalyzer()
        
        # Load classification rules
        self.load_classification_rules()
        
        # Scoring weights for multi-layer analysis
        self.weights = {
            'filename': 0.2,   # Reduced - filenames can be misleading
            'size': 0.2,       # File size is a good indicator
            'tensor': 0.5,     # Highest - tensor analysis is most reliable
            'metadata': 0.1    # SafeTensors metadata when available
        }
    
    def load_classification_rules(self):
        """Load classification rules from database or config fallback"""
        if self.database:
            try:
                self.size_rules = self.database.get_size_rules()
                self.sub_type_rules = self.database.get_sub_type_rules()
                self.architecture_patterns = self.database.get_architecture_patterns()
                self.external_apis = self.database.get_external_apis()
                self.model_types = self.database.get_model_types()
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
            'ip_adapter': {'min': 10_000_000, 'max': 2_000_000_000}
        })
        
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
        filename_key = file_path.name.lower()
        if filename_key in manual_overrides:
            override = manual_overrides[filename_key]
            return ClassificationResult(
                primary_type=override.get('primary_type', 'unknown'),
                sub_type=override.get('sub_type', 'unknown'),
                confidence=1.0,
                method="manual_override",
                triggers=override.get('triggers', [])
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
                base_model=base_model,
                triggers=civitai_result.get('triggers', [])
            )
        
        # STEP 4: SafeTensors metadata analysis
        if file_path.suffix.lower() == '.safetensors':
            return self.classify_safetensors(file_path, file_hash, quiet)
        
        # STEP 5: File size analysis fallback
        return self.classify_by_size(file_path, file_hash, quiet)
    
    def classify_safetensors(self, file_path: Path, file_hash: str, quiet: bool = False) -> ClassificationResult:
        """Classify SafeTensors files using metadata and tensor analysis"""
        if not quiet:
            print("    Analyzing SafeTensors metadata...")
        
        metadata = SafeTensorsExtractor.extract_metadata(file_path)
        if 'error' in metadata:
            if not quiet:
                print(f"    SafeTensors error: {metadata['error']}")
            return self.classify_by_size(file_path, file_hash, quiet)
        
        tensor_names = metadata.get('tensor_names', [])
        tensor_count = metadata.get('tensor_count', 0)
        st_metadata = metadata.get('metadata', {})
        
        # Analyze tensor patterns
        tensor_scores = self.tensor_analyzer.analyze_tensors(tensor_names)
        architecture = self.tensor_analyzer.get_architecture_hints(tensor_names, st_metadata)
        
        # Calculate individual scores
        filename_score = self.calculate_filename_score(file_path.name)
        size_score = self.calculate_size_score(file_path.stat().st_size)
        tensor_score = max(tensor_scores.values()) if tensor_scores else 0.0
        metadata_score = self.calculate_metadata_score(st_metadata)
        
        # Determine primary type based on highest tensor score
        if tensor_scores:
            primary_type = max(tensor_scores.items(), key=lambda x: x[1])[0]
            confidence = tensor_scores[primary_type]
        else:
            # Fallback to size-based classification
            primary_type, confidence = self.classify_by_size_only(file_path.stat().st_size)
        
        # Calculate weighted confidence
        weighted_confidence = (
            filename_score * self.weights['filename'] +
            size_score * self.weights['size'] +
            tensor_score * self.weights['tensor'] +
            metadata_score * self.weights['metadata']
        )
        
        # Use the higher of tensor-based or weighted confidence
        final_confidence = max(confidence, weighted_confidence)
        
        # Extract triggers for LoRA models
        triggers = []
        if primary_type == 'lora':
            triggers = self.extract_triggers_from_safetensors(st_metadata)
        
        # Determine sub-type
        base_model = self.extract_base_model_from_metadata(st_metadata)
        sub_type = self.determine_sub_type(primary_type, base_model, file_path.name)
        
        if not quiet:
            print(f"    SafeTensors: {primary_type} (confidence: {final_confidence:.2f})")
            if triggers:
                print(f"    Triggers: {', '.join(triggers)}")
        
        return ClassificationResult(
            primary_type=primary_type,
            sub_type=sub_type,
            confidence=final_confidence,
            method="safetensors_analysis",
            architecture=architecture,
            triggers=triggers,
            tensor_count=tensor_count,
            filename_score=filename_score,
            size_score=size_score,
            metadata_score=metadata_score,
            tensor_score=tensor_score
        )
    
    def classify_gguf(self, file_path: Path, file_hash: str, quiet: bool = False) -> ClassificationResult:
        """Classify GGUF files"""
        if not quiet:
            print("    Analyzing GGUF file...")
        
        metadata = GGUFExtractor.extract_metadata(file_path)
        if 'error' in metadata:
            return self.classify_by_size(file_path, file_hash, quiet)
        
        # GGUF files are typically quantized language models or diffusion models
        file_size = file_path.stat().st_size
        tensor_count = metadata.get('tensor_count', 0)
        
        # Determine type based on size and tensor count
        if file_size > 10_000_000_000:  # > 10GB
            primary_type = 'checkpoint'
            sub_type = 'gguf_checkpoint'
        elif file_size > 1_000_000_000:  # > 1GB
            primary_type = 'gguf'
            sub_type = 'quantized_model'
        else:
            primary_type = 'gguf'
            sub_type = 'small_model'
        
        confidence = 0.9  # High confidence for GGUF detection
        
        return ClassificationResult(
            primary_type=primary_type,
            sub_type=sub_type,
            confidence=confidence,
            method="gguf_classification",
            tensor_count=tensor_count
        )
    
    def classify_by_size(self, file_path: Path, file_hash: str, quiet: bool = False) -> ClassificationResult:
        """Classify model by file size as fallback"""
        file_size = file_path.stat().st_size
        primary_type, confidence = self.classify_by_size_only(file_size)
        
        # Calculate filename score for additional confidence
        filename_score = self.calculate_filename_score(file_path.name)
        
        # Combine size and filename scores
        combined_confidence = (confidence * 0.7) + (filename_score * 0.3)
        
        sub_type = self.determine_sub_type(primary_type, 'unknown', file_path.name)
        
        if not quiet:
            print(f"    Size-based: {primary_type} (confidence: {combined_confidence:.2f})")
        
        return ClassificationResult(
            primary_type=primary_type,
            sub_type=sub_type,
            confidence=combined_confidence,
            method="size_analysis",
            size_score=confidence,
            filename_score=filename_score
        )
    
    def classify_by_size_only(self, file_size: int) -> Tuple[str, float]:
        """Get type and confidence based on file size only"""
        best_match = 'unknown'
        best_confidence = 0.3
        
        for model_type, rules in self.size_rules.items():
            min_size = rules.get('min', 0)
            max_size = rules.get('max', float('inf'))
            
            if min_size <= file_size <= max_size:
                # Calculate confidence based on how well size fits
                size_range = max_size - min_size
                if size_range > 0:
                    distance_from_min = file_size - min_size
                    confidence = 0.5 + (0.4 * (1 - distance_from_min / size_range))
                else:
                    confidence = 0.9
                
                if confidence > best_confidence:
                    best_match = model_type
                    best_confidence = confidence
        
        return best_match, best_confidence
    
    def calculate_filename_score(self, filename: str) -> float:
        """Calculate confidence score based on filename patterns"""
        filename_lower = filename.lower()
        
        # Strong indicators
        if 'lora' in filename_lower:
            return 0.9
        elif 'controlnet' in filename_lower or 'control_' in filename_lower:
            return 0.9
        elif 'vae' in filename_lower:
            return 0.9
        elif 'ip-adapter' in filename_lower or 'ip_adapter' in filename_lower:
            return 0.9
        elif 'embedding' in filename_lower or 'textual_inversion' in filename_lower:
            return 0.9
        elif 'checkpoint' in filename_lower or 'ckpt' in filename_lower:
            return 0.8
        
        # Model architecture indicators
        elif any(arch in filename_lower for arch in ['flux', 'sdxl', 'sd15', 'sd3']):
            return 0.7
        
        # Weak indicators
        elif filename_lower.endswith('.safetensors'):
            return 0.5
        
        return 0.3
    
    def calculate_size_score(self, file_size: int) -> float:
        """Calculate confidence score based on file size"""
        _, confidence = self.classify_by_size_only(file_size)
        return confidence
    
    def calculate_metadata_score(self, metadata: Dict) -> float:
        """Calculate confidence score based on metadata content"""
        if not metadata:
            return 0.0
        
        score = 0.0
        
        # Check for specific metadata keys that indicate model type
        if 'ss_network_module' in metadata:  # LoRA training info
            score += 0.3
        if 'ss_base_model_version' in metadata:  # Base model info
            score += 0.2
        if 'ss_tag_frequency' in metadata:  # Training tags
            score += 0.2
        if 'modelspec.architecture' in metadata:  # Architecture info
            score += 0.3
        
        return min(score, 1.0)
    
    def extract_triggers_from_safetensors(self, metadata: Dict) -> List[str]:
        """Extract trigger words from SafeTensors metadata"""
        triggers = []
        
        # Method 1: Check ss_tag_frequency for training tags
        if 'ss_tag_frequency' in metadata:
            try:
                tag_freq = json.loads(metadata['ss_tag_frequency'])
                if isinstance(tag_freq, dict):
                    # Get most frequent tags (potential triggers)
                    for dataset, tags in tag_freq.items():
                        if isinstance(tags, dict):
                            # Sort by frequency, take top tags
                            sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
                            for tag, freq in sorted_tags[:5]:  # Top 5 tags
                                if freq > 10 and len(tag) > 2:  # Minimum frequency and length
                                    triggers.append(tag.upper())
            except:
                pass
        
        # Method 2: Check ss_output_name for trigger hints
        if 'ss_output_name' in metadata:
            output_name = metadata['ss_output_name']
            # Extract potential trigger from output name
            if isinstance(output_name, str) and len(output_name) > 0:
                # Clean up the output name to extract trigger
                clean_name = re.sub(r'[_\-]', ' ', output_name)
                triggers.append(clean_name.upper())
        
        # Method 3: Check modelspec.title
        if 'modelspec.title' in metadata:
            title = metadata['modelspec.title']
            if isinstance(title, str):
                triggers.append(title.upper())
        
        # Filter out common non-trigger words
        common_words = {'lora', 'model', 'style', 'concept', 'flux', 'sdxl', 'sd15', 'checkpoint'}
        filtered_triggers = []
        for trigger in triggers:
            if trigger.lower() not in common_words and len(trigger) > 1:
                filtered_triggers.append(trigger)
        
        return list(set(filtered_triggers))  # Remove duplicates
    
    def extract_base_model_from_metadata(self, metadata: Dict) -> str:
        """Extract base model information from metadata"""
        if 'ss_base_model_version' in metadata:
            base_version = metadata['ss_base_model_version'].lower()
            if 'flux' in base_version:
                return 'flux'
            elif 'xl' in base_version or 'sdxl' in base_version:
                return 'sdxl'
            elif '1.5' in base_version or 'sd15' in base_version:
                return 'sd15'
            elif 'sd3' in base_version:
                return 'sd3'
        
        if 'modelspec.architecture' in metadata:
            arch = metadata['modelspec.architecture'].lower()
            if 'flux' in arch:
                return 'flux'
            elif 'stable-diffusion-xl' in arch:
                return 'sdxl'
            elif 'stable-diffusion-v1' in arch:
                return 'sd15'
        
        return 'unknown'
    
    def determine_sub_type(self, primary_type: str, base_model: str, filename: str) -> str:
        """Determine sub-type based on primary type, base model, and filename"""
        filename_lower = filename.lower()
        
        if primary_type == 'lora':
            if base_model == 'flux':
                return 'flux_lora'
            elif base_model == 'sdxl':
                return 'sdxl_lora'
            elif base_model == 'sd15':
                return 'sd15_lora'
            elif base_model == 'wan':
                return 'wan_lora'
            elif base_model == 'hunyuan':
                return 'hunyuan_lora'
            elif base_model == 'ltxv':
                return 'ltxv_lora'
            elif base_model == 'video':
                return 'video_lora'
            elif 'pony' in filename_lower:
                return 'pony_lora'
            elif 'wan' in filename_lower:
                return 'wan_lora'
            elif 'hunyuan' in filename_lower:
                return 'hunyuan_lora'
            elif 'ltx' in filename_lower:
                return 'ltxv_lora'
            elif 'video' in filename_lower:
                return 'video_lora'
            else:
                return 'unknown_lora'
        
        elif primary_type == 'checkpoint':
            if base_model == 'flux':
                return 'flux_checkpoint'
            elif base_model == 'sdxl':
                return 'sdxl_checkpoint'
            elif base_model == 'sd15':
                return 'sd15_checkpoint'
            elif 'wan' in filename_lower:
                return 'wan_checkpoint'
            elif 'video' in filename_lower:
                return 'video_checkpoint'
            else:
                return 'unknown_checkpoint'
        
        elif primary_type == 'vae':
            if 'video' in filename_lower:
                return 'video_vae'
            elif base_model == 'sdxl':
                return 'sdxl_vae'
            else:
                return 'standard_vae'
        
        elif primary_type == 'controlnet':
            if 'openpose' in filename_lower:
                return 'openpose_controlnet'
            elif 'depth' in filename_lower:
                return 'depth_controlnet'
            elif 'canny' in filename_lower:
                return 'canny_controlnet'
            else:
                return 'unknown_controlnet'
        
        return 'unknown'
    
    def calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate SHA-256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {e}")
            return None