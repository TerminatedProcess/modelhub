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
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ClassificationDebugger:
    """Collects debug information during model classification"""
    classification_steps: List[Dict] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    tensor_analysis: Dict = field(default_factory=dict)
    file_analysis: Dict = field(default_factory=dict)
    external_api: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    classification_time: str = field(default_factory=lambda: datetime.now().isoformat())
    classifier_version: str = "enhanced_v1.3"
    
    def add_step(self, step: str, status: str, **kwargs):
        """Add a classification step with status and additional info"""
        step_info = {"step": step, "status": status}
        step_info.update(kwargs)
        self.classification_steps.append(step_info)
    
    def add_error(self, error: str):
        """Add an error message"""
        self.errors.append(error)
    
    def set_scores(self, filename_score: float = 0.0, size_score: float = 0.0, 
                  tensor_score: float = 0.0, metadata_score: float = 0.0):
        """Set classification scores"""
        self.scores = {
            "filename_score": filename_score,
            "size_score": size_score,
            "tensor_score": tensor_score,
            "metadata_score": metadata_score
        }
    
    def set_tensor_analysis(self, tensor_count: int, tensor_scores: Dict[str, float], 
                           tensor_names: List[str] = None):
        """Set tensor analysis results"""
        self.tensor_analysis = {
            "tensor_count": tensor_count,
            "patterns_checked": len(tensor_scores) if tensor_scores else 0,
            "pattern_matches": tensor_scores or {},
            "highest_score": max(tensor_scores.values()) if tensor_scores else 0.0,
            "tensor_names_count": len(tensor_names) if tensor_names else 0
        }
    
    def set_file_analysis(self, file_size: int, extension: str, filename: str):
        """Set file analysis info"""
        filename_lower = filename.lower()
        indicators = []
        for indicator in ['lora', 'flux', 'controlnet', 'vae', 'embedding', 'checkpoint', 'sdxl', 'sd15', 'sd3', 'pony']:
            if indicator in filename_lower:
                indicators.append(indicator)
        
        self.file_analysis = {
            "size_mb": round(file_size / (1024 * 1024), 1),
            "extension": extension,
            "filename_indicators": indicators
        }
    
    def set_external_api_info(self, attempted: bool, result: Dict = None, reason: str = None):
        """Set external API lookup info"""
        self.external_api = {
            "civitai_attempted": attempted,
            "result": result if result else {},
            "reason": reason
        }
    
    def to_json(self) -> str:
        """Convert debug info to JSON string"""
        return json.dumps({
            "classification_steps": self.classification_steps,
            "scores": self.scores,
            "tensor_analysis": self.tensor_analysis,
            "file_analysis": self.file_analysis,
            "external_api": self.external_api,
            "errors": self.errors,
            "classification_time": self.classification_time,
            "classifier_version": self.classifier_version
        }, indent=2)

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
    
    # Raw metadata for storage
    raw_metadata: Optional[Dict[str, Any]] = None
    
    # Debug information for troubleshooting
    debug_info: Optional[str] = None

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
                
        except UnicodeDecodeError as e:
            return {'error': f'SafeTensors header encoding error: {e}'}
        except json.JSONDecodeError as e:
            return {'error': f'SafeTensors header JSON parse error: {e}'}
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
                r'.*lora_B\..*',
                r'.*\.hada_w1_.*',
                r'.*\.hada_w2_.*',
                r'.*\.lora_.*\.weight$',
                r'.*\.lora_.*\.bias$',
                r'.*lora_down$',
                r'.*lora_up$',
                r'.*\.locon_.*',
                r'.*\.lokr_.*'
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
            
            # Improve classification for specific base models
            if primary_type == 'unknown' and base_model == 'pony':
                # Pony models are typically checkpoints
                primary_type = 'checkpoint'
            
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
                'version_id': version_info.get('id'),
                'raw_response': data  # Include complete raw response
            }
            
        except Exception as e:
            return {'found': False, 'source': f'civitai_parse_error_{type(e).__name__}'}
    
    def extract_base_model(self, model_info: Dict, version_info: Dict) -> str:
        """Extract base model from CivitAI data"""
        base_model = version_info.get('baseModel', '')
        base_model_lower = base_model.lower()
        
        # Check base model field first - preserve detailed Flux variants
        if 'flux.1 d' in base_model_lower or 'flux.1-d' in base_model_lower:
            return 'flux_dev'
        elif 'flux.1 s' in base_model_lower or 'flux.1-s' in base_model_lower:
            return 'flux_schnell'
        elif 'flux.1 kontext' in base_model_lower:
            return 'flux_kontext'
        elif 'flux' in base_model_lower:
            return 'flux'
        elif 'sdxl' in base_model_lower or 'xl' in base_model_lower:
            return 'sdxl'
        elif 'sd 3' in base_model_lower or 'sd3' in base_model_lower:
            return 'sd3'
        elif 'sd 1.5' in base_model_lower or 'sd15' in base_model_lower:
            return 'sd15'
        elif 'sd 2' in base_model_lower or 'sd2' in base_model_lower:
            return 'sd2'
        elif 'pony' in base_model_lower:
            return 'pony'
        elif 'wan' in base_model_lower or 'video' in base_model_lower:
            return 'wan'
        
        # Check model name for additional patterns
        name = model_info.get('name', '').lower()
        if 'flux' in name:
            return 'flux'
        elif 'xl' in name or 'sdxl' in name:
            return 'sdxl'
        elif 'sd3' in name:
            return 'sd3'
        elif 'pony' in name:
            return 'pony'
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

class DataDrivenClassificationEngine:
    """Data-driven classification engine that uses database rules instead of hard-coded logic"""
    
    def __init__(self, database):
        self.database = database
        self.civitai = None  # Will be initialized as needed
        
    def classify_model_data_driven(self, file_path: Path, civitai_data: Dict = None, metadata: Dict = None, 
                                 filename: str = None, file_size: int = None) -> Dict:
        """Main data-driven classification method"""
        import re
        
        filename = filename or file_path.name
        file_size = file_size or (file_path.stat().st_size if file_path.exists() else 0)
        
        results = {
            'primary_type': 'unknown',
            'base_model': 'unknown', 
            'sub_type': 'unknown',
            'confidence': 0.0,
            'method': 'data_driven',
            'modifiers': []
        }
        
        # Get classification workflow
        workflow_steps = self.database.get_classification_workflow()
        
        for step in workflow_steps:
            rule_group = step['rule_group']
            
            if rule_group == 'external_api_lookup' and civitai_data:
                self._apply_external_api_rules(civitai_data, results)
            elif rule_group == 'base_model_detection':
                self._apply_base_model_rules(filename, civitai_data, metadata, results)
            elif rule_group == 'primary_type_detection':
                self._apply_primary_type_rules(filename, civitai_data, results)
            elif rule_group == 'sub_type_detection':
                self._apply_sub_type_rules(results)
            elif rule_group == 'pattern_modifiers':
                self._apply_pattern_modifiers(filename, results)
            elif rule_group == 'file_size_fallback':
                self._apply_file_size_rules(file_size, results)
                
        return results
    
    def _apply_external_api_rules(self, civitai_data: Dict, results: Dict):
        """Apply external API mapping rules"""
        if not civitai_data:
            return
            
        mappings = self.database.get_external_api_mappings('civitai')
        
        for mapping in mappings:
            api_field = mapping['api_field']
            api_value = mapping['api_value']
            modelhub_field = mapping['modelhub_field']
            modelhub_value = mapping['modelhub_value']
            confidence = mapping['confidence']
            
            # Check if the API data matches this mapping
            if api_field in civitai_data:
                if civitai_data[api_field].lower() == api_value.lower():
                    results[modelhub_field] = modelhub_value
                    results['confidence'] = max(results['confidence'], confidence)
                    results['method'] = 'civitai_api'
    
    def _apply_base_model_rules(self, filename: str, civitai_data: Dict, metadata: Dict, results: Dict):
        """Apply base model detection rules"""
        import re
        
        rules = self.database.get_detection_rules(rule_type='base_model')
        
        for rule in rules:
            source_type = rule['source_type']
            pattern = rule['pattern']
            output_value = rule['output_value']
            confidence = rule['confidence']
            
            match_found = False
            
            if source_type == 'civitai_field' and civitai_data:
                field = rule['source_field']
                if field in civitai_data:
                    if re.search(pattern, civitai_data[field], re.IGNORECASE):
                        match_found = True
                        
            elif source_type == 'filename':
                if re.search(pattern, filename, re.IGNORECASE):
                    match_found = True
                    
            elif source_type == 'metadata_field' and metadata:
                field = rule['source_field']
                if field in metadata:
                    if re.search(pattern, str(metadata[field]), re.IGNORECASE):
                        match_found = True
            
            if match_found:
                results['base_model'] = output_value
                results['confidence'] = max(results['confidence'], confidence)
                break  # Use first matching rule (ordered by priority)
    
    def _apply_primary_type_rules(self, filename: str, civitai_data: Dict, results: Dict):
        """Apply primary type detection rules"""
        import re
        
        rules = self.database.get_detection_rules(rule_type='primary_type')
        
        for rule in rules:
            source_type = rule['source_type']
            pattern = rule['pattern']
            output_value = rule['output_value']
            confidence = rule['confidence']
            
            match_found = False
            
            if source_type == 'base_model_conditional':
                # Special rule type: if base_model matches source_field and current primary_type matches pattern
                if (results['base_model'] == rule['source_field'] and 
                    results['primary_type'] == pattern):
                    match_found = True
            
            if match_found:
                results['primary_type'] = output_value
                results['confidence'] = max(results['confidence'], confidence)
                break
    
    def _apply_sub_type_rules(self, results: Dict):
        """Apply sub-type detection rules based on base model and primary type"""
        # Use existing sub_type_rules logic but from database
        if results['base_model'] != 'unknown':
            # For now, use base_model as sub_type (can be enhanced with more complex rules)
            results['sub_type'] = results['base_model']
    
    def _apply_pattern_modifiers(self, filename: str, results: Dict):
        """Apply pattern modifiers to enhance sub-types"""
        import re
        
        modifiers = self.database.get_pattern_modifiers(
            applies_to_type=results['primary_type'],
            applies_to_subtype=results['sub_type']
        )
        
        applied_modifiers = []
        
        for modifier in modifiers:
            pattern = modifier['pattern']
            suffix = modifier['modifier_suffix']
            
            if re.search(pattern, filename, re.IGNORECASE):
                applied_modifiers.append(suffix)
        
        # Apply modifiers to sub_type
        if applied_modifiers:
            base_subtype = results['sub_type']
            results['sub_type'] = base_subtype + ''.join(applied_modifiers)
            results['modifiers'] = applied_modifiers
    
    def _apply_file_size_rules(self, file_size: int, results: Dict):
        """Apply file size-based classification as fallback"""
        if results['primary_type'] != 'unknown':
            return  # Don't override if we already have a type
            
        size_rules = self.database.get_size_rules()
        
        for model_type, rules in size_rules.items():
            min_size = rules.get('min_size', 0)
            max_size = rules.get('max_size', float('inf'))
            
            if min_size <= file_size <= max_size:
                results['primary_type'] = model_type
                results['confidence'] = max(results['confidence'], 0.3)  # Low confidence for size-only
                break

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
        
        # Initialize data-driven classification engine
        if self.database:
            self.data_driven_engine = DataDrivenClassificationEngine(self.database)
        else:
            self.data_driven_engine = None
        
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

    def classify_model(self, file_path: Path, file_hash: str = None, quiet: bool = False, model_id: int = None) -> ClassificationResult:
        """Classify a model using multi-layer analysis with comprehensive debug logging"""
        
        # Initialize debug tracker
        debugger = ClassificationDebugger()
        debugger.add_step("classification_start", "started", file_path=str(file_path))
        
        # Set file analysis info
        file_size = file_path.stat().st_size if file_path.exists() else 0
        debugger.set_file_analysis(file_size, file_path.suffix, file_path.name)
        
        # Note: Removed problematic "efficient reclassification" branching logic
        # All models now go through the same comprehensive classification process
        # This ensures consistent results between initial scan and reclassification
        
        # Full classification process for all models (unified approach)
        debugger.add_step("full_classification", "started")
        
        # Calculate hash if not provided
        if file_hash is None:
            debugger.add_step("hash_calculation", "started")
            file_hash = self.calculate_file_hash(file_path)
            if file_hash:
                debugger.add_step("hash_calculation", "completed", hash_length=len(file_hash))
            else:
                debugger.add_step("hash_calculation", "failed")
                debugger.add_error("Failed to calculate file hash")
                result = ClassificationResult(
                    primary_type="unknown",
                    sub_type="unknown",
                    confidence=0.0,
                    method="failed",
                    debug_info=debugger.to_json()
                )
                return result
        
        # STEP 1: LFS Pointer Detection
        debugger.add_step("lfs_check", "started")
        if SafeTensorsExtractor.is_lfs_pointer_file(file_path):
            debugger.add_step("lfs_check", "detected", result="lfs_pointer")
            result = ClassificationResult(
                primary_type="lfs_pointer",
                sub_type="undownloaded",
                confidence=1.0,
                method="lfs_detection",
                debug_info=debugger.to_json()
            )
            return result
        else:
            debugger.add_step("lfs_check", "passed", result="not_lfs_pointer")
        
        # STEP 2: GGUF Special Handling
        debugger.add_step("gguf_check", "started")
        if file_path.suffix.lower() == '.gguf':
            debugger.add_step("gguf_check", "detected", result="gguf_file")
            result = self.classify_gguf(file_path, file_hash, quiet)
            # Add debug info to result
            result.debug_info = debugger.to_json()
            return result
        else:
            debugger.add_step("gguf_check", "passed", result="not_gguf")
        
        # STEP 3: Manual Overrides (stubbed for now)
        debugger.add_step("manual_override", "started")
        # TODO: Implement full manual override system
        # For now, just check basic config overrides
        manual_overrides = self.class_config.get('manual_overrides', {})
        filename_key = file_path.name.lower()
        if filename_key in manual_overrides:
            override = manual_overrides[filename_key]
            debugger.add_step("manual_override", "applied", override_type=override.get('primary_type', 'unknown'))
            result = ClassificationResult(
                primary_type=override.get('primary_type', 'unknown'),
                sub_type=override.get('sub_type', 'unknown'),
                confidence=1.0,
                method="manual_override",
                triggers=override.get('triggers', []),
                debug_info=debugger.to_json()
            )
            return result
        else:
            debugger.add_step("manual_override", "skipped", reason="no_override_found")
        
        # STEP 4: Data-driven Classification Engine (temporarily disabled for debugging)
        debugger.add_step("data_driven_classification", "started")
        if self.data_driven_engine:
            debugger.add_step("data_driven_classification", "skipped", reason="temporarily_disabled_for_debugging")
            if not quiet:
                print("    Data-driven classification engine temporarily disabled for debugging...")
        else:
            debugger.add_step("data_driven_classification", "skipped", reason="no_engine")
        
        # STEP 5: CivitAI API Lookup (store result for later comparison)
        debugger.add_step("civitai_api_lookup", "started")
        if not quiet:
            print("    Querying CivitAI API...")
        
        civitai_result = self.civitai.lookup_by_hash(file_hash)
        civitai_classification = None
        
        if civitai_result.get('found'):
            primary_type = civitai_result['primary_type']
            base_model = civitai_result['base_model']
            sub_type = self.determine_sub_type(primary_type, base_model, file_path.name)
            
            debugger.add_step("civitai_api_lookup", "found", 
                            model_name=civitai_result.get('name', 'Unknown'),
                            primary_type=primary_type,
                            base_model=base_model)
            
            if not quiet:
                print(f"    CivitAI: {civitai_result.get('name', 'Unknown')} ({primary_type})")
            
            # Store CivitAI result for later comparison
            civitai_classification = ClassificationResult(
                primary_type=primary_type,
                sub_type=sub_type,
                confidence=civitai_result['confidence'],
                method="civitai_api",
                base_model=base_model,
                triggers=civitai_result.get('triggers', [])
            )
            civitai_classification.raw_metadata = {
                'civitai_response': civitai_result,
                'civitai_raw_data': civitai_result.get('raw_response', {})
            }
        else:
            debugger.add_step("civitai_api_lookup", "not_found")
        
        # STEP 6: SafeTensors Analysis
        debugger.add_step("safetensors_analysis", "started")
        local_result = None
        
        if file_path.suffix.lower() == '.safetensors':
            debugger.add_step("safetensors_analysis", "processing")
            local_result = self.classify_safetensors(file_path, file_hash, quiet)
            # Merge debug info
            if local_result.debug_info:
                existing_debug = json.loads(local_result.debug_info)
                debugger.classification_steps.extend(existing_debug.get('classification_steps', []))
                debugger.scores.update(existing_debug.get('scores', {}))
                debugger.tensor_analysis.update(existing_debug.get('tensor_analysis', {}))
            
            debugger.add_step("safetensors_analysis", "completed", 
                            primary_type=local_result.primary_type,
                            method=local_result.method)
        else:
            debugger.add_step("safetensors_analysis", "skipped", reason="not_safetensors")
            # STEP 7: Size-based Fallback
            debugger.add_step("size_fallback", "started")
            local_result = self.classify_by_size(file_path, file_hash, quiet)
            debugger.add_step("size_fallback", "completed", 
                            primary_type=local_result.primary_type,
                            method=local_result.method)
        
        # STEP 8: Compare CivitAI vs Local Classification
        final_result = self.choose_best_classification(civitai_classification, local_result, debugger)
        final_result.debug_info = debugger.to_json()
        return final_result
    
    def choose_best_classification(self, civitai_result: ClassificationResult, local_result: ClassificationResult, debugger: ClassificationDebugger) -> ClassificationResult:
        """Choose the best classification between CivitAI and local analysis"""
        if not civitai_result:
            debugger.add_step("classification_choice", "local_only", reason="no_civitai_result")
            return local_result
        
        if not local_result:
            debugger.add_step("classification_choice", "civitai_only", reason="no_local_result") 
            return civitai_result
        
        # Special rules for component classifications
        component_types = {'text_encoder', 'clip', 'vae'}
        if local_result.primary_type in component_types and civitai_result.primary_type not in component_types:
            debugger.add_step("classification_choice", "local_preferred", 
                            reason="component_vs_model", 
                            local_type=local_result.primary_type,
                            civitai_type=civitai_result.primary_type)
            return local_result
        
        # Confidence-based decision
        if local_result.confidence > civitai_result.confidence:
            debugger.add_step("classification_choice", "local_preferred",
                            reason="higher_confidence",
                            local_confidence=local_result.confidence,
                            civitai_confidence=civitai_result.confidence)
            return local_result
        else:
            debugger.add_step("classification_choice", "civitai_preferred",
                            reason="higher_confidence",
                            local_confidence=local_result.confidence, 
                            civitai_confidence=civitai_result.confidence)
            return civitai_result
    
    def classify_safetensors(self, file_path: Path, file_hash: str, quiet: bool = False) -> ClassificationResult:
        """Classify SafeTensors files using metadata and tensor analysis"""
        debugger = ClassificationDebugger()
        
        # Set file analysis info
        file_size = file_path.stat().st_size
        debugger.set_file_analysis(file_size, file_path.suffix, file_path.name)
        debugger.add_step("safetensors_classification", "started")
        
        if not quiet:
            print("    Analyzing SafeTensors metadata...")
        
        debugger.add_step("safetensors_extraction", "started")
        metadata = SafeTensorsExtractor.extract_metadata(file_path)
        if 'error' in metadata:
            debugger.add_step("safetensors_extraction", "failed", error=metadata['error'])
            debugger.add_error(f"SafeTensors extraction failed: {metadata['error']}")
            if not quiet:
                print(f"    SafeTensors error: {metadata['error']}")
            return self.classify_by_size(file_path, file_hash, quiet, debugger)
        
        debugger.add_step("safetensors_extraction", "success", tensor_count=metadata.get('tensor_count', 0))
        
        tensor_names = metadata.get('tensor_names', [])
        tensor_count = metadata.get('tensor_count', 0)
        st_metadata = metadata.get('metadata', {})
        
        # Analyze tensor patterns
        debugger.add_step("tensor_analysis", "started")
        tensor_scores = self.tensor_analyzer.analyze_tensors(tensor_names)
        architecture = self.tensor_analyzer.get_architecture_hints(tensor_names, st_metadata)
        debugger.add_step("tensor_analysis", "completed")
        
        # Calculate individual scores
        filename_score = self.calculate_filename_score(file_path.name)
        size_score = self.calculate_size_score(file_path.stat().st_size)
        tensor_score = max(tensor_scores.values()) if tensor_scores else 0.0
        metadata_score = self.calculate_metadata_score(st_metadata)
        
        # Set scores and tensor analysis in debugger
        debugger.set_scores(filename_score, size_score, tensor_score, metadata_score)
        debugger.set_tensor_analysis(tensor_count, tensor_scores, tensor_names)
        
        # Determine primary type based on highest tensor score
        has_meaningful_tensor_scores = tensor_scores and max(tensor_scores.values()) > 0.0
        if has_meaningful_tensor_scores:
            debugger.add_step("classification_decision", "tensor_based")
            primary_type = max(tensor_scores.items(), key=lambda x: x[1])[0]
            confidence = tensor_scores[primary_type]
        else:
            # Tensor analysis failed or all scores are 0 - use filename-based detection first
            debugger.add_step("classification_decision", "filename_based", reason="tensor_analysis_failed_or_zero")
            filename_lower = file_path.name.lower()
            
            # Always prioritize strong filename indicators, starting with components
            if any(pattern in filename_lower for pattern in ['t5xxl', 't5_xxl', 'umt5', 'text_encoder']):
                debugger.add_step("filename_detection", "success", detected="text_encoder", confidence=filename_score)
                primary_type = 'text_encoder'
                confidence = max(filename_score, 0.9)
            elif any(pattern in filename_lower for pattern in ['clip_l', 'clip_g', 'clip_vision', 'clip.']):
                debugger.add_step("filename_detection", "success", detected="clip", confidence=filename_score)
                primary_type = 'clip'
                confidence = max(filename_score, 0.9)
            elif 'vae' in filename_lower and 'encoder' not in filename_lower:
                debugger.add_step("filename_detection", "success", detected="vae", confidence=filename_score)
                primary_type = 'vae'
                confidence = max(filename_score, 0.9)
            elif 'lora' in filename_lower:
                debugger.add_step("filename_detection", "success", detected="lora", confidence=filename_score)
                primary_type = 'lora'
                confidence = max(filename_score, 0.8)
            elif 'controlnet' in filename_lower or 'control_' in filename_lower:
                debugger.add_step("filename_detection", "success", detected="controlnet", confidence=filename_score)
                primary_type = 'controlnet'
                confidence = max(filename_score, 0.8)
            elif 'embedding' in filename_lower or 'textual_inversion' in filename_lower:
                debugger.add_step("filename_detection", "success", detected="embedding", confidence=filename_score)
                primary_type = 'embedding'
                confidence = max(filename_score, 0.8)
            elif filename_score >= 0.8:  # Other high confidence filename matches
                debugger.add_step("filename_detection", "partial", reason="high_score_but_no_specific_match")
                # This catches cases like 'checkpoint', 'ip-adapter', etc.
                primary_type, confidence = self.classify_by_size_only(file_path.stat().st_size, file_path.suffix)
                confidence = max(confidence, filename_score)
            else:
                debugger.add_step("filename_detection", "failed", reason="low_confidence_score")
                # Fallback to size-based classification
                primary_type, confidence = self.classify_by_size_only(file_path.stat().st_size, file_path.suffix)
        
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
            if tensor_scores:
                print(f"    Tensor scores: {dict(sorted(tensor_scores.items(), key=lambda x: x[1], reverse=True))}")
            print(f"    Filename score: {filename_score:.2f}, Size score: {size_score:.2f}")
            print(f"    Has meaningful tensor scores: {has_meaningful_tensor_scores}")
            print(f"    Used filename-based detection: {not has_meaningful_tensor_scores and filename_score >= 0.8}")
            if triggers:
                print(f"    Triggers: {', '.join(triggers)}")
        
        # Prepare raw metadata for storage
        raw_metadata = {
            'safetensors_metadata': st_metadata,
            'tensor_names': tensor_names,
            'tensor_scores': tensor_scores,
            'file_size': file_path.stat().st_size
        }
        
        # Finalize debug info
        debugger.add_step("final_decision", "completed", 
                         primary_type=primary_type, 
                         sub_type=sub_type, 
                         final_confidence=final_confidence,
                         method="safetensors_analysis")
        
        classification_result = ClassificationResult(
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
            tensor_score=tensor_score,
            debug_info=debugger.to_json()
        )
        classification_result.raw_metadata = raw_metadata
        return classification_result
    
    def classify_gguf(self, file_path: Path, file_hash: str, quiet: bool = False) -> ClassificationResult:
        """Classify GGUF files"""
        if not quiet:
            print("    Analyzing GGUF file...")
        
        metadata = GGUFExtractor.extract_metadata(file_path)
        if 'error' in metadata:
            return self.classify_by_size(file_path, file_hash, quiet)
        
        # All GGUF files are quantized models by definition
        # File size is already stored separately in the database
        primary_type = 'gguf'
        sub_type = 'quantized_model'
        
        confidence = 0.9  # High confidence for GGUF detection
        
        # Get file size and tensor count for metadata
        file_size = file_path.stat().st_size
        tensor_count = metadata.get('tensor_count', 0)
        
        # Prepare raw metadata for storage
        raw_metadata = {
            'gguf_metadata': metadata,
            'file_size': file_size
        }
        
        classification_result = ClassificationResult(
            primary_type=primary_type,
            sub_type=sub_type,
            confidence=confidence,
            method="gguf_classification",
            tensor_count=tensor_count
        )
        classification_result.raw_metadata = raw_metadata
        return classification_result
    
    def classify_by_size(self, file_path: Path, file_hash: str, quiet: bool = False, debugger: ClassificationDebugger = None) -> ClassificationResult:
        """Classify model by file size as fallback"""
        if debugger is None:
            debugger = ClassificationDebugger()
            debugger.set_file_analysis(file_path.stat().st_size, file_path.suffix, file_path.name)
        
        debugger.add_step("size_classification", "started")
        
        file_size = file_path.stat().st_size
        primary_type, confidence = self.classify_by_size_only(file_size, file_path.suffix)
        
        debugger.add_step("size_analysis", "completed", 
                         detected=primary_type, 
                         confidence=confidence,
                         file_size_mb=round(file_size / (1024 * 1024), 1))
        
        # Calculate filename score for additional confidence
        filename_score = self.calculate_filename_score(file_path.name)
        
        # Combine size and filename scores
        combined_confidence = (confidence * 0.7) + (filename_score * 0.3)
        
        debugger.set_scores(filename_score=filename_score, size_score=confidence)
        debugger.add_step("final_decision", "completed", 
                         primary_type=primary_type, 
                         confidence=combined_confidence,
                         method="size_analysis")
        
        sub_type = self.determine_sub_type(primary_type, 'unknown', file_path.name)
        
        if not quiet:
            print(f"    Size-based: {primary_type} (confidence: {combined_confidence:.2f})")
        
        # Prepare raw metadata for storage
        raw_metadata = {
            'file_size': file_size,
            'filename': file_path.name,
            'file_extension': file_path.suffix
        }
        
        classification_result = ClassificationResult(
            primary_type=primary_type,
            sub_type=sub_type,
            confidence=combined_confidence,
            method="size_analysis",
            size_score=confidence,
            filename_score=filename_score,
            debug_info=debugger.to_json()
        )
        classification_result.raw_metadata = raw_metadata
        return classification_result
    
    def classify_by_size_only(self, file_size: int, file_extension: str = None) -> Tuple[str, float]:
        """Get type and confidence based on file size only"""
        best_match = 'unknown'
        best_confidence = 0.3
        
        for model_type, rules in self.size_rules.items():
            # Skip GGUF classification for non-.gguf files
            if model_type == 'gguf' and file_extension and file_extension.lower() != '.gguf':
                continue
                
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
        
        # Component-specific indicators (highest priority)
        if any(pattern in filename_lower for pattern in ['t5xxl', 't5_xxl', 'umt5', 'text_encoder']):
            return 0.95
        elif any(pattern in filename_lower for pattern in ['clip_l', 'clip_g', 'clip_vision', 'clip.']):
            return 0.95
        elif 'vae' in filename_lower and 'encoder' not in filename_lower:  # VAE but not text_encoder
            return 0.95
        
        # Model type indicators
        elif 'lora' in filename_lower:
            return 0.9
        elif 'controlnet' in filename_lower or 'control_' in filename_lower:
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
            if base_model in ['flux_dev', 'flux_schnell', 'flux_kontext', 'flux']:
                return base_model  # Return the specific flux variant
            elif base_model == 'sdxl':
                return 'sdxl'
            elif base_model == 'sd15':
                return 'sd15'
            elif base_model == 'sd3':
                return 'sd3'
            elif base_model == 'wan':
                return 'wan'
            elif base_model == 'pony':
                return 'pony'
            elif 'pony' in filename_lower:
                return 'pony'
            elif 'wan' in filename_lower:
                return 'wan'
            elif 'flux' in filename_lower:
                return 'flux'
            elif 'sd3' in filename_lower:
                return 'sd3'
            elif 'xl' in filename_lower or 'sdxl' in filename_lower:
                return 'sdxl'
            elif '1.5' in filename_lower or 'v1-5' in filename_lower:
                return 'sd15'
            else:
                return 'unknown'
        
        elif primary_type == 'checkpoint':
            if base_model in ['flux_dev', 'flux_schnell', 'flux_kontext', 'flux']:
                return base_model  # Return the specific flux variant
            elif base_model == 'sdxl':
                return 'sdxl'
            elif base_model == 'sd15':
                return 'sd15'
            elif base_model == 'sd3':
                return 'sd3'
            elif base_model == 'wan':
                return 'wan'
            elif base_model == 'pony':
                return 'pony'
            elif 'pony' in filename_lower:
                return 'pony'
            elif 'wan' in filename_lower:
                return 'wan'
            elif 'flux' in filename_lower:
                return 'flux'
            elif 'sd3' in filename_lower:
                return 'sd3'
            elif 'xl' in filename_lower or 'sdxl' in filename_lower:
                return 'sdxl'
            elif '1.5' in filename_lower or 'v1-5' in filename_lower:
                return 'sd15'
            elif 'upscal' in filename_lower or 'esrgan' in filename_lower:
                return 'upscale'
            else:
                return 'unknown'
        
        elif primary_type == 'vae':
            if 'video' in filename_lower:
                return 'video'
            elif base_model == 'flux':
                return 'flux'
            elif base_model == 'sdxl':
                return 'sdxl'
            elif base_model == 'sd3':
                return 'sd3'
            elif base_model == 'sd15':
                return 'sd15'
            elif 'flux' in filename_lower:
                return 'flux'
            elif 'sd3' in filename_lower:
                return 'sd3'
            elif 'xl' in filename_lower or 'sdxl' in filename_lower:
                return 'sdxl'
            elif '1.5' in filename_lower or 'v1-5' in filename_lower:
                return 'sd15'
            else:
                return 'unknown'
        
        elif primary_type == 'controlnet':
            if base_model == 'flux':
                return 'flux'
            elif base_model == 'sdxl':
                return 'sdxl'
            elif base_model == 'sd3':
                return 'sd3'
            elif base_model == 'sd15':
                return 'sd15'
            elif base_model == 'wan':
                return 'wan'
            elif 'flux' in filename_lower:
                return 'flux'
            elif 'sd3' in filename_lower:
                return 'sd3'
            elif 'xl' in filename_lower or 'sdxl' in filename_lower:
                return 'sdxl'
            elif 'wan' in filename_lower:
                return 'wan'
            elif '1.5' in filename_lower or 'v1-5' in filename_lower:
                return 'sd15'
            else:
                return 'unknown'
        
        elif primary_type == 'ultralytics':
            if 'detect' in filename_lower or 'bbox' in filename_lower:
                return 'bbox'
            elif 'segment' in filename_lower or 'segm' in filename_lower:
                return 'segm'
            else:
                return 'unknown'
        
        # For all other model types, use standard sub-type
        elif primary_type in ['facerestore', 'insightface', 'photomaker', 'style_model', 
                            'diffuser', 'gligen', 'grounding_dino', 'sam', 'rmbg', 
                            'vae_approx', 'clip', 'clip_vision', 'text_encoder', 
                            'unet', 'embedding', 'upscaler', 'hypernetwork']:
            return 'standard'
        
        return 'unknown'
    
    def calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate SHA-256 hash of file, using cached hash file if available"""
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
            return None