# Classification Logic Documentation

## Overview

This document provides a comprehensive overview of the ModelHub classification system, including file locations, entry points, and the complete process flow. The system has been extensively updated to fully support ComfyUI's model ecosystem with 24+ model types and comprehensive base model support.

## Recent Updates

**Database Architecture Split (Latest):**
- Split into dual databases: `modelhub.db` (model data) and `classification.db` (rules/config)
- Added 24 comprehensive model types aligned with ComfyUI
- Enhanced sub-type detection for flux, sd3, sdxl, wan, pony, sd15, and more
- Complete deployment mapping coverage for all ComfyUI folders

## Classification Database (`classification.db`)

**Location**: `/mnt/llm/model-hub/classification.db`

This separate database stores all classification rules, patterns, and configurations:

### Tables:
- **model_types**: Primary model type definitions (24 types: checkpoint, lora, vae, etc.)
- **sub_type_rules**: Pattern-based sub-type classification (flux, sdxl, pony, etc.)
- **size_rules**: File size ranges for model type detection
- **architecture_patterns**: Tensor pattern matching rules for deep analysis
- **external_apis**: External service configurations (CivitAI, etc.)

### Sub-Type Rules Structure:
```sql
CREATE TABLE sub_type_rules (
    id INTEGER PRIMARY KEY,
    primary_type TEXT NOT NULL,
    sub_type TEXT NOT NULL,
    pattern1 TEXT,
    pattern2 TEXT,
    pattern3 TEXT,
    pattern_type TEXT DEFAULT 'filename',
    confidence REAL DEFAULT 0.5,
    priority INTEGER DEFAULT 1,
    enabled BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Current Sub-Types by Model Type:
- **checkpoint**: flux, sdxl, sd3, sd15, pony, wan, upscale, unknown
- **lora**: flux, sdxl, sd3, sd15, pony, wan, unknown  
- **controlnet**: flux, sdxl, sd3, sd15, wan, unknown
- **vae**: flux, sdxl, sd3, sd15, video, unknown
- **gguf**: quantized_model
- **ultralytics**: bbox, segm, unknown

**Supported Model Types:**
- Core: checkpoint, lora, vae, controlnet, unet, gguf
- Encoders: clip, clip_vision, text_encoder  
- Specialized: embedding, upscaler, hypernetwork
- Face/Image: facerestore, insightface, photomaker, style_model
- Advanced AI: diffuser, gligen, grounding_dino, ultralytics, sam, rmbg
- Lightweight: vae_approx

## Classification Logic Location and Flow

### **Main Classification File**: `/home/dev/work/modelhub/classifier.py`

### **Entry Points and Flow**:

#### **1. Primary Entry Point** - `classify_model()` method
**Location**: `classifier.py:417-488`

This is the main method that orchestrates the entire classification process:

```python
def classify_model(self, file_path: Path, file_hash: str = None, quiet: bool = False) -> ClassificationResult:
```

#### **2. Classification Process Flow**:

1. **LFS Check** (`classifier.py:425-431`)
   - Checks if file is an undownloaded Git LFS pointer

2. **GGUF Special Handling** (`classifier.py:433-442`)
   - Special processing for GGUF quantized models

3. **Manual Override Check** (`classifier.py:444-451`)
   - Checks config for explicit model classifications

4. **CivitAI API Lookup** (`classifier.py:453-461`)
   - Queries external API for model metadata if enabled

5. **SafeTensors Analysis** (`classifier.py:463-475`)
   - Extracts and analyzes tensor metadata

6. **Final Classification** (`classifier.py:477-488`)
   - Combines all scoring methods and returns result

### **Key Components**:

#### **A. ModelClassifier Class** - `classifier.py:355-868`
Main classification engine with initialization and core methods

#### **B. Tensor Analysis** - `classifier.py:138-223`
**Method**: `analyze_tensors()`
Analyzes tensor patterns to identify model architectures:
- LoRA patterns: `lora_up`, `lora_down`
- Flux patterns: `double_blocks`, `single_blocks`
- ControlNet patterns: `control_model`
- VAE patterns: `encoder.down`, `decoder.up`

#### **C. SafeTensors Extraction** - `classifier.py:42-110`
**Method**: `extract_safetensors_metadata()`
Extracts metadata from SafeTensors files

#### **D. CivitAI Integration** - `classifier.py:224-353`
**Method**: `lookup_civitai_model()`
External API lookup with rate limiting and error handling

### **How Classification is Triggered**:

#### **1. From Database Import** - `database.py:773-947`
```python
# Comprehensive classification using new system
from classifier import ModelClassifier
classifier = ModelClassifier(raw_config, database=self)
classification = classifier.classify_model(storage_path, file_hash, quiet)
```

#### **2. From TUI Reclassification** - `tui.py:1457`
```python
from classifier import ModelClassifier
classifier = ModelClassifier(fast_config, database=self.db)
classification = classifier.classify_model(storage_path, model.file_hash, quiet=True)
```

#### **3. From CLI Operations** - Various entry points in CLI scripts

### **Configuration** - `config.py:75-95`
Classification behavior is controlled by:
- `confidence_threshold`: Minimum confidence for auto-classification
- `enable_external_apis`: Whether to use CivitAI lookup
- `manual_overrides`: Explicit model classifications
- `size_rules`: File size ranges for different model types

## Detailed Classification Components

### 1. **Main ModelClassifier Class** (`/home/dev/work/modelhub/classifier.py`)

**Location**: Lines 355-868 in `/home/dev/work/modelhub/classifier.py`

The `ModelClassifier` class is the core classification engine that performs multi-layer analysis to determine model types.

**Key Features**:
- Multi-layer analysis with weighted scoring
- External API integration (CivitAI)
- SafeTensors and GGUF metadata extraction
- Tensor pattern analysis
- Fallback size-based classification

### 2. **Main Classification Method** (`classify_model`)

**Location**: Lines 417-488 in `/home/dev/work/modelhub/classifier.py`

The `classify_model` method is the main entry point that orchestrates the classification process:

```python
def classify_model(self, file_path: Path, file_hash: str = None, quiet: bool = False) -> ClassificationResult:
```

**Classification Steps**:
1. **LFS Pointer Detection** - Checks for undownloaded Git LFS files
2. **GGUF File Handling** - Special handling for GGUF quantized models
3. **Manual Overrides** - Checks configuration for explicit model classifications
4. **CivitAI API Lookup** - Queries external API for model metadata
5. **SafeTensors Analysis** - Extracts and analyzes tensor metadata
6. **Size-based Fallback** - Uses file size as final classification method

### 3. **Classification Entry Points**

#### **A. Database Import Process**
**Location**: Lines 773-947 in `/home/dev/work/modelhub/database.py`

The `import_model` method automatically classifies models during import:
```python
def import_model(self, file_path: Path, model_hub_path: Path, quiet: bool = False, config_manager=None) -> Optional[Model]:
    # ... file handling ...
    
    # Comprehensive classification using new system
    from classifier import ModelClassifier
    classifier = ModelClassifier(raw_config, database=self)
    
    # Classify the model
    classification = classifier.classify_model(storage_path, file_hash, quiet)
```

#### **B. TUI Reclassification**
**Location**: Lines 1275-1432 in `/home/dev/work/modelhub/tui.py`

The TUI provides reclassification functionality:
- **Bulk Reclassification**: `reclassify_models()` method
- **Single Model Reclassification**: `reclassify_single_model()` method

#### **C. Application Entry Point**
**Location**: `/home/dev/work/modelhub/modelhub.py`

The main application initializes the system and launches the TUI interface.

### 4. **Classification Components**

#### **A. SafeTensors Extractor**
**Location**: Lines 42-110 in `/home/dev/work/modelhub/classifier.py`

Extracts metadata from SafeTensors files including:
- Tensor information and counts
- Model metadata
- LFS pointer detection

#### **B. GGUF Extractor**
**Location**: Lines 111-137 in `/home/dev/work/modelhub/classifier.py`

Handles GGUF format files with version and tensor count extraction.

#### **C. Tensor Analyzer**
**Location**: Lines 138-223 in `/home/dev/work/modelhub/classifier.py`

Analyzes tensor patterns to identify model architectures:
- LoRA patterns (`lora_up`, `lora_down`)
- Flux patterns (`double_blocks`, `single_blocks`)
- ControlNet patterns (`control_model`)
- VAE patterns (`encoder.down`, `decoder.up`)

#### **D. CivitAI API Integration**
**Location**: Lines 224-353 in `/home/dev/work/modelhub/classifier.py`

Provides external model lookup with:
- Hash-based model identification
- Rate limiting and error handling
- Trigger word extraction
- Base model detection

### 5. **Configuration and Initialization**

#### **A. Classification Configuration**
**Location**: Lines 75-95 in `/home/dev/work/modelhub/config.py`

Configuration includes:
- Confidence thresholds
- External API settings
- Manual overrides
- Size rules for different model types

#### **B. Database Rules**
**Location**: Lines 381-491 in `/home/dev/work/modelhub/database.py`

The database stores classification rules including:
- Size rules for model types
- Sub-type classification patterns
- Architecture tensor patterns
- External API configurations

### 6. **Scoring System**

The classifier uses a weighted scoring system:
- **Filename Score** (20%): Based on filename patterns
- **Size Score** (20%): Based on file size ranges
- **Tensor Score** (50%): Based on tensor analysis (most reliable)
- **Metadata Score** (10%): Based on SafeTensors metadata

### 7. **Key Features**

1. **Multi-format Support**: SafeTensors, GGUF, legacy formats
2. **External API Integration**: CivitAI lookup with caching
3. **Trigger Word Extraction**: Automatic extraction from metadata
4. **Architecture Detection**: Advanced tensor pattern analysis
5. **Fallback Mechanisms**: Multiple classification methods
6. **Database Integration**: Persistent storage of classification rules
7. **Performance Optimization**: Hash-based deduplication and caching

## Summary of Process Start

The classification process is comprehensive, using multiple analysis methods with weighted scoring to determine the most likely model type and characteristics.

### **Process Start Points**:
1. **Entry**: `classifier.py:417` - `classify_model()` method
2. **Core Logic**: `classifier.py:355-868` - `ModelClassifier` class
3. **Tensor Analysis**: `classifier.py:138-223` - `analyze_tensors()` method
4. **External APIs**: `classifier.py:224-353` - `lookup_civitai_model()` method

The classification system provides comprehensive model type detection with high accuracy through multiple analysis layers and external data sources.