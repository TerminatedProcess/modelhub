# ModelHub - AI Model Classification and Deployment System

A comprehensive database system for classifying, managing, and deploying machine learning models with automated organization and symlink-based deployment.

## Project Overview

ModelHub is a sophisticated model management system that:
- Automatically classifies AI models (LoRA, checkpoints, VAE, etc.) using multiple detection methods
- Stores models in a hash-based deduplication system
- Provides deployment automation with symlink management to target applications
- Offers both CLI and TUI interfaces for model management

## Current State

### Database Architecture
- **Main Database**: `/mnt/llm/model-hub/modelhub.db` (models, metadata, deployment)
- **Classification Database**: `/mnt/llm/model-hub/classification.db` (types, rules, patterns)
- **Working Copy**: `./modelhub.db` (empty, for development)
- **Architecture**: Dual SQLite databases for separation of concerns

### Model Statistics (Production Database)
- **Total Models**: 539 active models
- **Primary Types**:
  - LoRA: 360 models (67%)
  - Checkpoint: 98 models (18%)
  - UNet: 38 models (7%)
  - GGUF: 12 models (2%)
  - VAE: 8 models (1.5%)
  - Other: 23 models (embeddings, CLIP, text encoders, etc.)

### Storage Structure
```
/mnt/llm/model-hub/
├── modelhub.db                 # Main database (models, metadata, deployment)
├── classification.db           # Classification database (types, rules, patterns)
└── models/                     # Hash-organized model storage
    ├── 00674ab0e236.../        # Model hash directories
    │   └── model-file.safetensors
    ├── 00d7af617872.../
    │   └── another-model.safetensors
    └── ...
```

## Key Components

### Core Scripts
- `modelhub.py` - Main application entry point with TUI launcher
- `tui.py` - Terminal user interface with full model management
- `database.py` - Database operations and model management
- `classifier.py` - Model classification engine
- `create_default_deploy.py` - Deployment configuration setup

### Configuration
- `config.yaml` - Main configuration file
- `CLAUDE.md` - Project documentation and AI assistant instructions

### Database Schema

#### Main Database (modelhub.db)
- **models** - Primary model records with classification data
- **model_metadata** - Extended key-value metadata storage
- **deploy_targets** - Deployment destination configurations
- **deploy_mappings** - Model type to folder path mappings
- **deploy_links** - Symlink deployment tracking

#### Classification Database (classification.db)
- **model_types** - Supported model type definitions (checkpoint, lora, vae, etc.)
- **sub_type_rules** - Pattern-based sub-type classification (flux, sdxl, pony, etc.)
- **size_rules** - File size-based classification rules
- **architecture_patterns** - Tensor analysis patterns for deep classification
- **external_apis** - External service configurations (CivitAI, etc.)

*Note: See [ClassifyLogic.md](ClassifyLogic.md) for detailed classification system documentation.*

## Classification System

### Multi-Method Classification
1. **Filename Analysis** - Pattern matching on filenames
2. **File Size Analysis** - Size-based type detection
3. **Metadata Extraction** - SafeTensors/GGUF metadata parsing
4. **Tensor Analysis** - Deep model architecture inspection
5. **External APIs** - CivitAI and other service integration

### Confidence Scoring
- Weighted scoring across all detection methods
- Configurable confidence thresholds
- Manual override capabilities
- Reclassification support

## Deployment System

### Current Targets (All Disabled by Default)
- **ComfyUI**: `~/comfy/ComfyUI/models`
- **WAN Video**: `~/pinokio/api/wan.git/app`
- **Forge WebUI**: `~/pinokio/api/stable-diffusion-webui-forge.git/app/models`
- **Image Upscale**: `~/pinokio/api/Image-Upscale.git/models`

### Deployment Mappings
Each target has specific folder mappings for different model types:
- Checkpoints → `checkpoints/` or `Stable-diffusion/`
- LoRAs → `loras/` or `Lora/`
- VAE → `vae/` or `VAE/`
- ControlNet → `controlnet/` or `ControlNet/`
- etc.

### Planned Deployment Flow
Models are deployed via symlinks from hash-organized storage to target application folders, maintaining organization while avoiding duplication.

## Current Development Status

### Completed Features ✅
- [x] Database schema and infrastructure
- [x] Model classification system with multi-method detection
- [x] Hash-based storage with deduplication
- [x] Metadata extraction and storage
- [x] Basic deployment configuration
- [x] CLI interface
- [x] TUI interface with full model management
- [x] Runtime symlinks toggle (Shift+Y hotkey)
- [x] Model scanning with symlink conversion
- [x] Comprehensive model reclassification
- [x] Model metadata viewing and management
- [x] Hash file generation
- [x] Clipboard integration for symlink commands
- [x] Live filtering and sorting in TUI
- [x] Model deletion with soft delete support

### In Progress 🚧
- [ ] **Deploy Stub Creation** - Building symlink-based deployment system

### Planned Features 📋
- [ ] **Product Folder Creation** - Automated symlink deployment to target applications
- [ ] **Deployment Management** - Enable/disable targets, track deployment status
- [ ] **Broken Symlink Cleanup** - Maintenance utilities
- [ ] **Deployment Status Dashboard** - Monitor deployed vs available models
- [ ] **Batch Operations** - Deploy/undeploy multiple models
- [ ] **Model Deduplication Tools** - Find and manage duplicate models
- [ ] **Orphaned File Cleanup** - Clean up unused model files
- [ ] **Export/Import Functions** - Model sharing and backup

## Technical Architecture

### Dependencies
- Python 3.x
- SQLite database
- SafeTensors library support
- GGUF format support
- External API integrations

### Key Design Principles
1. **Hash-based Storage** - Prevents duplication, enables safe symlinks
2. **Multi-method Classification** - Robust model type detection
3. **Symlink Deployment** - Efficient space usage, maintains single source
4. **Extensible Architecture** - Easy to add new model types and targets
5. **Metadata Preservation** - Rich model information storage

## Configuration

### Model Hub Settings
- **Path**: `/mnt/llm/model-hub` (configurable in config.yaml)
- **Page Size**: 1000 models per page (configurable)
- **Symlinks**: Runtime toggle (Shift+Y), defaults to enabled each session

### Supported File Extensions
- `.safetensors` (primary format)
- `.ckpt` (legacy checkpoints)
- `.pth` / `.pt` (PyTorch models)
- `.gguf` (GGML Universal Format)

### Classification Thresholds
- **Confidence Threshold**: 0.5
- **External APIs**: Enabled
- **CivitAI Reclassification**: Enabled

## Usage Examples

### Basic Operations
```bash
# View model statistics
python modelhub_cli.py stats

# Launch TUI
python modelhub.py

# Create deployment configuration
python create_default_deploy.py

# Scan and import models (TUI has Shift+S hotkey)
python modelhub_cli.py scan /path/to/models
```

### TUI Hotkeys
```
Navigation:
  ↑/↓ - Navigate models
  PgUp/PgDn - Jump by page
  ←/→ - Previous/next page
  ENTER/'d' - Show model details
  
Filtering:
  Click filter fields or 'F' - Activate filtering
  Tab - Move between filter fields
  'n' - Toggle non-CivitAI filter
  'r' - Reset all filters
  
Model Management:
  'S' - Scan directory for new models
  'R' - Reclassify models
  'X' - Delete selected models
  'H' - Generate hash files
  'Y' - Toggle symlinks on/off (defaults to on)
  
Deployment:
  'D' - Deploy menu
  'c' - Configure deploy targets
  'e' - Export models
  
Other:
  'h' - Toggle help screen
  'q' - Quit application
```

### Database Queries
```bash
# View model types
sqlite3 /mnt/llm/model-hub/modelhub.db "SELECT primary_type, COUNT(*) FROM models WHERE deleted = 0 GROUP BY primary_type"

# Check deployment targets
sqlite3 /mnt/llm/model-hub/modelhub.db "SELECT name, display_name, enabled FROM deploy_targets"
```

## Next Steps

### Immediate Priorities
1. **Complete Deploy Stub Implementation** - Create the symlink deployment system
2. **Enable Deployment Targets** - Activate and test deployment to target applications
3. **Testing and Validation** - Ensure deployment system works correctly
4. **Documentation Updates** - User guides and deployment instructions

### Future Enhancements
1. **Web Interface** - Browser-based model management
2. **Model Versioning** - Track model updates and changes
3. **Performance Optimization** - Speed improvements for large collections
4. **Advanced Search** - Rich filtering and search capabilities
5. **Model Analytics** - Usage statistics and insights

## Development Notes

- The system uses a working directory setup with the main database in `/mnt/llm/model-hub/`
- Development database is at `./modelhub.db` for testing
- The project follows defensive coding practices with comprehensive error handling
- All database operations include proper transaction management
- The classification system is designed to be extensible and configurable

## File Organization

### Primary Python Modules
- **modelhub.py** - Application entry point and core logic
- **database.py** - Database abstraction layer and operations
- **classifier.py** - Model classification and analysis engine
- **tui.py** - Terminal user interface implementation
- **config.py** - Configuration management

### Utility Scripts
- **create_default_deploy.py** - Deployment setup utility
- **db_update.py** - Database migration and updates
- **hubclassify.py** - Standalone classification utility
- **clipboard_utils.py** - Cross-platform clipboard integration

### Legacy Scripts (in hold/ directory)
- **modelhub_cli.py** - Original CLI interface
- **Various utilities** - Legacy tools and migration scripts

### Configuration Files
- **config.yaml** - Main application configuration
- **requirements.txt** - Python dependencies
- **CLAUDE.md** - AI assistant instructions and project documentation

This README provides a comprehensive overview of the ModelHub project's current state, architecture, and development roadmap. It serves as both documentation and a reference for continuing development work.

## Maintenance Notes

**📋 README Update Protocol**: This README must be kept current as development progresses. When implementing new features, fixing bugs, or making architectural changes:

1. **Update Status Sections** - Move items from "Planned" to "In Progress" to "Completed"
2. **Modify Statistics** - Update model counts and database metrics
3. **Add New Features** - Document new capabilities and usage examples
4. **Update Architecture** - Reflect changes to database schema, file structure, or technical components
5. **Revise Configuration** - Update settings, paths, and configuration examples
6. **Maintain Accuracy** - Ensure all technical details remain current and accurate

**⚠️ Developer Reminder**: Always update this README as part of the development workflow to maintain project documentation quality and ensure smooth project handoffs.