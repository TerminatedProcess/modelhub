# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ModelHub is a comprehensive AI model classification and deployment system that automatically organizes machine learning models using hash-based deduplication and provides symlink-based deployment to target applications. The system features both CLI and TUI interfaces for model management.

## Architecture

### Core Components
- **modelhub.py** - Main application entry point with TUI launcher
- **database.py** - SQLite database abstraction layer with Model/DeployTarget dataclasses
- **classifier.py** - Multi-method model classification engine (filename, size, metadata, tensor analysis)
- **tui.py** - ncurses-based terminal user interface with filtering and navigation
- **config.py** - Configuration management with YAML support
- **clipboard_utils.py** - Cross-platform clipboard integration

### Database Architecture
- **Primary Database**: SQLite at path specified in config.yaml (`model_hub.path`)
- **Main Tables**: `models`, `model_metadata`, `deploy_targets`, `deploy_mappings`, `deploy_links`
- **Classification Tables**: `model_types`, `size_rules`, `sub_type_rules`, `architecture_patterns`
- **Hash-based Storage**: Models stored in `{model_hub_path}/models/{hash}/` directories

### Classification System
Multi-layer classification with weighted scoring:
1. **Filename Analysis** - Pattern matching on filenames
2. **Size Analysis** - File size-based type detection  
3. **Metadata Extraction** - SafeTensors/GGUF metadata parsing
4. **Tensor Analysis** - Deep model architecture inspection
5. **External APIs** - CivitAI and other service integration

## Common Commands

### Running the Application
```bash
# Launch TUI (main interface)
python3 modelhub.py

# Launch with custom config
python3 modelhub.py --config custom_config.yaml

# Legacy CLI interface (in hold/ directory)
python3 hold/modelhub_cli.py stats
```

### Database Operations
```bash
# Database path is in config.yaml - check with:
python3 -c "from config import ConfigManager; print(ConfigManager('config.yaml').get_db_path())"

# View database schema
sqlite3 $(python3 -c "from config import ConfigManager; print(ConfigManager('config.yaml').get_db_path())") ".schema"

# Count models by type
sqlite3 $(python3 -c "from config import ConfigManager; print(ConfigManager('config.yaml').get_db_path())") "SELECT primary_type, COUNT(*) FROM models WHERE deleted = 0 GROUP BY primary_type"
```

### Development Setup
```bash
# Install dependencies
pip3 install -r requirements.txt

# Create deployment configuration
python3 hold/create_default_deploy.py
```

## Configuration

### Main Configuration (config.yaml)
- **model_hub.path**: Base directory for model storage (default: `/mnt/llm/model-hub`)
- **model_hub.page_size**: Models per page in TUI (default: 1000)
- **classification.confidence_threshold**: Minimum confidence for auto-classification (default: 0.5)
- **file_extensions**: Supported formats (.safetensors, .ckpt, .pth, .pt, .gguf)

### Deploy Targets
Four preconfigured deployment targets (disabled by default):
- ComfyUI: `~/comfy/ComfyUI/models`
- WAN Video: `~/pinokio/api/wan.git/app` 
- Forge WebUI: `~/pinokio/api/stable-diffusion-webui-forge.git/app/models`
- Image Upscale: `~/pinokio/api/Image-Upscale.git/models`

## Development Notes

### Key Design Patterns
- **Context Managers**: Database connections use `with ModelHubDB(db_path) as db:`
- **Dataclasses**: Strong typing with `@dataclass` for Model, DeployTarget, ClassificationResult
- **Path Objects**: Use `pathlib.Path` throughout for cross-platform compatibility
- **Configuration-Driven**: All paths and settings externalized to config.yaml

### TUI Interface
- **Navigation**: Arrow keys, Page Up/Down, Home/End
- **Filtering**: F1-F3 for model/type/subtype filters, F4 for non-CivitAI toggle
- **Sorting**: Click column headers or use keyboard shortcuts
- **Mouse Support**: Click anywhere in interface for selection

### Classification Logic
- **Multi-score System**: Combines filename_score, size_score, metadata_score, tensor_score
- **Configurable Thresholds**: Size rules per model type in config.yaml
- **Reclassification Support**: Models can be marked for re-analysis
- **External API Integration**: CivitAI lookups for enhanced metadata

### Database Migration
- **Schema Evolution**: Use `hold/db_update.py` for database migrations
- **Hash Deduplication**: `file_hash` column with unique constraint prevents duplicates
- **Soft Deletion**: Models marked as deleted rather than removed

## File Organization

### Storage Structure
```
{model_hub_path}/
├── modelhub.db              # Main database
└── models/                  # Hash-organized model storage
    ├── {hash1}/
    │   └── model-file.safetensors
    ├── {hash2}/
    │   └── another-model.ckpt
    └── ...
```

### Deployment Structure
Models deployed via symlinks from hash storage to target applications:
```
~/comfy/ComfyUI/models/
├── checkpoints/
│   └── model1.safetensors -> /path/to/hash-storage/model1.safetensors
├── loras/
│   └── model2.safetensors -> /path/to/hash-storage/model2.safetensors
└── ...
```

## Dependencies

### Required Python Packages
- **PyYAML**: Configuration file parsing
- **requests**: External API integration
- **pyperclip**: Clipboard operations

### Built-in Libraries Used
- **sqlite3**: Database operations
- **curses**: TUI interface
- **pathlib**: Modern path handling
- **hashlib**: File hashing for deduplication
- **dataclasses**: Type-safe data structures

## Testing and Validation

### Database Testing
```bash
# Test database connection
python3 -c "from database import ModelHubDB; from config import ConfigManager; db_path = ConfigManager('config.yaml').get_db_path(); print(f'Models: {ModelHubDB(db_path).get_model_count()}')"

# Validate deploy targets
python3 -c "from database import ModelHubDB; from config import ConfigManager; db_path = ConfigManager('config.yaml').get_db_path(); print(f'Deploy targets: {len(ModelHubDB(db_path).get_deploy_targets())}')"
```

### Classification Testing
```bash
# Test classifier on sample file
python3 -c "from classifier import ModelClassifier; from pathlib import Path; classifier = ModelClassifier(); result = classifier.classify_model(Path('sample.safetensors')); print(f'Type: {result.primary_type}, Confidence: {result.confidence}')"
```

## Security Considerations

- **Path Traversal Prevention**: All file operations use `pathlib.Path.resolve()`
- **Database Injection Protection**: Parameterized queries throughout
- **Safe File Operations**: Proper error handling for file system operations
- **External API Rate Limiting**: Built-in delays for API calls