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
- **Dual Database System**: Separated concerns between model data and classification rules
  - **Main Database**: `{model_hub_path}/modelhub.db` - Model records, metadata, deployments
  - **Classification Database**: `{model_hub_path}/classification.db` - Rules, patterns, configurations
- **Main Tables**: `models`, `model_metadata`, `deploy_targets`, `deploy_mappings`, `deploy_links`
- **Classification Tables**: `model_types`, `size_rules`, `sub_type_rules`, `architecture_patterns`
- **Hash-based Storage**: Models stored in `{model_hub_path}/models/{hash}/` directories
- **Development Database**: `./modelhub.db` in project root (empty, for development)

### Classification System
Multi-step classification pipeline:
1. **GGUF Extension Check** - Direct classification for .gguf files
2. **LFS Pointer Detection** - Check for undownloaded Git LFS files
3. **CivitAI API Lookup** - Community-validated classification (highest priority)
4. **SafeTensors Analysis** - Database-driven tensor pattern matching
5. **Size-based Fallback** - File size classification for unsupported formats
6. **Final Decision Logic** - Compare CivitAI vs local analysis results

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
# Activate virtual environment (if using direnv)
# direnv allow (sets up venv automatically)

# Manual venv activation
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Create deployment configuration
python3 hold/create_default_deploy.py
```

### Development Shortcuts
```bash
# Common aliases (from .salias)
alias run='mhubd;python modelhub.py'  # Quick run command
alias vic='mhub;vi config.yaml'       # Edit config

# Direct execution
python3 modelhub.py                   # Launch TUI
python3 hold/modelhub_cli.py stats    # Legacy CLI stats
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

### Legacy Components
- **hold/ Directory**: Contains legacy CLI and utility scripts
  - `modelhub_cli.py`: Original CLI interface (still functional)
  - `create_default_deploy.py`: Deployment configuration setup
  - `db_update.py`: Database migration utilities
  - `hubclassify.py`: Standalone classification tool

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

### Python Version Requirement
- **Python 3.10+** (Project uses Python 3.12.10)

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

### Development Environment
- **Virtual Environment**: Use `venv/` directory for isolated dependencies
- **Environment Setup**: Project uses `.envrc` with direnv for automatic venv activation

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

## Additional Documentation

### Classification System Deep Dive
- **ClassifyLogic.md**: Comprehensive documentation of the classification system
  - Dual database architecture details
  - 24+ model types with ComfyUI alignment
  - Sub-type detection patterns (flux, sd3, sdxl, pony, etc.)
  - Tensor analysis and architecture patterns
  - External API integration details

### Project Status and Roadmap
- **README.md**: Current project status, feature roadmap, and development notes
  - Production statistics (539+ models classified)
  - Feature implementation status
  - Technical architecture overview
  - Usage examples and hotkey references