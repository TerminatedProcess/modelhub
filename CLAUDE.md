# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a model hub database system that classifies and manages machine learning models. The system uses SQLite to store model metadata and classification information.

## Database Architecture

The core data is stored in `modelhub.db`, a SQLite database with two main tables:

- `models` - Primary table storing model file information, classification results, and metadata
- `model_metadata` - Key-value pairs for extended model metadata

The database contains 435+ models across various types including:
- LoRA models (263)
- Checkpoints (94) 
- Video models (27)
- GGUF models (18)
- ControlNet models (13)
- VAE, text encoders, embeddings, and other specialized models

## Model Classification System

Models are classified with:
- `primary_type` and `sub_type` for categorization
- `confidence` score and `classification_method`
- Multiple scoring mechanisms: filename, size, metadata, and tensor scores
- Support for reclassification via the `reclassify` field

## Database Operations

Use SQLite commands to interact with the database:

```bash
# View database schema
sqlite3 modelhub.db ".schema"

# Count models by type
sqlite3 modelhub.db "SELECT primary_type, COUNT(*) FROM models GROUP BY primary_type"

# View model details
sqlite3 modelhub.db "SELECT filename, primary_type, confidence FROM models WHERE primary_type = 'lora' LIMIT 5"
```

## Configuration

The repository includes Claude Code permissions in `.claude/settings.local.json` allowing:
- SQLite database operations (`sqlite3:*`)
- File system listing (`ls:*`)

## Development Notes

- This appears to be a data-only repository with the database as the primary asset
- No build system, testing framework, or source code files are present
- The database uses file hashing for deduplication (`file_hash` column with unique constraint)
- Timestamps track when models were classified and last updated