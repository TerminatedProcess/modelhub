#!/usr/bin/env python3
"""
Default deployment configuration creator for ModelHub CLI
Creates default deploy targets and mappings in the database
Can be reused during application initialization
"""

import sqlite3
import os
from pathlib import Path

def create_default_deploy_data(db_path=None):
    """
    Create default deployment targets and mappings
    
    Args:
        db_path: Path to database file. If None, uses modelhub.db in current directory
    """
    if db_path is None:
        db_path = Path(__file__).parent / "modelhub.db"
    
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if deploy targets already exist
        cursor.execute("SELECT COUNT(*) FROM deploy_targets")
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            print(f"Deploy targets already exist ({existing_count} targets). Skipping default data creation.")
            return
        
        print("Creating default deploy targets...")
        
        # Default deploy targets
        deploy_targets = [
            ('comfyui', 'ComfyUI', '~/comfy/ComfyUI/models', False),
            ('wan', 'WAN Video', '~/pinokio/api/wan.git/app', False),
            ('forge', 'Forge WebUI', '~/pinokio/api/stable-diffusion-webui-forge.git/app/models', False),
            ('image_upscale', 'Image Upscale', '~/pinokio/api/Image-Upscale.git/models', False)
        ]
        
        # Insert deploy targets
        cursor.executemany("""
            INSERT INTO deploy_targets (name, display_name, base_path, enabled)
            VALUES (?, ?, ?, ?)
        """, deploy_targets)
        
        print(f"✓ Created {len(deploy_targets)} deploy targets")
        
        # Get target IDs for mappings
        cursor.execute("SELECT id, name FROM deploy_targets")
        target_ids = {name: id for id, name in cursor.fetchall()}
        
        # Default deploy mappings
        deploy_mappings = []
        
        # ComfyUI mappings
        comfyui_mappings = [
            ('checkpoint', 'checkpoints'),
            ('lora', 'loras'),
            ('vae', 'vae'),
            ('controlnet', 'controlnet'),
            ('embedding', 'embeddings'),
            ('upscaler', 'upscale_models'),
            ('text_encoder', 'text_encoders'),
            ('clip', 'clip'),
            ('unet', 'unet'),
            ('unknown', 'unknown')
        ]
        for model_type, folder_path in comfyui_mappings:
            deploy_mappings.append((target_ids['comfyui'], model_type, folder_path))
        
        # WAN mappings
        wan_mappings = [
            ('checkpoint', 'ckpts'),
            ('lora', 'loras'),
            ('video_lora', 'loras_i2v'),
            ('hunyuan_lora', 'loras_hunyuan'),
            ('ltxv_lora', 'loras_ltxv'),
            ('unknown', 'ckpts')  # Default unknown to ckpts for WAN
        ]
        for model_type, folder_path in wan_mappings:
            deploy_mappings.append((target_ids['wan'], model_type, folder_path))
        
        # Forge mappings
        forge_mappings = [
            ('checkpoint', 'Stable-diffusion'),
            ('lora', 'Lora'),
            ('vae', 'VAE'),
            ('controlnet', 'ControlNet'),
            ('embedding', 'embeddings'),
            ('upscaler', 'ESRGAN'),
            ('hypernetwork', 'hypernetworks'),
            ('text_encoder', 'text_encoder'),
            ('unknown', 'Stable-diffusion')  # Default unknown to checkpoints
        ]
        for model_type, folder_path in forge_mappings:
            deploy_mappings.append((target_ids['forge'], model_type, folder_path))
        
        # Image Upscale mappings
        image_upscale_mappings = [
            ('checkpoint', 'models/Stable-diffusion'),
            ('lora', 'Lora'),
            ('vae', 'VAE'),
            ('controlnet', 'ControlNet'),
            ('embedding', 'embeddings'),
            ('upscaler', 'upscalers'),
            ('unknown', 'models/Stable-diffusion')
        ]
        for model_type, folder_path in image_upscale_mappings:
            deploy_mappings.append((target_ids['image_upscale'], model_type, folder_path))
        
        # Insert deploy mappings
        cursor.executemany("""
            INSERT INTO deploy_mappings (target_id, model_type, folder_path)
            VALUES (?, ?, ?)
        """, deploy_mappings)
        
        print(f"✓ Created {len(deploy_mappings)} deploy mappings")
        
        # Commit changes
        conn.commit()
        print("✓ Default deploy configuration created successfully")
        
        # Display summary
        print("\nDefault Deploy Targets Created:")
        cursor.execute("""
            SELECT dt.display_name, dt.base_path, dt.enabled, COUNT(dm.id) as mappings
            FROM deploy_targets dt
            LEFT JOIN deploy_mappings dm ON dt.id = dm.target_id
            GROUP BY dt.id
            ORDER BY dt.name
        """)
        
        for display_name, base_path, enabled, mapping_count in cursor.fetchall():
            status = "enabled" if enabled else "disabled"
            print(f"  • {display_name}: {mapping_count} mappings ({status})")
            print(f"    Path: {base_path}")
        
        print("\nNote: All targets are disabled by default. Enable them in the CLI or by updating the database.")
        
    except sqlite3.Error as e:
        conn.rollback()
        raise Exception(f"Database error: {e}")
    except Exception as e:
        conn.rollback()
        raise Exception(f"Error creating default deploy data: {e}")
    finally:
        conn.close()

def main():
    """Main function when run as standalone script"""
    try:
        create_default_deploy_data()
        print("\n🎉 Default deployment configuration created successfully!")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())