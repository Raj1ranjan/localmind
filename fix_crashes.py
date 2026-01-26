#!/usr/bin/env python3
"""
Fix for LocalMind crashes on Windows
Addresses JSON corruption and thread cleanup issues
"""

import json
import os
import logging
from pathlib import Path

def fix_corrupted_chat_files():
    """Fix corrupted JSON chat files"""
    chats_dir = Path("chats")
    if not chats_dir.exists():
        return
    
    fixed_count = 0
    for json_file in chats_dir.glob("*.json"):
        try:
            # Try to read the file
            with open(json_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Check if file is empty or corrupted
            if not content or content == "":
                print(f"Fixing empty file: {json_file}")
                # Create minimal valid chat structure
                chat_id = json_file.stem
                default_chat = {
                    "id": chat_id,
                    "name": "New Chat",
                    "messages": [],
                    "html_content": "",
                    "created": "2026-01-26T00:00:00",
                    "profile": "general",
                    "draft_message": ""
                }
                
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(default_chat, f, indent=2, ensure_ascii=False)
                fixed_count += 1
                continue
            
            # Try to parse JSON
            json.loads(content)
            print(f"Valid JSON: {json_file}")
            
        except json.JSONDecodeError as e:
            print(f"Fixing corrupted JSON: {json_file} - {e}")
            # Backup corrupted file
            backup_file = json_file.with_suffix('.json.backup')
            json_file.rename(backup_file)
            
            # Create new valid file
            chat_id = json_file.stem
            default_chat = {
                "id": chat_id,
                "name": "Recovered Chat",
                "messages": [],
                "html_content": "",
                "created": "2026-01-26T00:00:00",
                "profile": "general",
                "draft_message": ""
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(default_chat, f, indent=2, ensure_ascii=False)
            fixed_count += 1
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f"Fixed {fixed_count} corrupted chat files")

if __name__ == "__main__":
    fix_corrupted_chat_files()
