#!/usr/bin/env python3
"""
Fix emoji issues in all Python files
"""

import re
import os

def fix_emojis_in_file(filename):
    """Remove emojis from a single file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove emojis from print statements
        emoji_patterns = [
            r'📥\s*', r'❌\s*', r'✅\s*', r'🔍\s*', r'🚀\s*', 
            r'💡\s*', r'🎯\s*', r'📊\s*', r'🎬\s*', r'🔄\s*',
            r'📝\s*', r'📈\s*', r'📋\s*', r'🎵\s*', r'🖼️\s*',
            r'⏱️\s*', r'🧠\s*', r'⚡\s*', r'📢\s*', r'🎭\s*',
            r'🔽\s*', r'🎪\s*', r'🎨\s*', r'⚠️\s*'
        ]
        
        for pattern in emoji_patterns:
            content = re.sub(pattern, '', content)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Emojis removed from {filename}")
        return True
        
    except Exception as e:
        print(f"Failed to fix {filename}: {e}")
        return False

def main():
    """Fix emojis in all Python files"""
    python_files = [
        'features.py',
        'suggest.py',
        'scoring.py'
    ]
    
    for file in python_files:
        if os.path.exists(file):
            fix_emojis_in_file(file)

if __name__ == "__main__":
    main()
