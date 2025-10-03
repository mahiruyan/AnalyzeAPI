#!/usr/bin/env python3
"""
Fix emoji issues in utils.py
"""

import re

def fix_emojis():
    """Remove emojis from utils.py"""
    
    # Read the file
    with open('utils.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove emojis from print statements
    emoji_patterns = [
        r'📥\s*', r'❌\s*', r'✅\s*', r'🔍\s*', r'🚀\s*', 
        r'💡\s*', r'🎯\s*', r'📊\s*', r'🎬\s*', r'🔄\s*',
        r'📝\s*', r'📈\s*', r'📋\s*', r'🎵\s*', r'🖼️\s*',
        r'⏱️\s*', r'🧠\s*', r'⚡\s*', r'📢\s*', r'🎭\s*'
    ]
    
    for pattern in emoji_patterns:
        content = re.sub(pattern, '', content)
    
    # Write back
    with open('utils.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Emojis removed from utils.py")

if __name__ == "__main__":
    fix_emojis()
