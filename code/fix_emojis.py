# Quick fix script to remove emojis from orchestrator.py
import re

file_path = r"c:\Users\user\Documents\GitHub\LLM-understand-graph\code\src\agents\orchestrator.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace emojis and special characters with text
content = content.replace('✅', '[OK]')
content = content.replace('❌', '[FAIL]')
content = content.replace('⚠️', '[WARN]')
content = content.replace('→', '->')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed emojis and special characters in orchestrator.py")
