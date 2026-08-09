import re
import glob

py_files = glob.glob("pages/*.py") + ["app.py", "llm/copilot.py"]

# Emoji pattern covering all emoji unicode ranges
emoji_pattern = re.compile(
    r'[\U0001F600-\U0001F64F'  # emoticons
    r'\U0001F300-\U0001F5FF'  # symbols & pictographs
    r'\U0001F680-\U0001F6FF'  # transport & map symbols
    r'\U0001F1E0-\U0001F1FF'  # flags (iOS)
    r'\U00002702-\U000027B0'  # dingbats
    r'\U000024C2-\U0001F251'
    r'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
    r'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
    r'\U00002600-\U000026FF]'  # Misc symbols (e.g. ⚠, ⚡)
)

for fpath in py_files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Clean emojis
    cleaned = emoji_pattern.sub('', content)
    
    # Write back
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(cleaned)

print("STRIP COMPLETE: All emojis stripped from Python files.")
