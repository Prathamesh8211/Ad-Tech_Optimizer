import re
import glob

py_files = glob.glob("pages/*.py") + ["app.py", "llm/copilot.py"]

emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]')

found_issues = []
for fpath in py_files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
        matches = emoji_pattern.findall(content)
        if matches:
            found_issues.append(f"{fpath}: Found {len(matches)} emojis: {set(matches)}")

with open("scratch/emoji_check_result.txt", "w", encoding="utf-8") as out:
    if found_issues:
        out.write("EMOJI CHECK FAILED:\n" + "\n".join(found_issues))
    else:
        out.write("EMOJI CHECK PASSED: Zero emojis found in UI code!")

print("Check finished.")
