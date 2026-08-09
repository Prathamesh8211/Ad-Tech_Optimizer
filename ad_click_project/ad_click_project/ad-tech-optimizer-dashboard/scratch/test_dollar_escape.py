import sys
sys.path.append("c:\\Users\\nshra\\Downloads\\ad_click_project\\ad-tech-optimizer-dashboard")

from utils.data_loader import load_sample_data
from pages._05_ai_copilot import render_simple_ai_response

df = load_sample_data()

# Simulate the exact type of query from the screenshot
query = "Summarize underperforming Food campaigns draining our budget."
response = render_simple_ai_response(query, df)

print("=== RESPONSE ===")
print(response[:600])
print()

# Check: $ should never appear UNESCAPED as a bare $ in the output
# Escaped form for Streamlit is \$
import re
bare_dollars = re.findall(r'(?<!\\)\$', response)
if bare_dollars:
    print(f"FAIL: Found {len(bare_dollars)} unescaped $ signs that could be eaten by LaTeX renderer")
else:
    print("PASS: All $ signs are properly escaped as \\$ for Streamlit markdown")
    print("Users will see dollar signs correctly rendered in the chat.")
