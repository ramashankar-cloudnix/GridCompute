import json
import re

html_path = "/mnt/c/Users/ramas/.gemini/antigravity-ide/brain/0fd2bdf9-da5b-44c3-81a4-3dafd9659941/.system_generated/steps/4/content.md"
output_path = "/mnt/d/wamp/www/REPO/GridCompute/share_text.md"

with open(html_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Let's extract script tags or anything that looks like conversation data
# In WIZ_global_data, the text is structured. We can also just extract all text between <p> and inside tags.
# But WIZ_global_data usually has lists of strings.
# Let's write a regex to find all text occurrences or extract lists.
# Let's extract strings in WIZ_global_data.
data_match = re.search(r'window\.WIZ_global_data\s*=\s*(\{.*?\});', content, re.DOTALL)
extracted_texts = []

if data_match:
    try:
        data_json = json.loads(data_match.group(1))
        # Let's search recursively for strings or look at specific keys like "TSDtV"
        def extract_strings(obj):
            if isinstance(obj, str):
                extracted_texts.append(obj)
            elif isinstance(obj, list):
                for item in obj:
                    extract_strings(item)
            elif isinstance(obj, dict):
                for val in obj.values():
                    extract_strings(val)
        extract_strings(data_json)
    except Exception as e:
        extracted_texts.append(f"JSON load failed: {str(e)}")

# Let's also extract plain paragraph contents or anything that looks like chat messages
body_text = re.findall(r'<div[^>]*class="[^"]*message-content[^"]*"[^>]*>(.*?)</div>', content, re.DOTALL)
if body_text:
    extracted_texts.extend(body_text)

# Clean up html tags
clean_texts = []
for text in extracted_texts:
    # remove html tags
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = clean.strip()
    if clean and len(clean) > 20: # ignore short strings
        clean_texts.append(clean)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write("# Extracted Share Content\n\n")
    for idx, text in enumerate(clean_texts):
        f.write(f"### Section {idx+1}\n{text}\n\n")

print("Done parsing!")
