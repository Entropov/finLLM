# scripts/data_processing/check_tokenizer_coverage.py
from transformers import AutoTokenizer
import json, collections

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

with open("data/sft/fin_instruct_train.json") as f:
    data = json.load(f)

fragmented_terms = collections.Counter()

for item in data[:5000]:
    for conv in item.get("conversations", []):
        text = conv.get("value", "")
        tokens = tokenizer.tokenize(text)
        # 统计被拆成多 token 的词（相邻 token 拼合）
        i = 0
        while i < len(tokens) - 1:
            bigram = tokens[i] + tokens[i+1]
            if len(bigram) >= 4:
                fragmented_terms[bigram] += 1
            i += 1

print("被过度分词的高频金融词 Top30:")
for w, c in fragmented_terms.most_common(30):
    print(f"  {w}: {c}")
