from datasets import load_dataset
import pandas as pd
import zipfile
import os
import tempfile
import shutil
from huggingface_hub import hf_hub_download

# 1. 下载 FinEval.zip
zip_path = hf_hub_download(
    repo_id="SUFE-AIFLM-Lab/FinEval",
    filename="FinEval.zip",
    repo_type="dataset",
)
print(f"Downloaded: {zip_path}")

# 2. 解压到临时目录并修复有问题的 CSV 文件
extract_dir = os.path.join(tempfile.gettempdir(), "fineval_fixed")
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_dir)

# 期望的列名
expected_cols = ["id", "question", "A", "B", "C", "D", "answer", "explanation"]

# 遍历所有 CSV 文件，修复列不一致的问题
csv_files = []
for root, dirs, files in os.walk(extract_dir):
    for f in files:
        if f.endswith(".csv"):
            csv_files.append(os.path.join(root, f))

fixed_count = 0
for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    needs_fix = False

    # 删除 Unnamed 列
    unnamed_cols = [c for c in cols if c.startswith("Unnamed")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
        needs_fix = True

    # 如果缺少 explanation 列，添加空列
    if "explanation" not in df.columns:
        df["explanation"] = ""
        needs_fix = True

    if needs_fix:
        df.to_csv(csv_path, index=False)
        print(f"Fixed: {csv_path}")
        fixed_count += 1

print(f"Total fixed: {fixed_count}/{len(csv_files)} files")

# 3. 从修复后的本地文件加载
dataset = load_dataset("csv", data_dir=extract_dir, name="default")
print(dataset)
