# ============================================================================
# CELL 1: Check GPU and Install Dependencies
# ============================================================================
# Check if GPU is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Install required packages with compatible versions
!pip install "numpy<2.0"
!pip install transformers
!pip install bert-score
!pip install scipy
!pip install pandas
!pip install POT
!pip install pyemd
!pip install tqdm
!pip install pytorch-pretrained-bert  # <-- ADD THIS LINE

print("\n⚠️  Runtime will restart now. After restart, run Cell 2 onwards.")
print("    (Skip re-running Cell 1)")

# Restart runtime
#import os
#os.kill(os.getpid(), 9)
# ============================================================================
# CELL 2: Download original repository
# ============================================================================
!git clone https://github.com/cyr19/Reproducibility.git
%cd Reproducibility/WMT18_BERT
# ============================================================================
# CELL 3: Fix directory structure and verify access
# ============================================================================
!bash download_wmt18.sh

# Fix the directory structure (the download script has permission issues)
!mkdir -p wmt18/wmt18
!mv wmt18-metrics-task-package wmt18/wmt18/ 2>/dev/null || echo "Already moved"
!mv wmt18/wmt18-metrics-task-nohybrids wmt18/wmt18/wmt18-metrics-task-package/input/ 2>/dev/null || echo "Already moved"

# Verify the data is there
!ls wmt18/wmt18/wmt18-metrics-task-package/manual-evaluation/RR-seglevel.csv

# ============================================================================
# CELL 4: Fix any encoding issues
# ============================================================================
import re

with open('reproduce_18.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add encoding='utf-8' to open() calls that don't have it
content = re.sub(
    r'with open\("wmt18/wmt18/wmt18-metrics-task-package/input/wmt18-metrics-task-nohybrids/system-outputs/newstest2018/\{\}/newstest2018\.\{\}\.\{\}"\.format\(lang_pair, system, lang_pair\)\) as f:',
    'with open("wmt18/wmt18/wmt18-metrics-task-package/input/wmt18-metrics-task-nohybrids/system-outputs/newstest2018/{}/newstest2018.{}.{}".format(lang_pair, system, lang_pair), encoding=\'utf-8\') as f:',
    content
)

content = re.sub(
    r'with open\("wmt18/wmt18/wmt18-metrics-task-package/input/wmt18-metrics-task-nohybrids/"\s*"references/\{\}"\.format\(\'newstest2018-\{\}\{\}-ref\.\{\}\'\.format\(src, tgt, tgt\)\)\s*, encoding=\'utf-8\'\) as f:',
    'with open("wmt18/wmt18/wmt18-metrics-task-package/input/wmt18-metrics-task-nohybrids/" "references/{}".format(\'newstest2018-{}{}-ref.{}\'.format(src, tgt, tgt)), encoding=\'utf-8\') as f:',
    content
)

with open('reproduce_18.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed encoding issues")

# ============================================================================
# CELL 5: Sample data; exclude this cell if you want to run on the whole dataset
# ============================================================================
with open('reproduce_18.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where to insert sampling code (before the return statement in get_wmt18_seg_data)
insert_line = None
for i, line in enumerate(lines):
    if 'return ref, cand_better, cand_worse' in line and 'get_wmt18_seg_data' in ''.join(lines[max(0,i-30):i]):
        insert_line = i
        break

if insert_line:
    sampling_code = """
    # Sample data for faster execution
    import random
    random.seed(42)
    sample_size = int(len(ref))  # Use 10% examples per language pair or 250
    if len(ref) > sample_size:
        indices = random.sample(range(len(ref)), sample_size)
        ref = [ref[i] for i in indices]
        cand_better = [cand_better[i] for i in indices]
        cand_worse = [cand_worse[i] for i in indices]

"""
    lines.insert(insert_line, sampling_code)

    with open('reproduce_18.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print("✓ Added data sampling")
else:
    print("⚠ Could not find insertion point for sampling")
# ============================================================================
# CELL 6: Set device
# ============================================================================
with open('reproduce_18.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Replace device='cpu' and device='cuda' with auto-detection
content = content.replace(
    "device='cpu'",
    f"device='{device}'"
)
content = content.replace(
    "device='cuda'",
    f"device='{device}'"
)

with open('reproduce_18.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✓ Set device to: {device}")
# ============================================================================
# CELL 7: BaryScore reproduction
# ============================================================================
print("\n" + "="*60)
print("Running BaryScore Evaluation")
print("="*60)
!python reproduce_18.py --metric baryscore
# ============================================================================
# CELL 8: MoverScore Reproduction
# ============================================================================
print("\n" + "="*60)
print("Running MoverScore Evaluation")
print("="*60)
!python reproduce_18.py --metric moverscore
# ============================================================================
# CELL 9: BERTScore Reproduction
# ============================================================================
print("\n" + "="*60)
print("Running BERTScore Evaluation")
print("="*60)
!python reproduce_18.py --metric bertscore










