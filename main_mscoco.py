import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

command = [
    "python", "FNC-UCH.py",
    "--data_name", "mscoco",
    "--task_epochs", "50",
    "--train_batch_size", "64",
    "--category_split_ratio", "(80,0)",
    "--bit", "128",
    "--lr", "0.0001",
    "--warmup_count", "5",
    "--prompt_len", "5",
    "--TOP_FNPS", "100",
    "--threshold", "0.95",
]
subprocess.run(command)
