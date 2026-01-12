import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

command = [
    "python", "FNC-UCH.py",
    "--data_name", "nuswide",
    "--task_epochs", "50",
    "--train_batch_size", "64",
    "--category_split_ratio", "(10,0)",
    "--bit", "128",
    "--lr", "0.0001",
    "--warmup_count", "1",
    "--prompt_len", "3",
    "--TOP_FNPS", "100",
    "--threshold", "0.95",
]
subprocess.run(command)
