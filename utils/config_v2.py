import argparse

parser = argparse.ArgumentParser(description='implementation')

parser.add_argument("--data_name", type=str, default="mirflickr25k",help="data name")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--pretrain_dir', type=str, default='all_checkpoints')
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--task_epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--bit', type=int, default=128, help='output shape')
parser.add_argument('--category_split_ratio', type=str, default="(23, 1)")
parser.add_argument('--warmup_count', type=int, default=5)
parser.add_argument('--prompt_len', type=int, default=5)
parser.add_argument('--TOP_FNPS', type=int, default=50)
parser.add_argument('--threshold', type=float, default=0.85)
args = parser.parse_args()

try:
    args.category_split_ratio = eval(args.category_split_ratio)
    if not isinstance(args.category_split_ratio, tuple) or len(args.category_split_ratio) != 2:
        raise ValueError("category_split_ratio must be a tuple with two elements")
except:
    raise ValueError("Invalid format for category_split_ratio. Use '(A, B)' format.")


