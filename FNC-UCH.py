import json
import os
import random as rn
import time
import warnings
import kornia.augmentation as K
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from peft import get_peft_model, PromptTuningConfig, TaskType
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import nets
from NCE.FNC import FNC
from src.load_mat import CMDataset
from utils.EDA import EDA
from utils.config_v2 import args

matplotlib.use('TkAgg')
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda'
formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S")
seed = 2025
print(f'seed={seed}')
np.random.seed(seed)
rn.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.benchmark = True
device_ids = [0, 1]
teacher_device_id = [0, 1]
best_acc = 0
start_epoch = 0
args.ckpt_dir = os.path.join(args.root_dir, 'ckpt', args.pretrain_dir)
os.makedirs(args.ckpt_dir, exist_ok=True)

def main():
    print(f'args.data_name={args.data_name}')
    print(f'args.train_batch_size={args.train_batch_size}')
    print(f'args.category_split_ratio={args.category_split_ratio}')
    print(f'args.bit={args.bit}')
    print(f'args.warmup_count={args.warmup_count}')
    print(f'args.prompt_len={args.prompt_len}')

    print('===> Preparing data ..')
    dataset = CMDataset(args.data_name, batch_size=args.train_batch_size, category_split_ratio=args.category_split_ratio)
    task_sequence = [dataset.visible_set, dataset.invisible_set]
    task_loaders = [DataLoader(task, batch_size=args.train_batch_size, shuffle=False) for task in task_sequence]
    visible_retrieval_loader = DataLoader(dataset.visible_retrieval_set, batch_size=args.train_batch_size, shuffle=False)
    visible_query_loader = DataLoader(dataset.visible_query_set, batch_size=args.train_batch_size, shuffle=False)
    invisible_retrieval_loader = DataLoader(dataset.invisible_retrieval_set, batch_size=args.train_batch_size, shuffle=False)
    invisible_query_loader = DataLoader(dataset.invisible_query_set, batch_size=args.train_batch_size, shuffle=False)

    print('===> Building ImageNet and TextNet..')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # IMAGE
    class ImageModel(nn.Module):
        def __init__(self, bit):
            super(ImageModel, self).__init__()
            self.dinov2 = AutoModel.from_pretrained('facebook/dinov2-small', local_files_only=True).to(device)
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.fea_net = nets.ImageNet(y_dim=384, bit=args.bit, hiden_layer=2).cuda()
            self.prompt_len = args.prompt_len
            self.hidden_dim = self.dinov2.config.hidden_size
            self.prompt = nn.Parameter(torch.zeros(1, self.prompt_len, self.hidden_dim))

        def forward(self, x, task):
            if task == 0:
                B = x.size(0)
                patch_embeds = self.dinov2.embeddings(x)
                prompt = self.prompt.expand(B, -1, -1)
                tokens = torch.cat([prompt, patch_embeds], dim=1)

                for blk in self.dinov2.encoder.layer:
                    tokens = blk(tokens)[0]
                tokens = self.dinov2.layernorm(tokens)

                patch_only = tokens[:, self.prompt_len:, :]
                pooled = patch_only.mean(dim=1)

                out = self.fea_net(pooled)

            else:
                origin_out = self.dinov2(x).pooler_output


                self.prompt.requires_grad = False
                B = x.size(0)
                patch_embeds = self.dinov2.embeddings(x)
                prompt = self.prompt.expand(B, -1, -1)
                tokens = torch.cat([prompt, patch_embeds], dim=1)
                for blk in self.dinov2.encoder.layer:
                    tokens = blk(tokens)[0]
                tokens = self.dinov2.layernorm(tokens)
                patch_only = tokens[:, self.prompt_len:, :]
                prompt_out = patch_only.mean(dim=1)

                fusion_out = (origin_out + prompt_out) / 2.
                out = self.fea_net(fusion_out)
            return out


    class TextModel(nn.Module):
        def __init__(self, bit):
            super(TextModel, self).__init__()

            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5", local_files_only=True)
            self.base_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5", local_files_only=True).to(device)
            for p in self.base_model.parameters():
                p.requires_grad = False


            self.prompt_len = args.prompt_len
            config = PromptTuningConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                num_virtual_tokens=self.prompt_len,
                tokenizer_name_or_path="BAAI/bge-small-en-v1.5",
            )


            self.bge = get_peft_model(self.base_model, config).to(device)
            self.bge.print_trainable_parameters()


            self.text_fea_net = nets.TextNet(y_dim=384, bit=bit, hiden_layer=2).cuda()

        def forward(self, sentences, task):
            inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            if task == 0:

                outputs = self.bge(**inputs)
                prompt_out = outputs.last_hidden_state[:, 0, :]
                out = self.text_fea_net(prompt_out)
            else:

                self.bge.prompt_encoder.default.embedding.weight.requires_grad = False


                base_outputs = self.base_model(**inputs)
                origin_out = base_outputs.last_hidden_state[:, 0, :]


                outputs = self.bge(**inputs)
                prompt_out = outputs.last_hidden_state[:, 0, :]


                fusion_out = (origin_out + prompt_out) / 2.
                out = self.text_fea_net(fusion_out)

            return out

    image_model = ImageModel(bit=args.bit).cuda()
    text_model = TextModel(bit=args.bit).cuda()

    parameters = list(image_model.parameters()) + list(text_model.parameters())

    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=1e-06)

    image_augmentation = K.AugmentationSequential(
        K.RandomResizedCrop((224, 224), scale=(0.5, 0.9), ratio=(3 / 4, 4 / 3), p=1.0),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation90(times=(1, 3), p=0.5),
    )
    image_augmentation = image_augmentation.cuda()
    text_augmentation = EDA(alpha_sr=0.5).cuda()

    n_data = len(dataset)
    print(f'n_data={n_data}')

    contrast = FNC(bit=args.bit, n_data=n_data, TOP_FNPS=args.TOP_FNPS, threshold=args.threshold).cuda()

    def set_train():
        image_model.train()
        text_model.train()

    def set_eval():
        image_model.eval()
        text_model.eval()

    global patience_counter
    def early_stopping(current_accuracy, best_accuracy, patience=10):
        global patience_counter
        if current_accuracy > best_accuracy:
            patience_counter = 0
        else:
            patience_counter += 1
        print(f'patience_counter={patience_counter}')
        if patience_counter >= patience:
            print(f"Training stopped after {patience} epochs without improvement.")
            patience_counter = 0
            return True
        else:
            return False

    def train():
        set_train()
        base_max_avg = 0.
        base_max_t2i = 0.
        base_max_i2t = 0.
        increment_max_avg = 0.
        increment_max_t2i = 0.
        increment_max_i2t = 0.
        for task, tr_loader in enumerate(task_loaders):
            count = len(tr_loader)

            if task == 1 and (args.category_split_ratio == (24, 0) or args.category_split_ratio == (10, 0) or args.category_split_ratio == (255, 0) or args.category_split_ratio == (80, 0)):
                print("Base only! End.")
                pass
            else:
                if task == 1:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr * 0.1
                warmup_count = args.warmup_count
                for n in range(int(args.task_epochs)):
                    start_time = time.time()
                    train_loss = 0.
                    for batch_idx, (idx, images, texts, tagets) in enumerate(tr_loader):

                        optimizer.zero_grad()
                        back_loss = 0.

                        idx = [idx.cuda()]
                        images_A = images.to(device)
                        texts_A = list(texts)

                        # warmup stage
                        if warmup_count > 0:
                            image_features_A = image_model(images_A, task)
                            text_features_A = text_model(texts_A, task)
                            Lc_loss = contrast(i_A=image_features_A, i_B=None, t_A=text_features_A,
                                               t_B=None, batch_idx=torch.cat(idx),
                                               warmup_count=warmup_count)
                        # hash memory stage
                        else:

                            torch.backends.cudnn.benchmark = True
                            torch.cuda.empty_cache()

                            images_B = image_augmentation(images_A)
                            texts_B = text_augmentation(texts_A)

                            image_features_A = image_model(images_A, task)
                            image_features_B = image_model(images_B, task)
                            text_features_A = text_model(texts_A, task)
                            text_features_B = text_model(texts_B, task)

                            Lc_loss = contrast(i_A=image_features_A, i_B=image_features_B, t_A=text_features_A, t_B=text_features_B, batch_idx = torch.cat(idx), warmup_count=warmup_count)  # n-args.warmup_epoch
                        train_loss += Lc_loss.item()

                        back_loss = Lc_loss
                        back_loss.backward()
                        optimizer.step()

                        clip_grad_norm_(parameters, 1.)

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f'epoch={n}, Train time={elapsed_time:.2f}')
                    if warmup_count > 0:
                        warmup_count -= 1
                    # else:
                    if task == 0:
                        print('Base hash representation Avg Loss: %.3f | LR: %g' % (train_loss / count, optimizer.param_groups[0]['lr']))
                        start_time = time.time()
                        i2t, t2i, avg = eval(visible_retrieval_loader, visible_query_loader, task=task)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f'Eval time={elapsed_time:.2f}')
                        stop_training = early_stopping(avg, base_max_avg)
                        if stop_training:
                            break
                        print(f'Base mAP: i2t={i2t:.3f}   t2i={t2i:.3f}   avg={avg:.3f}')
                        if avg > base_max_avg:
                            base_max_avg = avg
                            base_max_i2t = i2t
                            base_max_t2i = t2i
                            print(f'Saving Base Max mAP : i2t={base_max_i2t:.3f}   t2i={base_max_t2i:.3f}   avg={base_max_avg:.3f}')
                            state = {
                                'image_model_state_dict': image_model.state_dict(),
                                'text_model_state_dict': text_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'Avg': base_max_avg,
                                'Img2Txt': base_max_i2t,
                                'Txt2Img': base_max_t2i,
                            }
                            torch.save(state,os.path.join(args.ckpt_dir, 'demo_%d_%d_%d_%s_base_best_checkpoint.t7' % (args.category_split_ratio[0], args.category_split_ratio[1], args.bit, args.data_name)))
                    else:
                        print('Increment hash representation Avg Loss: %.3f | LR: %g' % (train_loss / count, optimizer.param_groups[0]['lr']))
                        start_time = time.time()
                        i2t, t2i, avg = eval(invisible_retrieval_loader, invisible_query_loader,task=task)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f'Eval time={elapsed_time:.2f}')
                        stop_training = early_stopping(avg, increment_max_avg)
                        if stop_training:
                            break
                        print(f'Increment mAP: i2t={i2t:.3f}   t2i={t2i:.3f}   avg={avg:.3f}')
                        if avg > increment_max_avg:
                            increment_max_avg = avg
                            increment_max_i2t = i2t
                            increment_max_t2i = t2i
                            print(f'Saving Increment Max mAP : i2t={increment_max_i2t:.3f}   t2i={increment_max_t2i:.3f}   avg={increment_max_avg:.3f}')
                            state = {
                                'image_model_state_dict': image_model.state_dict(),
                                'text_model_state_dict': text_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'Avg': increment_max_avg,
                                'Img2Txt': increment_max_i2t,
                                'Txt2Img': increment_max_t2i,
                            }
                            torch.save(state,os.path.join(args.ckpt_dir, 'demo_%d_%d_%d_%s_increment_best_checkpoint.t7' % (args.category_split_ratio[0], args.category_split_ratio[1], args.bit,args.data_name)))

        print()
        print(f'Base Max mAP : i2t={base_max_i2t:.3f}   t2i={base_max_t2i:.3f}  avg={base_max_avg:.3f}')
        print(f'Increment Max mAP : i2t={increment_max_i2t:.3f}   t2i={increment_max_t2i:.3f}   avg={increment_max_avg:.3f}')
        if args.category_split_ratio == (24, 0) or args.category_split_ratio == (10, 0) or args.category_split_ratio == (255, 0) or args.category_split_ratio == (80, 0):
            ckpt = torch.load(os.path.join(args.ckpt_dir, 'demo_%d_%d_%d_%s_base_best_checkpoint.t7' % (args.category_split_ratio[0], args.category_split_ratio[1], args.bit, args.data_name)))
            image_model.load_state_dict(ckpt['image_model_state_dict'])
            text_model.load_state_dict(ckpt['text_model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            eval(visible_retrieval_loader, visible_query_loader,PR=True,task=0)
        else:
            ckpt = torch.load(os.path.join(args.ckpt_dir, 'demo_%d_%d_%d_%s_increment_best_checkpoint.t7' % (args.category_split_ratio[0], args.category_split_ratio[1], args.bit, args.data_name)))
            image_model.load_state_dict(ckpt['image_model_state_dict'])
            text_model.load_state_dict(ckpt['text_model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            increment_base_i2t, increment_base_t2i, increment_base_avg = eval(visible_retrieval_loader, visible_query_loader, task=1)
            print(f'Increment_Base Max mAP : i2t={increment_base_i2t:.3f}   t2i={increment_base_t2i:.3f}   avg={increment_base_avg:.3f}')
            print(f'forgetting={(base_max_avg-increment_base_avg)*100:.4f}%')

    def eval(retrieval_loader, query_loader, PR=False, task=0):
        set_eval()
        imgs, txts, labs = [], [], []
        imgs_te, txts_te, labs_te = [], [], []
        with torch.no_grad():
            for batch_idx, (idx, images, texts, targets) in enumerate(retrieval_loader):
                images = images.to(device)
                texts = list(texts)

                images_outputs = image_model(images, task)
                texts_outputs = text_model(texts, task)

                imgs.append(images_outputs)
                txts.append(texts_outputs)
                labs.append(targets)
            retrieval_imgs = torch.cat(imgs).sign_().cuda()
            retrieval_txts = torch.cat(txts).sign_().cuda()
            retrieval_labs = torch.cat(labs).cuda()

            for batch_idx, (idx, images, texts, targets) in enumerate(query_loader):


                images = images.to(device)
                texts = list(texts)

                images_outputs = image_model(images, task)
                texts_outputs = text_model(texts, task)

                imgs_te.append(images_outputs)
                txts_te.append(texts_outputs)
                labs_te.append(targets)
            query_imgs = torch.cat(imgs_te).sign_().cuda()
            query_txts = torch.cat(txts_te).sign_().cuda()
            query_labs = torch.cat(labs_te).cuda()

            i2t = calculate_top_map(query_imgs, retrieval_txts, query_labs, retrieval_labs, topk=0)
            t2i = calculate_top_map(query_txts, retrieval_imgs, query_labs, retrieval_labs, topk=0)

            avg = (i2t + t2i) / 2.
            if PR:
                i2t_P, i2t_R = calculate_pr_curve(query_imgs, retrieval_txts, query_labs, retrieval_labs)
                print(f'i2t_P={i2t_P}')
                print(f'i2t_R={i2t_R}')
                # plot_pr_curve(i2t_P,i2t_R)
                t2i_P, t2i_R = calculate_pr_curve(query_txts, retrieval_imgs, query_labs, retrieval_labs)
                print(f't2i_P={t2i_P}')
                print(f't2i_R={t2i_R}')
                # plot_pr_curve(t2i_P, t2i_R)

                timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
                PR_save_dir = f"logs/{args.data_name}/PR"
                PR_file_name = f"demo_pr_{args.data_name}_{args.bit}bit_{timestamp}.json"
                PR_file_path = os.path.join(PR_save_dir, PR_file_name)
                os.makedirs(PR_save_dir, exist_ok=True)
                with open(PR_file_path, "w") as f:
                    json.dump({"i2t_P": i2t_P.tolist(), "i2t_R": i2t_R.tolist(), "t2i_P": t2i_P.tolist(), "t2i_R": t2i_R.tolist()}, f)

        return i2t, t2i, avg

    train()


def calculate_hamming(B1, B2):
    leng = B2.size(1)
    dot = torch.matmul(B1.unsqueeze(0), B2.t()).squeeze(0)
    distH = 0.5 * (leng - dot)
    return distH
def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):

    qu_L = qu_L.float()
    re_L = re_L.float()

    if topk == 0:
        topk = re_L.size(0)
    num_query = qu_L.size(0)
    topkmap = torch.tensor(0.0, device=qu_B.device)

    for i in range(num_query):
        gnd = (torch.matmul(qu_L[i], re_L.t()) > 0).float()
        hamm = calculate_hamming(qu_B[i], re_B)
        _, ind = torch.sort(hamm, descending=False)
        gnd = gnd[ind]
        tgnd = gnd[:topk]
        tsum = tgnd.sum()
        if tsum.item() == 0:
            continue

        steps = int(tsum.item())
        count = torch.linspace(1, float(steps), steps, device=tgnd.device)

        tindex = torch.nonzero(tgnd, as_tuple=True)[0].float() + 1.0

        topkmap_ = torch.mean(count / tindex)
        topkmap += topkmap_

    topkmap = topkmap / num_query
    return topkmap
def calculate_pr_curve(qB, rB, query_label, retrieval_label):
    query_label = query_label.float()
    retrieval_label = retrieval_label.float()
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qB[i, :], rB)
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    return P, R
def plot_pr_curve(P, R, title="Precision-Recall Curve"):
    plt.figure(figsize=(10, 10))
    plt.plot(R, P, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    plt.title(title, fontsize=16)
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()