import argparse
import logging
import os
import time

import wandb
import numpy as np
import torch

from hard_user_injector import HardUserInjector
from utils import set_logging, set_seed
from dataset import CrossDomain
from model import Model, Perceptor
from train import train


def debug_cold_item_counts(split_result, cold_item_id):
    """
    Debug: 計算冷門商品在 train/valid/test 出現次數
    """
    train_edges = split_result['target_train_edge_index']
    valid_edges = split_result['target_valid_edge_index']
    test_edges  = split_result['target_test_edge_index']

    train_count = (train_edges[1] == cold_item_id).sum().item()
    valid_count = (valid_edges[1] == cold_item_id).sum().item()
    test_count  = (test_edges[1] == cold_item_id).sum().item()

    print(f"[DEBUG] cold_item_id={cold_item_id}")
    print(f"   train 出現次數: {train_count}")
    print(f"   valid 出現次數: {valid_count}")
    print(f"   test  出現次數: {test_count}")

    return train_count, valid_count, test_count


def search(args):
    args.search = True

    wandb.init(project="BiGNAS", config=args)
    set_seed(args.seed)
    set_logging()

    logging.info(f"args: {args}")

    # === Load Data ===
    dataset = CrossDomain(
        root=args.root,
        categories=args.categories,
        target=args.target,
        use_source=args.use_source,
    )
    data = dataset[0]

    # === 基本資訊 ===
    args.num_users = data.num_users
    args.num_source_items = data.num_source_items
    args.num_target_items = data.num_target_items

    logging.info(f"data: {data}")

    # === Model Save Path ===
    DATE_FORMAT = "%Y-%m-%d_%H:%M:%S"
    args.model_path = os.path.join(
        args.model_dir,
        f'{time.strftime(DATE_FORMAT, time.localtime())}_{"_".join(args.categories)}.pt',
    )

    # === split_result ===
    split_result = {
        "source_train_edge_index": data.source_link,
        "target_train_edge_index": data.target_train_edge_index,
        "target_valid_edge_index": data.target_valid_edge_index,
        "target_test_edge_index":  data.target_test_edge_index,
    }

    # === Edge Export ===
    os.makedirs("logs/split_edges", exist_ok=True)

    def save_edge_index(name, edge_index):
        npy_path = f"logs/split_edges/{name}.npy"
        csv_path = f"logs/split_edges/{name}.csv"
        np.save(npy_path, edge_index.cpu().numpy())
        np.savetxt(csv_path, edge_index.cpu().numpy().T, fmt="%d", delimiter=",")
        logging.info(f"[Search] 已輸出 {name}: {edge_index.shape}")

    save_edge_index("source_train_edge_index", data.source_link)
    save_edge_index("target_train_edge_index", data.target_train_edge_index)
    save_edge_index("target_valid_edge_index", data.target_valid_edge_index)
    save_edge_index("target_test_edge_index",  data.target_test_edge_index)


    ###############################################################################
    # ====================== Hard User Injection (New Version) ===================
    ###############################################################################
    if args.use_hard_user_augment:
        logging.info("[HardUser] 使用新版 HardUserInjector（加 promoted、減 suppressed）...")

        injector = HardUserInjector(
            top_ratio=args.hard_top_ratio,
            log_dir="logs/hard_user"
        )

        # === Load SGL user embedding ===
        emb_path = os.path.join(args.sgl_dir_target, "user_embeddings_final.npy")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"[HardUser] 找不到 user embedding：{emb_path}")

        user_emb_target = torch.tensor(np.load(emb_path), dtype=torch.float)

        # === Run Injection ===
        summary = injector.run(
            split_result=split_result,
            user_emb_target=user_emb_target,
            num_users=args.num_users,
            num_source_items=args.num_source_items,
            num_target_items=args.num_target_items,
            cold_item_id=args.cold_item_id,
            add_promote_ratio=args.edge_ratio_target,       # 加冷門商品比例
            remove_suppress_ratio=args.remove_ratio,        # 減熱門商品比例
            popular_top_k=args.popular_top_k,               # 熱門池大小
        )

        logging.info(
            f"[HardUser] hard_users={len(summary['hard_users'])}, "
            f"promote_edges={summary['E_add_promote'].shape[1]}, "
            f"suppress_removed={summary['E_remove_suppress'].shape[1]}"
        )

        # === 替換 target_train_edge_index ===
        split_result["target_train_edge_index"] = summary["target_train_new"]

        # Debug 冷門商品出現次數
        debug_cold_item_counts(split_result, args.cold_item_id)


    ###############################################################################
    # ============================= Train Model ===================================
    ###############################################################################
    model = Model(args)
    perceptor = Perceptor(args)
    logging.info(f"model: {model}")

    train(model, perceptor, data, args, split_result)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Device
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--use-meta", action="store_true")
    parser.add_argument("--use-source", action="store_true")

    # Dataset
    parser.add_argument("--categories", type=str, nargs="+", default=["CD", "Kitchen"])
    parser.add_argument("--target", type=str, default="Kitchen")
    parser.add_argument("--root", type=str, default="data/")

    # Model
    parser.add_argument("--aggr", type=str, default="mean")
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--model-dir", type=str, default="./save/")

    # Supernet search space
    parser.add_argument("--space", type=str, nargs="+",
                        default=["gcn", "gatv2", "sage", "lightgcn", "linear"])
    parser.add_argument("--warm-up", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--T", type=int, default=1)
    parser.add_argument("--entropy", type=float, default=0.0)

    # Training
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eta-min", type=float, default=0.001)
    parser.add_argument("--T-max", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=15)

    # Meta-learning
    parser.add_argument("--meta-interval", type=int, default=50)
    parser.add_argument("--meta-num-layers", type=int, default=2)
    parser.add_argument("--meta-hidden-dim", type=int, default=32)
    parser.add_argument("--meta-batch-size", type=int, default=512)
    parser.add_argument("--conv-lr", type=float, default=1)
    parser.add_argument("--hpo-lr", type=float, default=0.01)
    parser.add_argument("--descent-step", type=int, default=10)
    parser.add_argument("--meta-op", type=str, default="gat")

    # Contrastive learning
    parser.add_argument("--ssl_aug_type", type=str, default='edge')
    parser.add_argument("--edge_drop_rate", type=float, default=0.2)
    parser.add_argument("--node_drop_rate", type=float, default=0.2)
    parser.add_argument("--ssl_reg", type=float, default=0.1)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--nce_temp", type=float, default=0.2)

    # Hard User Options
    parser.add_argument("--use-hard-user-augment", action="store_true",
                        help="啟用新版 Hard User 加邊/減邊")
    parser.add_argument("--hard-top-ratio", type=float, default=0.10)
    parser.add_argument("--cold-item-id", type=int, default=-1)

    # New Parameters
    parser.add_argument("--edge-ratio-target", type=float, default=1.0,
                        help="加 promoted item 的比例")
    parser.add_argument("--remove_ratio", type=float, default=0.1,
                        help="移除 popular item 的比例")
    parser.add_argument("--popular_top_k", type=int, default=50,
                        help="popular item pool 大小")

    # SGL embedding
    parser.add_argument("--sgl-dir-target", type=str,
        default="/mnt/sda1/sherry/BiGNAS/SGL-BiGNAS/BiGNAS-Attack/logs/sgl_emb/")

    args = parser.parse_args()
    search(args)
