import os
import math
import logging
import numpy as np
import torch
import torch.nn.functional as F


def _tensor2set(edge_index: torch.Tensor):
    """
    將 edge_index (shape = [2, E]) 轉成 Python 的 set，方便做「加邊 / 減邊」的集合運算。

    參數：
        edge_index: torch.LongTensor，shape=[2, num_edges]
            第一列：user id
            第二列：item id
            這裡傳進來的是「global id」版本（即 user / item 都是在同一個整體編號空間）。

    回傳：
        一個 set，裡面每個元素都是 (u, i) 的 tuple，
        例如 {(0, 1001), (0, 1005), (1, 1010), ...}
    """
    # edge_index.t() 會變成 shape=[num_edges, 2]，每一列是一條邊 (u, i)
    # map(tuple, ...) 把每一個 tensor 變成 Python 的 tuple
    # set(...) 把這些 tuple 收進 set，方便後續做增刪
    return set(map(tuple, edge_index.t().tolist()))


def _apply_add(edge_index: torch.Tensor, additions):
    """
    將「要新增的邊」加進原来的 edge_index，並回傳新的 edge_index。

    這裡一律用「global id」，不做任何 offset 轉換。

    參數：
        edge_index: torch.LongTensor, shape=[2, E]
            原始的圖邊集合 (global id)
        additions: list[tuple] 或 可迭代物件
            每個元素是一個 (u, i)，表示一條 user–item 邊要被加入

    回傳：
        torch.LongTensor, shape=[2, E_new]
            加完邊後的完整邊集合（會自動去重，因為用 set 存）
    """
    # 先把原本的邊轉成 set，方便做 union
    s = _tensor2set(edge_index)

    # 對每一條新增邊 (u, i)，放進 set 裡
    for u, i in additions:
        # 保險起見，轉成 int，避免有 numpy.int64 之類的型別
        s.add((int(u), int(i)))

    # 最後把 set 轉回 tensor，shape=[2, E_new]
    out = torch.tensor(list(s), dtype=torch.long).t()
    return out


def _apply_remove(edge_index: torch.Tensor, removals):
    """
    將「要刪除的邊」從原来的 edge_index 移除，並回傳新的 edge_index。

    同樣使用 global id，不做 offset。

    參數：
        edge_index: torch.LongTensor, shape=[2, E]
            原始邊集合 (global id)
        removals: list[tuple]
            要刪掉的 (u, i) 邊集合

    回傳：
        torch.LongTensor, shape=[2, E_new]
            刪完邊後的完整邊集合
    """
    s = _tensor2set(edge_index)

    # 逐條檢查 removals，有的話就從 set 移除
    for u, i in removals:
        key = (int(u), int(i))
        if key in s:
            s.remove(key)

    # 同樣轉回 tensor 格式
    out = torch.tensor(list(s), dtype=torch.long).t()
    return out


class HardUserInjector:
    """
    🔥 HardUserInjector：只動「target domain 的 user–item 邊」。

    功能分成兩個部分：

    1. target domain 加 promoted item（冷門商品）
       - 對「挑出來的 Hard Users」加一條邊：user -> cold_item_id
       - 加邊的比例由 add_promote_ratio 控制（例如 0.2 只加 20% 的 Hard Users）

    2. target domain 減 suppressed popular items（熱門商品池）
       - 先從 target_train_edge_index 中統計最熱門的 target item（出現次數多）
       - 選出 popular_top_k 個，當作 popular item pool
       - 對「Hard Users 中原本就有購買 popular item 的那些邊」做減少
       - 減邊的比例由 remove_suppress_ratio 控制（例如 0.3 表示只刪 30% 的符合條件邊）

    ⚠️ 特別注意：
    - 這個版本完全不會動到 source domain 的邊（source_train_edge_index），只動 target_train_edge_index。
    """

    def __init__(self,
                 top_ratio=0.10,
                 log_dir="logs/hard_user"):
        """
        建構子

        參數：
            top_ratio: float
                用來決定從 GroupB（非 groupA 的 user）中挑出多少比例的「Hard Users」。
                例如 top_ratio=0.1 表示從 GroupB 裡挑出距離最難的前 10% 當作 Hard Users。

            log_dir: str
                用來存放 log / npy 檔的路徑。
        """
        self.top_ratio = top_ratio
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    # -------------------------
    #   基本工具
    # -------------------------
    @staticmethod
    def _split_users_by_target_item(target_train_local, cold_item_local, num_users):
        """
        根據「指定的冷門商品 (local id)」將 user 分成 GroupA / GroupB。

        這裡用的是「local item id」，也就是 target domain 的 item 編號 0 ~ num_target_items-1。

        定義：
            - GroupA: 有買過 cold_item_local 的 user
            - GroupB: 其他 user（沒買過該 cold item）

        參數：
            target_train_local: torch.LongTensor, shape=[2, E]
                target domain 的 train 邊（但 item 已轉成本地編號）
                第 0 列：user id（0 ~ num_users-1）
                第 1 列：item id（0 ~ num_target_items-1）
            cold_item_local: int
                冷門商品的 local item id
            num_users: int
                user 總數（假設 user id 範圍為 [0, num_users-1]）

        回傳：
            groupA: list[int]
                有買 cold item 的 user 列表
            groupB: list[int]
                其他 user 列表
        """
        # 找出所有 target_train_local 中 item == cold_item_local 的邊
        mask = (target_train_local[1] == cold_item_local)

        # 把這些邊的 user 抓出來並唯一化
        ua = target_train_local[0][mask].unique()
        groupA = set(ua.tolist())  # 有買冷門商品的 user 集合

        # 所有 user id = {0, 1, 2, ..., num_users-1}
        all_users = set(range(num_users))

        # 不在 groupA 的就是 groupB
        groupB = list(all_users - groupA)

        return list(groupA), groupB

    @staticmethod
    def _pick_hard_users(user_emb_target, groupA, groupB, top_ratio):
        """
        從 groupB 中選出「Hard Users」：對 groupA 最不相似的那群人。

        直觀：對「有買冷門商品的人（groupA）」距離越遠，越難被遷移到也買冷門商品，
             所以我們稱為 Hard Users。

        做法：
            1. 對 groupA / groupB 的 user embedding 做 L2 normalize
            2. 算 sim = uB @ uA^T，取每個 groupB user 對所有 groupA 的最大相似度 max_sim
            3. 定義 dist = 1 - max_sim，距離越大越難
            4. 按 dist 由大到小排序，取前 top_ratio 比例當作 Hard Users

        參數：
            user_emb_target: torch.FloatTensor, shape=[num_users, dim]
                target domain 的 user embedding（例如 SGL 訓練出來的）
            groupA: list[int]
                有買冷門商品的 user
            groupB: list[int]
                沒買冷門商品的 user
            top_ratio: float
                要從 groupB 中挑出多少比例當 Hard User

        回傳：
            hard_users: list[int]
                被選中的 Hard User id 列表
        """
        if len(groupA) == 0 or len(groupB) == 0:
            # 沒有任一群 → 直接回傳空
            return []

        # 轉成 tensor，方便 index 到 embedding
        A = torch.tensor(groupA, device=user_emb_target.device)
        B = torch.tensor(groupB, device=user_emb_target.device)

        # 取出 groupA / groupB 的 embedding，並做 L2 normalize
        uA = F.normalize(user_emb_target[A], dim=-1)  # shape=[|A|, dim]
        uB = F.normalize(user_emb_target[B], dim=-1)  # shape=[|B|, dim]

        # cos similarity: sim[b, a] = uB[b] · uA[a]
        sim = torch.matmul(uB, uA.t())  # shape=[|B|, |A|]

        # 對每個 groupB 的 user，取與 groupA 的最大相似度 max_sim
        max_sim, _ = sim.max(dim=1)     # shape=[|B|]

        # 定義距離：1 - cos sim，越大表示越不相似
        dist = 1 - max_sim

        # 根據 top_ratio 決定要選幾個 Hard Users
        k = max(1, math.ceil(len(groupB) * top_ratio))
        k = min(k, len(dist))  # 避免超過 dist 的長度

        # 按照 dist 由大到小挑出前 k 個 index
        top_idx = torch.topk(dist, k=k, largest=True).indices

        # 把這些 index 對應回原本的 user id
        return [int(B[i]) for i in top_idx]

    @staticmethod
    def _get_popular_items(target_train, num_users, num_source_items,
                           popular_top_k):
        """
        根據 target_train_edge_index 中的「出現頻率」，找出 target domain 的熱門商品。

        注意：
            - target_train 是 global id 版本：
                  user: 0 ~ num_users-1
                  source item: num_users ~ num_users+num_source_items-1
                  target item: num_users+num_source_items ~ ...
            - 我們只關心 target domain item（global >= num_users+num_source_items）

        步驟：
            1. 在 target_train 中統計每個 item 的出現次數
            2. 依照次數由大到小排序
            3. 過濾掉非 target item（只保留 global id >= num_users+num_source_items）
            4. 取前 popular_top_k 個

        參數：
            target_train: torch.LongTensor, shape=[2, E]
                target domain 的 train 邊（global id）
            num_users: int
                user 數量
            num_source_items: int
                source domain item 數量
            popular_top_k: int
                要挑出多少個最熱門的 target item

        回傳：
            popular_items: list[int]
                最熱門的 target item（global id），最多 popular_top_k 個
        """
        # 取出所有 item id（global）
        item_ids = target_train[1]

        # 算每個 item 的出現次數
        unique_items, counts = item_ids.unique(return_counts=True)

        # 依照 counts 從大到小排序
        order = torch.argsort(counts, descending=True)
        sorted_items = unique_items[order].tolist()

        # target item 在 global 編號上至少要 >= num_users + num_source_items，
        # 所以把小於這個門檻的都視為非 target item（例如 source item）而丟掉。
        target_min = num_users + num_source_items
        popular_items = [i for i in sorted_items if i >= target_min][:popular_top_k]

        return popular_items

    # -------------------------
    #   主流程：加 promoted + 減 suppressed
    # -------------------------
    def run(
        self,
        split_result,
        user_emb_target,
        num_users,
        num_source_items,
        num_target_items,
        cold_item_id,                  # 已是 global id（target domain 冷門商品）
        add_promote_ratio,         # 加 promoted item 的邊比例（0~1）
        remove_suppress_ratio,     # 減 popular item 的邊比例（0~1）
        popular_top_k,              # popular pool 的大小
    ):
        """
        主函式：執行 Hard User 加 / 減邊策略。

        參數：
            split_result: dict
                {
                    "source_train_edge_index": Tensor([2, E_s]),
                    "target_train_edge_index": Tensor([2, E_t]),
                    "target_valid_edge_index": ...,
                    "target_test_edge_index":  ...
                }
                這裡只會動到 target_train_edge_index，其他都不修改。

            user_emb_target: torch.FloatTensor, shape=[num_users, dim]
                target domain 的 user embedding（SGL 模型產生）

            num_users: int
            num_source_items: int
            num_target_items: int
                用來判斷 id 範圍與做 local/global 轉換

            cold_item_id: int
                target domain 的冷門商品「global id」

            add_promote_ratio: float
                Hard Users 中，要有多少比例被加上 promoted 冷門商品邊。
                例如：
                    - 1.0 → 所有 Hard Users 都加邊
                    - 0.3 → 約 30% 的 Hard Users 隨機加邊

            remove_suppress_ratio: float
                在「Hard Users × popular items 中原本存在的邊」裡面，
                要刪掉多少比例。
                例如：
                    - 1.0 → 所有符合條件的邊都刪
                    - 0.2 → 只刪 20% 左右

            popular_top_k: int
                用來建 popular item pool 的大小：
                - 先算 item 出現次數排序
                - 取前 popular_top_k 個 target item 當「suppressed pool」

        回傳：
            dict 包含：
                "hard_users": list[int]
                "E_add_promote": Tensor([2, #added])
                "E_remove_suppress": Tensor([2, #removed])
                "target_train_new": Tensor([2, E_new])   ← 加減後的 target_train_edge_index
        """
        logging.info("🔥 [HardUser] 新版執行：加 promoted item + 減 popular item ...")

        # 取得 target domain 的 train 邊（global id）
        target_train_edge_index = split_result["target_train_edge_index"].clone()

        # -------------------------
        # 1️⃣ Cold item：global → local
        # -------------------------
        cold_item_global = cold_item_id

        # local id = global id - (num_users + num_source_items)
        # 也就是在 target domain 裡面的 0 ~ num_target_items-1
        cold_item_local = cold_item_global - (num_users + num_source_items)
        assert 0 <= cold_item_local < num_target_items, \
            f"cold_item_local={cold_item_local} 超出 [0, {num_target_items-1}]"

        # 產生 local 版本：
        #   user: 保持 [0, num_users-1]
        #   item: 減去 offset，變成 [0, num_target_items-1]
        target_train_local = target_train_edge_index.clone()
        target_train_local[1] -= (num_users + num_source_items)

        # -------------------------
        # 2️⃣ 切出 GroupA / GroupB
        # -------------------------
        groupA, groupB = self._split_users_by_target_item(
            target_train_local,
            cold_item_local,
            num_users
        )
        logging.info(f"[HardUser] GroupA={len(groupA)} users (有買冷門), "
                     f"GroupB={len(groupB)} users (沒買冷門)")

        # -------------------------
        # 3️⃣ 從 GroupB 中挑 Hard Users
        # -------------------------
        hard_users = self._pick_hard_users(
            user_emb_target, groupA, groupB, self.top_ratio
        )
        logging.info(f"[HardUser] 挑到 {len(hard_users)} 位 Hard Users (top_ratio={self.top_ratio})")

        if len(hard_users) == 0:
            logging.warning("⚠ [HardUser] 沒有 Hard User → 不做加減邊，直接回傳原圖")
            return {
                "hard_users": [],
                "E_add_promote": torch.empty((2, 0), dtype=torch.long),
                "E_remove_suppress": torch.empty((2, 0), dtype=torch.long),
                "target_train_new": target_train_edge_index
            }

        # -------------------------
        # 4️⃣ 對 Hard Users 加 promoted 冷門商品邊
        # -------------------------
        # 每個 Hard User 對應一條 (user, cold_item_global) 的邊
        promote_edges = []
        for u in hard_users:
            promote_edges.append((u, cold_item_global))

        # 轉成 tensor：[2, num_hard_users]
        promote_edges = torch.tensor(promote_edges, dtype=torch.long).t()

        # 若 add_promote_ratio < 1.0，則隨機只保留一定比例的 Hard Users 來加邊
        # 若 add_promote_ratio == 0 → 不加邊
        if add_promote_ratio == 0:
            promote_edges = torch.empty((2, 0), dtype=torch.long)
        
        elif add_promote_ratio < 1.0:
            k = int(promote_edges.size(1) * add_promote_ratio)
            k = max(1, k)  # 只有 add_promote_ratio > 0 時才會保底 1
            idx = torch.randperm(promote_edges.size(1))[:k]
            promote_edges = promote_edges[:, idx]


        logging.info(f"[HardUser] 加 promoted item 的邊數：{promote_edges.size(1)}")
        print("\n[HardUser] === 加邊（promote cold item） ===")
        print(f"加邊總數：{promote_edges.size(1)}")
        for u, i in promote_edges.t().tolist():
            print(f"  + user {u} -> item {i}")


        # -------------------------
        # 5️⃣ 建 popular item pool（target domain 的熱門商品）
        # -------------------------
        popular_items = self._get_popular_items(
            target_train_edge_index,
            num_users,
            num_source_items,
            popular_top_k
        )

        # logging.info(f"[HardUser] popular pool (top {popular_top_k}) "
        #              f"示例前 {popular_top_k} 個：{popular_items[:popular_top_k]}")
                # -------------------------
        # ⭐ 印出 popular items 買過次數（所有 user 與 Hard Users 各多少）
        # -------------------------
        print("\n==================== Popular Item 統計 ====================")
        print(f"Top-{popular_top_k} popular items（global id）:")
        print(popular_items)

        # 統計所有 user 對 popular items 的購買次數
        all_item_ids = target_train_edge_index[1].tolist()
        all_user_ids = target_train_edge_index[0].tolist()

        popular_stats = {}  # {item: {"all_user": X, "hard_user": Y}}

        for item in popular_items:
            popular_stats[item] = {
                "all_user": 0,
                "hard_user": 0
            }

        # 建 hash set 加速查詢
        hard_user_set = set(hard_users)

        # 遍歷所有真實邊（target train）
        for u, i in zip(all_user_ids, all_item_ids):
            if i in popular_stats:
                popular_stats[i]["all_user"] += 1
                if u in hard_user_set:
                    popular_stats[i]["hard_user"] += 1

        # 加入累積欄位
        cumulative_all = 0
        cumulative_hard = 0

        print("\n📊 Popular item 出現統計（含累積）：")
        print("(item, 全體 user 次數, Hard Users 次數, 全體累積, Hard累積)")

        for item in popular_items:
            stats = popular_stats[item]
            
            cumulative_all += stats["all_user"]
            cumulative_hard += stats["hard_user"]

            print(f"Item {item}:  "
                f"all_user={stats['all_user']},  "
                f"hard_user={stats['hard_user']},  "
                f"cumulative_all={cumulative_all},  "
                f"cumulative_hard={cumulative_hard}")

        print("===========================================================\n")


        # -------------------------
        # 6️⃣ 找出「Hard Users × popular items」中原本存在的邊 → 候選刪除邊
        # -------------------------
        remove_edges = []
        # 先把原本的 target_train_edge_index 變成 set，方便 O(1) 查詢某條邊是否存在
        exist_set = _tensor2set(target_train_edge_index)

        # 對每一個 Hard User，檢查他有沒有跟 popular_items 形成真實邊
        for u in hard_users:
            for i in popular_items:
                if (u, i) in exist_set:
                    remove_edges.append((u, i))

        # 若沒有任何候選邊就給空 tensor；有的話轉成 [2, num_candidate_remove]
        if len(remove_edges):
            remove_edges = torch.tensor(remove_edges, dtype=torch.long).t()
        else:
            remove_edges = torch.empty((2, 0), dtype=torch.long)

        # 若 remove_suppress_ratio < 1.0，則只刪掉其中一部分
        if remove_edges.numel() > 0 and remove_suppress_ratio < 1.0:
            k = max(1, int(remove_edges.size(1) * remove_suppress_ratio))
            idx = torch.randperm(remove_edges.size(1))[:k]
            remove_edges = remove_edges[:, idx]

        logging.info(f"[HardUser] 最終要減掉的 popular item 邊數：{remove_edges.size(1)}")
        print("\n[HardUser] === 減邊（suppress popular item） ===")
        print(f"減邊總數：{remove_edges.size(1)}")
        for u, i in remove_edges.t().tolist():
            print(f"  - user {u} -> item {i}")

        # -------------------------
        # 7️⃣ 套用「先減邊，再加邊」到 target_train_edge_index
        # -------------------------
        new_edge = target_train_edge_index

        # ① 先減掉 suppressed popular item 的邊
        if remove_edges.numel() > 0:
            new_edge = _apply_remove(new_edge, remove_edges.t().tolist())

        # ② 再加上 promoted cold item 的邊
        if promote_edges.numel() > 0:
            new_edge = _apply_add(new_edge, promote_edges.t().tolist())

        logging.info(
            f"[HardUser] target_train_edge_index 原本有 {target_train_edge_index.size(1)} 條邊，"
            f"現在有 {new_edge.size(1)} 條邊"
        )

        # -------------------------
        # 8️⃣ 存成 npy，方便 debug & 分析
        # -------------------------
        np.save(os.path.join(self.log_dir, "E_add_promote.npy"), promote_edges.cpu().numpy())
        np.save(os.path.join(self.log_dir, "E_remove_suppress.npy"), remove_edges.cpu().numpy())
        np.save(os.path.join(self.log_dir, "target_train_new.npy"), new_edge.cpu().numpy())

        return {
            "hard_users": hard_users,
            "E_add_promote": promote_edges,
            "E_remove_suppress": remove_edges,
            "target_train_new": new_edge,
        }
