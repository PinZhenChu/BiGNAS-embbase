# 🔧 Hard User 加邊 / 減邊方法

本方法只動 **target domain 的 user–item edge**，用來提升冷門商品（cold item）的排名，同時抑制熱門商品干擾。

## 1. 如何挑選 Hard Users？

1. 在 target domain 訓練集中找出「冷門商品」（cold_item_id）。
2. 根據是否購買過冷門商品，把使用者分成：

   * **GroupA**：有買過冷門商品
   * **GroupB**：沒買過冷門商品
3. 使用 SGL 的 target user embedding，計算 GroupB 使用者與 GroupA 的相似度。
4. 按照「距離（1 - cos sim）由大到小」排序，取前 `hard_top_ratio` 百分比作為 **Hard Users**。

Hard Users = 最不可能買冷門商品的一群人 → 需要人工干預。

## 2. 加邊（Promote 冷門商品）

對 Hard Users 人工加上冷門商品的邊：

```
(u, cold_item_id)
```

並可設定加邊比例：

```
edge_ratio_target  # 例如 0.2 表示只加 20% 的 Hard Users 的邊
```

## 3. 減邊（Suppress 熱門商品）

1. 在 target domain 中統計 item 出現頻率，挑出前 `popular_top_k` 個作為 **popular item pool**。
2. 找出 Hard Users 真的買過這些熱門 item 的邊。
3. 按 `remove_ratio` 刪除部分熱門邊，降低其影響。

## 4. 最後的邊集合

```
target_train_new = (原本的邊 - 減邊集合) + 加邊集合
```

只修改 target domain，不動 source domain。

