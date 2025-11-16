#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析分群結果
"""
import pandas as pd
import numpy as np

# 讀取完整資料
df = pd.read_csv('clustered_data/full_data_with_clusters.csv', encoding='utf-8-sig')

# 輸出到檔案
with open('cluster_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("分群統計報告\n")
    f.write("=" * 80 + "\n")

    f.write(f"\n總資料筆數: {len(df)}\n")

    # 計算各群統計
    cluster_stats = df['cluster'].value_counts().sort_index()

    f.write(f"\n總群數: {len(cluster_stats)}\n")
    f.write(f"群 0 的筆數: {cluster_stats.get(0, 0)} (佔比: {cluster_stats.get(0, 0) / len(df) * 100:.2f}%)\n")
    f.write(f"噪音點 (-1) 的筆數: {cluster_stats.get(-1, 0)}\n")

    # 排除群 0 和噪音點的統計
    other_clusters = cluster_stats[(cluster_stats.index != 0) & (cluster_stats.index != -1)]
    if len(other_clusters) > 0:
        f.write(f"\n其他群的統計:\n")
        f.write(f"  平均筆數: {other_clusters.mean():.2f}\n")
        f.write(f"  中位數: {other_clusters.median():.2f}\n")
        f.write(f"  最小值: {other_clusters.min()}\n")
        f.write(f"  最大值: {other_clusters.max()}\n")

    # 顯示前20個最大的群
    f.write(f"\n前20個最大的群:\n")
    top_20 = cluster_stats.head(20)
    for idx in top_20.index:
        count = top_20[idx]
        pct = count / len(df) * 100
        f.write(f"  群 {idx:>3}: {count:>6} 筆 ({pct:>5.2f}%)\n")

    # 分析群 0 的內容
    f.write("\n" + "=" * 80 + "\n")
    f.write("群 0 內容分析\n")
    f.write("=" * 80 + "\n")

    cluster_0 = df[df['cluster'] == 0]

    # 取樣前50筆標題
    f.write("\n群 0 前50筆書籍標題:\n")
    for i, title in enumerate(cluster_0['title'].head(50), 1):
        f.write(f"  {i:>2}. {title}\n")

    # 分析出版社分布
    f.write("\n群 0 的出版社分布 (前20):\n")
    publisher_counts = cluster_0['Clean_publisher'].value_counts().head(20)
    for pub, count in publisher_counts.items():
        if pd.notna(pub):
            f.write(f"  {pub}: {count} 筆\n")

    # 分析類別分布
    f.write("\n群 0 的 Readmoo 類別分布 (前20):\n")
    category_counts = cluster_0['readmoo_category'].value_counts().head(20)
    for cat, count in category_counts.items():
        if pd.notna(cat):
            f.write(f"  {cat[:80]}: {count} 筆\n")

print("分析完成，結果已儲存到 cluster_analysis.txt")
