#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只重新執行分群（使用已存在的 embedding）
"""
import pandas as pd
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from datetime import datetime

print("=" * 80)
print("重新分群工具 - 使用已存在的 embedding")
print("=" * 80)

# 解析命令行參數
parser = argparse.ArgumentParser(description='使用已存在的 embedding 重新執行 DBSCAN 分群')
parser.add_argument('--eps', type=float, default=0.3, help='DBSCAN eps 參數 (預設: 0.3)')
parser.add_argument('--min-samples', type=int, default=2, help='DBSCAN min_samples 參數 (預設: 2)')
parser.add_argument('--output-dir', type=str, default='clustered_data_new', help='輸出資料夾 (預設: clustered_data_new)')

args = parser.parse_args()

EPS = args.eps
MIN_SAMPLES = args.min_samples
OUTPUT_DIR = args.output_dir

print(f"\n參數設定:")
print(f"  - EPS: {EPS}")
print(f"  - MIN_SAMPLES: {MIN_SAMPLES}")
print(f"  - 輸出資料夾: {OUTPUT_DIR}")

# 步驟 1: 讀取包含 embedding 的資料
print(f"\n[1/4] 讀取資料: data_with_embeddings.csv")
df = pd.read_csv('data_with_embeddings.csv', encoding='utf-8-sig')
print(f">> 讀取完成！總共 {len(df)} 筆資料")

# 步驟 2: 準備分群資料
print(f"\n[2/4] 準備分群資料...")
df_valid = df[df['embedding'].notna()].copy()
print(f">> 有效資料: {len(df_valid)} 筆")

# 解析 embedding（從字串轉為列表）
print(">> 解析 embedding 向量...")
embeddings_list = []
for idx, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="解析中"):
    try:
        emb = row['embedding']
        if isinstance(emb, str):
            emb = json.loads(emb)
        embeddings_list.append(emb)
    except Exception as e:
        print(f"警告: 第 {idx} 筆資料解析失敗: {e}")
        embeddings_list.append(None)

df_valid['embedding_parsed'] = embeddings_list
df_valid = df_valid[df_valid['embedding_parsed'].notna()]

embeddings_array = np.array(df_valid['embedding_parsed'].tolist())
print(f">> Embedding 矩陣形狀: {embeddings_array.shape}")

# 步驟 3: 執行 DBSCAN 分群
print(f"\n[3/4] 執行 DBSCAN 分群...")
print(f"  - eps (鄰域半徑): {EPS}")
print(f"  - min_samples (最小樣本數): {MIN_SAMPLES}")

dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='cosine')
cluster_labels = dbscan.fit_predict(embeddings_array)

df_valid['cluster'] = cluster_labels

# 統計分群結果
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"\n>> 分群完成！")
print(f"  - 識別出的群數: {n_clusters}")
print(f"  - 噪音點: {n_noise} 筆 ({(n_noise/len(df_valid)*100):.2f}%)")

# 顯示各群統計
if n_clusters > 0:
    print(f"\n>> 各群的資料筆數統計：")
    cluster_counts = df_valid['cluster'].value_counts().sort_values(ascending=False)
    
    # 顯示前 20 個最大的群
    print("   前 20 個最大的群:")
    for cluster_id in cluster_counts.head(20).index:
        if cluster_id == -1:
            continue
        count = cluster_counts[cluster_id]
        percentage = (count / len(df_valid)) * 100
        print(f"   群 {cluster_id:>4}: {count:>5} 筆 ({percentage:>5.2f}%)")
    
    if n_noise > 0:
        percentage = (n_noise / len(df_valid)) * 100
        print(f"   噪音: {n_noise:>5} 筆 ({percentage:>5.2f}%)")

# 步驟 4: 拆分並儲存 CSV
print(f"\n[4/4] 拆分並儲存 CSV 檔案...")
print(f"  - 輸出資料夾: {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 移除 embedding 欄位（太大了）
df_to_save = df_valid.drop(columns=['embedding', 'embedding_parsed'])
saved_files = []

for cluster_id in tqdm(sorted(df_to_save['cluster'].unique()), desc="儲存檔案"):
    cluster_data = df_to_save[df_to_save['cluster'] == cluster_id]
    cluster_data_original = cluster_data.drop(columns=['cluster'])
    
    if cluster_id == -1:
        output_file = os.path.join(OUTPUT_DIR, "cluster_noise.csv")
        label = "噪音點"
    else:
        output_file = os.path.join(OUTPUT_DIR, f"cluster_{cluster_id}.csv")
        label = f"群 {cluster_id}"
    
    cluster_data_original.to_csv(output_file, index=False, encoding='utf-8-sig')
    saved_files.append(output_file)

# 儲存完整資料
full_output_file = os.path.join(OUTPUT_DIR, "full_data_with_clusters.csv")
df_to_save.to_csv(full_output_file, index=False, encoding='utf-8-sig')

print(f"\n>> 完成！")
print(f"  - 生成 {len(saved_files)} 個分群檔案")
print(f"  - 完整資料（含分群標籤）: {full_output_file}")

# 生成統計報告
report_file = os.path.join(OUTPUT_DIR, "clustering_report.txt")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("分群統計報告\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"參數設定:\n")
    f.write(f"  - EPS: {EPS}\n")
    f.write(f"  - MIN_SAMPLES: {MIN_SAMPLES}\n\n")
    f.write(f"分群結果:\n")
    f.write(f"  - 總資料筆數: {len(df_valid)}\n")
    f.write(f"  - 識別出的群數: {n_clusters}\n")
    f.write(f"  - 噪音點: {n_noise} 筆 ({(n_noise/len(df_valid)*100):.2f}%)\n\n")
    f.write(f"前 50 個最大的群:\n")
    for i, (cluster_id, count) in enumerate(cluster_counts.head(50).items(), 1):
        if cluster_id == -1:
            f.write(f"  噪音點: {count} 筆 ({(count/len(df_valid)*100):.2f}%)\n")
        else:
            f.write(f"  群 {cluster_id:>4}: {count:>5} 筆 ({(count/len(df_valid)*100):.2f}%)\n")

print(f"  - 統計報告: {report_file}")

print("\n" + "=" * 80)
print("重新分群完成！")
print("=" * 80)
print(f"\n下一步:")
print(f"1. 檢查 {OUTPUT_DIR}/ 資料夾中的分群檔案")
print(f"2. 查看 {report_file} 了解詳細統計")
print(f"3. 如果滿意結果，可以將 {OUTPUT_DIR}/ 取代原本的 clustered_data/")
print(f"4. 然後執行 LLM 去重：修改 process_books.py 中的 CLUSTERED_DATA_DIR")

