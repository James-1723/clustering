#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 log 提取已處理的分群檔案並複製到新資料夾
"""
import os
import shutil
import re
from pathlib import Path

print("=" * 80)
print("提取已處理的分群檔案")
print("=" * 80)

# 設定
LOG_FILE = 'process_books_log_20251108_201526.txt'
SOURCE_DIR = 'clustered_data'
TARGET_DIR = 'clustered_data_llm_processed'  # 已用 LLM 處理過的
REMAINING_DIR = 'clustered_data_remaining'   # 剩餘未處理的

print(f"\nLog 檔案: {LOG_FILE}")
print(f"來源資料夾: {SOURCE_DIR}")
print(f"已處理資料夾: {TARGET_DIR}")
print(f"剩餘檔案資料夾: {REMAINING_DIR}")

# 步驟 1: 從 log 提取已處理的檔案
print("\n[1/4] 分析 log 檔案...")

processed_files = []

with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()
    
    # 提取所有「處理檔案: cluster_XXX.csv」的記錄
    matches = re.findall(r'處理檔案: (cluster_\d+\.csv|cluster_noise\.csv)', content)
    processed_files = list(dict.fromkeys(matches))  # 去重但保持順序

print(f">> 找到 {len(processed_files)} 個已處理的分群檔案")

# 顯示前 10 個和後 10 個
if len(processed_files) > 0:
    print(f"\n前 10 個已處理:")
    for i, f in enumerate(processed_files[:10], 1):
        print(f"  {i}. {f}")
    
    if len(processed_files) > 20:
        print(f"  ...")
        print(f"\n後 10 個已處理:")
        for i, f in enumerate(processed_files[-10:], len(processed_files)-9):
            print(f"  {i}. {f}")

# 步驟 2: 建立目標資料夾
print(f"\n[2/4] 建立目標資料夾...")
os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(REMAINING_DIR, exist_ok=True)
print(f">> 資料夾建立完成")

# 步驟 3: 取得所有分群檔案
print(f"\n[3/4] 掃描所有分群檔案...")

import glob
all_cluster_files = glob.glob(os.path.join(SOURCE_DIR, "cluster_*.csv"))
all_cluster_files = [f for f in all_cluster_files if 'full' not in f]

all_cluster_names = [os.path.basename(f) for f in all_cluster_files]
print(f">> 總共 {len(all_cluster_names)} 個分群檔案")

# 找出未處理的檔案
remaining_files = [f for f in all_cluster_names if f not in processed_files]
print(f">> 已處理: {len(processed_files)} 個")
print(f">> 未處理: {len(remaining_files)} 個")

# 步驟 4: 複製檔案
print(f"\n[4/4] 複製檔案...")

# 複製已處理的檔案
print(f"\n複製已處理的檔案到 {TARGET_DIR}/...")
copied_processed = 0
for filename in processed_files:
    src = os.path.join(SOURCE_DIR, filename)
    dst = os.path.join(TARGET_DIR, filename)
    
    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied_processed += 1
        if copied_processed <= 5 or copied_processed % 100 == 0:
            print(f"  複製: {filename}")

print(f">> 已複製 {copied_processed} 個已處理的檔案")

# 複製未處理的檔案
print(f"\n複製未處理的檔案到 {REMAINING_DIR}/...")
copied_remaining = 0
for filename in remaining_files:
    src = os.path.join(SOURCE_DIR, filename)
    dst = os.path.join(REMAINING_DIR, filename)
    
    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied_remaining += 1
        if copied_remaining <= 5 or copied_remaining % 100 == 0:
            print(f"  複製: {filename}")

print(f">> 已複製 {copied_remaining} 個未處理的檔案")

# 統計資料
print("\n" + "=" * 80)
print("複製完成！統計資訊")
print("=" * 80)

# 計算資料筆數
print("\n資料筆數統計:")
import pandas as pd

processed_count = 0
for filename in processed_files:
    filepath = os.path.join(TARGET_DIR, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        processed_count += len(df)

remaining_count = 0
for filename in remaining_files:
    filepath = os.path.join(REMAINING_DIR, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        remaining_count += len(df)

print(f"  已處理分群: {len(processed_files)} 個檔案, {processed_count} 筆資料")
print(f"  未處理分群: {len(remaining_files)} 個檔案, {remaining_count} 筆資料")
print(f"  總計: {len(processed_files) + len(remaining_files)} 個檔案, {processed_count + remaining_count} 筆資料")

# 建立處理清單檔案
list_file = "processed_clusters_list.txt"
with open(list_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("已處理的分群檔案清單\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"總數: {len(processed_files)} 個\n")
    f.write(f"資料筆數: {processed_count} 筆\n\n")
    f.write("檔案清單:\n")
    for i, filename in enumerate(processed_files, 1):
        f.write(f"{i}. {filename}\n")

remaining_list_file = "remaining_clusters_list.txt"
with open(remaining_list_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("未處理的分群檔案清單\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"總數: {len(remaining_files)} 個\n")
    f.write(f"資料筆數: {remaining_count} 筆\n\n")
    f.write("檔案清單:\n")
    for i, filename in enumerate(remaining_files, 1):
        f.write(f"{i}. {filename}\n")

print(f"\n清單檔案:")
print(f"  - 已處理: {list_file}")
print(f"  - 未處理: {remaining_list_file}")

print("\n" + "=" * 80)
print("建議下一步:")
print("=" * 80)
print(f"1. 已處理的 {len(processed_files)} 個分群檔案已在: {TARGET_DIR}/")
print(f"   這些是用 LLM 處理過的，包含在 already_final_merged_output.csv 中")
print(f"\n2. 未處理的 {len(remaining_files)} 個分群檔案已在: {REMAINING_DIR}/")
print(f"   可以使用 BERT 方法處理這些檔案")
print(f"\n3. 執行 BERT 去重:")
print(f"   修改 bert_dedup_mini_test.py 的 CLUSTERED_DATA_DIR = '{REMAINING_DIR}'")
print(f"   然後處理剩餘的檔案")
print("=" * 80)

