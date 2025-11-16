#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析處理進度
"""
import re

log_file = 'process_books_log_20251108_201526.txt'

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# 提取處理的檔案
processed_files = re.findall(r'處理檔案: (cluster_\d+\.csv|cluster_noise\.csv)', content)
print(f"實際處理的檔案數: {len(processed_files)}")

# 提取 API 呼叫次數
api_calls = re.findall(r'API 呼叫 #(\d+):', content)
if api_calls:
    print(f"API 呼叫次數: {api_calls[-1]}")

# 提取合併次數
merge_count = re.findall(r'合併 #(\d+):', content)
if merge_count:
    print(f"實際合併次數: {merge_count[-1]}")

# 提取已寫入的資料
written = re.findall(r'已寫入 (\d+) 筆資料', content)
if written:
    total_written = sum(int(x) for x in written)
    print(f"總共寫入資料筆數: {total_written}")

# 提取 Rate Limit 錯誤
rate_limits = re.findall(r'Rate limit reached', content)
print(f"\nRate Limit 錯誤次數: {len(rate_limits)}")

if rate_limits:
    print("\n⚠️ 已達到 OpenAI API 每日限額 (10000 requests/day)")
    print("   程式被迫中斷，無法完成所有群的處理")

