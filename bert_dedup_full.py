#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT 去重完整版
使用 BERT 計算書籍標題相似度進行去重
處理所有分群檔案
"""
import pandas as pd
import numpy as np
import os
import re
import time
import logging
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

print("=" * 80)
print("BERT 去重系統 - 完整版")
print("=" * 80)

# ==================== 設定 ====================
CLUSTERED_DATA_DIR = "clustered_data"
OUTPUT_FILE = "final_merged_bert_processed.csv"
SIMILARITY_THRESHOLD = 0.99  # 相似度閾值（0-1），超過此值視為同一本書
LOG_FILE = f"bert_processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# ==================== 日誌設定 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
    ]
)

def log_and_print(message):
    """同時輸出到終端和 log"""
    print(message)
    logging.info(message)

# ==================== 載入 BERT 模型 ====================
log_and_print("\n[1/5] 載入 BERT 模型...")
log_and_print("使用模型: paraphrase-multilingual-MiniLM-L12-v2 (支援中文)")

try:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    log_and_print(">> 模型載入完成！")
except Exception as e:
    log_and_print(f"❌ 模型載入失敗: {e}")
    log_and_print("請先執行: pip install sentence-transformers")
    exit(1)

# ==================== 清理標題函數 ====================
def clean_title(title):
    """
    清理書籍標題，移除「電子書」等字樣
    """
    if pd.isna(title) or not str(title).strip():
        return ""
    
    title = str(title).strip()
    
    # 移除常見的電子書標記
    patterns_to_remove = [
        r'\(電子書\)',
        r'（電子書）',
        r'\[電子書\]',
        r'【電子書】',
        r'電子書',
        r'\(ebook\)',
        r'（ebook）',
        r'ebook',
        r'e-book',
    ]
    
    for pattern in patterns_to_remove:
        title = re.sub(pattern, ' ', title, flags=re.IGNORECASE)
    
    # 清理多餘空格
    title = ' '.join(title.split())
    
    return title.strip()

# ==================== BERT 相似度比較 ====================
def check_same_book_with_bert(title1, title2, model, threshold=SIMILARITY_THRESHOLD):
    """使用 BERT 計算兩個標題的相似度"""
    if not title1 or not title2:
        return False, 0.0
    
    # 清理標題
    cleaned_title1 = clean_title(title1)
    cleaned_title2 = clean_title(title2)
    
    if not cleaned_title1 or not cleaned_title2:
        return False, 0.0
    
    # 計算 embeddings
    embedding1 = model.encode(cleaned_title1, convert_to_tensor=True)
    embedding2 = model.encode(cleaned_title2, convert_to_tensor=True)
    
    # 計算 cosine 相似度
    similarity = util.cos_sim(embedding1, embedding2).item()
    
    # 判斷是否為同一本書
    is_same = similarity >= threshold
    
    return is_same, similarity

# ==================== 合併書籍函數 ====================
def merge_books_properly(book1, book2):
    """
    正確合併兩本書的資料（從 process_books.py 複製的邏輯）
    book1: 被合併者
    book2: 合併者
    """
    merged = {}
    
    # TAICCA_ID 系列：以斜線分隔
    for col in ['NEW_TAICCA_ID', '1106亦式ID', 'TAICCA_ID']:
        val1 = str(book1.get(col, '')).strip()
        val2 = str(book2.get(col, '')).strip()
        if pd.notna(book1.get(col)) and pd.notna(book2.get(col)):
            if val1 and val2 and val1 != val2:
                merged[col] = f"{val1} / {val2}"
            elif val1:
                merged[col] = val1
            elif val2:
                merged[col] = val2
        elif pd.notna(book1.get(col)):
            merged[col] = val1
        elif pd.notna(book2.get(col)):
            merged[col] = val2
        else:
            merged[col] = ''
    
    # isbn 系列：特殊處理
    for col in ['isbn', 'eisbn']:
        val1 = str(book1.get(col, '')).strip() if pd.notna(book1.get(col)) else ''
        val2 = str(book2.get(col, '')).strip() if pd.notna(book2.get(col)) else ''
        
        if val1 and val2 and val1 != val2:
            merged[col] = f"{val1} / {val2}"
        elif val1 and not val2:
            merged[col] = f"{val1} / （空白）"
        elif not val1 and val2:
            merged[col] = f"（空白）/ {val2}"
        elif val1:
            merged[col] = val1
        else:
            merged[col] = ''
    
    # 直接填補的欄位
    fill_cols = [
        'bookscom_isbn', 'kobo_isbn', 'readmoo_isbn', 'bookscom_eisbn', 'kobo_eisbn', 'readmoo_eisbn',
        'production_id', 'bookscom_production_id', 'kobo_production_id', 'readmoo_production_id',
        'bookscom_title', 'kobo_title', 'readmoo_title',
        'bookscom_processed_title', 'kobo_processed_title', 'readmoo_processed_title',
        'bookscom_original_title', 'kobo_original_title', 'readmoo_original_title',
        'bookscom_author', 'kobo_author', 'readmoo_author',
        'bookscom_translator', 'kobo_translator', 'readmoo_translator',
        'bookscom_publisher', 'kobo_publisher', 'readmoo_publisher',
        'bookscom_publish_date', 'kobo_publish_date', 'readmoo_publish_date',
        'bookscom_original_price', 'kobo_original_price', 'readmoo_original_price',
        'bookscom_category', 'kobo_category', 'readmoo_category',
        'kobo_type_ebook', 'readmoo_type_ebook',
        'bookscom_url', 'kobo_url', 'readmoo_url'
    ]
    
    for col in fill_cols:
        if pd.notna(book1.get(col)) and str(book1.get(col)).strip():
            merged[col] = book1[col]
        elif pd.notna(book2.get(col)) and str(book2.get(col)).strip():
            merged[col] = book2[col]
        else:
            merged[col] = ''
    
    # 保留被合併者的內容
    keep_from_book1 = [
        'title', '備註', 'processed_title', 'original_title',
        'author', 'translator', 'publisher'
    ]
    
    for col in keep_from_book1:
        merged[col] = book1.get(col, '')
    
    merged['Clean_publisher'] = book1.get('Clean_publisher', '')
    merged['未納入書目FIND'] = book1.get('未納入書目FIND', '')
    
    # min_publish_date：最早日期
    dates = []
    for col in ['min_publish_date', 'bookscom_publish_date', 'kobo_publish_date', 'readmoo_publish_date']:
        for book in [book1, book2]:
            if pd.notna(book.get(col)) and str(book.get(col)).strip():
                try:
                    date_str = str(book[col]).strip()
                    if '/' in date_str:
                        date_obj = datetime.strptime(date_str, '%Y/%m/%d')
                    elif '-' in date_str:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    else:
                        continue
                    dates.append(date_obj)
                except:
                    pass
    
    if dates:
        merged['min_publish_date'] = min(dates).strftime('%Y-%m-%d')
        merged['max_publish_date'] = max(dates).strftime('%Y-%m-%d')
    else:
        merged['min_publish_date'] = book1.get('min_publish_date', '')
        merged['max_publish_date'] = book1.get('max_publish_date', '')
    
    # price：最大值
    prices = []
    for col in ['price', 'bookscom_original_price', 'kobo_original_price', 'readmoo_original_price']:
        for book in [book1, book2]:
            if pd.notna(book.get(col)):
                try:
                    price = float(book[col])
                    prices.append(price)
                except:
                    pass
    
    if prices:
        merged['price'] = max(prices)
    else:
        merged['price'] = book1.get('price', '')
    
    return merged

# ==================== 處理分群檔案 ====================
def process_cluster_file_bert(csv_file, model):
    """
    使用 BERT 處理單個分群檔案
    """
    filename = os.path.basename(csv_file)
    
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    if len(df) == 0:
        logging.warning(f"{filename}: 沒有資料")
        return [], 0, 0
    
    # 如果只有 1 筆資料，直接返回
    if len(df) <= 1:
        logging.info(f"{filename}: 只有 {len(df)} 筆資料，直接輸出")
        return df.to_dict('records'), 0, 0
    
    logging.info(f"開始處理: {filename}, {len(df)} 筆資料")
    
    books = df.to_dict('records')
    merged_indices = set()
    result_books = []
    comparisons = 0
    merges = 0
    
    for i in range(len(books)):
        if i in merged_indices:
            continue
        
        current_book = books[i]
        found_match = False
        
        for j in range(i + 1, len(books)):
            if j in merged_indices:
                continue
            
            compare_book = books[j]
            
            # 使用 processed_title 或 title
            title1 = str(current_book.get('processed_title', current_book.get('title', ''))).strip()
            title2 = str(compare_book.get('processed_title', compare_book.get('title', ''))).strip()
            
            if not title1 or not title2:
                continue
            
            comparisons += 1
            is_same, similarity = check_same_book_with_bert(title1, title2, model)
            
            if is_same:
                # 記錄找到的相同書籍
                logging.info(f"{filename}: 找到相同書籍 (相似度: {similarity:.6f})")
                logging.info(f"  [{i}] {clean_title(title1)}")
                logging.info(f"  [{j}] {clean_title(title2)}")
                logging.info(f"  ID: {current_book.get('NEW_TAICCA_ID', '')} + {compare_book.get('NEW_TAICCA_ID', '')}")
                
                # 使用正確的合併邏輯
                merged_book = merge_books_properly(compare_book, current_book)
                result_books.append(merged_book)
                
                merged_indices.add(i)
                merged_indices.add(j)
                merges += 1
                found_match = True
                break
        
        if not found_match:
            result_books.append(current_book)
    
    logging.info(f"{filename}: 完成 - {len(df)} 筆 → {len(result_books)} 筆 (比較 {comparisons} 次, 合併 {merges} 次)")
    
    return result_books, comparisons, merges

# ==================== 讀取所有分群檔案 ====================
log_and_print(f"\n[2/5] 讀取分群檔案...")

import glob
cluster_files = sorted(glob.glob(os.path.join(CLUSTERED_DATA_DIR, "cluster_*.csv")))
cluster_files = [f for f in cluster_files if 'full' not in f]

log_and_print(f">> 找到 {len(cluster_files)} 個分群檔案")
logging.info(f"找到 {len(cluster_files)} 個分群檔案")

if len(cluster_files) == 0:
    log_and_print("❌ 沒有找到分群檔案")
    exit(1)

# 取得原始欄位順序
original_columns = pd.read_csv(cluster_files[0], encoding='utf-8-sig').columns.tolist()

# ==================== 處理所有分群 ====================
log_and_print(f"\n[3/5] 開始處理所有分群...")
log_and_print(f"相似度閾值: {SIMILARITY_THRESHOLD}")
log_and_print(f"輸出檔案: {OUTPUT_FILE}")
log_and_print(f"Log 檔案: {LOG_FILE}")

start_time = time.time()
total_original = 0
total_output = 0
total_comparisons = 0
total_merges = 0

# 處理每個分群檔案並即時寫入
for idx, cluster_file in enumerate(tqdm(cluster_files, desc="處理分群")):
    original_count = len(pd.read_csv(cluster_file, encoding='utf-8-sig'))
    total_original += original_count
    
    # 噪音檔案直接寫入
    is_noise_file = 'noise' in os.path.basename(cluster_file).lower()
    
    if is_noise_file:
        filename = os.path.basename(cluster_file)
        logging.info(f"處理噪音檔案: {filename}")
        
        df = pd.read_csv(cluster_file, encoding='utf-8-sig')
        df = df[[col for col in original_columns if col in df.columns]]
        
        if idx == 0:
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='w')
        else:
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='a', header=False)
        
        total_output += len(df)
        logging.info(f"{filename}: 噪音檔案，直接寫入 {len(df)} 筆")
        continue
    
    # 一般分群檔案：使用 BERT 比較
    results, comparisons, merges = process_cluster_file_bert(cluster_file, model)
    
    if results:
        result_df = pd.DataFrame(results)
        result_df = result_df[[col for col in original_columns if col in result_df.columns]]
        
        if idx == 0 and total_output == 0:
            result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='w')
        else:
            result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='a', header=False)
        
        total_output += len(result_df)
        total_comparisons += comparisons
        total_merges += merges

elapsed_time = time.time() - start_time

# ==================== 統計報告 ====================
log_and_print("\n[4/5] 處理完成！")
log_and_print("=" * 80)
log_and_print("統計報告")
log_and_print("=" * 80)
log_and_print(f"處理分群數: {len(cluster_files)}")
log_and_print(f"原始總筆數: {total_original}")
log_and_print(f"輸出資料筆數: {total_output}")
log_and_print(f"合併減少: {total_original - total_output} 筆 ({(total_original - total_output) / total_original * 100:.2f}%)")
log_and_print(f"總比較次數: {total_comparisons:,}")
log_and_print(f"實際合併次數: {total_merges}")
log_and_print(f"相似度閾值: {SIMILARITY_THRESHOLD}")
log_and_print(f"處理時間: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分鐘 / {elapsed_time/3600:.2f} 小時)")

logging.info("=" * 80)
logging.info(f"處理完成 - 總時間: {elapsed_time:.2f} 秒")
logging.info(f"原始: {total_original} 筆 → 輸出: {total_output} 筆")
logging.info(f"減少: {total_original - total_output} 筆")
logging.info("=" * 80)

# ==================== 驗證輸出 ====================
log_and_print(f"\n[5/5] 驗證輸出檔案...")

try:
    output_df = pd.read_csv(OUTPUT_FILE, encoding='utf-8-sig')
    log_and_print(f">> 輸出檔案: {OUTPUT_FILE}")
    log_and_print(f">> 驗證筆數: {len(output_df)} 筆")
    
    if len(output_df) == total_output:
        log_and_print(">> ✅ 筆數驗證通過！")
    else:
        log_and_print(f">> ⚠️ 筆數不符: 預期 {total_output}, 實際 {len(output_df)}")
    
    logging.info(f"輸出檔案驗證: {len(output_df)} 筆")
    
except Exception as e:
    log_and_print(f">> ❌ 驗證失敗: {e}")
    logging.error(f"輸出檔案驗證失敗: {e}")

# ==================== 完成 ====================
log_and_print("\n" + "=" * 80)
log_and_print("🎉 BERT 去重處理完成！")
log_and_print("=" * 80)
log_and_print(f"輸出檔案: {OUTPUT_FILE}")
log_and_print(f"Log 檔案: {LOG_FILE}")
log_and_print("=" * 80)

