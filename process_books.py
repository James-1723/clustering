#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
書籍處理完整流程
1. Embedding 分群
2. LLM 去重合併
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import time
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
from sklearn.cluster import DBSCAN

# ==================== 全域設定 ====================
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

# Embedding 設定
EMBEDDING_MODEL = "text-embedding-3-small"
EPS = 0.5           # DBSCAN 鄰域半徑
MIN_SAMPLES = 2     # DBSCAN 最小樣本數

# 輸出設定
CLUSTERED_DATA_DIR = "clustered_data"
FINAL_OUTPUT_FILE = "final_merged_output.csv"

# Log 檔案（使用時間戳記）
LOG_FILE = f"process_books_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# 全域計數器
api_call_count = 0
merge_count = 0

# ==================== 輔助函數 ====================

def log_and_print(message, level='info'):
    """同時輸出到終端和 log 檔案"""
    print(message)
    if level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'error':
        logging.error(message)

# ==================== 第一階段：Embedding 分群 ====================

def get_embedding(text, model=EMBEDDING_MODEL):
    """取得文字的 embedding 向量"""
    if pd.isna(text) or not str(text).strip():
        return None
    
    try:
        text = str(text).replace("\n", " ")
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Embedding 錯誤: {e}")
        return None

def stage1_embedding_clustering(input_file):
    """
    第一階段：讀取資料、計算 embedding、進行 DBSCAN 分群、拆分儲存
    """
    log_and_print("\n" + "=" * 80)
    log_and_print("📊 第一階段：Embedding 分群")
    log_and_print("=" * 80)
    
    # 步驟 1: 讀取資料
    log_and_print(f"\n📂 讀取資料: {input_file}")
    df = pd.read_csv(input_file)
    log_and_print(f"✅ 讀取完成！總共 {len(df)} 筆資料")
    logging.info(f"讀取檔案: {input_file}, 筆數: {len(df)}")
    
    # 步驟 2: 計算 Embedding
    log_and_print(f"\n🔄 開始計算 embeddings...")
    log_and_print(f"總共需要處理 {len(df)} 筆資料")
    
    embeddings = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="生成 Embeddings"):
        embedding = get_embedding(row['title'])
        embeddings.append(embedding)
    
    df['embedding'] = embeddings
    
    # 統計結果
    valid_embeddings = df['embedding'].notna().sum()
    invalid_embeddings = df['embedding'].isna().sum()
    
    log_and_print(f"\n✅ Embedding 生成完成！")
    log_and_print(f"  - 成功: {valid_embeddings} 筆")
    log_and_print(f"  - 失敗: {invalid_embeddings} 筆")
    logging.info(f"Embedding 統計: 成功 {valid_embeddings} 筆, 失敗 {invalid_embeddings} 筆")
    
    # 儲存包含 embedding 的資料
    embedding_file = 'data_with_embeddings.csv'
    df.to_csv(embedding_file, index=False, encoding='utf-8-sig')
    log_and_print(f"💾 已儲存包含 embeddings 的資料至: {embedding_file}")
    
    # 步驟 3: 準備分群資料
    log_and_print(f"\n📊 準備分群資料...")
    df_valid = df[df['embedding'].notna()].copy()
    log_and_print(f"  - 有效資料: {len(df_valid)} 筆")
    
    embeddings_array = np.array(df_valid['embedding'].tolist())
    log_and_print(f"  - Embedding 矩陣形狀: {embeddings_array.shape}")
    
    # 步驟 4: 執行 DBSCAN 分群
    log_and_print(f"\n🎯 執行 DBSCAN 分群...")
    log_and_print(f"  - eps (鄰域半徑): {EPS}")
    log_and_print(f"  - min_samples (最小樣本數): {MIN_SAMPLES}")
    logging.info(f"DBSCAN 參數: eps={EPS}, min_samples={MIN_SAMPLES}")
    
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='cosine')
    cluster_labels = dbscan.fit_predict(embeddings_array)
    
    df_valid['cluster'] = cluster_labels
    
    # 統計分群結果
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    log_and_print(f"\n✅ 分群完成！")
    log_and_print(f"  - 識別出的群數: {n_clusters}")
    log_and_print(f"  - 噪音點: {n_noise} 筆")
    logging.info(f"分群結果: {n_clusters} 個群, {n_noise} 個噪音點")
    
    # 顯示各群統計
    if n_clusters > 0:
        log_and_print(f"\n📊 各群的資料筆數：")
        for cluster_id in sorted(df_valid['cluster'].unique()):
            if cluster_id == -1:
                continue
            count = (df_valid['cluster'] == cluster_id).sum()
            percentage = (count / len(df_valid)) * 100
            log_and_print(f"  群 {cluster_id}: {count:>5} 筆 ({percentage:>5.1f}%)")
        
        if n_noise > 0:
            percentage = (n_noise / len(df_valid)) * 100
            log_and_print(f"  噪音: {n_noise:>5} 筆 ({percentage:>5.1f}%)")
    
    # 步驟 5: 拆分並儲存 CSV
    log_and_print(f"\n💾 開始拆分並儲存 CSV 檔案...")
    log_and_print(f"  - 輸出資料夾: {CLUSTERED_DATA_DIR}")
    
    os.makedirs(CLUSTERED_DATA_DIR, exist_ok=True)
    
    df_to_save = df_valid.drop(columns=['embedding'])
    saved_files = []
    
    for cluster_id in sorted(df_to_save['cluster'].unique()):
        cluster_data = df_to_save[df_to_save['cluster'] == cluster_id]
        cluster_data_original = cluster_data.drop(columns=['cluster'])
        
        if cluster_id == -1:
            output_file = os.path.join(CLUSTERED_DATA_DIR, "cluster_noise.csv")
            label = "噪音點"
        else:
            output_file = os.path.join(CLUSTERED_DATA_DIR, f"cluster_{cluster_id}.csv")
            label = f"群 {cluster_id}"
        
        cluster_data_original.to_csv(output_file, index=False, encoding='utf-8-sig')
        saved_files.append(output_file)
        log_and_print(f"  ✅ {label}: {len(cluster_data)} 筆 → {output_file}")
        logging.info(f"儲存分群檔案: {output_file}, 筆數: {len(cluster_data)}")
    
    # 儲存完整資料
    full_output_file = os.path.join(CLUSTERED_DATA_DIR, "full_data_with_clusters.csv")
    df_to_save.to_csv(full_output_file, index=False, encoding='utf-8-sig')
    log_and_print(f"\n  📊 完整資料（含分群標籤）: {full_output_file}")
    
    log_and_print("\n✅ 第一階段完成！分群檔案已儲存。")
    
    return {
        'cluster_files': saved_files,
        'total_records': len(df),
        'valid_records': len(df_valid),
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }

# ==================== 第二階段：LLM 去重合併 ====================

def check_same_book_with_llm(title1, title2):
    """使用 OpenAI LLM 判斷兩本書是否相同"""
    global api_call_count
    
    prompt = f"""請判斷以下兩本書的標題是否指向同一本書。
請只回答 "YES" 或 "NO"，不要有其他文字。

書籍1: {title1}
書籍2: {title2}

判斷標準：
- 標題完全相同或只有細微差異（如標點符號、空格）→ YES
- 同一系列但不同集數 → NO
- 完全不同的書 → NO

回答 (YES/NO):"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一個圖書館管理專家，專門判斷書籍是否相同。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        api_call_count += 1
        answer = response.choices[0].message.content.strip().upper()
        result = "YES" in answer
        
        logging.info(f"API 呼叫 #{api_call_count}: 比較 '{title1[:50]}...' vs '{title2[:50]}...' → {result}")
        
        return result
        
    except Exception as e:
        error_msg = f"  ⚠️ LLM 判斷錯誤: {e}"
        log_and_print(error_msg, 'error')
        return False

def merge_two_books(book1, book2):
    """合併兩本書的資料"""
    global merge_count
    merge_count += 1
    
    logging.info(f"合併 #{merge_count}:")
    logging.info(f"  被合併者 TAICCA_ID: {book1.get('NEW_TAICCA_ID', 'N/A')}, Title: {book1.get('title', 'N/A')[:50]}")
    logging.info(f"  合併者 TAICCA_ID: {book2.get('NEW_TAICCA_ID', 'N/A')}, Title: {book2.get('title', 'N/A')[:50]}")
    
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
    
    logging.info(f"  合併後 TAICCA_ID: {merged.get('NEW_TAICCA_ID', 'N/A')}")
    logging.info(f"  合併後 ISBN: {merged.get('isbn', 'N/A')}")
    
    return merged

def process_cluster_file(csv_file):
    """處理單個分群檔案"""
    filename = os.path.basename(csv_file)
    log_and_print(f"\n📂 處理檔案: {filename}")
    logging.info(f"開始處理: {csv_file}")
    
    df = pd.read_csv(csv_file)
    log_and_print(f"  - 讀取 {len(df)} 筆資料")
    
    if len(df) == 0:
        logging.warning(f"{filename} 沒有資料")
        return []
    
    books = df.to_dict('records')
    merged_indices = set()
    result_books = []
    
    for i in tqdm(range(len(books)), desc="  比較書籍"):
        if i in merged_indices:
            continue
        
        current_book = books[i]
        found_match = False
        
        for j in range(i + 1, len(books)):
            if j in merged_indices:
                continue
            
            compare_book = books[j]
            
            title1 = str(current_book.get('title', '')).strip()
            title2 = str(compare_book.get('title', '')).strip()
            
            if not title1 or not title2:
                continue
            
            is_same = check_same_book_with_llm(title1, title2)
            
            if is_same:
                log_and_print(f"    ✅ 找到相同書籍:")
                log_and_print(f"       [{i}] {title1}")
                log_and_print(f"       [{j}] {title2}")
                
                merged_book = merge_two_books(compare_book, current_book)
                result_books.append(merged_book)
                
                merged_indices.add(i)
                merged_indices.add(j)
                found_match = True
                break
        
        if not found_match:
            result_books.append(current_book)
    
    log_and_print(f"  ✅ 處理完成: {len(result_books)} 筆資料")
    logging.info(f"{filename} 處理結果: {len(df)} 筆 → {len(result_books)} 筆")
    
    return result_books

def stage2_llm_deduplication():
    """
    第二階段：讀取分群檔案、使用 LLM 判斷並合併
    """
    log_and_print("\n" + "=" * 80)
    log_and_print("🤖 第二階段：LLM 去重合併")
    log_and_print("=" * 80)
    
    # 讀取所有分群檔案
    cluster_files = glob.glob(os.path.join(CLUSTERED_DATA_DIR, "cluster_*.csv"))
    cluster_files = [f for f in cluster_files if 'full_data' not in f]
    
    log_and_print(f"\n找到 {len(cluster_files)} 個分群檔案:")
    for f in cluster_files:
        log_and_print(f"  - {os.path.basename(f)}")
    
    if not cluster_files:
        log_and_print("\n⚠️ 沒有找到任何分群檔案", 'warning')
        return None
    
    # 取得原始欄位順序
    original_columns = pd.read_csv(cluster_files[0]).columns.tolist()
    
    total_original = 0
    total_output = 0
    
    # 處理每個分群檔案並即時寫入
    for idx, cluster_file in enumerate(cluster_files):
        original_count = len(pd.read_csv(cluster_file))
        total_original += original_count
        
        is_noise_file = 'noise' in os.path.basename(cluster_file).lower()
        
        if is_noise_file:
            # 噪音檔案直接寫入
            filename = os.path.basename(cluster_file)
            log_and_print(f"\n📂 處理檔案: {filename}")
            logging.info(f"開始處理噪音檔案: {cluster_file}")
            
            df = pd.read_csv(cluster_file)
            log_and_print(f"  - 讀取 {len(df)} 筆資料")
            log_and_print(f"  ⚡ 噪音檔案，直接寫入（跳過比較）")
            
            if len(df) > 0:
                df = df[[col for col in original_columns if col in df.columns]]
                
                if idx == 0:
                    df.to_csv(FINAL_OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='w')
                    log_and_print(f"  💾 已寫入 {len(df)} 筆資料到 {FINAL_OUTPUT_FILE} (新建檔案)")
                else:
                    df.to_csv(FINAL_OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='a', header=False)
                    log_and_print(f"  💾 已追加 {len(df)} 筆資料到 {FINAL_OUTPUT_FILE}")
                
                logging.info(f"{filename}: 直接寫入 {len(df)} 筆資料（噪音檔案）")
                total_output += len(df)
        else:
            # 一般分群檔案：進行比較
            results = process_cluster_file(cluster_file)
            
            if results:
                result_df = pd.DataFrame(results)
                result_df = result_df[[col for col in original_columns if col in result_df.columns]]
                
                if idx == 0:
                    result_df.to_csv(FINAL_OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='w')
                    log_and_print(f"  💾 已寫入 {len(result_df)} 筆資料到 {FINAL_OUTPUT_FILE} (新建檔案)")
                else:
                    result_df.to_csv(FINAL_OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='a', header=False)
                    log_and_print(f"  💾 已追加 {len(result_df)} 筆資料到 {FINAL_OUTPUT_FILE}")
                
                logging.info(f"已寫入 {len(result_df)} 筆資料到 {FINAL_OUTPUT_FILE}")
                total_output += len(result_df)
    
    log_and_print("\n✅ 第二階段完成！去重合併已完成。")
    
    return {
        'total_original': total_original,
        'total_output': total_output,
        'api_calls': api_call_count,
        'merges': merge_count
    }

# ==================== 主程式 ====================

def main():
    global api_call_count, merge_count
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='書籍處理完整流程：Embedding 分群 + LLM 去重合併')
    parser.add_argument('input_file', type=str, help='輸入的 CSV 檔案路徑')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN eps 參數 (預設: 0.5)')
    parser.add_argument('--min-samples', type=int, default=2, help='DBSCAN min_samples 參數 (預設: 2)')
    parser.add_argument('--skip-embedding', action='store_true', help='跳過 embedding 階段，直接進行 LLM 去重')
    
    args = parser.parse_args()
    
    # 更新全域參數
    global EPS, MIN_SAMPLES
    EPS = args.eps
    MIN_SAMPLES = args.min_samples
    
    # 初始化 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
        ]
    )
    
    # 記錄開始時間
    start_time = time.time()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 輸出標題
    title = "=" * 80 + "\n📚 書籍處理完整流程系統\n" + "=" * 80
    log_and_print(title)
    logging.info(f"開始時間: {start_datetime}")
    logging.info(f"Log 檔案: {LOG_FILE}")
    logging.info(f"輸入檔案: {args.input_file}")
    logging.info(f"最終輸出檔案: {FINAL_OUTPUT_FILE}")
    logging.info(f"DBSCAN 參數: eps={EPS}, min_samples={MIN_SAMPLES}")
    
    log_and_print(f"\n📋 執行設定:")
    log_and_print(f"  - 輸入檔案: {args.input_file}")
    log_and_print(f"  - DBSCAN 參數: eps={EPS}, min_samples={MIN_SAMPLES}")
    log_and_print(f"  - 最終輸出: {FINAL_OUTPUT_FILE}")
    log_and_print(f"  - Log 檔案: {LOG_FILE}")
    
    # 檢查輸入檔案
    if not os.path.exists(args.input_file):
        log_and_print(f"\n❌ 錯誤: 找不到輸入檔案 '{args.input_file}'", 'error')
        return
    
    try:
        # 第一階段：Embedding 分群
        if not args.skip_embedding:
            stage1_result = stage1_embedding_clustering(args.input_file)
            log_and_print(f"\n📊 第一階段統計:")
            log_and_print(f"  - 總資料筆數: {stage1_result['total_records']}")
            log_and_print(f"  - 有效資料筆數: {stage1_result['valid_records']}")
            log_and_print(f"  - 識別出的群數: {stage1_result['n_clusters']}")
            log_and_print(f"  - 噪音點: {stage1_result['n_noise']}")
            log_and_print(f"  - 生成檔案數: {len(stage1_result['cluster_files'])}")
        else:
            log_and_print("\n⚠️ 跳過 embedding 階段，使用現有分群檔案")
        
        # 第二階段：LLM 去重合併
        stage2_result = stage2_llm_deduplication()
        
        if stage2_result:
            log_and_print(f"\n📊 第二階段統計:")
            log_and_print(f"  - 原始總筆數: {stage2_result['total_original']}")
            log_and_print(f"  - 輸出資料筆數: {stage2_result['total_output']}")
            log_and_print(f"  - 合併減少: {stage2_result['total_original'] - stage2_result['total_output']} 筆")
            log_and_print(f"  - LLM API 呼叫次數: {stage2_result['api_calls']}")
            log_and_print(f"  - 實際合併次數: {stage2_result['merges']}")
        
        # 計算總執行時間
        end_time = time.time()
        end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed_time = end_time - start_time
        
        # 最終總結
        log_and_print("\n" + "=" * 80)
        log_and_print("🎉 完整流程執行完畢！")
        log_and_print("=" * 80)
        log_and_print(f"  - 總處理時間: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分鐘)")
        log_and_print(f"  - 最終輸出檔案: {FINAL_OUTPUT_FILE}")
        log_and_print(f"  - Log 檔案: {LOG_FILE}")
        log_and_print(f"  - 分群檔案資料夾: {CLUSTERED_DATA_DIR}/")
        
        logging.info("=" * 80)
        logging.info(f"結束時間: {end_datetime}")
        logging.info(f"總執行時間: {elapsed_time:.2f} 秒")
        logging.info("=" * 80)
        
    except Exception as e:
        log_and_print(f"\n❌ 執行過程發生錯誤: {e}", 'error')
        logging.exception("執行錯誤")
        raise

if __name__ == "__main__":
    main()

