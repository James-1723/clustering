import pandas as pd
import os
import glob
from openai import OpenAI
from tqdm import tqdm
import json
from datetime import datetime
import logging
import time

# ==================== 設定 ====================
OPENAI_API_KEY = "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"  # 請替換成你的 API Key
client = OpenAI(api_key=OPENAI_API_KEY)

CLUSTERED_DATA_DIR = "clustered_data"
OUTPUT_FILE = "output.csv"

# Log 檔案設定（使用時間戳記）
LOG_FILE = f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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

# ==================== 函數定義 ====================

def check_same_book_with_llm(title1, title2):
    """
    使用 OpenAI LLM 判斷兩本書是否相同
    
    回傳: True (相同) 或 False (不同)
    """
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
            model="gpt-4o-mini",  # 使用較便宜的模型
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
        
        # 記錄 API 呼叫結果
        logging.info(f"API 呼叫 #{api_call_count}: 比較 '{title1[:50]}...' vs '{title2[:50]}...' → {result}")
        
        return result
        
    except Exception as e:
        error_msg = f"  ⚠️ LLM 判斷錯誤: {e}"
        log_and_print(error_msg, 'error')
        return False


def merge_two_books(book1, book2):
    """
    合併兩本書的資料
    book1: 被合併者（保留大部分資料）
    book2: 合併者（提供部分資料）
    
    回傳: 合併後的資料（dict）
    """
    global merge_count
    merge_count += 1
    
    # 記錄合併資訊
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
    
    # isbn 系列：特殊處理（空白要標註）
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
        # 優先使用 book1（被合併者）的資料，如果沒有才用 book2
        if pd.notna(book1.get(col)) and str(book1.get(col)).strip():
            merged[col] = book1[col]
        elif pd.notna(book2.get(col)) and str(book2.get(col)).strip():
            merged[col] = book2[col]
        else:
            merged[col] = ''
    
    # 保留「被合併者」(book1) 的內容
    keep_from_book1 = [
        'title', '備註', 'processed_title', 'original_title',
        'author', 'translator', 'publisher'
    ]
    
    for col in keep_from_book1:
        merged[col] = book1.get(col, '')
    
    # Clean_publisher：勿動（保留 book1 的）
    merged['Clean_publisher'] = book1.get('Clean_publisher', '')
    
    # 未納入書目FIND：保留 book1 的
    merged['未納入書目FIND'] = book1.get('未納入書目FIND', '')
    
    # min_publish_date：最早日期
    dates = []
    for col in ['min_publish_date', 'bookscom_publish_date', 'kobo_publish_date', 'readmoo_publish_date']:
        for book in [book1, book2]:
            if pd.notna(book.get(col)) and str(book.get(col)).strip():
                try:
                    date_str = str(book[col]).strip()
                    # 嘗試解析日期
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
    else:
        merged['min_publish_date'] = book1.get('min_publish_date', '')
    
    # max_publish_date：最晚日期
    if dates:
        merged['max_publish_date'] = max(dates).strftime('%Y-%m-%d')
    else:
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
    
    # 記錄合併結果
    logging.info(f"  合併後 TAICCA_ID: {merged.get('NEW_TAICCA_ID', 'N/A')}")
    logging.info(f"  合併後 ISBN: {merged.get('isbn', 'N/A')}")
    
    return merged


def process_cluster_file(csv_file):
    """
    處理單個分群檔案
    
    回傳: 處理後的資料列表
    """
    filename = os.path.basename(csv_file)
    log_and_print(f"\n📂 處理檔案: {filename}")
    logging.info(f"開始處理: {csv_file}")
    
    df = pd.read_csv(csv_file)
    log_and_print(f"  - 讀取 {len(df)} 筆資料")
    
    if len(df) == 0:
        logging.warning(f"{filename} 沒有資料")
        return []
    
    # 轉換成字典列表方便處理
    books = df.to_dict('records')
    
    # 標記哪些書已經被合併
    merged_indices = set()
    result_books = []
    
    # 兩兩比較
    for i in tqdm(range(len(books)), desc="  比較書籍"):
        if i in merged_indices:
            continue  # 已經被合併過，跳過
        
        current_book = books[i]
        found_match = False
        
        # 與後面的書比較
        for j in range(i + 1, len(books)):
            if j in merged_indices:
                continue
            
            compare_book = books[j]
            
            # 使用 LLM 判斷是否為同一本書
            title1 = str(current_book.get('title', '')).strip()
            title2 = str(compare_book.get('title', '')).strip()
            
            if not title1 or not title2:
                continue
            
            is_same = check_same_book_with_llm(title1, title2)
            
            if is_same:
                log_and_print(f"    ✅ 找到相同書籍:")
                log_and_print(f"       [{i}] {title1}")
                log_and_print(f"       [{j}] {title2}")
                
                # 合併兩本書（compare_book 是被合併者，保留其資料）
                merged_book = merge_two_books(compare_book, current_book)
                result_books.append(merged_book)
                
                # 標記兩本都已處理
                merged_indices.add(i)
                merged_indices.add(j)
                found_match = True
                break
        
        # 如果沒有找到配對，保留原書
        if not found_match:
            result_books.append(current_book)
    
    log_and_print(f"  ✅ 處理完成: {len(result_books)} 筆資料")
    logging.info(f"{filename} 處理結果: {len(df)} 筆 → {len(result_books)} 筆")
    
    return result_books


# ==================== 主程式 ====================

def main():
    global api_call_count, merge_count
    
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
    title = "=" * 80 + "\n📚 書籍去重合併系統 (使用 OpenAI LLM) - 即時寫入模式\n" + "=" * 80
    log_and_print(title)
    logging.info(f"開始時間: {start_datetime}")
    logging.info(f"Log 檔案: {LOG_FILE}")
    logging.info(f"輸出檔案: {OUTPUT_FILE}")
    
    # 讀取所有分群檔案
    cluster_files = glob.glob(os.path.join(CLUSTERED_DATA_DIR, "cluster_*.csv"))
    
    # 排除 full_data_with_clusters.csv
    cluster_files = [f for f in cluster_files if 'full_data' not in f]
    
    log_and_print(f"\n找到 {len(cluster_files)} 個分群檔案:")
    for f in cluster_files:
        log_and_print(f"  - {os.path.basename(f)}")
    
    # 取得原始欄位順序
    if cluster_files:
        original_columns = pd.read_csv(cluster_files[0]).columns.tolist()
    else:
        log_and_print("\n⚠️ 沒有找到任何分群檔案", 'warning')
        return
    
    # 統計資訊
    total_original = 0
    total_output = 0
    
    # 處理每個分群檔案並即時寫入
    for idx, cluster_file in enumerate(cluster_files):
        # 記錄原始筆數
        original_count = len(pd.read_csv(cluster_file))
        total_original += original_count
        
        # 判斷是否為 noise 檔案
        is_noise_file = 'noise' in os.path.basename(cluster_file).lower()
        
        if is_noise_file:
            # cluster_noise.csv 直接讀取並寫入，不進行比較
            filename = os.path.basename(cluster_file)
            log_and_print(f"\n📂 處理檔案: {filename}")
            logging.info(f"開始處理噪音檔案: {cluster_file}")
            
            df = pd.read_csv(cluster_file)
            log_and_print(f"  - 讀取 {len(df)} 筆資料")
            log_and_print(f"  ⚡ 噪音檔案，直接寫入（跳過比較）")
            
            if len(df) > 0:
                # 確保欄位順序與原始檔案相同
                df = df[[col for col in original_columns if col in df.columns]]
                
                # 第一個檔案：創建新檔案並寫入 header
                # 後續檔案：追加模式，不寫入 header
                if idx == 0:
                    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='w')
                    log_and_print(f"  💾 已寫入 {len(df)} 筆資料到 {OUTPUT_FILE} (新建檔案)")
                else:
                    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='a', header=False)
                    log_and_print(f"  💾 已追加 {len(df)} 筆資料到 {OUTPUT_FILE}")
                
                logging.info(f"{filename}: 直接寫入 {len(df)} 筆資料（噪音檔案）")
                total_output += len(df)
            else:
                log_and_print(f"  ⚠️ 此群組沒有資料", 'warning')
        else:
            # 一般分群檔案：進行比較處理
            results = process_cluster_file(cluster_file)
            
            if results:
                # 轉換成 DataFrame
                result_df = pd.DataFrame(results)
                
                # 確保欄位順序與原始檔案相同
                result_df = result_df[[col for col in original_columns if col in result_df.columns]]
                
                # 第一個檔案：創建新檔案並寫入 header
                # 後續檔案：追加模式，不寫入 header
                if idx == 0:
                    result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='w')
                    log_and_print(f"  💾 已寫入 {len(result_df)} 筆資料到 {OUTPUT_FILE} (新建檔案)")
                else:
                    result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='a', header=False)
                    log_and_print(f"  💾 已追加 {len(result_df)} 筆資料到 {OUTPUT_FILE}")
                
                logging.info(f"已寫入 {len(result_df)} 筆資料到 {OUTPUT_FILE}")
                total_output += len(result_df)
            else:
                log_and_print(f"  ⚠️ 此群組沒有資料輸出", 'warning')
    
    # 計算執行時間
    end_time = time.time()
    end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    elapsed_time = end_time - start_time
    
    # 最終統計
    log_and_print(f"\n" + "=" * 80)
    log_and_print(f"📊 最終統計")
    log_and_print(f"=" * 80)
    log_and_print(f"  - 原始總筆數: {total_original}")
    log_and_print(f"  - 輸出資料筆數: {total_output}")
    log_and_print(f"  - 合併減少: {total_original - total_output} 筆")
    log_and_print(f"  - LLM API 呼叫次數: {api_call_count}")
    log_and_print(f"  - 實際合併次數: {merge_count}")
    log_and_print(f"  - 處理時間: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分鐘)")
    log_and_print(f"\n✅ 結果已儲存至: {OUTPUT_FILE}")
    log_and_print(f"📄 Log 檔案已儲存至: {LOG_FILE}")
    
    # 記錄結束時間到 log
    logging.info("=" * 80)
    logging.info(f"結束時間: {end_datetime}")
    logging.info(f"總執行時間: {elapsed_time:.2f} 秒")
    logging.info("=" * 80)
    
    log_and_print(f"\n" + "=" * 80)
    log_and_print("🎉 處理完成！")
    log_and_print("=" * 80)


if __name__ == "__main__":
    main()
