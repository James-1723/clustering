import pandas as pd
from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from functools import lru_cache

app = Flask(__name__)
app.json.sort_keys = False

DATA_DIR = 'input_data'
FILE_7000 = os.path.join(DATA_DIR, 'raw_data.csv')
FILE_INPUT = os.path.join(DATA_DIR, 'paper_source_1124.csv')

merge_count = 0

# 快取資料和檔案修改時間
_cache = {
    'data': None,
    'input_data': None,
    'mtime': None,
    'input_mtime': None
}

def get_cached_data(filepath, cache_key, mtime_key):
    """從快取讀取資料，如果檔案有修改則重新讀取"""
    try:
        current_mtime = os.path.getmtime(filepath)
        if _cache[mtime_key] is None or _cache[mtime_key] != current_mtime or _cache[cache_key] is None:
            _cache[cache_key] = pd.read_csv(filepath)
            _cache[mtime_key] = current_mtime
        return _cache[cache_key].copy()
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return pd.read_csv(filepath)

def invalidate_cache():
    """清除快取"""
    _cache['data'] = None
    _cache['mtime'] = None

@lru_cache(maxsize=1024)
def parse_date(date_str):
    """快取日期解析結果"""
    date_formats = ['%Y/%m/%d', '%Y-%m-%d', '%m/%d/%y', '%m/%d/%Y', '%d/%m/%y', '%d/%m/%Y']
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None

def merge_two_books(book1, book2):
    """合併兩本書的資料（內部使用，不增加計數器）"""
    merged = {}
    
    # TAICCA_ID 系列：以斜線分隔
    for col in ['TAICCA_ID']:
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
    for col in ['isbn']:
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
        'bookscom_isbn', 'eslite_isbn', 'kingstone_isbn', 'sanmin_isbn',
        'bookscom_production_id', 'eslite_production_id', 'kingstone_production_id', 'sanmin_production_id',
        'bookscom_title', 'eslite_title', 'kingstone_title', 'sanmin_title',
        'bookscom_processed_title', 'eslite_processed_title', 'kingstone_processed_title', 'sanmin_processed_title',
        'bookscom_original_title', 'eslite_original_title', 'kingstone_original_title', 'sanmin_original_title',
        'bookscom_author', 'eslite_author', 'kingstone_author', 'sanmin_author',
        'bookscom_translator', 'eslite_translator', 'kingstone_translator', 'sanmin_translator',
        'bookscom_publisher', 'eslite_publisher', 'kingstone_publisher', 'sanmin_publisher',
        'bookscom_publish_date', 'eslite_publish_date', 'kingstone_publish_date', 'sanmin_publish_date',
        'bookscom_original_price', 'eslite_original_price', 'kingstone_original_price', 'sanmin_original_price',
        'bookscom_category', 'eslite_category', 'kingstone_category', 'sanmin_category', 'sanmin_vice_category',
        'bookscom_url', 'eslite_url', 'kingstone_url', 'sanmin_url'
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
        'title', 'processed_title', 'original_title',
        'author', 'translator', 'publisher'
    ]
    
    for col in keep_from_book1:
        merged[col] = book1.get(col, '')
    
    # min_publish_date 和 max_publish_date：只從四間書商的日期中比較
    dates = []
    date_formats = ['%Y/%m/%d', '%Y-%m-%d', '%m/%d/%y', '%m/%d/%Y', '%d/%m/%y', '%d/%m/%Y']
    
    for col in ['bookscom_publish_date', 'eslite_publish_date', 'kingstone_publish_date', 'sanmin_publish_date']:
        for book in [book1, book2]:
            if pd.notna(book.get(col)) and str(book.get(col)).strip():
                date_str = str(book[col]).strip()
                date_obj = parse_date(date_str)
                if date_obj:
                    dates.append(date_obj)
    
    if dates:
        merged['min_publish_date'] = min(dates).strftime('%Y-%m-%d')
        merged['max_publish_date'] = max(dates).strftime('%Y-%m-%d')
    else:
        merged['min_publish_date'] = ''
        merged['max_publish_date'] = ''
    
    # price：最大值
    prices = []
    for col in ['price', 'bookscom_original_price', 'eslite_original_price', 'kingstone_original_price', 'sanmin_original_price']:
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
        merged['price'] = ''
    
    return merged

def merge_multiple_books(books):
    """合併多本書的資料"""
    global merge_count
    
    if len(books) == 0:
        return None
    if len(books) == 1:
        return books[0]
    
    merge_count += 1
    
    # 以第一本書為基礎，逐一合併其他書
    result = books[0]
    for i in range(1, len(books)):
        result = merge_two_books(result, books[i])
    
    return result

def get_sorted_columns(df_columns):
    """Return columns sorted according to the user-specified order"""
    target_order = [
        'index',  # index 放在最左邊
        'TAICCA_ID', 
        'bookscom_production_id', 'eslite_production_id', 'kingstone_production_id', 'sanmin_production_id',
        'isbn', 'bookscom_isbn', 'eslite_isbn', 'kingstone_isbn', 'sanmin_isbn',
        'title', 'bookscom_title', 'eslite_title', 'kingstone_title', 'sanmin_title',
        'processed_title', 'bookscom_processed_title', 'eslite_processed_title', 'kingstone_processed_title', 'sanmin_processed_title',
        'original_title', 'bookscom_original_title', 'eslite_original_title', 'kingstone_original_title', 'sanmin_original_title',
        'author', 'bookscom_author', 'eslite_author', 'kingstone_author', 'sanmin_author',
        'translator', 'bookscom_translator', 'eslite_translator', 'kingstone_translator', 'sanmin_translator',
        'publisher', 'bookscom_publisher', 'eslite_publisher', 'kingstone_publisher', 'sanmin_publisher',
        'min_publish_date', 'max_publish_date', 'bookscom_publish_date', 'eslite_publish_date', 'kingstone_publish_date', 'sanmin_publish_date',
        'price', 'bookscom_original_price', 'eslite_original_price', 'kingstone_original_price', 'sanmin_original_price',
        'bookscom_category', 'eslite_category', 'kingstone_category', 'sanmin_category', 'sanmin_vice_category',
        'bookscom_url', 'eslite_url', 'kingstone_url', 'sanmin_url'
    ]
    
    # Filter to include only columns present in the dataframe, but in target order
    sorted_cols = [c for c in target_order if c in df_columns]
    
    # Append any extra columns that might be in the df but not in the target list
    extra_cols = [c for c in df_columns if c not in target_order]
    
    return sorted_cols + extra_cols

def ensure_index_column(df):
    """確保 DataFrame 有 index 欄位，如果沒有則建立"""
    if 'index' not in df.columns:
        df.insert(0, 'index', range(1, len(df) + 1))
    return df

def reindex_dataframe(df):
    """重新編號 index 欄位"""
    df['index'] = range(1, len(df) + 1)
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    df = get_cached_data(FILE_7000, 'data', 'mtime')
    
    # 確保有 index 欄位
    df = ensure_index_column(df)
    
    # Replace NaN with empty string for JSON serialization
    df = df.fillna('')
    
    # Sort columns
    cols = get_sorted_columns(df.columns)
    df = df[cols]
    
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/merge', methods=['POST'])
def merge_data():
    ids_to_merge = request.json.get('ids', [])
    if not ids_to_merge or len(ids_to_merge) < 2:
        return jsonify({'error': 'Need at least 2 items to merge'}), 400
    
    df = get_cached_data(FILE_7000, 'data', 'mtime')
    
    # 確保有 index 欄位
    df = ensure_index_column(df)
    
    # Filter rows to merge
    rows_to_merge = df[df['TAICCA_ID'].isin(ids_to_merge)].to_dict('records')
    
    if len(rows_to_merge) < 2:
         return jsonify({'error': 'Could not find all items to merge'}), 400

    # 找到要合併的資料中，最小的 index 位置
    merge_indices = df[df['TAICCA_ID'].isin(ids_to_merge)].index
    insert_position = merge_indices.min()
    
    merged_row = merge_multiple_books(rows_to_merge)
    
    # Remove original rows
    df = df[~df['TAICCA_ID'].isin(ids_to_merge)]
    
    # Convert merged_row back to DataFrame
    merged_df = pd.DataFrame([merged_row])
    
    # Ensure columns match
    for col in df.columns:
        if col not in merged_df.columns:
            merged_df[col] = ''
    
    # 將合併後的資料插入到原位置
    # 分成前後兩部分，然後插入合併的資料
    df_before = df.iloc[:insert_position]
    df_after = df.iloc[insert_position:]
    df = pd.concat([df_before, merged_df, df_after], ignore_index=True)
    
    # 重新編號 index
    df = reindex_dataframe(df)
    
    # Sort columns before saving
    cols = get_sorted_columns(df.columns)
    df = df[cols]
    
    df.to_csv(FILE_7000, index=False)
    
    # 清除快取以確保下次讀取最新資料
    invalidate_cache()
    
    return jsonify({'success': True})

@app.route('/api/unmerge', methods=['POST'])
def unmerge_data():
    print(f"[DEBUG] Received request.json: {request.json}")
    target_id = request.json.get('id')
    print(f"[DEBUG] target_id: '{target_id}'")
    
    if not target_id:
        print(f"[DEBUG] Error: No ID provided")
        return jsonify({'error': 'No ID provided'}), 400
        
    # Check if it is a merged ID (contains /)
    if '/' not in target_id:
        print(f"[DEBUG] Error: Not a merged ID (no '/' found)")
        return jsonify({'error': 'Not a merged ID'}), 400
    
    original_ids = [x.strip() for x in target_id.split('/')]
    
    # Read input.csv to get original data (使用快取)
    input_df = get_cached_data(FILE_INPUT, 'input_data', 'input_mtime')
    
    # Search for the original IDs
    mask = input_df['TAICCA_ID'].isin(original_ids)
    original_rows = input_df[mask]
    
    if original_rows.empty:
        return jsonify({'error': 'Original data not found'}), 404
          
    # Update system_test.csv (使用快取)
    df = get_cached_data(FILE_7000, 'data', 'mtime')
    
    # 確保有 index 欄位
    df = ensure_index_column(df)
    
    # 找到被合併資料的位置
    merge_position = df[df['TAICCA_ID'] == target_id].index
    
    if len(merge_position) == 0:
        return jsonify({'error': 'Merged data not found'}), 404
    
    insert_position = merge_position[0]
    
    # 移除被合併的資料
    df = df[df['TAICCA_ID'] != target_id]
    
    # 將原始資料插入到原位置
    df_before = df.iloc[:insert_position]
    df_after = df.iloc[insert_position:]
    df = pd.concat([df_before, original_rows, df_after], ignore_index=True)
    
    # 重新編號 index
    df = reindex_dataframe(df)
    
    # Sort columns before saving
    cols = get_sorted_columns(df.columns)
    df = df[cols]
    
    df.to_csv(FILE_7000, index=False)
    
    # 清除快取以確保下次讀取最新資料
    invalidate_cache()
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
