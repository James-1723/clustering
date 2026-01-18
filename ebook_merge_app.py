import pandas as pd
from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from functools import lru_cache

app = Flask(__name__)
app.json.sort_keys = False

# DATA_DIR = 'output_data'
TARGET_FILE = "input_data/ebook_output.csv"
SOURCE_FILE = "input_data/ebook_test.csv"

merge_count = 0
HAS_ID = False

# 快取資料和檔案修改時間
_cache = {
    'data': None,
    'input_data': None,
    'mtime': None,
    'input_mtime': None
}

def consolidate_row(row):
    """從書商欄位中遞補大欄位（Consolidate）- 電子書版本"""
    # Helper to check if value is empty or 'nan'
    def is_empty(val):
        if pd.isna(val): return True
        s = str(val).strip().lower()
        return not s or s == 'nan'

    # ISBN
    if is_empty(row.get('isbn')):
        for c in ['bookscom_isbn', 'readmoo_isbn', 'kobo_isbn']:
            if not is_empty(row.get(c)):
                row['isbn'] = row[c]
                break

    # Title (Standard)
    if is_empty(row.get('title')):
        for c in ['bookscom_title', 'readmoo_title', 'kobo_title']:
            if not is_empty(row.get(c)):
                row['title'] = row[c]
                break
                
    # Processed Title
    if is_empty(row.get('processed_title')):
        for c in ['bookscom_processed_title', 'readmoo_processed_title', 'kobo_processed_title']:
            if not is_empty(row.get(c)):
                row['processed_title'] = row[c]
                break

    # Title (original)
    if is_empty(row.get('original_title')):
        for c in ['bookscom_original_title', 'readmoo_original_title', 'kobo_original_title']:
            if not is_empty(row.get(c)):
                row['original_title'] = row[c]
                break
                
    # Author
    if is_empty(row.get('author')):
        for c in ['bookscom_author', 'readmoo_author', 'kobo_author']:
            if not is_empty(row.get(c)):
                row['author'] = row[c]
                break
                
    # Translator
    if is_empty(row.get('translator')):
        for c in ['bookscom_translator', 'readmoo_translator', 'kobo_translator']:
            if not is_empty(row.get(c)):
                row['translator'] = row[c]
                break
                
    # Publisher
    if is_empty(row.get('publisher')):
        for c in ['bookscom_publisher', 'readmoo_publisher', 'kobo_publisher']:
            if not is_empty(row.get(c)):
                row['publisher'] = row[c]
                break
    return row

def get_cached_data(filepath, cache_key, mtime_key):
    """從快取讀取資料，如果檔案有修改則重新讀取"""
    try:
        current_mtime = os.path.getmtime(filepath)
        if _cache[mtime_key] is None or _cache[mtime_key] != current_mtime or _cache[cache_key] is None:
            # 加入 dtype=str 避免 DtypeWarning
            _cache[cache_key] = pd.read_csv(filepath, dtype=str)
            _cache[mtime_key] = current_mtime
        return _cache[cache_key].copy()
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return pd.read_csv(filepath, dtype=str)

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
    
    if HAS_ID:
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
    
    # isbn 系列：特殊處理 (去重 + 保留空白)
    for col in ['isbn']:
        val1 = str(book1.get(col, '')).strip() if pd.notna(book1.get(col)) else ''
        val2 = str(book2.get(col, '')).strip() if pd.notna(book2.get(col)) else ''
        
        # 輔助函數：解析字串為列表，並回傳是否包含「空」意涵
        def parse_isbns(val):
            items = []
            has_empty = False
            if not val:
                has_empty = True
            else:
                parts = [x.strip() for x in val.split('/')]
                for p in parts:
                    if not p or p.lower() == 'nan':
                        has_empty = True
                    elif p == '（空白）':
                        has_empty = True
                    else:
                        items.append(p)
            return items, has_empty

        items1, empty1 = parse_isbns(val1)
        items2, empty2 = parse_isbns(val2)
        
        # 合併所有有效 ISBN 並去重 (保持順序)
        all_items = []
        seen = set()
        for x in items1 + items2:
            if x not in seen:
                all_items.append(x)
                seen.add(x)
        
        # 如果任一邊有空，則結果要包含「（空白）」
        result_parts = all_items[:]
        if empty1 or empty2:
            result_parts.append('（空白）')
            
        if result_parts:
            merged[col] = " / ".join(result_parts)
        else:
            merged[col] = ""
    
    # 直接填補的欄位
    fill_cols = [
        'bookscom_isbn', 'readmoo_isbn', 'kobo_isbn',
        'bookscom_production_id', 'readmoo_production_id', 'kobo_production_id',
        'bookscom_title', 'readmoo_title', 'kobo_title',
        'bookscom_processed_title', 'readmoo_processed_title', 'kobo_processed_title',
        'bookscom_original_title', 'readmoo_original_title', 'kobo_original_title',
        'bookscom_author', 'readmoo_author', 'kobo_author',
        'bookscom_translator', 'readmoo_translator', 'kobo_translator',
        'bookscom_publisher', 'readmoo_publisher', 'kobo_publisher',
        'bookscom_publish_date', 'readmoo_publish_date', 'kobo_publish_date',
        'bookscom_original_price', 'readmoo_original_price', 'kobo_original_price',
        'bookscom_category', 'readmoo_category', 'kobo_category',
        'bookscom_url', 'readmoo_url', 'kobo_url'
    ]
    
    for col in fill_cols:
        if pd.notna(book1.get(col)) and str(book1.get(col)).strip():
            merged[col] = book1[col]
        elif pd.notna(book2.get(col)) and str(book2.get(col)).strip():
            merged[col] = book2[col]
        else:
            merged[col] = ''
    
    # 填補 title、processed_title 等欄位（優先使用 book1，如果為空則使用 book2）
    fill_main_cols = [
        'title', 'processed_title', 'original_title',
        'author', 'translator', 'publisher'
    ]
    
    for col in fill_main_cols:
        val1 = book1.get(col, '')
        val2 = book2.get(col, '')
        
        # 優先使用 book1，如果 book1 為空才使用 book2
        # 注意：這裡要排除 'nan' 字串
        is_val1_ok = pd.notna(val1) and str(val1).strip() and str(val1).strip().lower() != 'nan'
        is_val2_ok = pd.notna(val2) and str(val2).strip() and str(val2).strip().lower() != 'nan'
        
        if is_val1_ok:
            merged[col] = val1
        elif is_val2_ok:
            merged[col] = val2
        else:
            merged[col] = ''
    
    # min_publish_date 和 max_publish_date：只從三間書商的日期中比較
    dates = []
    
    for col in ['bookscom_publish_date', 'kobo_publish_date', 'readmoo_publish_date']:
        for book in [book1, book2]:
            val = book.get(col)
            if pd.notna(val) and str(val).strip() and str(val).strip().lower() != 'nan':
                try:
                    date_str = str(val).strip()
                    # 使用 pd.to_datetime 更強大的解析能力
                    date_obj = pd.to_datetime(date_str, errors='coerce')
                    if pd.notna(date_obj):
                        dates.append(date_obj)
                except:
                    pass
    
    if dates:
        merged['min_publish_date'] = min(dates).strftime('%Y-%m-%d')
        merged['max_publish_date'] = max(dates).strftime('%Y-%m-%d')
    else:
        merged['min_publish_date'] = ''
        merged['max_publish_date'] = ''
    
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
    if HAS_ID:
        target_order = [
            'index',  # index 放在最左邊
            'TAICCA_ID',
            'title',  # title 緊接在 TAICCA_ID 旁邊
            'bookscom_production_id', 'kobo_production_id', 'readmoo_production_id',
            'isbn', 'bookscom_isbn', 'kobo_isbn', 'readmoo_isbn',
            'bookscom_title', 'kobo_title', 'readmoo_title',
            'processed_title', 'bookscom_processed_title', 'kobo_processed_title', 'readmoo_processed_title',
            'original_title', 'bookscom_original_title', 'kobo_original_title', 'readmoo_original_title',
            'author', 'bookscom_author', 'kobo_author', 'readmoo_author',
            'translator', 'bookscom_translator', 'kobo_translator', 'readmoo_translator',
            'publisher', 'bookscom_publisher', 'kobo_publisher', 'readmoo_publisher',
            'min_publish_date', 'max_publish_date', 'bookscom_publish_date', 'kobo_publish_date', 'readmoo_publish_date',
            'price', 'bookscom_original_price', 'kobo_original_price', 'readmoo_original_price',
            'bookscom_category', 'kobo_category', 'readmoo_category',
            'bookscom_url', 'kobo_url', 'readmoo_url'
        ]
    else:
        target_order = [
            'index',  # index 放在最左邊
            'title',
            'bookscom_production_id', 'kobo_production_id', 'readmoo_production_id',
            'isbn', 'bookscom_isbn', 'kobo_isbn', 'readmoo_isbn',
            'bookscom_title', 'kobo_title', 'readmoo_title',
            'processed_title', 'bookscom_processed_title', 'kobo_processed_title', 'readmoo_processed_title',
            'original_title', 'bookscom_original_title', 'kobo_original_title', 'readmoo_original_title',
            'author', 'bookscom_author', 'kobo_author', 'readmoo_author',
            'translator', 'bookscom_translator', 'kobo_translator', 'readmoo_translator',
            'publisher', 'bookscom_publisher', 'kobo_publisher', 'readmoo_publisher',
            'min_publish_date', 'max_publish_date', 'bookscom_publish_date', 'kobo_publish_date', 'readmoo_publish_date',
            'price', 'bookscom_original_price', 'kobo_original_price', 'readmoo_original_price',
            'bookscom_category', 'kobo_category', 'readmoo_category',
            'bookscom_url', 'kobo_url', 'readmoo_url'
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
    df = get_cached_data(TARGET_FILE, 'data', 'mtime')
    
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
    
    df = get_cached_data(TARGET_FILE, 'data', 'mtime')
    
    # 確保有 index 欄位
    df = ensure_index_column(df)
    
    # Determine ID column based on input format
    # Since we use dtype=str, handle IDs as strings
    is_numeric_ids = True
    try:
        [int(x) for x in ids_to_merge]
        ids_to_merge = [str(x) for x in ids_to_merge]
    except:
        is_numeric_ids = False
    
    if is_numeric_ids:
        id_col = 'index'
    else:
        id_col = 'TAICCA_ID'
            
    # Filter rows to merge
    rows_to_merge = df[df[id_col].astype(str).isin(ids_to_merge)].to_dict('records')
    
    if len(rows_to_merge) < 2:
         return jsonify({'error': 'Could not find all items to merge'}), 400

    # Consolidate data for each row before merging
    rows_to_merge = [consolidate_row(row) for row in rows_to_merge]

    # 找到要合併的資料中，最小的 index 位置
    merge_indices = df[df[id_col].astype(str).isin(ids_to_merge)].index
    insert_position = merge_indices.min()
    
    merged_row = merge_multiple_books(rows_to_merge)
    
    # Remove original rows
    df = df[~df[id_col].astype(str).isin(ids_to_merge)]
    
    # Convert merged_row back to DataFrame
    merged_df = pd.DataFrame([merged_row])
    
    # Ensure columns match and maintain same order as df
    for col in df.columns:
        if col not in merged_df.columns:
            merged_df[col] = ''
    
    # 確保 merged_df 的欄位順序與 df 完全一致
    merged_df = merged_df[df.columns]
    
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
    
    df.to_csv(TARGET_FILE, index=False)
    
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
        
    # Read input.csv to get original data (使用快取)
    input_df = get_cached_data(SOURCE_FILE, 'input_data', 'input_mtime')

    # Update system_test.csv (使用快取)
    df = get_cached_data(TARGET_FILE, 'data', 'mtime')
    
    # 確保有 index 欄位
    df = ensure_index_column(df)
    
    merge_position = []
    original_rows = pd.DataFrame()

    if HAS_ID:
        # Check if it is a merged ID (contains /)
        if '/' not in str(target_id):
            print(f"[DEBUG] Error: Not a merged ID (no '/' found)")
            return jsonify({'error': 'Not a merged ID'}), 400
    
        original_ids = [x.strip() for x in str(target_id).split('/')]
        
        # Search for the original IDs
        mask = input_df['TAICCA_ID'].isin(original_ids)
        original_rows = input_df[mask]
        
        # 找到被合併資料的位置
        merge_position = df[df['TAICCA_ID'] == target_id].index
        
    else:
        # HAS_ID = False: target_id is likely the index
        try:
            target_str = str(target_id)
            merge_position = df[df['index'].astype(str) == target_str].index
            
            if len(merge_position) > 0:
                target_idx = target_id
                target_row = df.loc[merge_position[0]]
                # print(f"[DEBUG] Processing Unmerge for Index: {target_idx}")
                
                # Check for IDs in production_ids columns
                prod_cols = ['bookscom_production_id', 'kobo_production_id', 'readmoo_production_id']
                original_ids_map = {col: [] for col in prod_cols}
                has_valid_id = False
                
                for col in prod_cols:
                    val = str(target_row.get(col, '')).strip()
                    if val and val.lower() != 'nan':
                        p_ids = []
                        if '/' in val:
                            p_ids = [x.strip() for x in val.split('/')]
                        else:
                            p_ids = [val]
                        
                        # Clean IDs: remove .0 suffix if present
                        p_ids = [x[:-2] if x.endswith('.0') else x for x in p_ids]
                        
                        # Filter empty and placeholders
                        p_ids = [x for x in p_ids if x and x != '（空白）' and x.lower() != 'nan']
                        
                        if p_ids:
                            has_valid_id = True
                            original_ids_map[col].extend(p_ids)

                if has_valid_id:
                    # Find original rows by any of the production IDs
                    mask = pd.Series([False] * len(input_df))
                    
                    for col, ids in original_ids_map.items():
                        if ids:
                            # Also handle .0 in input_df for comparison just in case
                            col_str = input_df[col].astype(str).apply(lambda x: x[:-2] if x.endswith('.0') else x)
                            sub_mask = col_str.isin(ids)
                            mask = mask | sub_mask
                            
                    original_rows = input_df[mask].copy() # Add .copy()
                    
                    if original_rows.empty:
                            return jsonify({'error': 'No original data found for the extracted IDs'}), 404
                    
                    # ✅ 對還原後的原始資料執行 Consolidate
                    original_rows_dict = original_rows.to_dict('records')
                    consolidated_rows = [consolidate_row(row) for row in original_rows_dict]
                    original_rows = pd.DataFrame(consolidated_rows)
                    
                else:
                    return jsonify({'error': 'Selected row has no valid Production IDs to trace'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid ID format for index'}), 400

    if original_rows.empty:
        return jsonify({'error': 'Original data not found'}), 404
    
    if len(merge_position) == 0:
        return jsonify({'error': 'Merged data not found'}), 404
    
    insert_position = merge_position[0]
    
    # 移除被合併的資料
    df = df.drop(merge_position)
    
    # 確保 original_rows 有所有需要的欄位，並且順序與 df 一致
    for col in df.columns:
        if col not in original_rows.columns:
            original_rows[col] = ''
    
    # 確保 original_rows 的欄位順序與 df 完全一致
    original_rows = original_rows[df.columns]
    
    # 將原始資料插入到原位置
    df_before = df.iloc[:insert_position]
    df_after = df.iloc[insert_position:]
    df = pd.concat([df_before, original_rows, df_after], ignore_index=True)
    
    # 重新編號 index
    df = reindex_dataframe(df)
    
    # Sort columns before saving
    cols = get_sorted_columns(df.columns)
    df = df[cols]
    
    df.to_csv(TARGET_FILE, index=False)
    
    # 清除快取以確保下次讀取最新資料
    invalidate_cache()
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
