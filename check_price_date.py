import pandas as pd

# 讀取 CSV 檔案
input_file = '補值_result.xlsx - 工作表2.csv'
output_file = '補值_result.xlsx - 工作表2_processed.csv'

# 讀取資料
df = pd.read_csv(input_file)

# 記錄處理的變更
changes_date = 0
changes_price = 0

# 處理每一筆資料
for idx, row in df.iterrows():
    # 1. 處理 min_publish_date 和 max_publish_date
    min_date = str(row['min_publish_date']).strip() if pd.notna(row['min_publish_date']) else ''
    max_date = str(row['max_publish_date']).strip() if pd.notna(row['max_publish_date']) else ''
    
    # 如果兩個日期都有資料且不同，則將 min_publish_date 的 2025 改成 2028
    if min_date and max_date and min_date != max_date:
        if '2025' in min_date:
            new_min_date = min_date.replace('2025', '2028')
            df.at[idx, 'min_publish_date'] = new_min_date
            changes_date += 1
            print(f"行 {idx + 2}: min_publish_date 從 '{min_date}' 改為 '{new_min_date}'")
    
    # 2. 處理 price 和 original_price 欄位
    price = str(row['price']).strip() if pd.notna(row['price']) else ''
    bookscom_price = str(row['bookscom_original_price']).strip() if pd.notna(row['bookscom_original_price']) else ''
    kobo_price = str(row['kobo_original_price']).strip() if pd.notna(row['kobo_original_price']) else ''
    readmoo_price = str(row['readmoo_original_price']).strip() if pd.notna(row['readmoo_original_price']) else ''
    
    # 收集有值的 original_price
    prices = []
    if bookscom_price and bookscom_price not in ['', 'nan', '0']:
        prices.append(bookscom_price)
    if kobo_price and kobo_price not in ['', 'nan', '0']:
        prices.append(kobo_price)
    if readmoo_price and readmoo_price not in ['', 'nan', '0']:
        prices.append(readmoo_price)
    
    # 如果有多個價格且不同，在 price 前加上 %
    if len(prices) > 1 and len(set(prices)) > 1:  # 有多個價格且值不同
        if price and not price.startswith('%'):
            new_price = '%' + price
            df.at[idx, 'price'] = new_price
            changes_price += 1
            print(f"行 {idx + 2}: price 從 '{price}' 改為 '{new_price}' (prices: {prices})")

# 儲存處理後的資料
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n處理完成！")
print(f"共處理 {len(df)} 筆資料")
print(f"min_publish_date 變更: {changes_date} 筆")
print(f"price 變更: {changes_price} 筆")
print(f"結果已儲存至: {output_file}")

