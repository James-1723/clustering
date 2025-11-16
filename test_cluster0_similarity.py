#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 cluster_0 中的標題相似度
"""
import re
from sentence_transformers import SentenceTransformer, util
try:
    import cn2an
except ImportError:
    cn2an = None

# 載入 BERT 模型
print("載入 BERT 模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ 模型載入完成\n")

def clean_title_for_bert(title):
    """清理標題用於 BERT 比較（移除電子書、限制級等標記）"""
    if not title or not str(title).strip():
        return ""
    
    title = str(title).strip()
    
    # 移除常見的標記和干擾字樣
    patterns_to_remove = [
        # 電子書相關
        r'\(電子書\)',
        r'（電子書）',
        r'\[電子書\]',
        r'【電子書】',
        r'電子書',
        r'\(ebook\)',
        r'（ebook）',
        r'ebook',
        r'e-book',
        # 限制級相關
        r'\(限\)',
        r'（限）',
        r'\[限\]',
        r'【限】',
        r'限$',  # 結尾的「限」
        r'限制級',
        r'18\+',
        r'18禁',
        # 其他常見干擾字樣
        r'\(完\)',
        r'（完）',
        r'\(新版\)',
        r'（新版）',
        r'\(修訂版\)',
        r'（修訂版）',
        r'\(全\)',
        r'（全）',
    ]
    
    for pattern in patterns_to_remove:
        title = re.sub(pattern, ' ', title, flags=re.IGNORECASE)
    
    # 清理多餘空格
    title = ' '.join(title.split())
    return title.strip()

def normalize_numbers_in_title(title):
    """將標題中的數字統一轉換為阿拉伯數字格式"""
    if not cn2an or not title:
        return title
    
    normalized = title
    
    try:
        # 1. 轉換全形數字為半形
        full_to_half = str.maketrans('０１２３４５６７８９', '0123456789')
        normalized = normalized.translate(full_to_half)
        
        # 2. 找出所有中文數字模式並轉換
        chinese_num_pattern = r'[一二三四五六七八九十百千萬零壹貳參肆伍陸柒捌玖拾佰仟]+'
        
        def replace_chinese_num(match):
            chinese_num = match.group(0)
            try:
                arabic_num = cn2an.cn2an(chinese_num, "smart")
                return str(arabic_num)
            except:
                return chinese_num
        
        normalized = re.sub(chinese_num_pattern, replace_chinese_num, normalized)
        
    except Exception as e:
        return title
    
    return normalized

def calculate_similarity(title1, title2):
    """計算兩個標題的相似度"""
    # 清理標題
    cleaned1 = clean_title_for_bert(title1)
    cleaned2 = clean_title_for_bert(title2)
    
    # 標準化數字
    normalized1 = normalize_numbers_in_title(cleaned1)
    normalized2 = normalize_numbers_in_title(cleaned2)
    
    # 計算相似度
    embedding1 = model.encode(normalized1, convert_to_tensor=True)
    embedding2 = model.encode(normalized2, convert_to_tensor=True)
    similarity = util.cos_sim(embedding1, embedding2).item()
    
    return similarity, cleaned1, cleaned2, normalized1, normalized2

# 測試案例
test_cases = [
    ("熟女記47", "熟女記47限"),
    ("熟女記48", "熟女記48限"),
    ("熟女記49", "熟女記49限"),
    ("熟女記50", "熟女記50限"),
    ("熟女記51", "熟女記51限"),
    ("熟女記52", "熟女記52限"),
    ("熟女記53", "熟女記53限電子書"),
    ("熟女記54", "熟女記54限電子書"),
]

print("=" * 80)
print("測試 cluster_0 中的標題相似度")
print("=" * 80)
print()

for title1, title2 in test_cases:
    similarity, cleaned1, cleaned2, normalized1, normalized2 = calculate_similarity(title1, title2)
    
    print(f"原始標題 1: {title1}")
    print(f"原始標題 2: {title2}")
    print(f"清理後 1:   {cleaned1}")
    print(f"清理後 2:   {cleaned2}")
    print(f"標準化 1:   {normalized1}")
    print(f"標準化 2:   {normalized2}")
    print(f"相似度:     {similarity:.6f}")
    
    if similarity >= 0.99:
        print(f"✅ 判斷為相同 (閾值: 0.99)")
    elif similarity >= 0.95:
        print(f"⚠️  接近相同 (閾值: 0.99)，建議降低閾值至 0.95")
    else:
        print(f"❌ 判斷為不同 (閾值: 0.99)")
    
    print("-" * 80)
    print()

print("\n建議:")
print("如果大部分相似度在 0.95-0.99 之間，建議將 SIMILARITY_THRESHOLD 降低至 0.95")
print("執行指令: python process_books.py input_data\\7000.csv --similarity 0.95 --skip-embedding")

