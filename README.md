# Book & Ebook Clustering and Merging System

這份專案旨在自動化整理與合併來自不同來源的書籍資料（包含紙本書與電子書）。流程分為「分群 (Clustering)」與「人工合併 (Merge App)」兩個階段。

## 專案結構

- **Clustering (分群)**
    - `book_clustering.ipynb`: 紙本書分群程式
    - `ebook_clustering.ipynb`: 電子書分群程式
- **Merge App (人工合併介面)**
    - `paper_merge_app.py`: 紙本書資料合併工具（Web 介面）
    - `ebook_merge_app.py`: 電子書資料合併工具（Web 介面）
    - `templates/index.html`: 合併工具的共用前端介面

## 使用流程

### 1. 資料分群 (Clustering)

首先執行 Jupyter Notebook 進行資料的前處理與自動分群。這些程式會讀取原始資料，計算文字 Embedding，並使用 DBSCAN 演算法將相似的書籍分到同一群（Cluster）。

*   **紙本書**：開啟並執行 `book_clustering.ipynb`
    *   **輸入**：`input_data/extracted_by_taicca_id.csv` (或其他指定的來源)
    *   **輸出**：生成分群後的結果 CSV 檔（如 `output_data/1128_output.csv` 或程式中指定的路徑）。

*   **電子書**：開啟並執行 `ebook_clustering.ipynb`
    *   **輸入**：`input_data/ebook_test.csv`
    *   **輸出**：生成分群後的結果 CSV 檔（如 `output_data/ebook_output.csv`）。

**注意**：執行 Notebook 前，請確保已設定好 OpenAI API Key 與相關 Python 套件（如 `pandas`, `sklearn`, `openai`, `sentence-transformers` 等）。

### 2. 人工合併 (Merge App)

分群完成後，使用 Flask Web App 來檢視分群結果，並進行人工確認與合併操作。

#### 啟動方式

確保所在的 Conda 環境已安裝 `flask` 與 `pandas`。

*   **紙本書合併工具**：
    ```bash
    python paper_merge_app.py
    ```
    啟動後，瀏覽器打開 `http://127.0.0.1:5001`。

*   **電子書合併工具**：
    ```bash
    python ebook_merge_app.py
    ```
    啟動後，瀏覽器打開 `http://127.0.0.1:5001`。

#### 功能說明

- **瀏覽資料**：網頁會顯示待處理的書籍資料（支援分頁顯示，每頁 50 筆）。
- **合併 (Merge)**：勾選多筆看起來是同一本書的資料，點擊 "Merge Selected"。系統會將選取的資料合併為一筆。
    - 支援依 `TAICCA_ID` 或 `index` 進行合併。
    - 合併後，系統會自動整合各欄位資訊（如 ISBN、書名、連結等），並保留所有來源的 Production ID。
- **取消合併 (Unmerge)**：勾選一筆已合併的資料（通常標示為灰色底），點擊 "Unmerge Selected"。
    - 系統會根據 ID（`TAICCA_ID` 或各通路的 `Production ID`）追朔並還原原始資料。
    - 即使是來自不同來源的「互補合併」（Production ID 無斜線分隔），只要能對應到原始資料，皆可還原。
- **編輯資料**：點擊欄位內容可直接編輯、插入或刪除列。

## 注意事項

1.  **環境設定**：確保正確啟動 Conda 環境（如 `conda activate condaEnv`）。
2.  **Port 衝突**：若發生 `Address already in use` 錯誤，請先關閉佔用 Port 5001 的程序（可使用 `lsof -i :5001` 查看）。
3.  **資料備份**：主要操作會直接修改讀入的 CSV 檔案（或快取中的資料），建議定期備份原始資料。

### 修改資料來源

若需要更改輸入或輸出的檔案路徑，請直接編輯對應的 Python 檔案開頭變數：

- **電子書 (`ebook_merge_app.py`)**
  ```python
  TARGET_FILE = "input_data/ebook_output.csv"  # 這是主要操作的合併檔案 (會被讀取與寫入)
  SOURCE_FILE = "input_data/ebook_test.csv"    # 這是原始未處理資料 (在 Unmerge 時用來查找原始紀錄)
  ```

- **紙本書 (`paper_merge_app.py`)**
  ```python
  DATA_DIR = 'input_data'
  FILE_7000 = os.path.join(DATA_DIR, 'raw_data.csv')          # 主要操作的合併檔案
  FILE_INPUT = os.path.join(DATA_DIR, 'paper_unproccessed.csv') # 原始未處理資料
  ```

## 更新日誌 (2026-01-26)

### 電子書 (Ebook) 支援優化
1. **EISBN 欄位支援**：
   - 新增 `eisbn` 欄位的完整合併邏輯（支援多組 ISBN 以 `/` 分隔）。
   - 支援從通路欄位 (`bookscom_eisbn`, `readmoo_eisbn`, `kobo_eisbn`) 自動填補主 `eisbn`。
   - `Unmerge` 功能現在支援透過 `eisbn` 追朔原始資料，解決無 `Production ID` 的電子書無法還原的問題。
2. **格式欄位與排序**：
   - 新增電子書格式欄位 (`bookscom_type_ebook`, `readmoo_type_ebook`, `kobo_type_ebook`) 的保留與合併。
   - 優化 CSV 輸出欄位排序，確保 URL 欄位固定排在最後，方便檢視。
3. **Notebook 預處理**：
   - `ebook_clustering.ipynb` 新增欄位整合邏輯 (`consolidate_eisbn` 等)，在分群前預先填補空缺欄位。

### 前端介面 (Merge App)
1. **電子書合併標示**：修正電子書合併列無法正確變色（藍底）的問題，現在正確支援 Readmoo 與 Kobo 的 ID 判斷。
2. **URL 顯示優化**：所有 URL 欄位在網頁中會顯示為 **深藍色並帶底線**，提升視覺辨識度。
3. **資料編輯與刪除**：新增行內資料編輯、插入新行與刪除行的功能。
