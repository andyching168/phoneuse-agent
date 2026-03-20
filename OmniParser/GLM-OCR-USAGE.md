# GLM-OCR 整合使用說明

## 概述
已成功將 GLM-OCR 整合到 OmniParser 中，用於提升文字辨識準確度。

**重要提示：** 
- 由於 GPU VRAM 限制（5.67 GB），GLM-OCR 運行在 **CPU 模式**
- CPU 模式速度**非常慢**（每個文字框 30-60 秒）
- **不建議用於大量文字辨識**，建議只用於：
  - 少量關鍵文字（<5 個文字框）
  - 特別複雜或模糊的文字
  - 需要高準確度的場景

## 使用方式

### 啟動服務

1. **啟動 GLM-OCR 伺服器**（Terminal 1）：
   ```bash
   cd /home/ac/MobileTagging
   python ocrServer.py
   ```
   
   等待看到：
   ```
   Model ready on CPU!
   INFO:     Uvicorn running on http://127.0.0.1:8000
   ```

2. **啟動 OmniParser Gradio Demo**（Terminal 2）：
   ```bash
   cd /home/ac/MobileTagging/OmniParser
   python gradio_demo.py
   ```

3. **在 Gradio 介面中**：
   - 上傳圖片
   - 勾選 ✅ "Use GLM-OCR (Better text recognition)"
   - 選擇 🔘 "API (Port 8000)"
   - 點擊 Submit
   - **耐心等待**（可能需要 5-10 分鐘處理完所有文字框）

## 效能考量

| 項目 | 說明 |
|------|------|
| **處理速度** | 30-60 秒/文字框（CPU 模式） |
| **記憶體使用** | 約 2-4 GB RAM |
| **Timeout** | 3 分鐘/請求 |
| **適用場景** | 少量、複雜、模糊文字 |

## 建議使用策略

1. **先用 PaddleOCR/EasyOCR** 快速處理
2. **僅對辨識錯誤的文字** 使用 GLM-OCR
3. 或考慮使用**雲端 OCR API**（Google Vision, AWS Textract）獲得更好的速度和準確度

## 效能比較

### PaddleOCR/EasyOCR
- ✅ 速度快（<1秒）
- ⚠️ 準確度中等
- ✅ 適合大量文字

### GLM-OCR (CPU)
- ❌ 速度慢（30-60秒）
- ✅ 準確度高
- ❌ 僅適合少量文字

### 雲端 API（建議）
- ✅ 速度快（1-3秒）
- ✅ 準確度高
- ⚠️ 需要 API Key 和費用

## 運作流程

```
輸入圖片
    ↓
OmniParser YOLO 偵測邊界框（精準）
    ↓
裁剪每個文字區域
    ↓
GLM-OCR 辨識文字（準確）
    ↓
替換原有 OCR content
    ↓
輸出結果
```

## 效果比較

### 原本（EasyOCR/PaddleOCR）
```
icon 0: {'type': 'text', 'content': '$49', 'source': 'box_ocr_content_ocr'}
icon 1: {'type': 'text', 'content': '4.4', 'source': 'box_ocr_content_ocr'}
```

### 使用 GLM-OCR 後
```
icon 0: {'type': 'text', 'content': '$49.00', 'source': 'glm_ocr_api'}
icon 1: {'type': 'text', 'content': '4.4 ⭐', 'source': 'glm_ocr_api'}
```

## 故障排除

### API 模式連接失敗
如果看到 `API error: ...`，檢查：
1. `ocrServer.py` 是否正在運行
2. 端口 8000 是否被佔用
3. 修改 `util/utils.py` 中的 `GLM_OCR_API_URL` 如果使用不同端口

### 本地模型載入失敗
如果看到 `Failed to load GLM-OCR model`：
1. 確認 transformers 版本 >= 4.30
2. 確認有足夠的 GPU 記憶體（建議 8GB+）
3. 使用 API 模式作為替代方案

## 配置

可在 `util/utils.py` 中修改：
- `GLM_OCR_API_URL`: API 端點位址（預設: `http://localhost:8000/ocr`）
- `max_new_tokens`: 最大生成 token 數（預設: 8192）

## 性能考量

- **API 模式**: 更穩定，適合生產環境
- **本地模式**: 更快（避免網路延遲），但需要更多記憶體
- **處理時間**: 每個文字框約 0.5-2 秒（取決於圖片大小）

## 程式碼修改記錄

- `util/utils.py`: 新增 GLM-OCR 函數和 API 支援
- `gradio_demo.py`: 新增 GLM-OCR 選項和模式選擇
