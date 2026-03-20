# PhoneUse

手機自動化 Agent 框架，支援 Android 裝置控制、螢幕標記、OCR 辨識與 AI 視覺理解。

---

## 模型權重下載

本專案使用 OmniParser 作為視覺解析引擎，需要下載以下模型權重：

### 1. OmniParser 模型（ICON 偵測 + 圖片描述）

```bash
cd OmniParser

# 建立 weights 目錄
mkdir -p weights

# 下載 V2 模型
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do
    huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done

# 重新命名資料夾（程式碼預設名稱）
mv weights/icon_caption weights/icon_caption_florence
```

### 2. GLM-OCR 模型（用於 ocr_server.py）

本專案的 `ocr_server.py` 提供基於 GLM-OCR 的本地 OCR 服務。

#### 前置需求

```bash
# 安裝 transformers（用于 GLM-OCR 模型）
pip install transformers>=4.30
```

#### 模型下載

GLM-OCR 模型會在首次執行時自動下載，無需手動下載。

#### 啟動 OCR 伺服器

```bash
# 啟動伺服器（預設 port 8765）
python ocr_server.py

# 或指定 port
python ocr_server.py --port 8080
```

伺服器啟動後會顯示：
```
Model ready!
INFO:     Uvicorn running on http://127.0.0.1:8765
```

#### API 使用方式

```bash
# 直接呼叫
curl -X POST http://127.0.0.1:8765/ocr \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "base64編碼的圖片"}'
```

詳細說明請參考 [OmniParser/GLM-OCR-USAGE.md](OmniParser/GLM-OCR-USAGE.md)

### 3. 權重資料夾結構

```
OmniParser/weights/
├── icon_detect/
│   ├── model.pt
│   ├── model.yaml
│   └── train_args.yaml
└── icon_caption_florence/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    └── ...
```

---

## 功能特色

### 🎯 螢幕標記工具 (TaggingTool)

基於 PyQt6 的 GUI 工具，用於標記手機畫面中的互動元素：

- **狀態管理**：為每個 App 畫面建立狀態節點
- **標定點**：支援點擊 (`tap`) 與滑動 (`swipe`) 兩種類型
- **狀態跳轉**：可設定標定點點擊後要跳转到哪個狀態
- **JSON 匯出**：自動產生包含座標、類型、狀態關聯的標記檔案

產出的 `markers.json` 可讓 Agent 理解：
- 畫面中有哪些可互動元素
- 各元素的精確座標
- 點擊後會跳转到哪個新狀態

### 🔍 Screen Overview 兩種模式

#### OCR 模式（快速）
```
phoneuse.py screen_overview --provider ocr
```
直接將截圖送往 GLM-OCR 本機伺服器，取得畫面文字內容。

#### Full 管線（詳細）
```
phoneuse.py screen_overview --provider full
```
完整 AI 視覺理解流程：

```
截圖 → OmniParser (ICON偵測 + Florence2 Caption)
              ↓
         每個偵測到的區域
              ↓
         GLM-OCR (區域文字辨識)
              ↓
         Gemini (AI 精煉 + 統一描述)
              ↓
         輸出像素座標 + 詳細標籤
```

輸出格式：
```
icon 0: {id: 0, type: button, content: "確認購買", bbox: [120, 540, 360, 600], ...}
icon 1: {id: 1, type: icon_button, content: "返回", bbox: [30, 30, 90, 90], ...}
```

### 📸 其他功能

| 命令 | 說明 |
|------|------|
| `screenshot` | 取得截圖 |
| `hierarchy` | 取得 UI 階層 XML |
| `click x y` | 點擊座標 |
| `swipe sx sy ex ey` | 滑動 |
| `press key` | 按鍵 (home/back/menu...) |
| `annotate` | Gemini 視覺標注（需要 GEMINI_API_KEY） |
| `list_states` | 列出 JSON 中的狀態 |
| `run_marker` | 執行標記點動作 |

---

## 快速安裝

### 1. 建立 conda 環境

```bash
conda env create -f environment.yaml
conda activate phoneuse
```

### 2. 安裝依賴

```bash
pip install python-dotenv

# 、行動自動化
pip install uiautomator2

# GUI 標記工具
pip install PyQt6

# OCR 伺服器（可選）
# 請參考 OmniParser/GLM-OCR-USAGE.md
```

### 3. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env 填入你的 API Keys
```

### 4. 連接 Android 裝置

確保 ADB 已安裝並啟用 USB 偵錯：

```bash
adb devices
# 確認你的裝置在列表中
```

---

## 使用範例

### 建立標記檔

1. 啟動 TaggingTool：
   ```bash
   python main.py
   ```

2. 點「擷取螢幕」或「擷取手機畫面」取得截圖

3. 在右側面板點「新增標定點」，然後在圖片上點擊要標記的位置

4. 設定標定點名稱與類型（點擊/滑動）

5. 匯出 JSON：`檔案 → 匯出 JSON`

### 產出的 markers.json 格式

```json
{
  "base_resolution": [1080, 2400],
  "states": [
    {
      "name": "首頁",
      "description": "App 首頁",
      "image_path": "images/01_首頁.png",
      "markers": [
        {
          "name": "開始遊戲按鈕",
          "x": 540,
          "y": 1200,
          "type": "tap",
          "target_state": "戰鬥頁面"
        },
        {
          "name": "滑動條",
          "x": 300,
          "y": 1800,
          "type": "swipe",
          "bx": 600,
          "by": 1600
        }
      ]
    }
  ]
}
```

### 執行 Agent 自動化

```bash
# 查看目前畫面（OCR 模式）
python phoneuse.py screen_overview --provider ocr

# 查看目前畫面（Full 管線）
python phoneuse.py screen_overview --provider full

# 執行特定標記點
python phoneuse.py run_marker "首頁" "開始遊戲按鈕" --json markers.json

# 執行並跟隨狀態跳轉
python phoneuse.py run_marker_follow "首頁" "開始遊戲按鈕" --json markers.json
```

---

## 環境變數

| 變數 | 說明 | 預設值 |
|------|------|--------|
| `GEMINI_API_KEY` | Google Gemini API Key（用於 annotate 和 full 管線） | - |
| `OPENROUTER_API_KEY` | OpenRouter API Key | - |
| `GLM_OCR_SERVER_URL` | GLM-OCR 本機服務位址 | `http://192.168.0.212:8765/ocr` |
| `OLLAMA_MODEL` | Ollama 模型名稱 | `qwen3.5:0.8b` |

---

## 資料夾結構

```
PhoneUse/
├── main.py              # TaggingTool GUI 工具
├── phoneuse.py          # CLI 主程式
├── ocr_server.py        # OCR 伺服器
├── markers.json          # 標記檔案
├── screenshot/          # 截圖存放
├── images/              # TaggingTool 管理的圖片
├── OmniParser/          # OmniParser 模型
├── session_3095/        # 範例 session 資料
├── .env                 # 環境變數（不上傳）
├── .env.example         # 環境變數範本
└── requirements.txt
```

---

## 依賴說明

- **uiautomator2**：Android 自動化控制
- **PyQt6**：TaggingTool GUI
- **requests**：API 呼叫
- **python-dotenv**：環境變數管理
- **OmniParser**（可選）：ICON 偵測 + Captioning
- **GLM-OCR**（可選）：本地端 OCR 服務

---

## API Keys 安全性

⚠️ **重要**：所有 API Keys 請勿寫入程式碼！

- 使用 `.env` 檔案管理敏感資訊
- `.env` 已在 `.gitignore` 中排除
- 參考 `.env.example` 建立你的 `.env`

---

## OmniParser 致謝與授權

本專案內嵌了 [Microsoft OmniParser](https://github.com/microsoft/OmniParser) 作為視覺解析引擎。

### 模型授權

| 模型 | 授權條款 |
|------|----------|
| `icon_detect`（ICON 偵測模型） | AGPL |
| `icon_caption_blip2` & `icon_caption_florence`（圖片描述模型） | MIT |

詳細授權資訊請參考：
- [OmniParser/LICENSE](OmniParser/LICENSE)
- [OmniParser 官方模型頁面](https://huggingface.co/microsoft/OmniParser)

### 本專案魔改項目

本專案對 OmniParser 進行了以下客製化修改：

1. **簡化相依性**：移除不必要的 Azure 雲端服務依賴
2. **整合進 screen overview 管線**：與 GLM-OCR、Gemini 串聯成完整流程
3. **支援本機部署**：可直接使用本地 OCR 服務
4. **JSON 標記格式**：與 TaggingTool 整合，支援狀態機制

### 引用 OmniParser

如果您在學術研究中使用了 OmniParser，請引用原文：

```bibtex
@misc{lu2024omniparserpurevisionbased,
      title={OmniParser for Pure Vision Based GUI Agent},
      author={Yadong Lu and Jianwei Yang and Yelong Shen and Ahmed Awadallah},
      year={2024},
      eprint={2408.00203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00203},
}
```

---

## 參考連結

- [OmniParser 原始論文 (arXiv)](https://arxiv.org/abs/2408.00203)
- [OmniParser GitHub](https://github.com/microsoft/OmniParser)
- [OmniParser V2 模型 (HuggingFace)](https://huggingface.co/microsoft/OmniParser-v2.0)
- [OmniParser V1.5 模型 (HuggingFace)](https://huggingface.co/microsoft/OmniParser)
