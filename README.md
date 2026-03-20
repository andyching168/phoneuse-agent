# PhoneUse

手機自動化 Agent 框架，支援 Android 裝置控制、螢幕標記、OCR 辨識與 AI 視覺理解。

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
