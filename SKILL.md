```skill
---
name: phone-use
description: Control Android phone via ADB/uiautomator2 and drive flow with shopee.json state machine.
---

# Phone Use Skill

控制 Android 裝置，並可使用 `shopee.json`（含 `target_state`）形成可循環的流程控制。
適合特殊app自動化（例如蝦皮購物）與一般 App 操作。

## ⚡️ 預設模式：本地模型（免費）

**預設使用本地 GLM-OCR 伺服器** 進行畫面辨識，完全免費！

若要切換到 **Z.AI MaaS API**，請設定環境變數：
```bash
USE_LOCAL_GLM_OCR=false
```

> ⚠️ **安全提醒**：API Key 請勿寫在 code 裡！請透過 `.env` 檔案或環境變數設定。`.env` 已在 `.gitignore` 中，不會被 commit。

## Available Tools

### phone_screenshot
Get current screen as base64 image.
```bash
python phoneuse.py screenshot
```

### phone_hierarchy
Get UI hierarchy as XML (compressed). Useful for finding element positions and resource IDs.
```bash
python phoneuse.py hierarchy
```

### phone_click
Click at coordinates (x, y).
```bash
python phoneuse.py click <x> <y>
```

### phone_long_click
Long click at coordinates with duration.
```bash
python phoneuse.py long_click <x> <y> --duration <seconds>
```

### phone_swipe
Swipe from (sx, sy) to (ex, ey).
```bash
python phoneuse.py swipe <sx> <sy> <ex> <ey> --duration <seconds>
```

### phone_press
Press key (home/back/menu/volume_up/volume_down/power).
```bash
python phoneuse.py press <key>
```

### phone_text
Input text (uses fastinput IME).
```bash
python phoneuse.py text "<content>"
```

### phone_clear
Clear input field.

### phone_tap_text
Tap element by text content.
```bash
python phoneuse.py tap_text "<text>" --fuzzy
```

### phone_exists
Check if element with text exists. Returns "true" or "false".
```bash
python phoneuse.py exists "<text>"
```

### phone_wait_id
Wait for element by resource ID. Returns "true" or "false".
```bash
python phoneuse.py wait_id <resource_id> --timeout <seconds>
```

### phone_screen_size
Get screen resolution (e.g., "1080x1920").

### phone_current_app
Get currently running app info (JSON).

### phone_start_app
Start app by package name.
```bash
python phoneuse.py start_app <package>
```

### phone_stop_app
Stop app by package name.
```bash
python phoneuse.py stop_app <package>
```

### phone_annotate
Annotate UI elements using Gemini vision (for games/apps without hierarchy). Returns JSON with element bounding boxes, types, and coordinates.
```bash
python phoneuse.py annotate
```
除非是連screen_overview full都無法應對的情況,再使用phone_annotate.
The output includes `center` coordinates for direct `phone_click`.

### phone_list_states
讀取標記檔中的狀態清單（含 base_resolution / description / marker_count）。
```bash
python phoneuse.py list_states --json markers.json
```

### phone_list_markers
列出某個狀態的標記。
```bash
python phoneuse.py list_markers "主頁" --json markers.json
```

### phone_run_marker
執行某個標記（tap / swipe）。
```bash
python phoneuse.py run_marker "主頁" "任務" --json markers.json
```

### phone_run_marker_follow
執行標記後，回傳 `target_state` 的摘要，便於流程迴圈。
```bash
python phoneuse.py run_marker_follow "主頁" "任務" --json shopee.json
```

## screen_overview - 畫面辨識（重要！）

### 預設：本地模型 ⭐ 免費
```bash
python phoneuse.py screen_overview 
```

輸出格式（純文字）：
```
公告
更新情報
活動資訊
問題說明

黎明界迷宮追加公會
★3必中白金轉蛋
```

### --provider api（Z.AI 雲端 💰）
若設定 `USE_LOCAL_GLM_OCR=false`，會改用 Z.AI MaaS GLM-OCR API。
```bash
USE_LOCAL_GLM_OCR=false python phoneuse.py screen_overview --provider api
```

輸出格式（包含 Bounding Box 座標）：
```
[IMAGE] size=1080x2220
[image] bbox=[20, 2, 237, 38]
[text] bbox=[135, 610, 256, 638] "MQTT_Hiyo"
[text] bbox=[493, 821, 589, 852] "蝦皮購物"
```

> ⚠️ **重要限制**：Z.AI GLM-OCR API 在遇到**手機遊戲畫面**時，可能會把整個遊戲畫面當成圖片處理，導致無法正確辨識遊戲內的文字。若需要自動化遊戲，建議使用 `--provider full` 或純 `markers.json` 方案。

### --provider full（超詳細模式）⚠️ 需要 Z.AI API
使用 OmniParser + GLM-OCR + Gemini，提供每個 UI 元素的：
- 精確座標 (`bbox`, `bbox_ratio`)
- 文字內容 (`content`)
- 是否可互動 (`interactivity`)
- 圖示語意標籤 (`icon_label`)

```bash
USE_LOCAL_GLM_OCR=false python phoneuse.py screen_overview --provider full
```

> ⚠️ full 模式需要 `omni` conda 環境才能正常載入 OmniParser。
> ```bash
> /home/ac/miniconda3/bin/conda run -n omni python phoneuse.py screen_overview --provider full
> ```

## Usage Pattern

1. 先用 `phone_list_states` / `phone_list_markers` 了解 JSON 狀態機
2. 進迴圈前先用 `phone_screen_overview` 或 `phone_hierarchy` 判斷目前畫面
3. 以當前狀態挑選對應標記，用 `phone_run_marker_follow` 執行
4. 讀取回傳的 `next_state` 作為下一輪狀態
5. 直到命中終點狀態或步數上限才停止

## 特殊程式迴圈（推薦）

目標：在「主頁 → 檢查畫面 → 關閉可能的廣告彈窗 → 到「我的」等路徑中執行。

### 1) 初始化
```bash
python phoneuse.py start_app com.shopee.tw
python phoneuse.py list_states --json shopee.json
```

### 2) 每輪流程
1. 用 `phone_screen_overview` 取得目前畫面大意（避免卡在未知彈窗）
2. 依目前狀態名稱，呼叫：
   ```bash
   python phoneuse.py run_marker_follow "<目前狀態>" "<要點的標記>" --json shopee.json
   ```
3. 解析輸出中的 `next_state.name`，當成下一輪 `<目前狀態>`

### 3) 例外處理
- 若 `next_state` 為 `null`：維持原狀態，改選另一個標記
- 若畫面與狀態不符：先 `phone_screen_overview`，必要時執行 `phone_press back`
- 若是活動 UI 變動：先回標記工具補點，再重跑

### 4) 終止條件
- 達到指定步數（例如 30 步）
- 進入「結束/待機」狀態
- 連續 N 次無法前進（例如 3 次 `next_state = null`）

## 給 Agent 的呼叫策略（特殊程式）

1. 優先走 `markers.json`：`list_states` → `run_marker_follow`
2. 畫面不確定時才呼叫 `screen_overview`
3. 不要盲點座標；優先使用已標記的 `name + target_state`
4. 每輪記錄：`current_state`, `marker`, `next_state`, `result`
5. 若偵測到卡關，執行一次 `press back` 後再回主循環

## Setup

```bash
pip install -r requirements.txt
python -m uiautomator2 init
```

## 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `ZAI_API_KEY` | (選填) | Z.AI API Key（用於 `--provider api` 或 `--provider full`） |
| `USE_LOCAL_GLM_OCR` | `true` | 設為 `false` 可切換到 Z.AI 雲端 API |
| `GLM_OCR_SERVER_URL` | `http://192.168.0.212:8765/ocr` | 本地 GLM-OCR server URL |
| `GEMINI_API_KEY` | - | Gemini API Key (用於 annotate) |

> ⚠️ 所有 API Key 請勿寫在 code 裡，請透過 `.env` 設定！

## ⚠️ 實作心得（2026-03-17 新增）

1. **優先使用 JSON 標示**：有狀態機 JSON 時，優先用 `run_marker_follow` 導航，不要直接用 hierarchy 找座標
2. **辨識目前狀態**：根據畫面上的元素來判斷（例如：所有訂單顯示「已完成」代表在「全部」或「訂單已完成」tab）
3. **每步驟後檢查螢幕**：點完之後要馬上用 `screen_overview` 檢查，確認狀態變化是否符合預期
4. **狀態機會自動識別**：呼叫 `run_marker_follow` 時，如果狀態名稱正確，會回傳目前狀態，可用來確認
