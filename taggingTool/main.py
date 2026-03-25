import sys
import os
import json
import shutil
import subprocess
import pyscreenshot
from PIL import Image
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QLineEdit, QListWidget, QListWidgetItem,
    QScrollArea, QFrame, QDialog, QDialogButtonBox, QFormLayout, QMessageBox,
    QFileDialog, QSpinBox, QGroupBox, QGraphicsScene, QGraphicsView, QInputDialog, QComboBox,
    QProgressDialog, QMenu,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsTextItem,
    QGraphicsItem
)
from PyQt6.QtCore import Qt, QPointF, QRectF, QRect, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QScreen, QTransform, QBrush, QFont


@dataclass
class Marker:
    name: str
    x: int
    y: int
    marker_type: str = "tap"
    bx: Optional[int] = None
    by: Optional[int] = None
    target_state: Optional[str] = None

    def __post_init__(self):
        self.x = int(round(float(self.x)))
        self.y = int(round(float(self.y)))
        marker_type = str(self.marker_type).lower()
        if marker_type in ["swipe", "slide", "滑動"]:
            self.marker_type = "swipe"
        else:
            self.marker_type = "tap"

        if self.marker_type == "swipe":
            if self.bx is None or self.by is None:
                self.bx = self.x
                self.by = self.y
            else:
                self.bx = int(round(float(self.bx)))
                self.by = int(round(float(self.by)))
        else:
            self.bx = None
            self.by = None

        if self.target_state is not None:
            self.target_state = str(self.target_state).strip() or None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "x": int(self.x),
            "y": int(self.y),
            "type": self.marker_type
        }
        if self.marker_type == "swipe":
            data["bx"] = int(self.bx)
            data["by"] = int(self.by)
        if self.target_state:
            data["target_state"] = self.target_state
        return data


@dataclass
class State:
    name: str
    image_path: str
    description: str = ""
    markers: List[Marker] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "image_path": self.image_path,
            "description": self.description,
            "markers": [m.to_dict() for m in self.markers]
        }


@dataclass
class StateTabContext:
    graphics_view: "AutoFitGraphicsView"
    scene: "MarkerGraphicsScene"
    state_name_edit: QLineEdit
    state_desc_edit: QLineEdit
    marker_list: QListWidget


class ScreenCapture:
    last_capture_message: str = ""
    last_capture_message_level: str = ""

    @staticmethod
    def is_wayland_session() -> bool:
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        return session_type == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))

    @staticmethod
    def _clear_capture_message():
        ScreenCapture.last_capture_message = ""
        ScreenCapture.last_capture_message_level = ""

    @staticmethod
    def _set_capture_message(level: str, message: str):
        ScreenCapture.last_capture_message_level = level
        ScreenCapture.last_capture_message = message

    @staticmethod
    def _load_temp_pixmap(temp_file: str) -> Optional[QPixmap]:
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            pixmap = QPixmap(temp_file)
            try:
                os.unlink(temp_file)
            except:
                pass
            if not pixmap.isNull():
                return pixmap
        return None

    @staticmethod
    def get_android_devices() -> List[Dict[str, str]]:
        ScreenCapture._clear_capture_message()

        try:
            result = subprocess.run(
                ["adb", "devices", "-l"],
                capture_output=True,
                text=True,
                check=False
            )
        except FileNotFoundError:
            ScreenCapture._set_capture_message(
                "warning",
                "找不到 adb，請先安裝 android-platform-tools。"
            )
            return []
        except Exception as e:
            ScreenCapture._set_capture_message("warning", f"讀取 Android 裝置失敗: {e}")
            return []

        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            ScreenCapture._set_capture_message(
                "warning",
                "執行 adb devices 失敗。"
                + (f"\n\n系統訊息: {error_message}" if error_message else "")
            )
            return []

        devices: List[Dict[str, str]] = []
        for line in result.stdout.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            serial = parts[0]
            status = parts[1]
            extra_info = " ".join(parts[2:])
            model = ""

            for part in parts[2:]:
                if part.startswith("model:"):
                    model = part.split(":", 1)[1].replace("_", " ")
                    break

            devices.append({
                "serial": serial,
                "status": status,
                "model": model,
                "description": extra_info
            })

        return devices

    @staticmethod
    def capture_android_screen(device_serial: Optional[str] = None) -> Optional[QPixmap]:
        image_data, message_level, message = ScreenCapture.capture_android_screen_bytes(device_serial)
        if message:
            ScreenCapture._set_capture_message(message_level, message)
            return None

        pixmap = QPixmap()
        if pixmap.loadFromData(image_data, "PNG"):
            return pixmap

        ScreenCapture._set_capture_message("warning", "Android 螢幕資料格式無法解析為 PNG。")
        return None

    @staticmethod
    def capture_android_screen_bytes(device_serial: Optional[str] = None) -> tuple[Optional[bytes], str, str]:
        command = ["adb"]
        if device_serial:
            command.extend(["-s", device_serial])
        command.extend(["exec-out", "screencap", "-p"])

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=False
            )
        except FileNotFoundError:
            return None, "warning", "找不到 adb，請先安裝 android-platform-tools。"
        except Exception as e:
            return None, "warning", f"Android 截圖失敗: {e}"

        if result.returncode != 0:
            stderr_text = result.stderr.decode(errors="ignore").strip() if result.stderr else ""
            return (
                None,
                "warning",
                "無法從 Android 裝置擷取畫面。"
                + (f"\n\n系統訊息: {stderr_text}" if stderr_text else "")
            )

        image_data = result.stdout or b""
        if not image_data:
            return None, "warning", "Android 裝置沒有回傳任何畫面資料。"

        return image_data, "", ""

    @staticmethod
    def get_windows() -> List[Dict[str, Any]]:
        if ScreenCapture.is_wayland_session():
            print("Wayland session detected: xdotool window enumeration is not supported.")
            return []

        try:
            result = subprocess.run(
                ["xdotool", "search", "--onlyvisible", "--class", ""],
                capture_output=True, text=True
            )
            windows = []
            for win_id in result.stdout.strip().split('\n'):
                if not win_id:
                    continue
                try:
                    title = subprocess.run(
                        ["xdotool", "getwindowname", win_id],
                        capture_output=True, text=True
                    ).stdout.strip()
                    
                    geo = subprocess.run(
                        ["xdotool", "getwindowgeometry", "--shell", win_id],
                        capture_output=True, text=True
                    )
                    
                    x, y, w, h = 0, 0, 0, 0
                    for line in geo.stdout.split('\n'):
                        if line.startswith("X="):
                            x = int(line.split('=')[1])
                        elif line.startswith("Y="):
                            y = int(line.split('=')[1])
                        elif line.startswith("WIDTH="):
                            w = int(line.split('=')[1])
                        elif line.startswith("HEIGHT="):
                            h = int(line.split('=')[1])
                    
                    if w > 50 and h > 50 and not title.startswith("Desktop"):
                        windows.append({
                            "id": win_id,
                            "title": title if title else f"Window {win_id}",
                            "x": x, "y": y, "w": w, "h": h
                        })
                except:
                    continue
            return windows
        except Exception as e:
            print(f"Error getting windows: {e}")
            return []
    
    @staticmethod
    def capture_window(window_info: Dict[str, Any], screen: QScreen) -> Optional[QPixmap]:
        if ScreenCapture.is_wayland_session():
            ScreenCapture._set_capture_message(
                "warning",
                "GNOME Wayland 不支援目前這種以 xdotool / X11 擷取單一視窗的方式。"
            )
            return None

        try:
            win_id = window_info["id"]
            
            result = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", win_id],
                capture_output=True, text=True
            )
            
            x, y, w, h = 0, 0, 0, 0
            for line in result.stdout.split('\n'):
                if line.startswith("X="):
                    x = int(line.split('=')[1])
                elif line.startswith("Y="):
                    y = int(line.split('=')[1])
                elif line.startswith("WIDTH="):
                    w = int(line.split('=')[1])
                elif line.startswith("HEIGHT="):
                    h = int(line.split('=')[1])
            
            if w > 0 and h > 0:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    temp_file = f.name
                
                subprocess.run([
                    'import', '-window', win_id, '-crop', f'{w}x{h}+{x}+{y}', temp_file
                ], check=True)
                
                pixmap = QPixmap(temp_file)
                try:
                    os.unlink(temp_file)
                except:
                    pass
                
                if not pixmap.isNull():
                    return pixmap
                
                win_id_int = int(win_id)
                return screen.grabWindow(win_id_int, 0, 0, w, h)
            return None
        except Exception as e:
            print(f"Error capturing window: {e}")
            return None
    
    @staticmethod
    def capture_full_screen(screen: QScreen) -> QPixmap:
        return screen.grabWindow(0)
    
    @staticmethod
    def capture_region(x: int, y: int, w: int, h: int) -> Optional[QPixmap]:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_file = f.name
            
            subprocess.run([
                'gnome-screenshot', '-a', '-f', temp_file
            ], check=True)
            
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                pixmap = QPixmap(temp_file)
                try:
                    os.unlink(temp_file)
                except:
                    pass
                return pixmap
            return None
        except FileNotFoundError:
            print("gnome-screenshot not found, using pyscreenshot")
            img = pyscreenshot.grab(bbox=(x, y, x + w, y + h))
            img = img.convert('RGBA')
            qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format.Format_RGBA8888)
            return QPixmap.fromImage(qimg)
        except Exception as e:
            print(f"Error capturing region: {e}")
            return None
    
    @staticmethod
    def capture_full(screen: QScreen) -> Optional[QPixmap]:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_file = f.name
            
            subprocess.run([
                'gnome-screenshot', '-f', temp_file
            ], check=True)
            
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                pixmap = QPixmap(temp_file)
                try:
                    os.unlink(temp_file)
                except:
                    pass
                return pixmap
            return None
        except FileNotFoundError:
            print("gnome-screenshot not found, using pyscreenshot")
            img = pyscreenshot.grab()
            img = img.convert('RGBA')
            qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format.Format_RGBA8888)
            return QPixmap.fromImage(qimg)
        except Exception as e:
            print(f"Error capturing full screen: {e}")
            return None
    
    @staticmethod
    def capture_gnome_screenshot(mode: str = "interactive") -> Optional[QPixmap]:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_file = f.name
            
            cmd = ['gnome-screenshot']
            if mode == "area":
                cmd.append('-a')
            elif mode == "window":
                cmd.append('-w')
            
            result = subprocess.run(cmd, capture_output=True)
            
            result = subprocess.run(
                ['gnome-screenshot', '--file', temp_file],
                capture_output=True
            )
            
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                pixmap = QPixmap(temp_file)
                try:
                    os.unlink(temp_file)
                except:
                    pass
                if not pixmap.isNull():
                    return pixmap
            
            return None
        except FileNotFoundError:
            print("gnome-screenshot not found, install with: sudo dnf install gnome-screenshot")
            return None
        except Exception as e:
            print(f"Error capturing with gnome-screenshot: {e}")
            return None
    
    @staticmethod
    def capture_gnome_interactive(parent_window=None) -> Optional[QPixmap]:
        ScreenCapture._clear_capture_message()

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_file = f.name
            
            if parent_window:
                parent_window.hide()
            QApplication.processEvents()
            import time
            time.sleep(0.2)
            
            result = subprocess.run(
                ['gnome-screenshot', '--area', '--file', temp_file],
                check=False,
                capture_output=True,
                text=True
            )

            pixmap = ScreenCapture._load_temp_pixmap(temp_file)
            if result.returncode == 0 and pixmap:
                return pixmap

            stderr_text = (result.stderr or "").strip()
            if ScreenCapture.is_wayland_session():
                fallback_result = subprocess.run(
                    ['gnome-screenshot', '--file', temp_file],
                    check=False,
                    capture_output=True,
                    text=True
                )
                fallback_pixmap = ScreenCapture._load_temp_pixmap(temp_file)
                if fallback_result.returncode == 0 and fallback_pixmap:
                    ScreenCapture._set_capture_message(
                        "info",
                        "GNOME Wayland 無法使用區域擷取，已自動退回全螢幕擷取。"
                    )
                    return fallback_pixmap

                fallback_stderr = (fallback_result.stderr or "").strip()
                details = fallback_stderr or stderr_text
                ScreenCapture._set_capture_message(
                    "warning",
                    "GNOME Wayland 區域截圖失敗。請確認系統截圖權限與 gnome-shell / xdg-desktop-portal 正常。"
                    + (f"\n\n系統訊息: {details}" if details else "")
                )
                return None

            ScreenCapture._set_capture_message(
                "warning",
                "截圖失敗。"
                + (f"\n\n系統訊息: {stderr_text}" if stderr_text else "")
            )
            return None
        except FileNotFoundError:
            ScreenCapture._set_capture_message(
                "warning",
                "找不到 gnome-screenshot，請先安裝 gnome-screenshot。"
            )
            return None
        except Exception as e:
            ScreenCapture._set_capture_message("warning", f"截圖時發生錯誤: {e}")
            return None
        finally:
            if parent_window:
                parent_window.show()


class MarkerGraphicsScene(QGraphicsScene):
    markerClicked = pyqtSignal(int)
    markerAdded = pyqtSignal()
    markerRightClicked = pyqtSignal(int)
    markerEditConfirmed = pyqtSignal(int, int, int, int, int)  # index, ax, ay, bx, by
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.marker_items: List[QGraphicsItem] = []
        self.marker_labels: List[QGraphicsTextItem] = []
        self.markers: List[Marker] = []
        self.image_item: Optional[QGraphicsItem] = None
        self.target_resolution = (1920, 1080)
        self.source_resolution = (1920, 1080)
        self.pending_swipe_start: Optional[tuple[int, int]] = None
        self.setSceneRect(0, 0, 800, 600)
        
        # Edit mode state
        self.edit_mode = False
        self.edit_marker_index = -1
        self.edit_preview_item: Optional[QGraphicsItem] = None
        
    def set_image(self, pixmap: QPixmap):
        self.clear()
        self.marker_items.clear()
        self.marker_labels.clear()
        
        self.image_item = self.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.source_resolution = (pixmap.width(), pixmap.height())
        self.pending_swipe_start = None

    def set_placeholder_with_resolution(self, width: int, height: int):
        """Set a gray placeholder background when image is missing but we know the resolution."""
        self.clear()
        self.marker_items.clear()
        self.marker_labels.clear()
        
        self.source_resolution = (width, height)
        self.pending_swipe_start = None
        
        # Create a gray background rectangle
        self.image_item = self.addRect(
            0, 0, width, height,
            QPen(QColor(100, 100, 100), 2),
            QBrush(QColor(60, 60, 60))
        )
        self.image_item.setZValue(-1)  # Put behind markers
        
        # Add resolution text
        res_text = self.addText(f"解析度: {width} x {height}\n(圖片遺失)")
        res_text.setDefaultTextColor(QColor(180, 180, 180))
        res_text.setFont(res_text.font())
        self.marker_labels.append(res_text)
        
        self.setSceneRect(0, 0, width, height)
        
    def start_edit_mode(self, marker_index: int):
        """Enter edit mode for a specific marker, allowing user to click canvas to set new position."""
        self.edit_mode = True
        self.edit_marker_index = marker_index
        self._draw_edit_preview()
        
    def cancel_edit_mode(self):
        """Cancel edit mode without applying changes."""
        self.edit_mode = False
        self.edit_marker_index = -1
        self._clear_edit_preview()
        
    def _draw_edit_preview(self):
        """Draw a preview of where the marker will be moved to."""
        self._clear_edit_preview()
        if self.edit_marker_index < 0 or self.edit_marker_index >= len(self.markers):
            return
            
        marker = self.markers[self.edit_marker_index]
        scaled_x = marker.x * self.sceneRect().width() / self.target_resolution[0]
        scaled_y = marker.y * self.sceneRect().height() / self.target_resolution[1]
        
        marker_diameter = max(16.0, min(self.sceneRect().width(), self.sceneRect().height()) * 0.02)
        marker_radius = marker_diameter / 2.0
        
        # Draw dashed circle to indicate edit mode
        ellipse = QGraphicsEllipseItem(
            scaled_x - marker_radius,
            scaled_y - marker_radius,
            marker_diameter,
            marker_diameter
        )
        ellipse.setPen(QPen(QColor(255, 255, 0), 3, Qt.PenStyle.DashLine))
        ellipse.setBrush(QBrush(QColor(255, 255, 0, 50)))
        ellipse.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        ellipse.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.addItem(ellipse)
        self.edit_preview_item = ellipse
        
        # Add text label
        text = self.addText(f"[編輯模式] 點擊設定新座標")
        text.setPos(scaled_x + marker_radius + 4, scaled_y - marker_radius)
        text.setDefaultTextColor(QColor(255, 255, 0))
        text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.marker_labels.append(text)
        
    def _clear_edit_preview(self):
        """Clear the edit preview."""
        if self.edit_preview_item:
            self.removeItem(self.edit_preview_item)
            self.edit_preview_item = None
        # Also clear any edit-related labels
        for label in list(self.marker_labels):
            if isinstance(label, QGraphicsTextItem):
                if "[編輯模式]" in (label.toPlainText() or ""):
                    self.removeItem(label)
                    if label in self.marker_labels:
                        self.marker_labels.remove(label)
        
    def set_target_resolution(self, width: int, height: int):
        self.target_resolution = (width, height)
        
    def add_marker(
        self,
        name: str,
        x: int,
        y: int,
        marker_type: str = "tap",
        bx: Optional[int] = None,
        by: Optional[int] = None
    ):
        marker = Marker(name=name, x=x, y=y, marker_type=marker_type, bx=bx, by=by)
        self.markers.append(marker)
        self._draw_marker(len(self.markers) - 1)
        self.markerAdded.emit()
        
    def _draw_marker(self, index: int):
        marker = self.markers[index]

        if self.target_resolution[0] <= 0 or self.target_resolution[1] <= 0:
            return
        
        scaled_x = marker.x * self.sceneRect().width() / self.target_resolution[0]
        scaled_y = marker.y * self.sceneRect().height() / self.target_resolution[1]

        marker_diameter = max(16.0, min(self.sceneRect().width(), self.sceneRect().height()) * 0.02)
        marker_radius = marker_diameter / 2.0

        ellipse = QGraphicsEllipseItem(
            scaled_x - marker_radius,
            scaled_y - marker_radius,
            marker_diameter,
            marker_diameter
        )
        ellipse.setBrush(QColor(255, 0, 0, 180))
        ellipse.setPen(QPen(QColor(255, 0, 0), 2))
        ellipse.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        ellipse.setData(0, index)
        self.addItem(ellipse)
        self.marker_items.append(ellipse)

        marker_label_prefix = ""
        if marker.marker_type == "swipe":
            scaled_bx = marker.bx * self.sceneRect().width() / self.target_resolution[0]
            scaled_by = marker.by * self.sceneRect().height() / self.target_resolution[1]

            line_item = self.addLine(
                scaled_x,
                scaled_y,
                scaled_bx,
                scaled_by,
                QPen(QColor(0, 255, 255), 2)
            )
            line_item.setData(0, index)
            self.marker_items.append(line_item)

            end_ellipse = QGraphicsEllipseItem(
                scaled_bx - marker_radius,
                scaled_by - marker_radius,
                marker_diameter,
                marker_diameter
            )
            end_ellipse.setBrush(QColor(0, 120, 255, 170))
            end_ellipse.setPen(QPen(QColor(0, 120, 255), 2))
            end_ellipse.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            end_ellipse.setData(0, index)
            self.addItem(end_ellipse)
            self.marker_items.append(end_ellipse)
            marker_label_prefix = "[滑] "
        else:
            marker_label_prefix = "[點] "
        
        if marker.marker_type == "swipe":
            coord_text = f"A({marker.x}, {marker.y}) -> B({marker.bx}, {marker.by})"
        else:
            coord_text = f"({marker.x}, {marker.y})"

        text = self.addText(f"{marker_label_prefix}{marker.name}\n{coord_text}")
        text.setPos(scaled_x + marker_radius + 2, scaled_y - marker_radius)
        text.setDefaultTextColor(QColor(255, 255, 0))
        self.marker_labels.append(text)
        
    def update_markers_display(self):
        for item in self.marker_items:
            self.removeItem(item)
        for label in self.marker_labels:
            self.removeItem(label)
        self.marker_items.clear()
        self.marker_labels.clear()
        
        for i in range(len(self.markers)):
            self._draw_marker(i)
            
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            clicked_item = self.itemAt(event.scenePos(), self.views()[0].transform() if self.views() else QTransform())
            if clicked_item is not None:
                marker_idx = clicked_item.data(0)
                if isinstance(marker_idx, int) and 0 <= marker_idx < len(self.markers):
                    self.markerRightClicked.emit(marker_idx)
                    event.accept()
                    return

        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            if self.sceneRect().contains(pos):
                scale_x = self.target_resolution[0] / self.sceneRect().width()
                scale_y = self.target_resolution[1] / self.sceneRect().height()
                
                x = int(round(pos.x() * scale_x))
                y = int(round(pos.y() * scale_y))

                # Handle edit mode - clicking canvas sets new position
                if self.edit_mode and self.edit_marker_index >= 0:
                    if self.edit_marker_index < len(self.markers):
                        marker = self.markers[self.edit_marker_index]
                        # If it's a swipe marker, we need both start and end points
                        if marker.marker_type == "swipe":
                            if self.pending_swipe_start is None:
                                # First click: set A point
                                self.pending_swipe_start = (x, y)
                                # Update preview position
                                self._draw_edit_preview_for_point(x, y, is_a=True)
                            else:
                                # Second click: set B point and confirm
                                ax, ay = self.pending_swipe_start
                                self.pending_swipe_start = None
                                # Emit signal with marker index and type only; 
                                # MainWindow will update using stored pending A point
                                self.markerEditConfirmed.emit(self.edit_marker_index, ax, ay, x, y)
                                self.edit_mode = False
                                self.edit_marker_index = -1
                                self._clear_edit_preview()
                        else:
                            # Tap marker - just set the single point
                            self.markerEditConfirmed.emit(self.edit_marker_index, x, y, x, y)
                            self.edit_mode = False
                            self.edit_marker_index = -1
                            self._clear_edit_preview()
                    return
                
                # Normal mode - add new marker
                is_ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                if is_ctrl:
                    if self.pending_swipe_start is None:
                        self.pending_swipe_start = (x, y)
                    else:
                        ax, ay = self.pending_swipe_start
                        self.pending_swipe_start = None

                        dialog = MarkerInputDialog(ax, ay, marker_type="swipe", bx=x, by=y)
                        if dialog.exec() == QDialog.DialogCode.Accepted:
                            name, tx, ty, marker_type, bx, by = dialog.get_values()
                            if name:
                                self.add_marker(name, tx, ty, marker_type, bx, by)
                else:
                    self.pending_swipe_start = None
                    dialog = MarkerInputDialog(x, y, marker_type="tap")
                    if dialog.exec() == QDialog.DialogCode.Accepted:
                        name, tx, ty, marker_type, bx, by = dialog.get_values()
                        if name:
                            self.add_marker(name, tx, ty, marker_type, bx, by)
                    
        super().mousePressEvent(event)
    
    def _draw_edit_preview_for_point(self, x: int, y: int, is_a: bool = True):
        """Update preview to show current pending point for swipe editing."""
        self._clear_edit_preview()
        
        scaled_x = x * self.sceneRect().width() / self.target_resolution[0]
        scaled_y = y * self.sceneRect().height() / self.target_resolution[1]
        
        marker_diameter = max(16.0, min(self.sceneRect().width(), self.sceneRect().height()) * 0.02)
        marker_radius = marker_diameter / 2.0
        
        prefix = "A" if is_a else "B"
        ellipse = QGraphicsEllipseItem(
            scaled_x - marker_radius,
            scaled_y - marker_radius,
            marker_diameter,
            marker_diameter
        )
        ellipse.setPen(QPen(QColor(255, 255, 0), 3, Qt.PenStyle.DashLine))
        ellipse.setBrush(QBrush(QColor(255, 255, 0, 50)))
        ellipse.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.addItem(ellipse)
        self.edit_preview_item = ellipse
        
        text = self.addText(f"[編輯模式] A點已設定，點擊設定B點")
        text.setPos(scaled_x + marker_radius + 4, scaled_y - marker_radius)
        text.setDefaultTextColor(QColor(255, 255, 0))
        text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.marker_labels.append(text)


class AutoFitGraphicsView(QGraphicsView):
    resized = pyqtSignal()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()


class AndroidCaptureWorker(QObject):
    finished = pyqtSignal(object, str, str)

    def __init__(self, device_serial: str):
        super().__init__()
        self.device_serial = device_serial

    def run(self):
        image_data, message_level, message = ScreenCapture.capture_android_screen_bytes(self.device_serial)
        self.finished.emit(image_data, message_level, message)


class MarkerInputDialog(QDialog):
    def __init__(
        self,
        x: int,
        y: int,
        parent=None,
        marker_type: str = "tap",
        bx: Optional[int] = None,
        by: Optional[int] = None
    ):
        super().__init__(parent)
        self.setWindowTitle("新增標定點")
        layout = QFormLayout(self)
        
        self.name_edit = QLineEdit()
        self.x_spin = QSpinBox()
        self.y_spin = QSpinBox()
        
        self.x_spin.setRange(0, 100000)
        self.y_spin.setRange(0, 100000)
        self.x_spin.setValue(int(x))
        self.y_spin.setValue(int(y))

        self.type_combo = QComboBox()
        self.type_combo.addItem("點擊", "tap")
        self.type_combo.addItem("滑動", "swipe")

        self.bx_spin = QSpinBox()
        self.by_spin = QSpinBox()
        self.bx_spin.setRange(0, 100000)
        self.by_spin.setRange(0, 100000)
        self.bx_spin.setValue(int(bx if bx is not None else x))
        self.by_spin.setValue(int(by if by is not None else y))

        if marker_type == "swipe":
            self.type_combo.setCurrentIndex(1)
        
        layout.addRow("名稱:", self.name_edit)
        layout.addRow("類別:", self.type_combo)
        layout.addRow("X 座標:", self.x_spin)
        layout.addRow("Y 座標:", self.y_spin)
        layout.addRow("B 點 X:", self.bx_spin)
        layout.addRow("B 點 Y:", self.by_spin)

        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        self.on_type_changed()
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
    def get_values(self):
        marker_type = self.type_combo.currentData()
        if marker_type == "swipe":
            bx = self.bx_spin.value()
            by = self.by_spin.value()
        else:
            bx = None
            by = None
        return self.name_edit.text(), self.x_spin.value(), self.y_spin.value(), marker_type, bx, by

    def on_type_changed(self):
        is_swipe = self.type_combo.currentData() == "swipe"
        self.bx_spin.setEnabled(is_swipe)
        self.by_spin.setEnabled(is_swipe)


class MarkerEditDialog(QDialog):
    def __init__(self, marker: Marker, parent=None):
        super().__init__(parent)
        self.setWindowTitle("編輯標定點")
        layout = QFormLayout(self)
        
        self.name_edit = QLineEdit(marker.name)
        self.x_spin = QSpinBox()
        self.y_spin = QSpinBox()
        
        self.x_spin.setRange(0, 100000)
        self.y_spin.setRange(0, 100000)
        self.x_spin.setValue(int(marker.x))
        self.y_spin.setValue(int(marker.y))

        self.type_combo = QComboBox()
        self.type_combo.addItem("點擊", "tap")
        self.type_combo.addItem("滑動", "swipe")
        self.type_combo.setCurrentIndex(1 if marker.marker_type == "swipe" else 0)

        self.bx_spin = QSpinBox()
        self.by_spin = QSpinBox()
        self.bx_spin.setRange(0, 100000)
        self.by_spin.setRange(0, 100000)
        self.bx_spin.setValue(int(marker.bx if marker.bx is not None else marker.x))
        self.by_spin.setValue(int(marker.by if marker.by is not None else marker.y))
        
        layout.addRow("名稱:", self.name_edit)
        layout.addRow("類別:", self.type_combo)
        layout.addRow("X 座標:", self.x_spin)
        layout.addRow("Y 座標:", self.y_spin)
        layout.addRow("B 點 X:", self.bx_spin)
        layout.addRow("B 點 Y:", self.by_spin)

        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        self.on_type_changed()
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
    def get_values(self):
        marker_type = self.type_combo.currentData()
        if marker_type == "swipe":
            bx = self.bx_spin.value()
            by = self.by_spin.value()
        else:
            bx = None
            by = None
        return self.name_edit.text(), self.x_spin.value(), self.y_spin.value(), marker_type, bx, by

    def on_type_changed(self):
        is_swipe = self.type_combo.currentData() == "swipe"
        self.bx_spin.setEnabled(is_swipe)
        self.by_spin.setEnabled(is_swipe)


class WindowSelectionDialog(QDialog):
    def __init__(self, windows: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("選擇視窗")
        self.selected_window = None
        layout = QVBoxLayout(self)
        
        label = QLabel("選擇要擷取的視窗:")
        layout.addWidget(label)
        
        self.list_widget = QListWidget()
        for win in windows:
            self.list_widget.addItem(f"{win['title']} ({win['w']}x{win['h']})")
        layout.addWidget(self.list_widget)
        
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(QLabel("手動輸入視窗ID:"))
        self.manual_id = QLineEdit()
        self.manual_id.setPlaceholderText("輸入 xdotool 視窗 ID")
        manual_layout.addWidget(self.manual_id)
        manual_btn = QPushButton("使用")
        manual_btn.clicked.connect(self.use_manual_id)
        manual_layout.addWidget(manual_btn)
        layout.addLayout(manual_layout)
        
        full_screen_btn = QPushButton("全螢幕擷取")
        full_screen_btn.clicked.connect(self.select_full_screen)
        layout.addWidget(full_screen_btn)
        
        region_btn = QPushButton("選取區域擷取")
        region_btn.clicked.connect(self.select_region)
        layout.addWidget(region_btn)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.windows = windows
        
    def use_manual_id(self):
        win_id = self.manual_id.text().strip()
        if win_id:
            try:
                result = subprocess.run(
                    ["xdotool", "getwindowgeometry", "--shell", win_id],
                    capture_output=True, text=True
                )
                x, y, w, h = 0, 0, 0, 0
                for line in result.stdout.split('\n'):
                    if line.startswith("X="):
                        x = int(line.split('=')[1])
                    elif line.startswith("Y="):
                        y = int(line.split('=')[1])
                    elif line.startswith("WIDTH="):
                        w = int(line.split('=')[1])
                    elif line.startswith("HEIGHT="):
                        h = int(line.split('=')[1])
                
                if w > 0 and h > 0:
                    self.selected_window = {
                        "id": win_id,
                        "title": f"Window {win_id}",
                        "x": x, "y": y, "w": w, "h": h
                    }
                    self.accept()
            except:
                pass
    
    def select_full_screen(self):
        self.selected_window = {"fullscreen": True}
        self.accept()
        
    def select_region(self):
        self.selected_window = {"region": True}
        self.accept()
        
    def get_selected_window(self):
        if self.selected_window and self.selected_window.get("fullscreen"):
            return self.selected_window
        if self.selected_window and self.selected_window.get("region"):
            return self.selected_window
        idx = self.list_widget.currentRow()
        if idx >= 0 and idx < len(self.windows):
            return self.windows[idx]
        return None


class AndroidDeviceSelectionDialog(QDialog):
    def __init__(self, devices: List[Dict[str, str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("選擇 Android 裝置")
        self.devices = devices

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("偵測到多個 Android 裝置，請選擇要擷取的裝置:"))

        self.list_widget = QListWidget()
        for device in devices:
            model = device.get("model") or "未知型號"
            status = device.get("status") or "unknown"
            serial = device.get("serial") or ""
            self.list_widget.addItem(f"{model} [{serial}] - {status}")
        self.list_widget.setCurrentRow(0)
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_device(self) -> Optional[Dict[str, str]]:
        idx = self.list_widget.currentRow()
        if 0 <= idx < len(self.devices):
            return self.devices[idx]
        return None


class RegionSelectDialog(QDialog):
    def __init__(self, screen_geometry, parent=None):
        super().__init__(parent)
        self.setWindowTitle("選取區域")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setGeometry(screen_geometry)
        self.setCursor(Qt.CursorShape.CrossCursor)
        
        self.start_point = None
        self.end_point = None
        self.selection_rect = None
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        
        if self.selection_rect:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(self.selection_rect, Qt.GlobalColor.white)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.drawRect(self.selection_rect)
            
            w = self.selection_rect.width()
            h = self.selection_rect.height()
            text = f"{w} x {h}"
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawText(
                self.selection_rect.center().x() - 30,
                self.selection_rect.center().y(),
                text
            )
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.selection_rect = None
            
    def mouseMoveEvent(self, event):
        if self.start_point:
            self.end_point = event.pos()
            x = min(self.start_point.x(), self.end_point.x())
            y = min(self.start_point.y(), self.end_point.y())
            w = abs(self.end_point.x() - self.start_point.x())
            h = abs(self.end_point.y() - self.start_point.y())
            self.selection_rect = QRect(x, y, w, h)
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.start_point:
            self.end_point = event.pos()
            x = min(self.start_point.x(), self.end_point.x())
            y = min(self.start_point.y(), self.end_point.y())
            w = abs(self.end_point.x() - self.start_point.x())
            h = abs(self.end_point.y() - self.start_point.y())
            
            if w > 10 and h > 10:
                self.selection_rect = QRect(x, y, w, h)
                self.accept()
            else:
                self.selection_rect = None
                self.start_point = None
                
    def get_region(self):
        if self.selection_rect:
            return (self.selection_rect.x(), self.selection_rect.y(), 
                    self.selection_rect.width(), self.selection_rect.height())
        return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("螢幕標記工具")
        self.resize(1200, 800)
        
        self.states: List[State] = []
        self.base_resolution = (1920, 1080)
        self.current_state_index = -1
        self.is_loading_state = False
        self.is_android_capture_running = False
        self.android_capture_thread: Optional[QThread] = None
        self.android_capture_worker: Optional[AndroidCaptureWorker] = None
        self.android_capture_target_index: Optional[int] = None
        self.capture_progress_dialog: Optional[QProgressDialog] = None
        self.current_json_path: Optional[str] = None
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        
        self.capture_btn = QPushButton("擷取螢幕")
        self.capture_btn.clicked.connect(self.capture_screen)
        toolbar_layout.addWidget(self.capture_btn)

        self.capture_android_btn = QPushButton("擷取手機畫面")
        self.capture_android_btn.clicked.connect(self.capture_android_screen)
        toolbar_layout.addWidget(self.capture_android_btn)
        
        self.export_btn = QPushButton("匯出 JSON")
        self.export_btn.clicked.connect(self.export_json)
        toolbar_layout.addWidget(self.export_btn)

        self.import_btn = QPushButton("匯入 JSON")
        self.import_btn.clicked.connect(self.import_json)
        toolbar_layout.addWidget(self.import_btn)

        self.cleanup_images_btn = QPushButton("清理未引用圖片")
        self.cleanup_images_btn.clicked.connect(self.cleanup_unreferenced_images)
        toolbar_layout.addWidget(self.cleanup_images_btn)
        
        toolbar_layout.addWidget(QLabel("基準解析度:"))
        
        self.res_width = QSpinBox()
        self.res_width.setRange(100, 10000)
        self.res_width.setValue(self.base_resolution[0])
        self.res_width.setPrefix("寬: ")
        self.res_width.valueChanged.connect(self.on_resolution_changed)
        toolbar_layout.addWidget(self.res_width)
        
        self.res_height = QSpinBox()
        self.res_height.setRange(100, 10000)
        self.res_height.setValue(self.base_resolution[1])
        self.res_height.setPrefix("高: ")
        self.res_height.valueChanged.connect(self.on_resolution_changed)
        toolbar_layout.addWidget(self.res_height)
        
        toolbar_layout.addStretch()
        
        layout.addWidget(toolbar)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.tab_widget.tabCloseRequested.connect(self.close_state_tab)
        layout.addWidget(self.tab_widget)
        
        self.add_state_tab()
        
    def add_state_tab(self, state: Optional[State] = None):
        if state is None:
            state_num = len(self.states) + 1
            state = State(name=f"狀態{state_num}", image_path="")
            self.states.append(state)
        
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        graphics_view = AutoFitGraphicsView()
        scene = MarkerGraphicsScene()
        graphics_view.setScene(scene)
        graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        graphics_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        graphics_view.resized.connect(self.fit_image_to_view)
        scene.markerAdded.connect(self.on_scene_marker_added)
        scene.markerRightClicked.connect(self.on_scene_marker_right_clicked)
        scene.markerEditConfirmed.connect(self.on_marker_edit_confirmed)
        scene.set_target_resolution(self.base_resolution[0], self.base_resolution[1])
        left_layout.addWidget(graphics_view)
        
        layout.addWidget(left_panel, 3)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        state_name_group = QGroupBox("狀態名稱")
        state_name_layout = QHBoxLayout(state_name_group)
        state_name_edit = QLineEdit(state.name)
        state_name_edit.textChanged.connect(self.on_state_name_changed)
        state_name_layout.addWidget(state_name_edit)
        right_layout.addWidget(state_name_group)

        state_desc_group = QGroupBox("狀態描述")
        state_desc_layout = QVBoxLayout(state_desc_group)
        state_desc_edit = QLineEdit(state.description)
        state_desc_edit.setPlaceholderText("輸入此狀態的描述")
        state_desc_edit.textChanged.connect(self.on_state_description_changed)
        state_desc_layout.addWidget(state_desc_edit)
        right_layout.addWidget(state_desc_group)
        
        marker_group = QGroupBox("標定點")
        marker_layout = QVBoxLayout(marker_group)
        
        marker_list = QListWidget()
        marker_list.itemClicked.connect(self.on_marker_clicked)
        marker_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        marker_list.customContextMenuRequested.connect(self.on_marker_list_context_menu)
        marker_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        marker_layout.addWidget(marker_list)
        
        marker_btn_layout = QHBoxLayout()
        
        add_marker_btn = QPushButton("新增標定點")
        add_marker_btn.clicked.connect(self.add_marker)
        marker_btn_layout.addWidget(add_marker_btn)
        
        edit_marker_btn = QPushButton("編輯")
        edit_marker_btn.clicked.connect(self.edit_marker)
        marker_btn_layout.addWidget(edit_marker_btn)
        
        delete_marker_btn = QPushButton("刪除")
        delete_marker_btn.clicked.connect(self.delete_marker)
        marker_btn_layout.addWidget(delete_marker_btn)

        move_markers_btn = QPushButton("平移多選")
        move_markers_btn.clicked.connect(self.translate_selected_markers)
        marker_btn_layout.addWidget(move_markers_btn)

        copy_markers_btn = QPushButton("從某狀態複製")
        copy_markers_btn.clicked.connect(self.copy_markers_from_state)
        marker_btn_layout.addWidget(copy_markers_btn)
        
        marker_layout.addLayout(marker_btn_layout)
        right_layout.addWidget(marker_group)
        
        new_screenshot_btn = QPushButton("新截圖到此狀態")
        new_screenshot_btn.clicked.connect(self.capture_to_current_state)
        right_layout.addWidget(new_screenshot_btn)

        android_screenshot_btn = QPushButton("手機截圖到此狀態")
        android_screenshot_btn.clicked.connect(self.capture_android_to_current_state)
        right_layout.addWidget(android_screenshot_btn)
        
        layout.addWidget(right_panel, 1)
        
        idx = self.tab_widget.addTab(widget, state.name)
        widget._tab_context = StateTabContext(
            graphics_view=graphics_view,
            scene=scene,
            state_name_edit=state_name_edit,
            state_desc_edit=state_desc_edit,
            marker_list=marker_list
        )
        self.tab_widget.setCurrentIndex(idx)
        self.current_state_index = self.states.index(state)
        self.on_tab_changed(idx)

    def get_tab_context(self, index: Optional[int] = None) -> Optional[StateTabContext]:
        if index is None:
            index = self.tab_widget.currentIndex()
        if index < 0:
            return None

        widget = self.tab_widget.widget(index)
        if not widget:
            return None

        return getattr(widget, "_tab_context", None)

    def on_tab_changed(self, index):
        if index < 0 or index >= len(self.states):
            self.current_state_index = -1
            return

        self.current_state_index = index
        self.load_state(self.states[index], index)

    def on_scene_marker_added(self):
        self.update_marker_list()
        
    def close_state_tab(self, index):
        if self.tab_widget.count() <= 1:
            return
        
        self.tab_widget.removeTab(index)
        if index < len(self.states):
            self.states.pop(index)
        
        if self.tab_widget.count() > 0:
            self.on_tab_changed(self.tab_widget.currentIndex())
    
    def capture_screen(self):
        pixmap = ScreenCapture.capture_gnome_interactive(self)

        if ScreenCapture.last_capture_message:
            if ScreenCapture.last_capture_message_level == "info":
                QMessageBox.information(self, "提示", ScreenCapture.last_capture_message)
            else:
                QMessageBox.warning(self, "截圖失敗", ScreenCapture.last_capture_message)

        if pixmap:
            self.add_new_state_with_screenshot(pixmap)

    def select_android_device(self) -> Optional[str]:
        devices = ScreenCapture.get_android_devices()
        if ScreenCapture.last_capture_message:
            QMessageBox.warning(self, "Android 裝置", ScreenCapture.last_capture_message)
            return None

        online_devices = [device for device in devices if device.get("status") == "device"]
        if not online_devices:
            QMessageBox.warning(self, "Android 裝置", "沒有偵測到可用的 Android 裝置。")
            return None

        if len(online_devices) == 1:
            return online_devices[0].get("serial")

        dialog = AndroidDeviceSelectionDialog(online_devices, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected = dialog.get_selected_device()
            if selected:
                return selected.get("serial")
        return None

    def capture_android_screen(self):
        if self.is_android_capture_running:
            QMessageBox.information(self, "Android 截圖", "目前正在擷取中，請稍候。")
            return

        device_serial = self.select_android_device()
        if not device_serial:
            return
        self.start_android_capture(device_serial, None)
    
    def capture_to_current_state(self):
        pixmap = ScreenCapture.capture_gnome_interactive(self)

        if ScreenCapture.last_capture_message:
            if ScreenCapture.last_capture_message_level == "info":
                QMessageBox.information(self, "提示", ScreenCapture.last_capture_message)
            else:
                QMessageBox.warning(self, "截圖失敗", ScreenCapture.last_capture_message)

        if pixmap and 0 <= self.current_state_index < len(self.states):
            self.save_screenshot_to_state(pixmap, self.states[self.current_state_index])

    def capture_android_to_current_state(self):
        if not (0 <= self.current_state_index < len(self.states)):
            return

        if self.is_android_capture_running:
            QMessageBox.information(self, "Android 截圖", "目前正在擷取中，請稍候。")
            return

        device_serial = self.select_android_device()
        if not device_serial:
            return

        self.start_android_capture(device_serial, self.current_state_index)

    def start_android_capture(self, device_serial: str, target_index: Optional[int]):
        self.is_android_capture_running = True
        self.android_capture_target_index = target_index
        self.capture_android_btn.setEnabled(False)

        self.capture_progress_dialog = QProgressDialog("正在從 Android 擷取畫面...", None, 0, 0, self)
        self.capture_progress_dialog.setWindowTitle("請稍候")
        self.capture_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.capture_progress_dialog.setMinimumDuration(0)
        self.capture_progress_dialog.setAutoClose(False)
        self.capture_progress_dialog.setAutoReset(False)
        self.capture_progress_dialog.show()

        self.android_capture_thread = QThread(self)
        self.android_capture_worker = AndroidCaptureWorker(device_serial)
        self.android_capture_worker.moveToThread(self.android_capture_thread)

        self.android_capture_thread.started.connect(self.android_capture_worker.run)
        self.android_capture_worker.finished.connect(self.on_android_capture_finished)
        self.android_capture_worker.finished.connect(self.android_capture_thread.quit)
        self.android_capture_worker.finished.connect(self.android_capture_worker.deleteLater)
        self.android_capture_thread.finished.connect(self.android_capture_thread.deleteLater)

        self.android_capture_thread.start()

    def on_android_capture_finished(self, image_data, message_level: str, message: str):
        if self.capture_progress_dialog:
            self.capture_progress_dialog.close()
            self.capture_progress_dialog.deleteLater()
            self.capture_progress_dialog = None

        self.is_android_capture_running = False
        self.capture_android_btn.setEnabled(True)

        self.android_capture_worker = None
        self.android_capture_thread = None

        if message:
            if message_level == "info":
                QMessageBox.information(self, "Android 截圖", message)
            else:
                QMessageBox.warning(self, "Android 截圖", message)
            return

        pixmap = QPixmap()
        if not image_data or not pixmap.loadFromData(image_data, "PNG"):
            QMessageBox.warning(self, "Android 截圖", "Android 螢幕資料格式無法解析為 PNG。")
            return

        if self.android_capture_target_index is None:
            self.add_new_state_with_screenshot(pixmap)
            return

        target_index = self.android_capture_target_index
        if 0 <= target_index < len(self.states):
            self.save_screenshot_to_state(pixmap, self.states[target_index])

    def keyPressEvent(self, event):
        # Handle ESC to cancel edit mode
        if event.key() == Qt.Key.Key_Escape:
            context = self.get_tab_context()
            if context and context.scene.edit_mode:
                context.scene.cancel_edit_mode()
                if hasattr(self, '_edit_pending_a'):
                    self._edit_pending_a = None
                if hasattr(self, '_pending_b_for_edit'):
                    self._pending_b_for_edit = None
                QMessageBox.information(self, "取消編輯", "已取消編輯模式。")
                return
        super().keyPressEvent(event)
    
    def closeEvent(self, event):
        if self.is_android_capture_running:
            QMessageBox.information(self, "請稍候", "正在從 Android 擷取畫面，請等待完成後再關閉。")
            event.ignore()
            return

        super().closeEvent(event)
    
    def add_new_state_with_screenshot(self, pixmap: QPixmap):
        state_num = len(self.states) + 1
        state = State(name=f"狀態{state_num}", image_path="")
        self.states.append(state)
        
        self.save_screenshot_to_state(pixmap, state, refresh_ui=False)
        self.add_state_tab(state)
        
    def save_screenshot_to_state(self, pixmap: QPixmap, state: State, refresh_ui: bool = True):
        os.makedirs("images", exist_ok=True)

        self._sync_base_resolution_with_pixmap(pixmap)

        managed_images_dir = os.path.abspath("images")
        old_image_path = state.image_path
        old_image_abs = os.path.abspath(old_image_path) if old_image_path else ""

        if old_image_abs and self._is_path_inside(old_image_abs, managed_images_dir):
            image_path_abs = old_image_abs
        else:
            try:
                state_index = self.states.index(state) + 1
            except ValueError:
                state_index = len(self.states)
            safe_state_name = self._sanitize_filename(state.name)
            file_name = f"{state_index:02d}_{safe_state_name}.png"
            image_path_abs = self._build_unique_path(managed_images_dir, file_name)

        pixmap.save(image_path_abs)
        state.image_path = os.path.relpath(image_path_abs, os.getcwd())

        if old_image_abs and old_image_abs != image_path_abs:
            self._remove_managed_image(old_image_abs)

        if refresh_ui and 0 <= self.current_state_index < len(self.states):
            if self.states[self.current_state_index] is state:
                self.load_state(state, self.current_state_index)

    def _is_path_inside(self, path: str, directory: str) -> bool:
        try:
            return os.path.commonpath([os.path.abspath(path), os.path.abspath(directory)]) == os.path.abspath(directory)
        except ValueError:
            return False

    def _remove_managed_image(self, path: str):
        managed_images_dir = os.path.abspath("images")
        abs_path = os.path.abspath(path)
        if not self._is_path_inside(abs_path, managed_images_dir):
            return
        if not os.path.isfile(abs_path):
            return
        try:
            os.remove(abs_path)
        except Exception:
            pass

    def cleanup_unreferenced_images(self):
        default_path = self.current_json_path if self.current_json_path and os.path.exists(self.current_json_path) else "markers.json"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇要清理的 JSON",
            default_path,
            "JSON Files (*.json)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "清理失敗", f"無法讀取 JSON：{e}")
            return

        raw_states = data.get("states", [])
        if not isinstance(raw_states, list):
            QMessageBox.warning(self, "清理失敗", "JSON 格式錯誤：缺少 states 陣列。")
            return

        base_dir = os.path.dirname(os.path.abspath(file_path))
        referenced_images = set()
        candidate_dirs = set()

        for raw_state in raw_states:
            if not isinstance(raw_state, dict):
                continue
            raw_image_path = str(raw_state.get("image_path") or "").strip()
            if not raw_image_path:
                continue

            if os.path.isabs(raw_image_path):
                abs_image_path = os.path.normpath(raw_image_path)
            else:
                abs_image_path = os.path.normpath(os.path.join(base_dir, raw_image_path))

            referenced_images.add(abs_image_path)
            candidate_dirs.add(os.path.dirname(abs_image_path))

        json_name = os.path.splitext(os.path.basename(file_path))[0]
        default_images_dir = os.path.join(base_dir, f"{json_name}_images")
        if os.path.isdir(default_images_dir):
            candidate_dirs.add(default_images_dir)

        candidate_dirs = {d for d in candidate_dirs if d and os.path.isdir(d)}
        if not candidate_dirs:
            QMessageBox.information(self, "清理未引用圖片", "此 JSON 沒有可清理的圖片資料夾。")
            return

        answer = QMessageBox.question(
            self,
            "清理未引用圖片",
            f"將依據\n{file_path}\n清理未被 image_path 引用的圖片檔，是否繼續？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        removed_files = []
        kept_count = 0
        failed_files = []

        for directory in sorted(candidate_dirs):
            try:
                entries = os.listdir(directory)
            except Exception:
                continue

            for entry in entries:
                file_path_abs = os.path.normpath(os.path.join(directory, entry))
                if not os.path.isfile(file_path_abs):
                    continue
                _, ext = os.path.splitext(entry)
                if ext.lower() not in image_extensions:
                    continue
                if file_path_abs in referenced_images:
                    kept_count += 1
                    continue

                try:
                    os.remove(file_path_abs)
                    removed_files.append(os.path.relpath(file_path_abs, base_dir))
                except Exception:
                    failed_files.append(os.path.relpath(file_path_abs, base_dir))

        summary = [
            f"保留：{kept_count} 張",
            f"刪除：{len(removed_files)} 張"
        ]
        if failed_files:
            summary.append(f"刪除失敗：{len(failed_files)} 張")
        if removed_files:
            summary.append("\n已刪除檔案：\n" + "\n".join(removed_files[:30]))
            if len(removed_files) > 30:
                summary.append(f"... 其餘 {len(removed_files) - 30} 張")
        if failed_files:
            summary.append("\n刪除失敗檔案：\n" + "\n".join(failed_files[:30]))

        self.current_json_path = file_path
        QMessageBox.information(self, "清理完成", "\n".join(summary))

    def _sync_base_resolution_with_pixmap(self, pixmap: QPixmap):
        width = int(pixmap.width())
        height = int(pixmap.height())
        if width <= 0 or height <= 0:
            return

        if self.base_resolution == (width, height):
            return

        self.base_resolution = (width, height)

        self.res_width.blockSignals(True)
        self.res_height.blockSignals(True)
        self.res_width.setValue(width)
        self.res_height.setValue(height)
        self.res_width.blockSignals(False)
        self.res_height.blockSignals(False)

        for tab_index in range(self.tab_widget.count()):
            context = self.get_tab_context(tab_index)
            if not context:
                continue
            context.scene.set_target_resolution(width, height)
            context.scene.update_markers_display()

        self.update_marker_list()
        self.fit_image_to_view()
    
    def fit_image_to_view(self):
        context = self.get_tab_context()
        if not context or not context.scene.image_item:
            return

        image_rect = context.scene.image_item.sceneBoundingRect()
        if image_rect.isEmpty():
            return

        context.graphics_view.resetTransform()
        context.graphics_view.fitInView(
                image_rect,
                Qt.AspectRatioMode.KeepAspectRatio
            )
        context.graphics_view.centerOn(image_rect.center())
    
    def load_state(self, state: State, tab_index: Optional[int] = None):
        context = self.get_tab_context(tab_index)
        if not context:
            return

        self.is_loading_state = True
        context.state_name_edit.blockSignals(True)
        context.state_desc_edit.blockSignals(True)
        context.state_name_edit.setText(state.name)
        context.state_desc_edit.setText(state.description)
        context.state_name_edit.blockSignals(False)
        context.state_desc_edit.blockSignals(False)
        
        if state.image_path and os.path.exists(state.image_path):
            pixmap = QPixmap(state.image_path)

            context.scene.set_image(pixmap)
            context.scene.set_target_resolution(
                self.base_resolution[0],
                self.base_resolution[1]
            )

            self.fit_image_to_view()
        else:
            # Use placeholder with resolution info even when image is missing
            context.scene.set_placeholder_with_resolution(
                self.base_resolution[0],
                self.base_resolution[1]
            )
            # Still load markers so they can be viewed/edited
            context.scene.markers = list(state.markers)
        
        self.update_marker_list(tab_index)
        self.is_loading_state = False
        
    def update_marker_list(self, tab_index: Optional[int] = None):
        context = self.get_tab_context(tab_index)
        if not context:
            return

        context.marker_list.clear()
        state_index = self.current_state_index if tab_index is None else tab_index
        if 0 <= state_index < len(self.states):
            state = self.states[state_index]
            for marker in state.markers:
                target_text = f" -> {marker.target_state}" if marker.target_state else ""
                if marker.marker_type == "swipe":
                    context.marker_list.addItem(
                        f"{marker.name} [滑動] A({marker.x}, {marker.y}) -> B({marker.bx}, {marker.by}){target_text}"
                    )
                else:
                    context.marker_list.addItem(f"{marker.name} [點擊] ({marker.x}, {marker.y}){target_text}")
            
            context.scene.markers = state.markers
            context.scene.update_markers_display()
    
    def on_state_name_changed(self, text):
        if self.is_loading_state:
            return

        if 0 <= self.current_state_index < len(self.states):
            old_name = self.states[self.current_state_index].name
            self.states[self.current_state_index].name = text
            self.tab_widget.setTabText(self.tab_widget.currentIndex(), text)

            if old_name != text:
                for state in self.states:
                    for marker in state.markers:
                        if marker.target_state == old_name:
                            marker.target_state = text
                self.update_marker_list()

    def on_state_description_changed(self, text):
        if self.is_loading_state:
            return

        if 0 <= self.current_state_index < len(self.states):
            self.states[self.current_state_index].description = text
    
    def on_resolution_changed(self):
        if self.is_loading_state:
            return

        width = self.res_width.value()
        height = self.res_height.value()
        self.base_resolution = (width, height)

        for tab_index in range(self.tab_widget.count()):
            context = self.get_tab_context(tab_index)
            if not context:
                continue
            context.scene.set_target_resolution(width, height)
            context.scene.update_markers_display()

        context = self.get_tab_context()
        if not context:
            return

        self.update_marker_list()
        self.fit_image_to_view()
    
    def on_marker_clicked(self, item):
        context = self.get_tab_context()
        if not context:
            return

        idx = context.marker_list.row(item)
        if 0 <= self.current_state_index < len(self.states):
            state = self.states[self.current_state_index]
            if 0 <= idx < len(state.markers):
                marker = state.markers[idx]
                if context.scene.image_item:
                    scale_x = context.scene.sceneRect().width() / context.scene.target_resolution[0]
                    scale_y = context.scene.sceneRect().height() / context.scene.target_resolution[1]
                    x = marker.x * scale_x
                    y = marker.y * scale_y
                    context.graphics_view.centerOn(x, y)

    def _find_tab_index_by_marker_list(self, marker_list: QListWidget) -> int:
        for index in range(self.tab_widget.count()):
            context = self.get_tab_context(index)
            if context and context.marker_list is marker_list:
                return index
        return -1

    def on_marker_list_context_menu(self, pos):
        marker_list = self.sender()
        if not isinstance(marker_list, QListWidget):
            return

        row = marker_list.indexAt(pos).row()
        if row < 0:
            return

        tab_index = self._find_tab_index_by_marker_list(marker_list)
        if tab_index < 0 or tab_index >= len(self.states):
            return

        menu = QMenu(self)
        set_target_action = menu.addAction("設定跳轉狀態")
        selected_action = menu.exec(marker_list.mapToGlobal(pos))
        if selected_action is set_target_action:
            self.set_marker_target_state(tab_index, row)

    def set_marker_target_state(self, state_index: int, marker_index: int):
        if not (0 <= state_index < len(self.states)):
            return

        state = self.states[state_index]
        if not (0 <= marker_index < len(state.markers)):
            return

        marker = state.markers[marker_index]
        options = ["不跳轉"] + [s.name for s in self.states]
        current_target = marker.target_state if marker.target_state in options else "不跳轉"
        current_index = options.index(current_target)

        selected, ok = QInputDialog.getItem(
            self,
            "設定跳轉狀態",
            f"標定點「{marker.name}」跳轉到:",
            options,
            current_index,
            False
        )
        if not ok:
            return

        marker.target_state = None if selected == "不跳轉" else selected
        if self.current_state_index == state_index:
            self.update_marker_list()
        else:
            self.update_marker_list(state_index)

    def on_scene_marker_right_clicked(self, marker_index: int):
        if not (0 <= self.current_state_index < len(self.states)):
            return
        self.set_marker_target_state(self.current_state_index, marker_index)
    
    def add_marker(self):
        dialog = MarkerInputDialog(0, 0, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name, x, y, marker_type, bx, by = dialog.get_values()
            if name and 0 <= self.current_state_index < len(self.states):
                marker = Marker(name=name, x=x, y=y, marker_type=marker_type, bx=bx, by=by, target_state=None)
                self.states[self.current_state_index].markers.append(marker)
                self.update_marker_list()
    
    def edit_marker(self):
        context = self.get_tab_context()
        if not context:
            return

        idx = context.marker_list.currentRow()
        if 0 <= self.current_state_index < len(self.states):
            state = self.states[self.current_state_index]
            if 0 <= idx < len(state.markers):
                marker = state.markers[idx]
                if marker.marker_type == "swipe":
                    QMessageBox.information(
                        self,
                        "編輯模式",
                        f"已进入编辑模式！\n\n"
                        f"标定点「{marker.name}」是滑動類型。\n"
                        f"• 第一次點擊：設定 A 點（新座標）\n"
                        f"• 第二次點擊：設定 B 點（結束座標）\n\n"
                        f"按 ESC 可取消編輯。"
                    )
                else:
                    QMessageBox.information(
                        self,
                        "編輯模式",
                        f"已进入编辑模式！\n\n"
                        f"标定点「{marker.name}」是點擊類型。\n"
                        f"直接點擊畫面設定新座標。\n\n"
                        f"按 ESC 可取消編輯。"
                    )
                context.scene.start_edit_mode(idx)
    
    def on_marker_edit_confirmed(self, marker_index: int, ax: int, ay: int, bx: int, by: int):
        """Handle marker edit confirmation from canvas click."""
        if not (0 <= self.current_state_index < len(self.states)):
            return
        state = self.states[self.current_state_index]
        if not (0 <= marker_index < len(state.markers)):
            return
        
        marker = state.markers[marker_index]
        if marker.marker_type == "swipe":
            # Update both A and B points
            marker.x = ax
            marker.y = ay
            marker.bx = bx
            marker.by = by
        else:
            # For tap, just update the single point
            marker.x = ax
            marker.y = ay
        self.update_marker_list()
    
    def delete_marker(self):
        context = self.get_tab_context()
        if not context:
            return

        idx = context.marker_list.currentRow()
        if 0 <= self.current_state_index < len(self.states):
            state = self.states[self.current_state_index]
            if 0 <= idx < len(state.markers):
                state.markers.pop(idx)
                self.update_marker_list()

    def translate_selected_markers(self):
        context = self.get_tab_context()
        if not context:
            return

        if not (0 <= self.current_state_index < len(self.states)):
            return

        selected_rows = sorted(
            {
                context.marker_list.row(item)
                for item in context.marker_list.selectedItems()
                if context.marker_list.row(item) >= 0
            }
        )
        if not selected_rows:
            QMessageBox.warning(self, "平移標記", "請先在標定點清單多選至少一個標記。")
            return

        offset_x, ok = QInputDialog.getInt(
            self,
            "平移標記",
            "X 偏移量（可負數）:",
            0,
            -100000,
            100000,
            1
        )
        if not ok:
            return

        offset_y, ok = QInputDialog.getInt(
            self,
            "平移標記",
            "Y 偏移量（可負數）:",
            0,
            -100000,
            100000,
            1
        )
        if not ok:
            return

        state = self.states[self.current_state_index]
        moved_count = 0
        for row in selected_rows:
            if not (0 <= row < len(state.markers)):
                continue
            marker = state.markers[row]
            marker.x += offset_x
            marker.y += offset_y
            if marker.marker_type == "swipe":
                marker.bx += offset_x
                marker.by += offset_y
            moved_count += 1

        if moved_count <= 0:
            return

        self.update_marker_list()

    def copy_markers_from_state(self):
        if not (0 <= self.current_state_index < len(self.states)):
            return

        source_candidates = []
        for index, state in enumerate(self.states):
            if index == self.current_state_index:
                continue
            source_candidates.append((index, state))

        if not source_candidates:
            QMessageBox.warning(self, "複製標記", "目前沒有其他可複製的狀態。")
            return

        option_labels = [
            f"{index + 1}. {state.name} ({len(state.markers)} 個標記)"
            for index, state in source_candidates
        ]

        selected_label, ok = QInputDialog.getItem(
            self,
            "從某狀態複製",
            "請選擇來源狀態:",
            option_labels,
            0,
            False
        )
        if not ok or not selected_label:
            return

        selected_pos = option_labels.index(selected_label)
        _, source_state = source_candidates[selected_pos]
        target_state = self.states[self.current_state_index]

        answer = QMessageBox.question(
            self,
            "複製標記",
            f"要以「{source_state.name}」的標記覆蓋目前狀態「{target_state.name}」嗎？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        source_marker_dict = {m.name: m for m in source_state.markers}
        target_marker_dict = {m.name: m for m in target_state.markers}
        
        # If target is empty, copy all from source
        if not target_state.markers:
            target_state.markers = [
                Marker(
                    name=marker.name,
                    x=marker.x,
                    y=marker.y,
                    marker_type=marker.marker_type,
                    bx=marker.bx,
                    by=marker.by,
                    target_state=marker.target_state
                )
                for marker in source_state.markers
            ]
            self.update_marker_list()
            QMessageBox.information(
                self,
                "複製標記",
                f"目標狀態為空，已複製來源的所有 {len(target_state.markers)} 個標記。"
            )
            return
        
        # Otherwise, only overwrite markers where name matches
        matched_names = set(source_marker_dict.keys()) & set(target_marker_dict.keys())
        updated_count = 0
        
        for name in matched_names:
            source_marker = source_marker_dict[name]
            target_marker_dict[name].x = source_marker.x
            target_marker_dict[name].y = source_marker.y
            target_marker_dict[name].bx = source_marker.bx
            target_marker_dict[name].by = source_marker.by
            target_marker_dict[name].marker_type = source_marker.marker_type
            updated_count += 1
        
        if updated_count > 0:
            self.update_marker_list()
            QMessageBox.information(
                self,
                "複製標記",
                f"已更新 {updated_count} 個同名標記（只覆蓋名字相同的）。"
            )
        else:
            QMessageBox.information(
                self,
                "複製標記",
                f"來源狀態與目標狀態沒有同名的標記。\n\n"
                f"來源：{list(source_marker_dict.keys())}\n"
                f"目標：{list(target_marker_dict.keys())}"
            )

    def _sanitize_filename(self, name: str) -> str:
        invalid_chars = '<>:"/\\|?*'
        safe_name = ''.join('_' if char in invalid_chars else char for char in name.strip())
        safe_name = safe_name.rstrip('.')
        return safe_name or "state"

    def _build_unique_path(self, directory: str, filename: str) -> str:
        base_name, ext = os.path.splitext(filename)
        candidate = os.path.join(directory, filename)
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(directory, f"{base_name}_{counter}{ext}")
            counter += 1
        return candidate

    def _load_states_from_data(self, data: Dict[str, Any], base_dir: str) -> List[State]:
        raw_states = data.get("states")
        if not isinstance(raw_states, list):
            raise ValueError("JSON 格式錯誤：缺少 states 陣列")

        loaded_states: List[State] = []
        for idx, raw_state in enumerate(raw_states, start=1):
            if not isinstance(raw_state, dict):
                continue

            state_name = str(raw_state.get("name") or f"狀態{idx}")
            state_description = str(raw_state.get("description") or "")
            raw_image_path = str(raw_state.get("image_path") or "").strip()

            resolved_image_path = ""
            if raw_image_path:
                if os.path.isabs(raw_image_path):
                    candidate_path = raw_image_path
                else:
                    candidate_path = os.path.join(base_dir, raw_image_path)
                if os.path.exists(candidate_path):
                    resolved_image_path = os.path.normpath(candidate_path)

            markers: List[Marker] = []
            raw_markers = raw_state.get("markers", [])
            if isinstance(raw_markers, list):
                for marker_idx, raw_marker in enumerate(raw_markers, start=1):
                    if not isinstance(raw_marker, dict):
                        continue
                    marker_name = str(raw_marker.get("name") or f"標定點{marker_idx}")
                    marker_target_state = raw_marker.get("target_state")
                    if marker_target_state is not None:
                        marker_target_state = str(marker_target_state)
                    raw_type = str(raw_marker.get("type") or "").strip().lower()
                    has_b_point = ("bx" in raw_marker and "by" in raw_marker)
                    marker_type = "swipe" if raw_type in ["swipe", "slide", "滑動"] or has_b_point else "tap"
                    try:
                        marker_x = int(round(float(raw_marker.get("x", 0))))
                        marker_y = int(round(float(raw_marker.get("y", 0))))
                    except (TypeError, ValueError):
                        marker_x, marker_y = 0, 0

                    marker_bx = None
                    marker_by = None
                    if marker_type == "swipe":
                        try:
                            marker_bx = int(round(float(raw_marker.get("bx", marker_x))))
                            marker_by = int(round(float(raw_marker.get("by", marker_y))))
                        except (TypeError, ValueError):
                            marker_bx, marker_by = marker_x, marker_y

                    markers.append(
                        Marker(
                            name=marker_name,
                            x=marker_x,
                            y=marker_y,
                            marker_type=marker_type,
                            bx=marker_bx,
                            by=marker_by,
                            target_state=marker_target_state
                        )
                    )

            loaded_states.append(
                State(
                    name=state_name,
                    image_path=resolved_image_path,
                    description=state_description,
                    markers=markers
                )
            )

        return loaded_states

    def _parse_base_resolution(self, data: Dict[str, Any]) -> tuple[int, int]:
        raw = data.get("base_resolution")
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            try:
                width = int(round(float(raw[0])))
                height = int(round(float(raw[1])))
                if width > 0 and height > 0:
                    return width, height
            except (TypeError, ValueError):
                pass

        if isinstance(raw, dict):
            try:
                width = int(round(float(raw.get("width", 0))))
                height = int(round(float(raw.get("height", 0))))
                if width > 0 and height > 0:
                    return width, height
            except (TypeError, ValueError):
                pass

        return self.base_resolution

    def _replace_all_states(self, new_states: List[State]):
        self.is_loading_state = True
        self.tab_widget.blockSignals(True)
        self.tab_widget.clear()
        self.states = []
        self.current_state_index = -1

        if new_states:
            self.states = new_states
            for state in self.states:
                self.add_state_tab(state)
            self.tab_widget.setCurrentIndex(0)
            self.on_tab_changed(0)
        else:
            self.add_state_tab()

        self.tab_widget.blockSignals(False)
        self.is_loading_state = False
    
    def export_json(self):
        if not self.states:
            QMessageBox.warning(self, "警告", "沒有資料可匯出")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "匯出 JSON", "markers.json", "JSON Files (*.json)"
        )
        
        if file_path:
            self.current_json_path = file_path
            export_dir = os.path.dirname(file_path)
            export_name = os.path.splitext(os.path.basename(file_path))[0]
            images_dir = os.path.join(export_dir, f"{export_name}_images")
            os.makedirs(images_dir, exist_ok=True)

            exported_states = []
            missing_images = []

            for index, state in enumerate(self.states, start=1):
                exported_state = {
                    "name": state.name,
                    "description": state.description,
                    "image_path": "",
                    "markers": [m.to_dict() for m in state.markers]
                }
                
                if state.image_path:
                    source_path = state.image_path
                    if not os.path.isabs(source_path):
                        source_path = os.path.abspath(source_path)

                    if os.path.exists(source_path):
                        source_ext = os.path.splitext(source_path)[1].lower() or ".png"
                        safe_state_name = self._sanitize_filename(state.name)
                        new_file_name = f"{index:02d}_{safe_state_name}{source_ext}"
                        destination_path = self._build_unique_path(images_dir, new_file_name)
                        shutil.copy2(source_path, destination_path)
                        exported_state["image_path"] = os.path.relpath(destination_path, export_dir)
                    else:
                        missing_images.append(state.name)
                
                exported_states.append(exported_state)

            output = {
                "base_resolution": [self.base_resolution[0], self.base_resolution[1]],
                "states": exported_states
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            if missing_images:
                QMessageBox.warning(
                    self,
                    "部分匯出",
                    f"已匯出至 {file_path}\n\n以下狀態的圖片找不到，已匯出但無圖片：\n" + "\n".join(missing_images)
                )
            else:
                QMessageBox.information(self, "成功", f"已匯出至 {file_path}")

    def import_json(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "匯入 JSON", "", "JSON Files (*.json)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "匯入失敗", f"無法讀取 JSON：{e}")
            return

        self.base_resolution = self._parse_base_resolution(data)
        self.res_width.blockSignals(True)
        self.res_height.blockSignals(True)
        self.res_width.setValue(self.base_resolution[0])
        self.res_height.setValue(self.base_resolution[1])
        self.res_width.blockSignals(False)
        self.res_height.blockSignals(False)

        try:
            loaded_states = self._load_states_from_data(data, os.path.dirname(file_path))
        except ValueError as e:
            QMessageBox.warning(self, "匯入失敗", str(e))
            return

        if not loaded_states:
            QMessageBox.warning(self, "匯入失敗", "JSON 內沒有可用的狀態資料。")
            return

        self.current_json_path = file_path

        missing_image_states = [state.name for state in loaded_states if not state.image_path]
        self._replace_all_states(loaded_states)

        if missing_image_states:
            QMessageBox.warning(
                self,
                "匯入完成（部分圖片遺失）",
                "已匯入狀態與標定點，但以下狀態找不到圖片：\n" + "\n".join(missing_image_states)
            )
        else:
            QMessageBox.information(self, "匯入成功", f"已匯入 {len(loaded_states)} 個狀態。")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
