import json
import os
import queue
import sys
from pathlib import Path
import customtkinter as ctk
import numpy as np
import sounddevice as sd


APP_TITLE = "延遲語音回授 (DAF)"
DEFAULT_SETTINGS = {
    "theme": "dark",
    "input_device_name": "",
    "output_device_name": "",
    "delay_ms": 150,
    "volume_percent": 100,
}

# 現代極簡扁平化配色 (淺色, 深色)
COLORS = {
    "bg": ("#F4F5F7", "#111317"),
    "card": ("#FFFFFF", "#1B1E24"),
    "card_border": ("#E5E7EB", "#2A2E37"),
    "input_bg": ("#F9FAFB", "#232730"),
    "input_border": ("#D1D5DB", "#343A46"),
    "text": ("#111827", "#F3F4F6"),
    "text_muted": ("#6B7280", "#9CA3AF"),
    "accent": ("#10B981", "#10B981"),
    "accent_hover": ("#059669", "#059669"),
    "danger": ("#EF4444", "#EF4444"),
    "danger_hover": ("#DC2626", "#DC2626"),
    "subtle_btn": ("#F3F4F6", "#262A34"),
    "subtle_btn_hover": ("#E5E7EB", "#323744"),
    "seg_bg": ("#E5E7EB", "#232730"),
    "seg_selected": ("#FFFFFF", "#323846"),
    "seg_selected_hover": ("#F9FAFB", "#3B4252"),
}

FONT_FAMILY = "Microsoft JhengHei UI"


def get_settings_path() -> Path:
    """取得設定檔路徑，優先使用 APPDATA，若無法寫入則退回程式所在目錄。"""
    candidates: list[Path] = []
    base_dir = os.environ.get("APPDATA")
    if base_dir:
        candidates.append(Path(base_dir) / "DAF" / "settings.json")
    candidates.append(Path.home() / ".config" / "DAF" / "settings.json")

    app_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    candidates.append(app_dir / "daf_settings.json")

    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if os.access(path.parent, os.W_OK):
                return path
        except OSError:
            continue
    return candidates[-1]


def load_settings() -> dict:
    """讀取上次儲存的設定，若不存在或格式錯誤則回傳預設值。"""
    settings = DEFAULT_SETTINGS.copy()
    path = get_settings_path()
    if not path.exists():
        return settings
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if data.get("theme") in ("light", "dark"):
                settings["theme"] = data["theme"]
            if isinstance(data.get("input_device_name"), str):
                settings["input_device_name"] = data["input_device_name"]
            if isinstance(data.get("output_device_name"), str):
                settings["output_device_name"] = data["output_device_name"]
            delay_val = data.get("delay_ms")
            if isinstance(delay_val, (int, float)) and 0 <= delay_val <= 5000:
                settings["delay_ms"] = int(delay_val) if float(delay_val).is_integer() else float(delay_val)
            vol_val = data.get("volume_percent")
            if isinstance(vol_val, (int, float)) and 0 <= vol_val <= 200:
                settings["volume_percent"] = int(vol_val)
    except Exception:
        pass
    return settings


def save_settings(settings: dict) -> None:
    """將目前設定寫入 JSON 檔案。"""
    path = get_settings_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


class ErrorDialog(ctk.CTkToplevel):
    """符合整體極簡風格的自訂提示視窗。"""

    def __init__(self, parent, title: str, message: str):
        super().__init__(parent)
        self.title(title)
        self.geometry("360x185")
        self.resizable(False, False)
        self.configure(fg_color=COLORS["bg"])
        self.transient(parent)
        self.grab_set()

        # 置中顯示於主視窗上方
        parent.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 360) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 185) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

        container = ctk.CTkFrame(
            self,
            fg_color=COLORS["card"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["card_border"],
        )
        container.pack(fill="both", expand=True, padx=14, pady=14)

        ctk.CTkLabel(
            container,
            text=title,
            font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w", padx=18, pady=(16, 6))

        ctk.CTkLabel(
            container,
            text=message,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            text_color=COLORS["text_muted"],
            wraplength=296,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 14))

        ctk.CTkButton(
            container,
            text="確定",
            width=96,
            height=34,
            corner_radius=8,
            fg_color=COLORS["danger"],
            hover_color=COLORS["danger_hover"],
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
            command=self.destroy,
        ).pack(anchor="e", padx=18, pady=(0, 14))

        self.bind("<Escape>", lambda _e: self.destroy())
        self.bind("<Return>", lambda _e: self.destroy())


class DAFApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.settings = load_settings()
        ctk.set_appearance_mode(self.settings["theme"])
        ctk.set_default_color_theme("blue")

        self.title(APP_TITLE)
        self.resizable(False, False)
        self.configure(fg_color=COLORS["bg"])

        # 音訊設定與狀態變數
        self.is_running = False
        self.delay_ms = self.settings["delay_ms"]
        self.volume_percent = self.settings["volume_percent"]
        self.volume_level = self.volume_percent / 100.0
        self.sample_rate = 44100
        self.channels = 1
        self.blocksize = 1024

        self.audio_queue = queue.Queue()
        self.stream = None

        self.input_devices: list[tuple[int, str]] = []
        self.output_devices: list[tuple[int, str]] = []
        self.input_label_to_dev: dict[str, tuple[int, str]] = {}
        self.output_label_to_dev: dict[str, tuple[int, str]] = {}

        self._build_ui()
        self._refresh_devices(restore_saved=True)

        # 依實際排版高度自動調整視窗大小，確保所有按鈕與狀態列完整顯示
        self.update_idletasks()
        target_h = max(620, self.winfo_reqheight() + 8)
        self.geometry(f"500x{target_h}")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _build_ui(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=24, pady=22)

        # --- 頂部標題與主題切換 ---
        header = ctk.CTkFrame(main, fg_color="transparent")
        header.pack(fill="x", pady=(0, 18))

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left")

        ctk.CTkLabel(
            title_box,
            text="延遲語音回授",
            font=ctk.CTkFont(family=FONT_FAMILY, size=22, weight="bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_box,
            text="Delayed Auditory Feedback",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w")

        self.theme_switch = ctk.CTkSegmentedButton(
            header,
            values=["淺色", "深色"],
            width=116,
            height=32,
            corner_radius=10,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            fg_color=COLORS["seg_bg"],
            selected_color=COLORS["seg_selected"],
            selected_hover_color=COLORS["seg_selected_hover"],
            unselected_color=COLORS["seg_bg"],
            unselected_hover_color=COLORS["subtle_btn_hover"],
            text_color=COLORS["text"],
            command=self._on_theme_change,
        )
        self.theme_switch.set("深色" if self.settings["theme"] == "dark" else "淺色")
        self.theme_switch.pack(side="right", anchor="n", pady=4)

        # --- 卡片 1：音訊裝置 ---
        device_card = ctk.CTkFrame(
            main,
            fg_color=COLORS["card"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["card_border"],
        )
        device_card.pack(fill="x", pady=(0, 14))

        device_header = ctk.CTkFrame(device_card, fg_color="transparent")
        device_header.pack(fill="x", padx=18, pady=(14, 10))

        ctk.CTkLabel(
            device_header,
            text="音訊裝置",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            text_color=COLORS["text"],
        ).pack(side="left")

        self.refresh_btn = ctk.CTkButton(
            device_header,
            text="重新整理",
            width=76,
            height=26,
            corner_radius=8,
            fg_color=COLORS["subtle_btn"],
            hover_color=COLORS["subtle_btn_hover"],
            text_color=COLORS["text_muted"],
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            command=lambda: self._refresh_devices(restore_saved=False),
        )
        self.refresh_btn.pack(side="right")

        # 輸入裝置
        ctk.CTkLabel(
            device_card,
            text="輸入裝置 (麥克風)",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w", padx=18, pady=(0, 4))

        self.input_var = ctk.StringVar(value="")
        self.input_menu = ctk.CTkOptionMenu(
            device_card,
            variable=self.input_var,
            values=[""],
            height=36,
            corner_radius=9,
            fg_color=COLORS["input_bg"],
            button_color=COLORS["subtle_btn"],
            button_hover_color=COLORS["subtle_btn_hover"],
            text_color=COLORS["text"],
            dropdown_fg_color=COLORS["card"],
            dropdown_hover_color=COLORS["subtle_btn_hover"],
            dropdown_text_color=COLORS["text"],
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            dropdown_font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            anchor="w",
        )
        self.input_menu.pack(fill="x", padx=18, pady=(0, 12))

        # 輸出裝置
        ctk.CTkLabel(
            device_card,
            text="輸出裝置 (耳機 / 喇叭)",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w", padx=18, pady=(0, 4))

        self.output_var = ctk.StringVar(value="")
        self.output_menu = ctk.CTkOptionMenu(
            device_card,
            variable=self.output_var,
            values=[""],
            height=36,
            corner_radius=9,
            fg_color=COLORS["input_bg"],
            button_color=COLORS["subtle_btn"],
            button_hover_color=COLORS["subtle_btn_hover"],
            text_color=COLORS["text"],
            dropdown_fg_color=COLORS["card"],
            dropdown_hover_color=COLORS["subtle_btn_hover"],
            dropdown_text_color=COLORS["text"],
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            dropdown_font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            anchor="w",
        )
        self.output_menu.pack(fill="x", padx=18, pady=(0, 16))

        # --- 卡片 2：參數設定 ---
        param_card = ctk.CTkFrame(
            main,
            fg_color=COLORS["card"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["card_border"],
        )
        param_card.pack(fill="x", pady=(0, 18))

        ctk.CTkLabel(
            param_card,
            text="參數設定",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w", padx=18, pady=(14, 10))

        # 延遲時間設定列
        delay_row = ctk.CTkFrame(param_card, fg_color="transparent")
        delay_row.pack(fill="x", padx=18, pady=(0, 14))

        delay_left = ctk.CTkFrame(delay_row, fg_color="transparent")
        delay_left.pack(side="left")

        ctk.CTkLabel(
            delay_left,
            text="延遲時間",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            text_color=COLORS["text"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            delay_left,
            text="常見設定 50 - 200 ms (上限 5000)",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w")

        delay_right = ctk.CTkFrame(delay_row, fg_color="transparent")
        delay_right.pack(side="right")

        self.delay_entry = ctk.CTkEntry(
            delay_right,
            width=88,
            height=34,
            corner_radius=9,
            fg_color=COLORS["input_bg"],
            border_width=1,
            border_color=COLORS["input_border"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
            justify="center",
        )
        self.delay_entry.insert(0, str(self.delay_ms))
        self.delay_entry.pack(side="left", padx=(0, 6))

        ctk.CTkLabel(
            delay_right,
            text="ms",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=COLORS["text_muted"],
        ).pack(side="left")

        # 輸出音量滑桿
        vol_header = ctk.CTkFrame(param_card, fg_color="transparent")
        vol_header.pack(fill="x", padx=18, pady=(0, 4))

        ctk.CTkLabel(
            vol_header,
            text="輸出音量放大",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            text_color=COLORS["text"],
        ).pack(side="left")

        self.vol_label = ctk.CTkLabel(
            vol_header,
            text=f"{self.volume_percent}%",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
            text_color=COLORS["accent"],
        )
        self.vol_label.pack(side="right")

        self.vol_slider = ctk.CTkSlider(
            param_card,
            from_=0,
            to=200,
            number_of_steps=200,
            height=16,
            button_length=14,
            button_corner_radius=7,
            progress_color=COLORS["accent"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            fg_color=COLORS["seg_bg"],
            command=self._on_vol_change,
        )
        self.vol_slider.set(self.volume_percent)
        self.vol_slider.pack(fill="x", padx=18, pady=(2, 16))

        # --- 控制按鈕與狀態 ---
        self.toggle_btn = ctk.CTkButton(
            main,
            text="開啟 DAF 功能",
            height=46,
            corner_radius=12,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color="#FFFFFF",
            font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"),
            command=self.toggle_daf,
        )
        self.toggle_btn.pack(fill="x", pady=(0, 12))

        status_row = ctk.CTkFrame(main, fg_color="transparent")
        status_row.pack()

        self.status_dot = ctk.CTkLabel(
            status_row,
            text="●",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=COLORS["text_muted"],
        )
        self.status_dot.pack(side="left", padx=(0, 6))

        self.status_label = ctk.CTkLabel(
            status_row,
            text="已停止",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=COLORS["text_muted"],
        )
        self.status_label.pack(side="left")

    def _on_theme_change(self, value: str):
        mode = "dark" if value == "深色" else "light"
        ctk.set_appearance_mode(mode)
        self.settings["theme"] = mode
        save_settings(self.settings)

    def _refresh_devices(self, restore_saved: bool = False):
        """取得系統中的音訊裝置列表（優先使用 Windows WASAPI 以避免重複與名稱截斷），並優先還原上次選取的裝置。"""
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception:
            devices = []
            hostapis = []

        self.input_devices.clear()
        self.output_devices.clear()
        self.input_label_to_dev.clear()
        self.output_label_to_dev.clear()

        # 依優先順序挑選單一 Host API，避免同一裝置因 MME / DirectSound / WASAPI 重複出現
        preferred_api_order = ("Windows WASAPI", "Windows DirectSound", "MME")
        selected_api_idx = None
        for api_name in preferred_api_order:
            for idx, api in enumerate(hostapis):
                if api.get("name") == api_name and api.get("devices"):
                    selected_api_idx = idx
                    break
            if selected_api_idx is not None:
                break

        default_in_id = -1
        default_out_id = -1
        if selected_api_idx is not None:
            default_in_id = hostapis[selected_api_idx].get("default_input_device", -1)
            default_out_id = hostapis[selected_api_idx].get("default_output_device", -1)
        else:
            try:
                default_in_id = sd.default.device[0]
                default_out_id = sd.default.device[1]
            except Exception:
                pass

        seen_inputs: set[str] = set()
        seen_outputs: set[str] = set()
        input_labels = []
        output_labels = []

        for idx, dev in enumerate(devices):
            if selected_api_idx is not None and dev.get("hostapi") != selected_api_idx:
                continue

            name = dev["name"].strip()
            if dev["max_input_channels"] > 0 and name not in seen_inputs:
                seen_inputs.add(name)
                self.input_devices.append((idx, name))
                self.input_label_to_dev[name] = (idx, name)
                input_labels.append(name)
            if dev["max_output_channels"] > 0 and name not in seen_outputs:
                seen_outputs.add(name)
                self.output_devices.append((idx, name))
                self.output_label_to_dev[name] = (idx, name)
                output_labels.append(name)

        self.input_menu.configure(values=input_labels if input_labels else ["無可用輸入裝置"])
        self.output_menu.configure(values=output_labels if output_labels else ["無可用輸出裝置"])

        saved_in_name = self.settings["input_device_name"] if restore_saved else ""
        saved_out_name = self.settings["output_device_name"] if restore_saved else ""

        self._select_device(
            menu_var=self.input_var,
            labels=input_labels,
            dev_list=self.input_devices,
            saved_name=saved_in_name,
            sys_default_id=default_in_id,
            is_input=True,
        )
        self._select_device(
            menu_var=self.output_var,
            labels=output_labels,
            dev_list=self.output_devices,
            saved_name=saved_out_name,
            sys_default_id=default_out_id,
            is_input=False,
        )

    def _select_device(
        self,
        menu_var: ctk.StringVar,
        labels: list[str],
        dev_list: list[tuple[int, str]],
        saved_name: str,
        sys_default_id: int,
        is_input: bool,
    ):
        if not labels:
            menu_var.set("無可用輸入裝置" if is_input else "無可用輸出裝置")
            return

        # 1. 優先比對上次儲存的裝置名稱
        if saved_name:
            for label, (_idx, name) in zip(labels, dev_list):
                if name == saved_name:
                    menu_var.set(label)
                    return

        # 2. 其次比對該 Host API 的系統預設裝置
        if sys_default_id >= 0:
            for label, (idx, _name) in zip(labels, dev_list):
                if idx == sys_default_id:
                    menu_var.set(label)
                    return

        # 3. 最後選取第一個可用裝置
        menu_var.set(labels[0])

    def _on_vol_change(self, val: float):
        val_int = int(round(float(val)))
        self.volume_percent = val_int
        self.volume_level = val_int / 100.0
        self.vol_label.configure(text=f"{val_int}%")

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """sounddevice 串流處理回呼函數"""
        if status:
            print(f"Audio Status: {status}")

        self.audio_queue.put(indata.copy())

        target_delay_sec = self.delay_ms / 1000.0
        required_buffered_frames = int(target_delay_sec * self.sample_rate)
        current_buffered_frames = self.audio_queue.qsize() * self.blocksize

        if current_buffered_frames < required_buffered_frames:
            outdata.fill(0)
        else:
            try:
                data = self.audio_queue.get_nowait()
                outdata[:] = np.clip(data * self.volume_level, -1.0, 1.0)
            except queue.Empty:
                outdata.fill(0)

    def _collect_current_settings(self) -> dict:
        """收集當前介面上的設定，準備寫入設定檔。"""
        in_dev = self.input_label_to_dev.get(self.input_var.get())
        out_dev = self.output_label_to_dev.get(self.output_var.get())

        delay_val = self.delay_ms
        try:
            parsed = float(self.delay_entry.get().strip())
            if 0 <= parsed <= 5000:
                delay_val = int(parsed) if parsed.is_integer() else parsed
        except ValueError:
            pass

        return {
            "theme": self.settings["theme"],
            "input_device_name": in_dev[1] if in_dev else self.settings["input_device_name"],
            "output_device_name": out_dev[1] if out_dev else self.settings["output_device_name"],
            "delay_ms": delay_val,
            "volume_percent": self.volume_percent,
        }

    def toggle_daf(self):
        if not self.is_running:
            self.start_daf()
        else:
            self.stop_daf()

    def _find_fallback_device_pair(self, input_name: str, output_name: str) -> tuple[int, int] | None:
        """若 WASAPI 裝置取樣率不相容，嘗試尋找同名的 DirectSound 裝置作為備援。"""
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception:
            return None

        for api_name in ("Windows DirectSound", "MME"):
            api_idx = next((i for i, a in enumerate(hostapis) if a.get("name") == api_name), None)
            if api_idx is None:
                continue
            in_id = next(
                (i for i, d in enumerate(devices)
                 if d.get("hostapi") == api_idx and d["max_input_channels"] > 0 and d["name"].strip() == input_name),
                None,
            )
            out_id = next(
                (i for i, d in enumerate(devices)
                 if d.get("hostapi") == api_idx and d["max_output_channels"] > 0 and d["name"].strip() == output_name),
                None,
            )
            if in_id is not None and out_id is not None:
                return (in_id, out_id)
        return None

    def _open_audio_stream(self, input_dev_id: int, output_dev_id: int, input_name: str, output_name: str) -> sd.Stream:
        """依裝置支援的取樣率自動挑選並開啟音訊串流。"""
        candidates: list[int] = []
        try:
            in_info = sd.query_devices(input_dev_id)
            out_info = sd.query_devices(output_dev_id)
            for sr in (int(in_info["default_samplerate"]), int(out_info["default_samplerate"]), 48000, 44100):
                if sr > 0 and sr not in candidates:
                    candidates.append(sr)
        except Exception:
            candidates = [48000, 44100]

        last_err: Exception | None = None
        pairs_to_try: list[tuple[int, int]] = [(input_dev_id, output_dev_id)]
        fallback_pair = self._find_fallback_device_pair(input_name, output_name)
        if fallback_pair and fallback_pair not in pairs_to_try:
            pairs_to_try.append(fallback_pair)

        for dev_pair in pairs_to_try:
            for sr in candidates:
                stream = None
                try:
                    stream = sd.Stream(
                        device=dev_pair,
                        samplerate=sr,
                        blocksize=self.blocksize,
                        channels=self.channels,
                        dtype="float32",
                        callback=self._audio_callback,
                    )
                    stream.start()
                    self.sample_rate = sr
                    return stream
                except Exception as e:
                    last_err = e
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass

        assert last_err is not None
        raise last_err

    def start_daf(self):
        try:
            val = float(self.delay_entry.get().strip())
            if val < 0 or val > 5000:
                raise ValueError("範圍請介於 0 到 5000 ms 之間")
            self.delay_ms = int(val) if val.is_integer() else val
        except ValueError as e:
            ErrorDialog(self, "輸入錯誤", f"請輸入有效的數值 (例如: 150)\n{e}")
            return

        in_dev = self.input_label_to_dev.get(self.input_var.get())
        out_dev = self.output_label_to_dev.get(self.output_var.get())

        if not in_dev or not out_dev:
            ErrorDialog(self, "裝置錯誤", "請選擇有效的輸入與輸出裝置！")
            return

        input_dev_id, input_name = in_dev
        output_dev_id, output_name = out_dev

        while not self.audio_queue.empty():
            self.audio_queue.get()

        try:
            self.stream = self._open_audio_stream(input_dev_id, output_dev_id, input_name, output_name)
            self.is_running = True

            # 同步保存本次啟動使用的設定
            self.settings = self._collect_current_settings()
            save_settings(self.settings)

            self.toggle_btn.configure(
                text="停止 DAF 功能",
                fg_color=COLORS["danger"],
                hover_color=COLORS["danger_hover"],
            )
            self.status_dot.configure(text_color=COLORS["accent"])
            self.status_label.configure(
                text=f"運作中 · 延遲 {int(self.delay_ms)} ms",
                text_color=COLORS["accent"],
            )
            self.input_menu.configure(state="disabled")
            self.output_menu.configure(state="disabled")
            self.refresh_btn.configure(state="disabled")
            self.delay_entry.configure(state="disabled")
        except Exception as e:
            ErrorDialog(self, "音訊啟動失敗", f"無法啟動音訊串流:\n{e}")

    def stop_daf(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        self.is_running = False
        self.toggle_btn.configure(
            text="開啟 DAF 功能",
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
        )
        self.status_dot.configure(text_color=COLORS["text_muted"])
        self.status_label.configure(text="已停止", text_color=COLORS["text_muted"])
        self.input_menu.configure(state="normal")
        self.output_menu.configure(state="normal")
        self.refresh_btn.configure(state="normal")
        self.delay_entry.configure(state="normal")

    def on_closing(self):
        self.settings = self._collect_current_settings()
        save_settings(self.settings)
        self.stop_daf()
        self.destroy()


if __name__ == "__main__":
    app = DAFApp()
    app.mainloop()
