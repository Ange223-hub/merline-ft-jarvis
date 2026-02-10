import os
import time
import math
import threading
from dataclasses import dataclass


@dataclass
class GestureConfig:
    camera_index: int = 0
    show_preview: bool = False
    max_fps: int = 20
    cooldown_s: float = 0.7


class GestureController:
    def __init__(self, config: GestureConfig | None = None):
        self.config = config or GestureConfig(
            camera_index=int(os.environ.get("MERLINE_CAMERA_INDEX", "0")),
            show_preview=os.environ.get("MERLINE_VISION_PREVIEW", "0").strip() in {"1", "true", "yes"},
            max_fps=int(os.environ.get("MERLINE_VISION_MAX_FPS", "20")),
            cooldown_s=float(os.environ.get("MERLINE_VISION_COOLDOWN_S", "0.7")),
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_action_ts = 0.0
        self._pinch_base: float | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _cooldown_ok(self) -> bool:
        now = time.time()
        if now - self._last_action_ts >= self.config.cooldown_s:
            self._last_action_ts = now
            return True
        return False

    def _run(self) -> None:
        try:
            import cv2  # type: ignore
            import mediapipe as mp  # type: ignore
        except Exception as e:
            print(f"\033[33m[VISION] Disabled (missing dependency): {e}\033[0m")
            return

        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            print("\033[33m[VISION] Webcam not available\033[0m")
            return

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        prev_ts = 0.0
        try:
            while not self._stop.is_set():
                now = time.time()
                if self.config.max_fps > 0 and now - prev_ts < (1.0 / float(self.config.max_fps)):
                    time.sleep(0.001)
                    continue
                prev_ts = now

                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    lm = result.multi_hand_landmarks[0].landmark
                    gesture = self._classify(lm)
                    if gesture is not None:
                        self._apply_gesture(gesture, lm)

                if self.config.show_preview:
                    cv2.imshow("MERLINE Vision", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

        finally:
            try:
                hands.close()
            except Exception:
                pass
            cap.release()
            if self.config.show_preview:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    def _dist(self, a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _classify(self, lm):
        thumb_tip = lm[4]
        index_tip = lm[8]
        middle_tip = lm[12]
        ring_tip = lm[16]
        pinky_tip = lm[20]

        index_pip = lm[6]
        middle_pip = lm[10]
        ring_pip = lm[14]
        pinky_pip = lm[18]

        index_up = index_tip.y < index_pip.y
        middle_up = middle_tip.y < middle_pip.y
        ring_up = ring_tip.y < ring_pip.y
        pinky_up = pinky_tip.y < pinky_pip.y

        pinch = self._dist(thumb_tip, index_tip)

        if index_up and middle_up and ring_up and pinky_up:
            if pinch < 0.05:
                return "PINCH"
            return "OPEN_PALM"

        if (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up):
            return "FIST"

        if index_up and not middle_up and not ring_up and not pinky_up:
            return "INDEX_UP"

        return None

    def _apply_gesture(self, gesture: str, lm) -> None:
        if gesture == "PINCH":
            thumb_tip = lm[4]
            index_tip = lm[8]
            pinch = self._dist(thumb_tip, index_tip)
            if self._pinch_base is None:
                self._pinch_base = pinch
                return
            delta = pinch - self._pinch_base
            if abs(delta) < 0.01:
                return
            if delta > 0:
                self._volume_up(step=2)
            else:
                self._volume_down(step=2)
            self._pinch_base = pinch
            return

        self._pinch_base = None

        if gesture == "OPEN_PALM":
            if self._cooldown_ok():
                self._media_play_pause()
            return

        if gesture == "FIST":
            if self._cooldown_ok():
                self._lock_workstation()
            return

        if gesture == "INDEX_UP":
            if self._cooldown_ok():
                self._brightness_up(step=10)
            return

    def _volume_up(self, step: int = 2) -> None:
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore
            from comtypes import CLSCTX_ALL  # type: ignore
            import ctypes

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))
            current = volume.GetMasterVolumeLevelScalar()
            volume.SetMasterVolumeLevelScalar(min(1.0, current + (step / 100.0)), None)
        except Exception:
            self._send_media_key("VOLUMEUP")

    def _volume_down(self, step: int = 2) -> None:
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore
            from comtypes import CLSCTX_ALL  # type: ignore
            import ctypes

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))
            current = volume.GetMasterVolumeLevelScalar()
            volume.SetMasterVolumeLevelScalar(max(0.0, current - (step / 100.0)), None)
        except Exception:
            self._send_media_key("VOLUMEDOWN")

    def _brightness_up(self, step: int = 10) -> None:
        try:
            import screen_brightness_control as sbc  # type: ignore

            current = sbc.get_brightness(display=0)
            if isinstance(current, list):
                current = current[0]
            sbc.set_brightness(min(100, int(current) + int(step)), display=0)
        except Exception:
            pass

    def _media_play_pause(self) -> None:
        self._send_media_key("PLAYPAUSE")

    def _lock_workstation(self) -> None:
        try:
            import ctypes

            ctypes.windll.user32.LockWorkStation()
        except Exception:
            pass

    def _send_media_key(self, key: str) -> None:
        try:
            import keyboard  # type: ignore

            mapping = {
                "PLAYPAUSE": "play/pause media",
                "VOLUMEUP": "volume up",
                "VOLUMEDOWN": "volume down",
            }
            hotkey = mapping.get(key)
            if hotkey:
                keyboard.send(hotkey)
        except Exception:
            pass
