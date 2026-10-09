#!/usr/bin/env python3
"""
listener.py - offline microphone speech-to-text for speak_gui.py.

Uses Vosk's small English model (~40 MB download, runs in real time on weak
CPUs). Everything stays on your machine.

    pip install vosk sounddevice
"""

import json
import queue
import threading
import urllib.request
import zipfile
from pathlib import Path

import sounddevice as sd

STT_DIR = Path.home() / ".cache" / "speak_py_stt"
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"


def input_devices():
    """Return [(index, name)] for every device that can record."""
    out = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                out.append((i, d["name"]))
    except Exception:
        pass
    return out


def ensure_stt_model(emit=None) -> Path:
    STT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = STT_DIR / MODEL_NAME
    if model_dir.exists():
        return model_dir
    if emit:
        emit("status", "Downloading speech model (~40 MB, first time only)...")
    zip_path = STT_DIR / (MODEL_NAME + ".zip.part")
    urllib.request.urlretrieve(MODEL_URL, zip_path)
    if emit:
        emit("status", "Unpacking speech model...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(STT_DIR)
    zip_path.unlink()
    return model_dir


class Listener:
    """Streams the microphone into Vosk on a background thread.

    emit(kind, text) is called from that thread with kinds:
      status   progress message while starting
      ready    microphone is open and listening
      partial  words heard so far in the current phrase
      final    a finished phrase
      error    something went wrong
      stopped  the thread has ended (always sent last)

    is_muted() lets the caller pause recognition (e.g. while the speakers are
    talking) so the program does not transcribe its own voice.
    """

    _model = None  # loaded once, reused between sessions

    def __init__(self, emit, is_muted=lambda: False):
        self.emit = emit
        self.is_muted = is_muted
        self._stop = threading.Event()
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, device=None):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(device,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------
    def _run(self, device):
        try:
            try:
                from vosk import KaldiRecognizer, Model, SetLogLevel
            except ImportError:
                self.emit("error", "Speech-to-text needs vosk: pip install vosk")
                return
            SetLogLevel(-1)

            if Listener._model is None:
                path = ensure_stt_model(self.emit)
                self.emit("status", "Loading speech model...")
                Listener._model = Model(str(path))

            audio_q: "queue.Queue[bytes]" = queue.Queue()

            def callback(indata, frames, time_info, status):
                if audio_q.qsize() < 60:  # drop audio rather than grow without bound
                    audio_q.put(bytes(indata))

            def open_stream(rate):
                return sd.RawInputStream(samplerate=rate, blocksize=4000, device=device,
                                         dtype="int16", channels=1, callback=callback)

            rate = 16000
            try:
                stream = open_stream(rate)
            except sd.PortAudioError:
                # some devices refuse 16 kHz; use their native rate (Vosk handles it)
                info = sd.query_devices(device, "input")
                rate = int(info["default_samplerate"])
                stream = open_stream(rate)

            rec = KaldiRecognizer(Listener._model, rate)
            was_muted = False
            with stream:
                self.emit("ready", "")
                while not self._stop.is_set():
                    try:
                        data = audio_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if self.is_muted():
                        was_muted = True
                        continue
                    if was_muted:
                        rec.Reset()
                        was_muted = False
                    if rec.AcceptWaveform(data):
                        text = json.loads(rec.Result()).get("text", "").strip()
                        if text:
                            self.emit("final", text)
                        self.emit("partial", "")
                    else:
                        self.emit("partial", json.loads(rec.PartialResult()).get("partial", ""))
                text = json.loads(rec.FinalResult()).get("text", "").strip()
                if text:
                    self.emit("final", text)
        except Exception as exc:
            self.emit("error", f"Microphone error: {exc}")
        finally:
            self.emit("stopped", "")
