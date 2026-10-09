#!/usr/bin/env python3
"""
speak_gui.py - lightweight tkinter front end for speak.py (text to speech)
and listener.py (microphone speech to text).

Keep speak.py, listener.py and this file in the same folder, then run:
    python speak_gui.py

Linux Mint:  sudo apt install python3-tk libportaudio2
             pip install piper-tts sounddevice numpy vosk
"""

import json
import queue
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, scrolledtext

import speak
import listener

DEFAULT_TEXT = "Hello! Type something here and press Speak."
SYSTEM_DEFAULT = "System default microphone"


def prettify(text: str) -> str:
    """Vosk returns lowercase text with no punctuation; tidy it so the voice
    gets sensible sentence breaks and pauses."""
    text = re.sub(r"\bi\b", "I", text.strip())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Speak - offline text to speech")
        self.minsize(720, 660)

        self.spk = speak.Speaker(speak.Settings())
        self.spk.on_error = self._set_note
        self._note = ""            # status message set from background threads
        self._loading = 0          # number of voice downloads in progress
        self._listen_msg = ""      # mic start-up progress
        self._partial = ""         # words heard so far
        self.listening = False

        # events from the microphone thread; handled on the Tk thread in _poll
        self.events: "queue.Queue" = queue.Queue()
        self.listener = listener.Listener(
            emit=lambda kind, text: self.events.put((kind, text)),
            is_muted=self._tts_busy)

        self.voice_var = tk.StringVar()
        self.personality_var = tk.StringVar(value="neutral")
        self.speaker_var = tk.IntVar(value=0)
        self.pitch_var = tk.DoubleVar(value=0.0)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.volume_var = tk.DoubleVar(value=1.0)
        self.interrupt_var = tk.BooleanVar(value=True)
        self.auto_speak_var = tk.BooleanVar(value=False)
        self.device_var = tk.StringVar(value=SYSTEM_DEFAULT)
        self.status_var = tk.StringVar(value="Ready")
        self.persona_info = tk.StringVar()
        self._devices = {}

        self._build_ui()
        self._voice_labels = self._voice_values()
        self.voice_var.set(self._voice_labels[0])
        self.on_voice_change()
        self.on_personality_change()
        self.refresh_devices()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self._poll)

    # ------------------------------------------------------------------ UI
    @staticmethod
    def _voice_values():
        return [f"{k} - {desc}" for k, (_m, desc) in speak.VOICES.items()]

    def _build_ui(self):
        pad = dict(padx=10, pady=4)
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # text input
        self.text = scrolledtext.ScrolledText(root, wrap="word", height=10,
                                              font=("TkDefaultFont", 12))
        self.text.grid(row=0, column=0, sticky="nsew", padx=10, pady=(8, 4))
        self.text.insert("1.0", DEFAULT_TEXT)
        self.text.bind("<Control-Return>", self.on_speak)
        self.text.focus_set()

        # buttons
        bar = ttk.Frame(root)
        bar.grid(row=1, column=0, sticky="ew", **pad)
        ttk.Button(bar, text="Speak  (Ctrl+Enter)", command=self.on_speak).pack(side="left")
        ttk.Button(bar, text="Stop", command=self.spk.stop).pack(side="left", padx=6)
        ttk.Button(bar, text="Clear", command=lambda: self.text.delete("1.0", "end")).pack(side="left")
        ttk.Checkbutton(bar, text="Interrupt current speech",
                        variable=self.interrupt_var).pack(side="right")

        # microphone
        mic_box = ttk.LabelFrame(root, text="Microphone (speech to text)", padding=8)
        mic_box.grid(row=2, column=0, sticky="ew", **pad)
        mic_box.columnconfigure(1, weight=1)
        self.mic_btn = ttk.Button(mic_box, text="Start Mic", width=11, command=self.toggle_mic)
        self.mic_btn.grid(row=0, column=0, sticky="w")
        self.device_cb = ttk.Combobox(mic_box, textvariable=self.device_var, state="readonly")
        self.device_cb.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(mic_box, text="Refresh", width=8,
                   command=self.refresh_devices).grid(row=0, column=2)
        ttk.Checkbutton(mic_box, variable=self.auto_speak_var,
                        text="Voice changer: speak what I say in the selected voice").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(mic_box,
                  text="Use headphones: the mic pauses while the voice is talking, "
                       "so it can't hear itself.",
                  foreground="gray40").grid(row=2, column=0, columnspan=3, sticky="w")

        # voice settings
        voice_box = ttk.LabelFrame(root, text="Voice", padding=8)
        voice_box.grid(row=3, column=0, sticky="ew", **pad)
        voice_box.columnconfigure(1, weight=1)

        ttk.Label(voice_box, text="Voice").grid(row=0, column=0, sticky="w")
        self.voice_cb = ttk.Combobox(voice_box, textvariable=self.voice_var,
                                     values=self._voice_values())
        self.voice_cb.grid(row=0, column=1, sticky="ew", padx=6)
        self.voice_cb.bind("<<ComboboxSelected>>", lambda _e: self.on_voice_change())
        self.voice_cb.bind("<Return>", lambda _e: self.on_voice_change())

        ttk.Label(voice_box, text="Speaker #").grid(row=0, column=2, padx=(10, 4))
        self.speaker_sb = ttk.Spinbox(voice_box, from_=0, to=0, width=6,
                                      textvariable=self.speaker_var, state="disabled")
        self.speaker_sb.grid(row=0, column=3)

        ttk.Label(voice_box,
                  text="Tip: you can type any Piper model name (e.g. de_DE-thorsten-medium) "
                       "and press Enter. It downloads on first use.",
                  foreground="gray40").grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # personality
        pers_box = ttk.LabelFrame(root, text="Personality (controls pacing)", padding=8)
        pers_box.grid(row=4, column=0, sticky="ew", **pad)
        pers_box.columnconfigure(1, weight=1)
        ttk.Label(pers_box, text="Personality").grid(row=0, column=0, sticky="w")
        pers_cb = ttk.Combobox(pers_box, textvariable=self.personality_var,
                               values=list(speak.PERSONALITIES), state="readonly", width=18)
        pers_cb.grid(row=0, column=1, sticky="w", padx=6)
        pers_cb.bind("<<ComboboxSelected>>", lambda _e: self.on_personality_change())
        ttk.Label(pers_box, textvariable=self.persona_info,
                  foreground="gray40").grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # sliders
        sl_box = ttk.LabelFrame(root, text="Tone", padding=8)
        sl_box.grid(row=5, column=0, sticky="ew", **pad)
        sl_box.columnconfigure(1, weight=1)
        self._slider(sl_box, 0, "Depth / pitch (semitones)", self.pitch_var,
                     -6, 6, 0.5, "negative = deeper, positive = higher", 0.0)
        self._slider(sl_box, 1, "Speed", self.speed_var, 0.5, 2.0, 0.05,
                     "multiplies the personality's pace", 1.0)
        self._slider(sl_box, 2, "Volume", self.volume_var, 0.0, 2.0, 0.05, "", 1.0)

        # status
        ttk.Label(root, textvariable=self.status_var, anchor="w").grid(
            row=6, column=0, sticky="ew", padx=10, pady=(4, 8))

    def _slider(self, parent, row, label, var, lo, hi, step, hint, default):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        scale = tk.Scale(parent, variable=var, from_=lo, to=hi, resolution=step,
                         orient="horizontal", showvalue=True, length=280)
        scale.grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(parent, text="Reset", width=6,
                   command=lambda: var.set(default)).grid(row=row, column=2)
        if hint:
            ttk.Label(parent, text=hint, foreground="gray40").grid(
                row=row, column=3, sticky="w", padx=8)

    # -------------------------------------------------------------- voice
    def voice_key(self) -> str:
        raw = self.voice_var.get().strip()
        key = raw.split(" - ", 1)[0].strip()
        return key if key in speak.VOICES else raw

    def on_voice_change(self):
        key = self.voice_key()
        if not key:
            return
        self.speaker_var.set(0)
        self._loading += 1
        self._note = "Preparing voice (first use downloads it)..."
        threading.Thread(target=self._prepare_voice, args=(key,), daemon=True).start()

    def _prepare_voice(self, key):
        try:
            model = speak.resolve_model_name(key)
            onnx = speak.ensure_model(model)
            cfg = json.loads(Path(str(onnx) + ".json").read_text())
            n = int(cfg.get("num_speakers", 1))
            self.after(0, lambda: self._set_speakers(n))
            self._note = ""
        except SystemExit as exc:
            self._note = f"Error: {exc}"
        except Exception as exc:
            self._note = f"Could not load voice: {exc}"
        finally:
            self._loading -= 1

    def _set_speakers(self, n):
        if n > 1:
            self.speaker_sb.configure(state="normal", to=n - 1)
        else:
            self.speaker_sb.configure(state="disabled", to=0)
            self.speaker_var.set(0)

    def on_personality_change(self):
        p = speak.PERSONALITIES[self.personality_var.get()]
        self.persona_info.set(
            f"pace x{p['speed']:.2f}   pauses x{p['pause']:.2f}   "
            f"expressiveness {p['expressive']:.2f}   timing variation {p['rhythm']:.2f}   "
            f"base pitch {p['pitch']:+.1f} st")

    def apply_settings(self):
        st = self.spk.settings
        st.voice = self.voice_key()
        try:
            st.speaker = max(0, int(self.speaker_var.get()))
        except (tk.TclError, ValueError):
            st.speaker = 0
        st.personality = self.personality_var.get()
        st.pitch = float(self.pitch_var.get())
        st.speed = float(self.speed_var.get())
        st.volume = float(self.volume_var.get())

    def on_speak(self, _event=None):
        text = self.text.get("1.0", "end").strip()
        if not text:
            return "break"
        if self.interrupt_var.get():
            self.spk.stop()
        self.apply_settings()
        self._note = ""
        self.spk.say(text)
        return "break"

    # --------------------------------------------------------- microphone
    def refresh_devices(self):
        devs = listener.input_devices()
        self._devices = {SYSTEM_DEFAULT: None}
        for idx, name in devs:
            self._devices[f"{idx}: {name}"] = idx
        self.device_cb.configure(values=list(self._devices))
        if self.device_var.get() not in self._devices:
            self.device_var.set(SYSTEM_DEFAULT)

    def toggle_mic(self):
        if self.listening:
            self.listener.stop()
            self.mic_btn.state(["disabled"])   # re-enabled when the thread ends
            return
        if self.listener.running:
            return
        self._note = ""
        self._listen_msg = "Starting microphone..."
        self.listening = True
        self.mic_btn.configure(text="Stop Mic")
        self.listener.start(self._devices.get(self.device_var.get()))

    def _tts_busy(self) -> bool:
        return bool(self.spk.jobs.unfinished_tasks or self.spk.audio.unfinished_tasks)

    def _on_phrase(self, raw: str):
        phrase = prettify(raw)
        if not phrase:
            return
        # always log it in the text box so it can be edited and re-spoken
        current = self.text.get("1.0", "end").strip()
        if current == DEFAULT_TEXT:
            self.text.delete("1.0", "end")
            current = ""
        self.text.insert("end", ("\n" if current else "") + phrase)
        self.text.see("end")
        if self.auto_speak_var.get():
            self.apply_settings()
            self.spk.say(phrase)   # queued, so phrases never cut each other off

    def _handle_event(self, kind: str, text: str):
        if kind == "status":
            self._listen_msg = text
        elif kind == "ready":
            self._listen_msg = ""
        elif kind == "partial":
            self._partial = text
        elif kind == "final":
            self._partial = ""
            self._on_phrase(text)
        elif kind == "error":
            self._note = text
        elif kind == "stopped":
            self.listening = False
            self._listen_msg = ""
            self._partial = ""
            self.mic_btn.configure(text="Start Mic")
            self.mic_btn.state(["!disabled"])

    # -------------------------------------------------------------- status
    def _set_note(self, msg):
        self._note = msg

    def _poll(self):
        try:
            while True:
                self._handle_event(*self.events.get_nowait())
        except queue.Empty:
            pass

        if self._note:
            msg = self._note
        elif self._listen_msg:
            msg = self._listen_msg
        elif self._loading:
            msg = "Preparing voice..."
        elif self._tts_busy():
            msg = "Speaking..."
        elif self.listening and self._partial:
            msg = f"Hearing: {self._partial}"
        elif self.listening:
            msg = "Listening..."
        else:
            msg = "Ready"
        self.status_var.set(msg)
        self.after(100, self._poll)

    def on_close(self):
        self.listener.stop()
        self.spk.stop()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
