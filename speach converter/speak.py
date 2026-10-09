#!/usr/bin/env python3
"""
speak.py - lightweight, offline, neural text-to-speech for low-end hardware.

Engine : Piper (VITS models exported to ONNX; runs faster than real time on a
         Raspberry Pi class CPU, no GPU, ~60-120 MB RAM).
Audio  : sounddevice (PortAudio) with a streaming pipeline, so speech starts
         after the FIRST sentence is synthesized instead of the whole text.

Install:
    pip install piper-tts sounddevice numpy
    (Linux)  sudo apt install libportaudio2

Run:
    python speak.py                      # interactive mode
    python speak.py "Hello there"        # one-shot
    python speak.py --list               # list voices / personalities
"""

import argparse
import dataclasses
import queue
import random
import re
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper import PiperVoice

try:  # piper-tts >= 1.3
    from piper import SynthesisConfig
    NEW_API = True
except ImportError:  # older piper-tts (1.2.x)
    SynthesisConfig = None
    NEW_API = False

MODELS_DIR = Path.home() / ".cache" / "speak_py_voices"

# --------------------------------------------------------------------------
# Voices. nickname -> (piper model name, description)
# Any other Piper model name (e.g. "de_DE-thorsten-medium") also works via
# --voice; it is downloaded automatically on first use.
# "low" models are the fastest; use them on very weak CPUs.
# --------------------------------------------------------------------------
VOICES = {
    "lessac":   ("en_US-lessac-medium",   "US female, clear, balanced"),
    "lessac-hq": ("en_US-lessac-high",    "US female, highest quality (slower)"),
    "lessac-fast": ("en_US-lessac-low",   "US female, fastest"),
    "amy":      ("en_US-amy-medium",      "US female, warm"),
    "amy-fast": ("en_US-amy-low",         "US female, fastest"),
    "kristin":  ("en_US-kristin-medium",  "US female, mature"),
    "kathleen": ("en_US-kathleen-low",    "US female, soft, fast"),
    "hfc-f":    ("en_US-hfc_female-medium", "US female, natural"),
    "ryan":     ("en_US-ryan-medium",     "US male, natural"),
    "ryan-hq":  ("en_US-ryan-high",       "US male, highest quality (slower)"),
    "joe":      ("en_US-joe-medium",      "US male, deeper"),
    "danny":    ("en_US-danny-low",       "US male, low and fast"),
    "hfc-m":    ("en_US-hfc_male-medium", "US male, natural"),
    "alan":     ("en_GB-alan-medium",     "British male"),
    "north":    ("en_GB-northern_english_male-medium", "Northern English male"),
    "alba":     ("en_GB-alba-medium",     "Scottish female"),
    "jenny":    ("en_GB-jenny_dioco-medium", "British female"),
    "libritts": ("en_US-libritts_r-medium", "900+ speakers, pick with --speaker N"),
}

# --------------------------------------------------------------------------
# Personalities: they change pacing, pauses, expressiveness and pitch.
#   speed      speaking-rate multiplier (>1 faster)
#   expressive Piper noise_scale  (higher = more varied intonation)
#   rhythm     Piper noise_w      (higher = more irregular timing)
#   pause      multiplier for pauses between sentences / paragraphs
#   pitch      base pitch offset in semitones
#   pitch_var  random per-sentence pitch wobble (semitones, std-dev)
#   jitter     random per-sentence speed variation (fraction)
# --------------------------------------------------------------------------
PERSONALITIES = {
    "neutral":       dict(speed=1.00, expressive=0.667, rhythm=0.80, pause=1.0, pitch=0.0,  pitch_var=0.15, jitter=0.03),
    "calm":          dict(speed=0.90, expressive=0.500, rhythm=0.60, pause=1.4, pitch=-0.3, pitch_var=0.10, jitter=0.02),
    "cheerful":      dict(speed=1.08, expressive=0.800, rhythm=0.90, pause=0.8, pitch=1.0,  pitch_var=0.35, jitter=0.05),
    "energetic":     dict(speed=1.22, expressive=0.900, rhythm=1.00, pause=0.55, pitch=1.5, pitch_var=0.50, jitter=0.07),
    "nervous":       dict(speed=1.18, expressive=0.850, rhythm=1.10, pause=0.60, pitch=1.0, pitch_var=0.50, jitter=0.10),
    "authoritative": dict(speed=0.92, expressive=0.450, rhythm=0.60, pause=1.3, pitch=-1.5, pitch_var=0.10, jitter=0.02),
    "sad":           dict(speed=0.82, expressive=0.500, rhythm=0.70, pause=1.6, pitch=-1.0, pitch_var=0.15, jitter=0.03),
    "thoughtful":    dict(speed=0.88, expressive=0.600, rhythm=0.90, pause=1.8, pitch=-0.5, pitch_var=0.20, jitter=0.06),
}

# Base pause (seconds) after each kind of sentence ending.
PAUSE_BASE = {".": 0.35, "!": 0.38, "?": 0.40, "…": 0.60, ";": 0.25, ":": 0.25}
PARAGRAPH_EXTRA = 0.45


@dataclasses.dataclass
class Settings:
    voice: str = "lessac"
    speaker: int = 0
    personality: str = "neutral"
    pitch: float = 0.0     # extra semitones on top of personality (-6..+6). negative = deeper
    speed: float = 1.0     # extra speed multiplier
    volume: float = 1.0


# --------------------------------------------------------------------------
# Model download / loading
# --------------------------------------------------------------------------
def resolve_model_name(voice: str) -> str:
    return VOICES[voice][0] if voice in VOICES else voice


def ensure_model(model_name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx = MODELS_DIR / f"{model_name}.onnx"
    cfg = MODELS_DIR / f"{model_name}.onnx.json"
    if onnx.exists() and cfg.exists():
        return onnx
    try:
        lang_region, name, quality = model_name.split("-", 2)
    except ValueError:
        sys.exit(f"Unrecognised voice '{model_name}'. Expected e.g. en_US-lessac-medium")
    lang = lang_region.split("_")[0]
    base = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            f"{lang}/{lang_region}/{name}/{quality}/{model_name}")
    for url, dest in ((base + ".onnx.json", cfg), (base + ".onnx", onnx)):
        print(f"Downloading {dest.name} ...", file=sys.stderr)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return onnx


# --------------------------------------------------------------------------
# DSP helper: pitch shift by resampling. Playing a signal back "faster" raises
# pitch (and formants, which is what makes a voice sound deeper/smaller), so we
# compensate by slowing synthesis by the same factor to keep the pace unchanged.
# --------------------------------------------------------------------------
def resample(audio: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3 or len(audio) < 2:
        return audio.astype(np.float32)
    n_out = max(2, int(len(audio) / factor))
    x_new = np.linspace(0, len(audio) - 1, n_out, dtype=np.float32)
    return np.interp(x_new, np.arange(len(audio), dtype=np.float32),
                     audio.astype(np.float32))


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------
_SENT_RE = re.compile(r"[^.!?…;:]+[.!?…;:]*|[.!?…;:]+")


def split_sentences(text: str, max_len: int = 220):
    """Yield (sentence, ending_char). Very long sentences are cut at commas so
    the first audio can start sooner."""
    for raw in _SENT_RE.findall(text):
        s = raw.strip()
        if not s or not re.search(r"\w", s):
            continue
        end = s[-1] if s[-1] in PAUSE_BASE else "."
        if len(s) <= max_len:
            yield s, end
            continue
        parts = re.split(r"(?<=,)\s+", s)
        buf = ""
        for p in parts:
            if buf and len(buf) + len(p) > max_len:
                yield buf, ","
                buf = p
            else:
                buf = f"{buf} {p}".strip()
        if buf:
            yield buf, end


# --------------------------------------------------------------------------
# Engine: synth thread -> bounded queue -> playback thread
# --------------------------------------------------------------------------
class Speaker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.voice = None
        self.voice_model = None
        self.jobs: "queue.Queue" = queue.Queue()
        self.audio: "queue.Queue" = queue.Queue(maxsize=3)  # bounded -> low RAM
        self.epoch = 0
        self.on_error = None  # optional callback(str) so a GUI can show errors
        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    # ---- public API
    def say(self, text: str):
        self.jobs.put((self.epoch, text, dataclasses.replace(self.settings)))

    def stop(self):
        self.epoch += 1  # everything queued with an older epoch is discarded

    def wait(self):
        self.jobs.join()
        self.audio.join()

    # ---- internals
    def _load(self, voice_key: str):
        model = resolve_model_name(voice_key)
        if model != self.voice_model:
            path = ensure_model(model)
            self.voice = PiperVoice.load(str(path))
            self.voice_model = model
        return self.voice

    def _synth_raw(self, voice, text, length_scale, noise_scale, noise_w, speaker):
        sid = speaker if getattr(voice.config, "num_speakers", 1) > 1 else None
        if NEW_API:
            cfg = SynthesisConfig(speaker_id=sid, length_scale=length_scale,
                                  noise_scale=noise_scale, noise_w_scale=noise_w)
            chunks = [np.frombuffer(c.audio_int16_bytes, dtype=np.int16)
                      for c in voice.synthesize(text, syn_config=cfg)]
        else:
            chunks = [np.frombuffer(b, dtype=np.int16)
                      for b in voice.synthesize_stream_raw(
                          text, speaker_id=sid, length_scale=length_scale,
                          noise_scale=noise_scale, noise_w=noise_w)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)

    def _put(self, ep, sr, data):
        while ep == self.epoch:
            try:
                self.audio.put((ep, sr, data), timeout=0.2)
                return
            except queue.Full:
                continue

    def _synth_loop(self):
        while True:
            ep, text, st = self.jobs.get()
            try:
                if ep == self.epoch:
                    self._speak(ep, text, st)
            except Exception as exc:  # keep the app alive on any failure
                print(f"[error] {exc}", file=sys.stderr)
                if self.on_error:
                    self.on_error(f"Error: {exc}")
            finally:
                self.jobs.task_done()

    def _speak(self, ep, text, st: Settings):
        voice = self._load(st.voice)
        sr = voice.config.sample_rate
        p = PERSONALITIES[st.personality]
        paragraphs = [x for x in re.split(r"\n+", text) if x.strip()]

        for pi, para in enumerate(paragraphs):
            for sentence, end in split_sentences(para):
                if ep != self.epoch:
                    return
                # humanising: tiny random variation in pace and pitch per sentence
                jitter = 1.0 + random.uniform(-p["jitter"], p["jitter"])
                semitones = st.pitch + p["pitch"] + random.gauss(0, p["pitch_var"])
                semitones = max(-8.0, min(8.0, semitones))
                factor = 2.0 ** (semitones / 12.0)
                rate = max(0.4, p["speed"] * st.speed)
                length_scale = (1.0 / rate) * jitter * factor  # compensate pitch speed-up

                pcm = self._synth_raw(voice, sentence, length_scale,
                                      p["expressive"], p["rhythm"], st.speaker)
                pcm = resample(pcm, factor) * st.volume
                pcm = np.clip(pcm, -32768, 32767).astype(np.int16)

                # pause after the sentence, scaled by personality
                pause = 0.0 if end == "," else PAUSE_BASE.get(end, 0.15) * p["pause"]
                pause *= random.uniform(0.85, 1.15)
                silence = np.zeros(int(sr * pause), dtype=np.int16)
                self._put(ep, sr, np.concatenate([pcm, silence]))

            if pi < len(paragraphs) - 1:
                extra = int(sr * PARAGRAPH_EXTRA * p["pause"])
                self._put(ep, sr, np.zeros(extra, dtype=np.int16))

    def _play_loop(self):
        stream, cur_sr = None, None
        while True:
            ep, sr, data = self.audio.get()
            try:
                if ep != self.epoch:
                    continue
                if sr != cur_sr:
                    if stream:
                        stream.stop()
                        stream.close()
                    stream = sd.OutputStream(samplerate=sr, channels=1, dtype="int16")
                    stream.start()
                    cur_sr = sr
                block = max(1, sr // 10)  # 100 ms blocks keep /stop responsive
                for i in range(0, len(data), block):
                    if ep != self.epoch:
                        break
                    stream.write(np.ascontiguousarray(data[i:i + block].reshape(-1, 1)))
            except Exception as exc:
                print(f"[audio error] {exc}", file=sys.stderr)
                if self.on_error:
                    self.on_error(f"Audio error: {exc}")
            finally:
                self.audio.task_done()


# --------------------------------------------------------------------------
# CLI / interactive mode
# --------------------------------------------------------------------------
HELP = """Commands:
  /voices                list voices
  /voice NAME            choose voice (nickname or any Piper model name)
  /speaker N             speaker id (multi-speaker voices such as 'libritts')
  /personality NAME      neutral calm cheerful energetic nervous authoritative sad thoughtful
  /pitch N               semitones, -6..+6 (negative = deeper, positive = higher)
  /speed X               speed multiplier, e.g. 0.8 slower, 1.3 faster
  /volume X              0.0 - 2.0
  /stop                  cut off current speech
  /settings              show current settings
  /help  /quit
Anything else you type is spoken immediately."""


def print_lists():
    print("Voices:")
    for k, (model, desc) in VOICES.items():
        print(f"  {k:<12} {model:<38} {desc}")
    print("\nPersonalities:", ", ".join(PERSONALITIES))
    print("\nMore voices: https://huggingface.co/rhasspy/piper-voices (use the model name with --voice)")


def handle_command(line: str, spk: Speaker) -> bool:
    st = spk.settings
    cmd, _, arg = line[1:].partition(" ")
    arg = arg.strip()
    try:
        if cmd in ("quit", "exit", "q"):
            return False
        elif cmd == "help":
            print(HELP)
        elif cmd == "voices":
            print_lists()
        elif cmd == "voice":
            st.voice = arg
            st.speaker = 0
            print(f"voice -> {resolve_model_name(arg)} (downloads on first use)")
        elif cmd == "speaker":
            st.speaker = int(arg)
        elif cmd == "personality":
            if arg not in PERSONALITIES:
                print("options:", ", ".join(PERSONALITIES))
            else:
                st.personality = arg
        elif cmd == "pitch":
            st.pitch = max(-6.0, min(6.0, float(arg)))
        elif cmd == "speed":
            st.speed = max(0.4, min(2.5, float(arg)))
        elif cmd == "volume":
            st.volume = max(0.0, min(2.0, float(arg)))
        elif cmd == "stop":
            spk.stop()
        elif cmd == "settings":
            print(st)
        else:
            print("Unknown command. /help for the list.")
    except ValueError:
        print("Bad value. /help for usage.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Offline neural text-to-speech")
    ap.add_argument("text", nargs="*", help="text to speak (omit for interactive mode)")
    ap.add_argument("--voice", default="lessac")
    ap.add_argument("--speaker", type=int, default=0)
    ap.add_argument("--personality", default="neutral", choices=PERSONALITIES)
    ap.add_argument("--pitch", type=float, default=0.0, help="semitones, negative = deeper")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--volume", type=float, default=1.0)
    ap.add_argument("--list", action="store_true", help="list voices and personalities")
    args = ap.parse_args()

    if args.list:
        print_lists()
        return

    settings = Settings(args.voice, args.speaker, args.personality,
                        max(-6.0, min(6.0, args.pitch)), args.speed, args.volume)
    spk = Speaker(settings)

    if args.text:
        spk.say(" ".join(args.text))
        spk.wait()
        return

    print("Type text and press Enter to hear it. /help for commands.")
    spk.say("Ready.")  # also pre-loads the model so the first real line is instant
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not handle_command(line, spk):
                break
        else:
            spk.say(line)
    spk.stop()
    spk.wait()


if __name__ == "__main__":
    main()
