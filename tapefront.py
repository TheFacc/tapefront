#!/usr/bin/env python3
"""tapefront: record a meeting as one stereo FLAC (L = me/mic, R = them/system),
with an optional live transcript and an accurate final pass with whisper.cpp.

  tapefront [--title T] [--no-live] [--out ROOT] [--live-model PATH] [FINAL OPTIONS]   record
  tapefront DIR|FILE [FINAL OPTIONS]      final pass of a session dir, or transcribe any audio/video file
  tapefront devices | models
FINAL OPTIONS: [--model PATH] [--device auto|cpu|N] [--lang LANG] [--only me|them]
"""

import argparse
import ctypes
import datetime
import glob
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import wave
from array import array

DEFAULT_OUT = "~/Meetings"
SPEAKERS = ("me", "them")  # channel 0 (L) = mic, channel 1 (R) = system audio
LIVE_RATE = 16000
LIVE_CHUNK_BYTES = LIVE_RATE // 2 * 2 * 4  # 0.5 s of interleaved stereo float32
# Models are looked up in the Hugging Face cache, where `hf download REPO FILE`
# puts them (Handy downloads the live model there too): (repo, file, note).
LIVE_MODEL = ("handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf",
              "nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf", "0.7 GB")
# Default whisper models, best first: full large-v3 when it fits on the GPU,
# else q5_0 (fits beside other GPU users). Missing ones are skipped.
FINAL_MODELS = (("ggerganov/whisper.cpp", "ggml-large-v3.bin", "3.1 GB, ~4 GB VRAM"),
                ("ggerganov/whisper.cpp", "ggml-large-v3-q5_0.bin", "1.1 GB"))
FINAL_MODEL_DIR = "~/.cache/nix-transcribe"  # also looked in, for older setups
VAD_MODEL = ("ggml-org/whisper-vad", "ggml-silero-v5.1.2.bin", "1 MB")
VAD_MODEL_ENV = "TAPEFRONT_VAD_MODEL"  # set by the Nix wrapper, else VAD_MODEL
SENTENCE_END = re.compile(r"[.?!](?=\s|$)|[。？！]")
# Whitespace, or punctuation not starting a number (".5", ",000"), that must
# not start a line (e.g. a sentence end arriving after an idle flush).
LEADING_JUNK = re.compile(r"(?:\s|[.,;:?!。，？！](?!\d))*")
# A segment line on whisper-cli's stdout: "[00:01:02.340 --> 00:01:05.000]  text"
WHISPER_SEGMENT = re.compile(r"\[[\d:.]+ --> (\d+):(\d\d):(\d\d(?:\.\d+)?)\]")


def warn(msg):
    print(f"tapefront: {msg}", file=sys.stderr, flush=True)


# --- pure helpers (unit-tested) -------------------------------------------

def fmt_ts(seconds):
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def format_line(seconds, speaker, text):
    return f"[{fmt_ts(seconds)}] {speaker + ': ' if speaker else ''}{text}"


def header(title, started):
    return f"# {title or 'Meeting'} — started {started} (local)"


def slugify(title):
    return re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")[:40].strip("-")


def session_name(started, title):
    name = started.strftime("%Y-%m-%d_%H%M")
    slug = slugify(title)
    return f"{name}_{slug}" if slug else name


def make_session_dir(root, name):
    """Create ROOT/NAME, or NAME_2, NAME_3, ... if taken. Returns the path."""
    os.makedirs(root, exist_ok=True)
    n = 1
    while True:
        path = os.path.join(root, name if n == 1 else f"{name}_{n}")
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            n += 1


def point_latest(root, target):
    """Atomically repoint ROOT/latest at target (relative link)."""
    tmp = os.path.join(root, f".latest.{os.getpid()}")
    if os.path.lexists(tmp):
        os.unlink(tmp)
    os.symlink(os.path.basename(target), tmp)
    os.replace(tmp, os.path.join(root, "latest"))


class LiveLines:
    """Cuts one channel's streaming text into transcript lines.

    transcribe.cpp 0.4.1 stops advancing `committed` for the Nemotron model
    after the first sentence end (commit_policy auto and stable_prefix n=2/3
    alike), while `full` keeps growing and was never revised in tests, so the
    live view follows `full`. `written` is the part of `full` already turned
    into lines; the pending text after it is recomputed on every update, so a
    revision inside it needs no special handling. If a revision reaches into
    written text, pending text resumes where the end of the written text shows
    up again in the revised one, else at the next word boundary after the
    revision point: a revised word may be skipped or repeated, but later
    sentences are neither replayed nor lost.

    A line ends at a sentence end once text follows it (so "3." can still
    become "3.5"), after `idle` seconds without change (all pending text), or
    `max_age` seconds since the line started while text keeps coming (cut at
    the last space, keeping the word that may still be growing)."""

    def __init__(self, idle=1.5, max_age=15.0):
        self.idle, self.max_age = idle, max_age
        self.full = self.written = ""
        self.start = self.changed = None  # audio times: pending text began / `full` last changed

    def update(self, full, now, final=False):
        """Take the stream's current `full` text at audio time `now`.
        Returns a finished (start, text) line or None; final=True emits all pending text."""
        if full != self.full:
            self.full, self.changed = full, now
        skip = LEADING_JUNK.match(full, self._resume(full)).end()
        pending = full[skip:]
        if not pending:
            self.start = None
            return None
        if self.start is None:
            self.start = now
        cut = self._cut(pending, now, final)
        if not cut:
            return None
        start, line = self.start, pending[:cut].strip()
        self.written = full[:skip + cut]
        self.start = now if pending[cut:].strip() else None
        return (start, line) if line else None

    def _resume(self, full):
        """Index in `full` where the not yet written text starts."""
        if full.startswith(self.written):
            return len(self.written)
        n = len(os.path.commonprefix([self.written, full]))
        # The written text after the revision point was already emitted: resume
        # after the longest tail of it (3+ chars, last 40 only) found again.
        tail = self.written[n:][-40:]
        for i in range(len(tail) - 2):
            j = full.find(tail[i:], n)
            if j >= 0:
                return j + len(tail) - i
        # Not found: resume at the next word boundary, or at the revision
        # point when there is none (CJK), so text may repeat but is never lost.
        m = re.compile(r"\s").search(full, n)
        return m.start() if m else n

    def _cut(self, pending, now, final):
        """How many chars of `pending` make a finished line (0 = none yet)."""
        # Final, or no new text for `idle` seconds: the last word is done.
        if final or now - self.changed >= self.idle:
            return len(pending)
        ends = [m.end() for m in SENTENCE_END.finditer(pending)]
        followed = [e for e in ends if pending[e:].strip()]
        if followed:
            return followed[-1]
        if now - self.start < self.max_age:
            return 0
        if ends and not pending[ends[-1]:].strip():
            return len(pending)
        # Overdue while text keeps growing: cut at the last space, since the
        # last word may be a fragment the model is still extending.
        # No space at all (e.g. CJK): keep the last char pending.
        spaces = [m.start() for m in re.finditer(r"\s", pending)]
        return spaces[-1] if spaces else len(pending) - 1


def parse_whisper_json(data, speaker):
    """whisper-cli -oj output -> [(start_seconds, speaker, text)], empty texts dropped."""
    out = []
    for seg in data.get("transcription", []):
        text = seg.get("text", "").strip()
        if text:
            out.append((seg["offsets"]["from"] / 1000, speaker, text))
    return out


def whisper_attempts(models, device):
    """(model, GPU device or None for CPU) tries in order: each model on the
    GPU, then the last (smallest) one on the CPU."""
    gpu = [] if device is None else [(m, device) for m in models]
    return gpu + [(models[-1], None)]


# ggml's list of Vulkan devices, e.g. "ggml_vulkan: 1 = NVIDIA RTX ... | uma: 0 | ..."
# (uma: 1 = integrated GPU sharing system memory).
VK_DEVICE = re.compile(r"ggml_vulkan: (\d+) = (.+?) \| uma: (\d)")


def gpu_devices():
    """[(index, name, integrated)] of the GPUs whisper-cli can use, in its own
    numbering. ggml prints them before opening the model, so a dummy model
    gives the list in a fraction of a second."""
    try:
        res = subprocess.run(["whisper-cli", "-m", os.devnull, "-f", os.devnull],
                             capture_output=True, text=True, errors="replace", timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [(int(m[1]), m[2], m[3] == "1") for m in VK_DEVICE.finditer(res.stdout + res.stderr)]


def pick_device(devices):
    """The first discrete GPU, else the first GPU, else None (CPU). whisper's
    own default is device 0, which on laptops is usually the integrated one."""
    for index, _, integrated in devices:
        if not integrated:
            return index
    return devices[0][0] if devices else None


def device_arg(value):
    if value in ("auto", "cpu"):
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected auto, cpu or a GPU index (see: tapefront devices)")


def list_devices():
    devices = gpu_devices()
    if not devices:
        print("No GPU found: whisper runs on the CPU.")
        return 0
    auto = pick_device(devices)
    for index, name, integrated in devices:
        print(f"{index}  {name} ({'integrated' if integrated else 'discrete'})"
              + ("  <- auto" if index == auto else ""))
    print("Use one with --device N, or --device cpu.")
    return 0


def segment_end(line):
    """End time in seconds of a whisper-cli segment line, else None."""
    m = WHISPER_SEGMENT.match(line)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else None


def file_transcript_path(path):
    """meeting.m4a -> meeting.m4a.transcript.md (never clashes with notes)."""
    return path + ".transcript.md"


def merge_segments(*channels):
    """Merge per-channel segment lists by start time (stable: me before them on ties)."""
    return sorted((seg for ch in channels for seg in ch), key=lambda seg: seg[0])


# --- processes ----------------------------------------------------------

_libc = ctypes.CDLL(None, use_errno=True)


def _die_with_parent():
    # preexec_fn: get SIGTERM if the parent dies, so a SIGKILLed supervisor
    # still leaves ffmpeg to close the FLAC instead of recording forever.
    _libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def capture_cmd(rate, output_args):
    """ffmpeg reading mic (default source) and system audio (default sink
    monitor), each downmixed to mono, merged into stereo L = mic, R = system.
    The pulse defaults are used on purpose: if the default device changes
    mid-meeting (headset plugged in), the recording follows it."""
    mono = f"aresample={rate}:async=1,aformat=channel_layouts=mono"
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-loglevel", "error",
        "-f", "pulse", "-i", "default",
        "-f", "pulse", "-i", "@DEFAULT_MONITOR@",
        "-filter_complex", f"[0:a]{mono}[me];[1:a]{mono}[them];[me][them]amerge=inputs=2[out]",
        "-map", "[out]", *output_args,
    ]


def stop_child(proc, sig, timeout):
    """Ask a child to finish with `sig`; SIGKILL it after `timeout`."""
    if proc.poll() is None:
        proc.send_signal(sig)
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            warn(f"{proc.args[0]} did not stop in {timeout:.0f}s, killing it")
            proc.kill()
            proc.wait()


def hf_cache_file(repo, name):
    """`name` from `repo` in the Hugging Face cache (newest snapshot), or None."""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.join(os.environ.get("XDG_CACHE_HOME") or "~/.cache", "huggingface"),
        "hub")
    pattern = os.path.join(os.path.expanduser(hub), "models--" + repo.replace("/", "--"), "snapshots", "*", name)
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    return hits[-1] if hits else None


def find_model(spec):
    """Where a (repo, file, note) model is on disk, or None."""
    repo, name, _ = spec
    if spec in FINAL_MODELS:
        old = os.path.join(os.path.expanduser(FINAL_MODEL_DIR), name)
        if os.path.isfile(old):
            return old
    if spec == VAD_MODEL and os.path.isfile(os.environ.get(VAD_MODEL_ENV, "")):
        return os.environ[VAD_MODEL_ENV]
    return hf_cache_file(repo, name)


def find_live_model(path):
    if path:
        return os.path.expanduser(path) if os.path.isfile(os.path.expanduser(path)) else None
    return find_model(LIVE_MODEL)


def list_models():
    """Show which models are on disk, and how to get the missing ones."""
    missing = []
    for title, specs in (("Live transcript", [LIVE_MODEL]),
                         ("Final pass (whisper, best first; one is enough)", FINAL_MODELS),
                         ("Final pass, silence detection (needed)", [VAD_MODEL])):
        print(title + ":")
        for spec in specs:
            path = find_model(spec)
            print(f"  {'found  ' if path else 'missing'}  {spec[1]} ({spec[2]})" + (f"\n           {path}" if path else ""))
            if not path:
                missing.append(spec)
    if missing:
        print("\nGet the missing ones (into the Hugging Face cache) with the hf CLI from huggingface_hub:")
        for repo, name, _ in missing:
            print(f"  hf download {repo} {name}")
    return 0


# --- record (supervisor) ------------------------------------------------

def record(args):
    root = os.path.abspath(os.path.expanduser(args.out or DEFAULT_OUT))
    started = datetime.datetime.now()
    sdir = make_session_dir(root, session_name(started, args.title))
    point_latest(root, sdir)
    stamp = started.strftime("%Y-%m-%d %H:%M:%S")

    stop_sig = None

    def on_signal(sig, _frame):
        nonlocal stop_sig
        stop_sig = sig

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    flac = os.path.join(sdir, "audio.flac")
    # Title and start time ride along as FLAC tags so the final pass can rebuild the header.
    rec = subprocess.Popen(
        capture_cmd(48000, ["-c:a", "flac", "-metadata", f"title={args.title or ''}",
                            "-metadata", f"date={stamp}", flac]),
        start_new_session=True, preexec_fn=_die_with_parent,
    )
    t0 = time.time()
    worker = None
    failed = False
    try:
        print(f"Recording to {sdir}", flush=True)
        if not args.no_live:
            model = find_live_model(args.live_model)
            if model is None:
                warn("live model not found (pass --live-model PATH); recording without live transcript")
            else:
                cmd = [sys.executable, os.path.abspath(__file__), "_live", sdir, model,
                       "--title", args.title or "", "--started", stamp, "--t0", repr(t0)]
                try:
                    # Own session: a terminal Ctrl-C reaches only the supervisor,
                    # which then stops the children in order.
                    worker = subprocess.Popen(cmd, start_new_session=True, preexec_fn=_die_with_parent)
                    print(f"Live transcript: {os.path.join(sdir, 'live.md')}", flush=True)
                except OSError as e:
                    warn(f"live transcript disabled: {e}")
        print("Press Ctrl-C to stop.", flush=True)

        while stop_sig is None:
            if rec.poll() is not None:
                warn(f"recording ffmpeg exited unexpectedly (code {rec.returncode})")
                failed = True
                break
            if worker is not None and worker.poll() is not None:
                warn(f"live transcript stopped (worker exit code {worker.returncode}); recording continues")
                worker = None
            time.sleep(0.2)
    finally:
        # Recording first, then live.
        stop_child(rec, signal.SIGINT, 10)
        if worker is not None:
            stop_child(worker, signal.SIGTERM, 20)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

    if failed:
        return 1
    print(f"Saved {flac}", flush=True)
    if stop_sig == signal.SIGINT and sys.stdin.isatty():
        try:
            answer = input("Run final transcription now? [Y/n] ").strip().lower()
        except EOFError:
            answer = "n"
        if answer in ("", "y", "yes"):
            return final(sdir, args.model, args.device, args.lang, args.only)
    print(f"Run later with: tapefront {sdir}", flush=True)
    return 0


# --- live worker (separate process: a native crash can't hurt the recording) ---

def live_worker(args):
    stop, ff = [], None

    def on_stop(*_):
        stop.append(True)
        if ff is not None:
            ff.kill()  # a read() blocked on the capture pipe then returns EOF

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)

    path = os.path.join(args.dir, "live.md")
    out = open(path, "a", encoding="utf-8")
    if out.tell() == 0:
        out.write(header(args.title, args.started) + "\n\n")
        out.flush()

    try:
        import transcribe_cpp as tc
        tc.set_log_callback(lambda level, msg: level in (2, 3) and warn(msg.rstrip()))  # warn/error only
        # One active stream per Model, so one Model per channel (weights are mmapped, shared).
        streams = [tc.Model(args.model, backend="cpu").session(n_threads=4).stream()
                   for _ in SPEAKERS]
    except Exception as e:  # missing/broken native lib or model
        warn(f"live transcript disabled: {e}")
        return 1

    ff = subprocess.Popen(
        capture_cmd(LIVE_RATE, ["-f", "f32le", "-c:a", "pcm_f32le", "pipe:1"]),
        stdout=subprocess.PIPE, preexec_fn=_die_with_parent,
    )
    # Live timestamps are relative to the recording start, not to this
    # (later, after model load) capture start.
    offset = time.time() - args.t0
    lines = [LiveLines() for _ in SPEAKERS]

    def take(i, now, final=False):
        line = lines[i].update(streams[i].text().full, now, final)
        if not line:
            return
        text = format_line(line[0], SPEAKERS[i], line[1])
        out.write(text + "\n")
        out.flush()
        try:
            print(text, flush=True)
        except OSError:  # terminal gone; live.md is what matters
            pass

    now = offset
    try:
        frames = 0
        while not stop:
            data = ff.stdout.read(LIVE_CHUNK_BYTES)
            if not data:
                break
            pcm = array("f", data[:len(data) - len(data) % 8])
            frames += len(pcm) // 2
            now = offset + frames / LIVE_RATE
            for i, st in enumerate(streams):
                st.feed(pcm[i::2])  # deinterleave: L = me, R = them
                take(i, now)
    finally:
        ff.kill()  # its output is disposable; SIGTERM would make it complain about the pipe
        ff.wait()
    for i, st in enumerate(streams):
        st.finalize()
        take(i, now, final=True)
    out.close()
    return 0


# --- final pass ---------------------------------------------------------

def flac_header(flac):
    """Rebuild the transcript header from the FLAC's title/date tags."""
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", flac],
        capture_output=True, text=True,
    )
    try:
        tags = {k.lower(): v for k, v in json.loads(res.stdout)["format"]["tags"].items()}
    except (ValueError, KeyError):
        tags = {}
    return header(tags.get("title"), tags.get("date", "unknown"))


def _run_whisper(cmd, env, wav):
    """Run whisper-cli, showing how far into the audio it is (on a TTY) from the
    segments it prints as it goes. Returns (exit code, stderr)."""
    tty = sys.stdout.isatty()
    try:
        with wave.open(wav) as w:
            total = w.getnframes() / w.getframerate()
    except (OSError, wave.Error):
        total = 0
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8",
                                errors="replace", env=env, preexec_fn=_die_with_parent)
        for line in proc.stdout:
            end = segment_end(line)
            if tty and end is not None and total:
                print(f"\r    {fmt_ts(end)} / {fmt_ts(total)} ({min(100, int(100 * end / total))}%)",
                      end="", flush=True)
        rc = proc.wait()
        if tty and total:
            print(flush=True)
        err.seek(0)
        return rc, err.read()


def whisper(attempts, vad, wav, base, lang="auto"):
    """Run whisper-cli down the `attempts` list until one works. Failed tries
    are dropped from the list, so the next channel starts where this one ended."""
    while True:
        model, dev = attempts[0]
        where = "CPU" if dev is None else f"GPU {dev}"
        print(f"  {os.path.basename(model)} on {where}", flush=True)
        # VAD skips long silences, where whisper would otherwise hallucinate.
        # -mc 0: don't feed the previous text back as context; with it, one
        # repeated phrase can loop for the rest of the file.
        cmd = ["whisper-cli", "-m", model, "-f", wav, "-l", lang, "-mc", "0", "-oj", "-of", base,
               "--vad", "-vm", vad]
        cmd += ["-ng"] if dev is None else ["-dev", str(dev)]
        # Without this, ggml uploads the weights through host-visible VRAM, which
        # on NVIDIA takes ~5 min for the 3 GB full model (2-3 s with it).
        env = {"GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM": "1", **os.environ}
        rc, err = _run_whisper(cmd, env, wav)
        if rc == 0 and os.path.exists(base + ".json"):
            break
        if rc == 0:
            # whisper-cli exits 0 after an argument error, such as an unknown
            # --lang; another model or device would hit it too.
            errors = [l for l in err.splitlines() if l.startswith("error:")]
            raise SystemExit("tapefront: whisper-cli wrote no transcript:\n" + ("\n".join(errors) or err[-1500:]))
        err = err[-1500:]
        if len(attempts) == 1:
            raise SystemExit(f"tapefront: whisper-cli failed:\n{err}")
        warn(f"whisper-cli failed ({os.path.basename(model)} on {where}):\n{err}")
        attempts.pop(0)
    with open(base + ".json", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def final_models(model):
    """The whisper models to try: --model alone, else the defaults on disk."""
    if model:
        model = os.path.expanduser(model)
        if not os.path.isfile(model):
            raise SystemExit(f"tapefront: whisper model not found: {model}")
        return [model]
    found = [path for path in map(find_model, FINAL_MODELS) if path]
    if not found:
        raise SystemExit("tapefront: no whisper model found; `tapefront models` shows how to get one, "
                         "or pass --model PATH")
    if not find_model(FINAL_MODELS[0]):
        warn(f"full large-v3 not found (see `tapefront models`), using {os.path.basename(found[0])}")
    return found


def final(path, model, device, lang="auto", only=None):
    """Final transcript of a session dir (per channel, me/them, or just the
    `only` one) or of any audio/video file (downmixed to mono, no speaker labels)."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        src = os.path.join(path, "audio.flac")
        if not os.path.isfile(src):
            raise SystemExit(f"tapefront: no audio.flac in {path}")
        out, head = os.path.join(path, "transcript.md"), flac_header(src)
        tracks = [(speaker, ["-af", f"pan=mono|c0=c{ch}"]) for ch, speaker in enumerate(SPEAKERS)
                  if only in (None, speaker)]
    elif only:
        raise SystemExit("tapefront: --only works on session dirs, not on a single file")
    elif os.path.isfile(path):
        src, out, head = path, file_transcript_path(path), f"# {os.path.basename(path)}"
        tracks = [(None, ["-map", "0:a:0", "-ac", "1"])]
    else:
        raise SystemExit(f"tapefront: no such file or directory: {path}")
    models = final_models(model)
    if device == "auto":
        device = pick_device(gpu_devices())
    attempts = whisper_attempts(models, None if device == "cpu" else device)
    vad = find_model(VAD_MODEL)
    if not vad:
        raise SystemExit("tapefront: Silero VAD model not found; `tapefront models` shows how to get it")

    channels = []
    with tempfile.TemporaryDirectory(prefix="tapefront-") as tmp:
        for n, (speaker, pick) in enumerate(tracks, 1):
            name = speaker or "audio"
            print(f"Final transcription: {name} ({n}/{len(tracks)})", flush=True)
            wav = os.path.join(tmp, f"{name}.wav")
            res = subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                 *pick, "-ar", "16000", wav],
                stderr=subprocess.PIPE, text=True, preexec_fn=_die_with_parent,
            )
            if res.returncode != 0:
                raise SystemExit(f"tapefront: ffmpeg could not read audio from {src}:\n{res.stderr[-800:]}")
            data = whisper(attempts, vad, wav, os.path.join(tmp, name), lang)
            channels.append(parse_whisper_json(data, speaker))

    lines = [head, ""]
    lines += [format_line(t, spk, text) for t, spk, text in merge_segments(*channels)]
    part = f"{out}.{os.getpid()}.tmp"  # unique per run (not NamedTemporaryFile: that would be 0600)
    with open(part, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(part, out)
    print(f"Transcript: {out}", flush=True)
    return 0


# --- CLI ----------------------------------------------------------------

def add_final_opts(p):
    p.add_argument("--model", help="whisper ggml model (default: full large-v3, else q5_0; "
                                   "see `tapefront models`)")
    p.add_argument("--device", type=device_arg, default="auto",
                   help="whisper GPU index from `tapefront devices`, or cpu (default: auto, the first discrete GPU)")
    p.add_argument("--lang", default="auto",
                   help="spoken language, e.g. en or it (default: auto, guessed from the first 30 s)")
    p.add_argument("--only", choices=SPEAKERS,
                   help="transcribe only this channel of a session: me (mic) or them (system audio)")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv[:1] == ["devices"]:
            return list_devices()
        if argv[:1] == ["models"]:
            return list_models()
        if argv[:1] == ["_live"]:  # internal: the live worker
            p = argparse.ArgumentParser(prog="tapefront _live")
            p.add_argument("dir")
            p.add_argument("model")
            p.add_argument("--title", default="")
            p.add_argument("--started", required=True)
            p.add_argument("--t0", type=float, required=True)
            return live_worker(p.parse_args(argv[1:]))
        p = argparse.ArgumentParser(prog="tapefront", description=__doc__.split("\n\n")[0],
                                    epilog="Also: tapefront devices (list GPUs), tapefront models (check models)")
        p.add_argument("path", nargs="?",
                       help="session dir (re-run its final pass) or any audio/video file to transcribe; "
                            "without it, tapefront records")
        rec = p.add_argument_group("recording")
        rec.add_argument("--title", help="meeting title (also used in the folder name)")
        rec.add_argument("--no-live", action="store_true", help="record only, no live transcript")
        rec.add_argument("--out", help=f"sessions root (default {DEFAULT_OUT})")
        rec.add_argument("--live-model", help="streaming transcribe.cpp model (default: Handy's Nemotron)")
        add_final_opts(p.add_argument_group("final pass"))
        a = p.parse_args(argv)
        if a.path is None:
            return record(a)
        if a.title or a.no_live or a.out or a.live_model:
            p.error("--title, --no-live, --out and --live-model are for recording, not with a path")
        return final(a.path, a.model, a.device, a.lang, a.only)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
