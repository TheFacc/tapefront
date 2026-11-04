#!/usr/bin/env python3
"""tapefront: record a meeting as one stereo FLAC (L = me/mic, R = them/system),
with an optional live transcript and an accurate final pass with whisper.cpp.

  tapefront [--title T] [--no-live] [--out ROOT] [--live-model PATH] [--model PATH] [--device N]
  tapefront final DIR [--model PATH] [--device N]
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
from array import array

SPEAKERS = ("me", "them")  # channel 0 (L) = mic, channel 1 (R) = system audio
LIVE_RATE = 16000
LIVE_CHUNK_BYTES = LIVE_RATE // 2 * 2 * 4  # 0.5 s of interleaved stereo float32
LIVE_MODEL_GLOB = (
    "~/.cache/huggingface/hub/models--handy-computer--nemotron-3.5-asr-streaming-0.6b-gguf"
    "/snapshots/*/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"
)
FINAL_MODEL = "~/.cache/nix-transcribe/ggml-large-v3-q5_0.bin"
VAD_MODEL_ENV = "TAPEFRONT_VAD_MODEL"  # set by the Nix wrapper
SENTENCE_END = re.compile(r"[.?!](?=\s|$)|[。？！]")
# Whitespace, or punctuation not starting a number (".5", ",000"), that must
# not start a line (e.g. a sentence end arriving after an idle flush).
LEADING_JUNK = re.compile(r"(?:\s|[.,;:?!。，？！](?!\d))*")


def warn(msg):
    print(f"tapefront: {msg}", file=sys.stderr, flush=True)


# --- pure helpers (unit-tested) -------------------------------------------

def fmt_ts(seconds):
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def format_line(seconds, speaker, text):
    return f"[{fmt_ts(seconds)}] {speaker}: {text}"


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


def find_live_model(path):
    if path:
        return os.path.expanduser(path) if os.path.isfile(os.path.expanduser(path)) else None
    hits = sorted(glob.glob(os.path.expanduser(LIVE_MODEL_GLOB)))
    return hits[-1] if hits else None


# --- record (supervisor) ------------------------------------------------

def record(args):
    root = os.path.abspath(os.path.expanduser(args.out))
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
    # Title and start time ride along as FLAC tags so `final` can rebuild the header.
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
            return final(sdir, args.model, args.device)
    print(f"Run later with: tapefront final {sdir}", flush=True)
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


def whisper(model, vad, wav, base, device):
    # VAD skips long silences, where whisper would otherwise hallucinate.
    cmd = ["whisper-cli", "-m", model, "-f", wav, "-l", "auto", "-oj", "-of", base, "--vad", "-vm", vad]
    run = dict(stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, preexec_fn=_die_with_parent)
    res = subprocess.run(cmd + ["-dev", str(device)], **run)
    if res.returncode != 0:
        warn(f"whisper-cli failed on GPU device {device}:\n{res.stderr[-1500:]}\nretrying on CPU (-ng)")
        res = subprocess.run(cmd + ["-ng"], **run)
        if res.returncode != 0:
            raise SystemExit(f"tapefront: whisper-cli failed:\n{res.stderr[-1500:]}")
    with open(base + ".json", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def final(sdir, model, device):
    sdir = os.path.abspath(sdir)
    flac = os.path.join(sdir, "audio.flac")
    if not os.path.isfile(flac):
        raise SystemExit(f"tapefront: no audio.flac in {sdir}")
    model = os.path.expanduser(model)
    if not os.path.isfile(model):
        raise SystemExit(f"tapefront: whisper model not found: {model} (pass --model PATH)")
    vad = os.environ.get(VAD_MODEL_ENV)
    if not vad or not os.path.isfile(vad):
        raise SystemExit(f"tapefront: Silero VAD model not found: set {VAD_MODEL_ENV} "
                         "(the Nix wrapper does) to a ggml-silero-*.bin")

    channels = []
    with tempfile.TemporaryDirectory(prefix="tapefront-") as tmp:
        for ch, speaker in enumerate(SPEAKERS):
            print(f"Final transcription: {speaker} (channel {ch + 1}/{len(SPEAKERS)})", flush=True)
            wav = os.path.join(tmp, f"{speaker}.wav")
            subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", flac,
                 "-af", f"pan=mono|c0=c{ch}", "-ar", "16000", wav],
                check=True, preexec_fn=_die_with_parent,
            )
            data = whisper(model, vad, wav, os.path.join(tmp, speaker), device)
            channels.append(parse_whisper_json(data, speaker))

    lines = [flac_header(flac), ""]
    lines += [format_line(t, spk, text) for t, spk, text in merge_segments(*channels)]
    path = os.path.join(sdir, "transcript.md")
    part = f"{path}.{os.getpid()}.tmp"  # unique per run (not NamedTemporaryFile: that would be 0600)
    with open(part, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(part, path)
    print(f"Transcript: {path}", flush=True)
    return 0


# --- CLI ----------------------------------------------------------------

def add_final_opts(p):
    p.add_argument("--model", default=FINAL_MODEL, help=f"whisper ggml model (default {FINAL_MODEL})")
    p.add_argument("--device", type=int, default=1, help="whisper GPU device index (default 1)")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv[:1] == ["final"]:
            p = argparse.ArgumentParser(prog="tapefront final",
                                        description="Run (or re-run) the final transcription of a session.")
            p.add_argument("dir")
            add_final_opts(p)
            a = p.parse_args(argv[1:])
            return final(a.dir, a.model, a.device)
        if argv[:1] == ["_live"]:  # internal: the live worker
            p = argparse.ArgumentParser(prog="tapefront _live")
            p.add_argument("dir")
            p.add_argument("model")
            p.add_argument("--title", default="")
            p.add_argument("--started", required=True)
            p.add_argument("--t0", type=float, required=True)
            return live_worker(p.parse_args(argv[1:]))
        p = argparse.ArgumentParser(prog="tapefront", description=__doc__.split("\n\n")[0],
                                    epilog="Subcommand: tapefront final DIR [--model PATH] [--device N]")
        p.add_argument("--title", help="meeting title (also used in the folder name)")
        p.add_argument("--no-live", action="store_true", help="record only, no live transcript")
        p.add_argument("--out", default="~/Meetings", help="sessions root (default ~/Meetings)")
        p.add_argument("--live-model", help="streaming transcribe.cpp model (default: Handy's Nemotron)")
        add_final_opts(p)
        return record(p.parse_args(argv))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
