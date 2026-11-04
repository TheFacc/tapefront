# tapefront

A small CLI meeting recorder. It records your mic and the system audio as the two channels of one stereo FLAC
(L = me, R = them), writes a live transcript as the meeting goes, and at the end can run an accurate final
transcription with whisper.cpp.

The live transcript is a plain file, so you can ask an assistant "what did I miss in the last 5 minutes" at any time.
Summaries and speaker diarization are not part of the tool.

## Usage

```sh
tapefront --title "Weekly sync"     # record; Ctrl-C stops, then asks to run the final pass
tapefront --no-live                 # record only
tapefront final ~/Meetings/latest   # run (or re-run) the final transcription of a session
```

Options: `--out ROOT` (default `~/Meetings`), `--live-model PATH`, `--model PATH` (whisper model), `--device N`
(whisper GPU index, default 1). The prompt only appears on a TTY; SIGTERM or a non-interactive run just stops.

Each session gets a folder `ROOT/YYYY-MM-DD_HHMM[_title-slug]`, and `ROOT/latest` points at the newest one:

| File | Content |
|---|---|
| `audio.flac` | stereo, L = mic (me), R = system audio (them); stays readable even if the process is killed |
| `live.md` | live transcript, one line per sentence: `[00:12:05] them: text` |
| `transcript.md` | final transcript, same format, both channels merged by time |

## How it works

- **Recording**: one ffmpeg reads the pulse `default` source and `@DEFAULT_MONITOR@`, downmixes each to mono and
  merges them. It follows the default devices, so plugging in a headset mid-meeting is picked up.
- **Live**: a separate worker process runs a second ffmpeg (16 kHz) and streams each channel through
  [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) with the Nemotron streaming model that
  [Handy](https://handy.computer) already downloads. If the worker crashes, recording continues.
- **Final**: each channel is extracted and transcribed with `whisper-cli` (Silero VAD to skip silences), then merged
  into `transcript.md`. Models are tried in order: full large-v3 on the GPU, large-v3 q5_0 on the GPU (fits when
  something else holds most of the VRAM), then q5_0 on the CPU. A try that fails is skipped for the second channel
  too. `--model PATH` uses only that model: GPU, then CPU.

Models are not downloaded at runtime. The live model comes from Handy's cache
(`~/.cache/huggingface/hub/models--handy-computer--nemotron-3.5-asr-streaming-0.6b-gguf`). The whisper models live
in `~/.cache/nix-transcribe/`; either one is enough, missing ones are skipped:

```sh
cd ~/.cache/nix-transcribe
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin       # 3.1 GB, ~4 GB VRAM
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin  # 1.1 GB
```

## Nix

```sh
nix run . -- --title test
nix build    # also runs the unit tests
```

The flake packages the transcribe.cpp release (prebuilt native libs, autopatchelf'd, and the Python binding),
puts ffmpeg and whisper-cpp-vulkan on the wrapper's PATH and fetches the Silero VAD model. x86_64-linux only.
