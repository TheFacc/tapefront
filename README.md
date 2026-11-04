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
- **Final**: each channel is extracted and transcribed with `whisper-cli` (large-v3 q5_0 by default, Silero VAD to
  skip silences), on the GPU with a CPU retry, then merged into `transcript.md`.

Models are not downloaded at runtime: the live model comes from Handy's cache
(`~/.cache/huggingface/hub/models--handy-computer--nemotron-3.5-asr-streaming-0.6b-gguf`), the whisper model from
`~/.cache/nix-transcribe/ggml-large-v3-q5_0.bin` unless `--model` says otherwise.

## Nix

```sh
nix run . -- --title test
nix build    # also runs the unit tests
```

The flake packages the transcribe.cpp release (prebuilt native libs, autopatchelf'd, and the Python binding),
puts ffmpeg and whisper-cpp-vulkan on the wrapper's PATH and fetches the Silero VAD model. x86_64-linux only.
