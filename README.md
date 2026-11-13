# tapefront - Live Nix Notetaker

A small CLI meeting notetaker for Linux, with two uses:

- **Record** a meeting: your mic and the system audio go into the two channels of one stereo FLAC (L = me,
  R = them), with an optional live transcript as the meeting goes and an accurate final transcription with
  whisper.cpp at the end.
- **Transcribe** an earlier session, or any audio/video file, offline.

The live transcript is a plain file, so you can ask an assistant "what did I miss in the last 5 minutes" at any time.
Summaries and speaker diarization are not part of the tool.

## Record a meeting

```sh
tapefront                          # record, with a live transcript
tapefront --title "Weekly sync"    # same, with a title
tapefront --no-live                # record only
```

Ctrl-C stops the recording and asks whether to run the final transcription now. You can also run it later with
`tapefront ~/Meetings/latest` (see below). The question only appears on a TTY; SIGTERM or a non-interactive run
just stops.

| Option | What it does |
|---|---|
| `--title T` | meeting title, also used in the folder name |
| `--no-live` | no live transcript, record only |
| `--out ROOT` | where sessions go (default `~/Meetings`) |
| `--live-model PATH` | another streaming model for the live transcript |

The transcription options below also apply to the final pass offered at the end.

Each session gets a folder `ROOT/YYYY-MM-DD_HHMM[_title-slug]`, and `ROOT/latest` points at the newest one:

| File | Content |
|---|---|
| `audio.flac` | stereo, L = mic (me), R = system audio (them); stays readable even if the process is killed |
| `live.md` | live transcript, one line per sentence: `[00:12:05] them: text` |
| `transcript.md` | final transcript, same format, both channels merged by time |

## Transcribe a file

```sh
tapefront ~/Meetings/latest        # a session: writes its transcript.md, with me:/them: lines
tapefront call.m4a                 # any audio/video file: writes call.m4a.transcript.md
tapefront call.m4a --lang en
```

A file other than a session is downmixed to mono, so its lines have timestamps but no `me:`/`them:` labels.
Re-running replaces the transcript.

| Option | What it does |
|---|---|
| `--lang LANG` | spoken language, e.g. `en` or `it`. Default `auto`: whisper guesses from the first 30 s, and a wrong guess comes out translated into that language |
| `--only me\|them` | sessions only: transcribe just one channel, e.g. to skip a muted or noisy mic |
| `--device auto\|cpu\|N` | where whisper runs. `auto` takes the first discrete GPU, else the first GPU, else the CPU; `tapefront devices` lists the indexes |
| `--model PATH` | use only this whisper model (default: full large-v3, else q5_0) |

## Tools

```sh
tapefront devices                  # GPUs whisper can use, and which one --device auto picks
tapefront models                   # which models are on disk, and how to get the missing ones
```

## How it works

- **Recording**: one ffmpeg reads the pulse `default` source and `@DEFAULT_MONITOR@`, downmixes each to mono and
  merges them. It follows the default devices, so plugging in a headset mid-meeting is picked up.
- **Live**: a separate worker process runs a second ffmpeg (16 kHz) and streams each channel through
  [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) with a Nemotron streaming model (the one
  [Handy](https://handy.computer) uses). If the worker crashes, recording continues.
- **Final**: each channel is extracted and transcribed with `whisper-cli` (Silero VAD to skip silences), then merged
  into `transcript.md`. Models are tried in order: full large-v3 on the GPU, large-v3 q5_0 on the GPU (fits when
  something else holds most of the VRAM), then q5_0 on the CPU. A try that fails is skipped for the second channel
  too. `--model PATH` uses only that model: GPU, then CPU. With `--device cpu` (or no GPU) only the CPU try runs.

## Models

tapefront never downloads models itself. It looks for them in the Hugging Face cache (`~/.cache/huggingface/hub`,
or `$HF_HOME/hub`), where `hf download` puts them and where Handy already keeps its live model. `tapefront models`
shows which ones it found and prints the download command for the missing ones:

```sh
tapefront models
hf download handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf  # live, 0.7 GB
hf download ggerganov/whisper.cpp ggml-large-v3.bin       # final, 3.1 GB, ~4 GB VRAM
hf download ggerganov/whisper.cpp ggml-large-v3-q5_0.bin  # final, 1.1 GB
```

- `hf` comes with `huggingface_hub`: `pipx install huggingface_hub`, or
  `nix shell nixpkgs#python3Packages.huggingface-hub`.
- One whisper model is enough: without the full one, the final pass uses q5_0.
- Without the live model, tapefront records without a live transcript.
- The Nix package brings the Silero VAD model (silence detection). Elsewhere get it with
  `hf download ggml-org/whisper-vad ggml-silero-v5.1.2.bin`, or point `TAPEFRONT_VAD_MODEL` at it.

## Nix

```sh
nix run . -- --title test
nix build    # also runs the unit tests
```

The flake packages the transcribe.cpp release (prebuilt native libs, autopatchelf'd, and the Python binding),
puts ffmpeg and whisper-cpp-vulkan on the wrapper's PATH and fetches the Silero VAD model. x86_64-linux only.

## License

MIT, see [LICENSE](LICENSE).
