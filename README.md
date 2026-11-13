# tapefront: live meeting notes your LLM can read

A meeting recorder for Linux that writes the transcript to a Markdown file while the meeting is still going, so
any LLM can read it.

```
$ tail -3 ~/Meetings/latest/live.md
[00:12:05] them: so the release moves to Thursday
[00:12:09] me: fine, I'll tell the docs team
[00:12:14] them: and we still need someone for the on-call rota
```

Point Claude Code, Codex or any assistant that can read files at `~/Meetings/latest/live.md` and ask
"what did I miss in the last 5 minutes?" or "what was I asked to do?" mid-meeting. When the meeting ends, tapefront runs a slower, more accurate
whisper.cpp pass and writes `transcript.md` next to it. It can also transcribe any audio or video file offline.

tapefront records your mic and the system audio as separate channels, so lines say `me:` or `them:`. It doesn't
tell the other speakers apart and doesn't write summaries; leave that to the LLMs, they're scarily smart.

I wrote it because [tapeback](https://github.com/yastcher/tapeback), a fuller Linux recorder with speaker
diarization, transcribes live in 60-second batches, only when you turn that on, and its README doesn't describe a
live file another tool can read. tapefront streams each channel and appends every sentence to `live.md` as soon
as it ends. And is overall much simpler for my use.

## Installation

tapefront needs Linux with PipeWire or PulseAudio. With Nix, try it without installing anything:

```sh
nix run github:TheFacc/tapefront -- --title "Weekly sync"
```

To keep it, add the flake to your NixOS or Home Manager config:

```nix
# flake.nix
inputs.tapefront.url = "github:TheFacc/tapefront";

# a NixOS module (or home.packages in Home Manager)
environment.systemPackages = [ inputs.tapefront.packages.x86_64-linux.default ];
```

The flake brings ffmpeg, whisper.cpp with Vulkan and the live transcription library. You only fetch the models
(see [Models](#models)). It builds for x86_64-linux only.

Without Nix it should work on any distro, but nobody has tested that yet. You need:

- Python 3 and ffmpeg built with PulseAudio input (most distro builds are);
- `whisper-cli` from [whisper.cpp](https://github.com/ggml-org/whisper.cpp), built with Vulkan to run on the GPU;
- `pip install transcribe-cpp` for the live transcript (without it, tapefront records and skips the live part);
- the models, including the Silero VAD model (see [Models](#models)).

Then run `./tapefront.py` from a clone instead of `tapefront`.

## Record a meeting

```sh
tapefront                          # record, with a live transcript
tapefront --title "Weekly sync"    # same, with a title
tapefront --no-live                # record only
```

Ctrl-C stops the recording and asks whether to run the final transcription now. You can also run it later with
`tapefront ~/Meetings/latest`. Without a terminal (SIGTERM, a script) tapefront just stops.

| Option | What it does |
|---|---|
| `--title T` | meeting title, also used in the folder name |
| `--no-live` | record only, no live transcript |
| `--out ROOT` | where sessions go (default `~/Meetings`) |
| `--live-model PATH` | another streaming model for the live transcript |

The options in [Transcribe a file](#transcribe-a-file) also work here, for the final pass at the end.

Each session gets a folder `ROOT/YYYY-MM-DD_HHMM[_title-slug]`, and `ROOT/latest` points at the newest one:

| File | Content |
|---|---|
| `audio.flac` | stereo, left = your mic (`me`), right = system audio (`them`); still readable if tapefront is killed |
| `live.md` | live transcript, one line per sentence: `[00:12:05] them: text` |
| `transcript.md` | final transcript, same format, both channels merged by time |

## Transcribe a file

```sh
tapefront ~/Meetings/latest        # a session folder: writes its transcript.md, with me:/them: lines
tapefront call.m4a                 # any other audio/video file: writes call.m4a.transcript.md
tapefront call.m4a --lang en
```

Only a session folder keeps the two channels apart, because only there tapefront knows that left is your mic and
right is everyone else. Pass the folder, not the `audio.flac` inside it. Any other file, stereo or not, gets mixed
down to mono, so its lines have timestamps but no `me:`/`them:` labels. Running it again replaces the transcript.

| Option | What it does |
|---|---|
| `--lang LANG` | spoken language, e.g. `en` or `it`. The default `auto` guesses from the first 30 s; a wrong guess comes out translated into that language |
| `--only me\|them` | sessions only: transcribe one channel, e.g. to skip a muted or noisy mic |
| `--device auto\|cpu\|N` | where whisper runs. `auto` takes the first discrete GPU, else the first GPU, else the CPU |
| `--model PATH` | use only this whisper model |

## Other commands

```sh
tapefront devices                  # GPUs whisper can use, with their --device numbers and auto's pick
tapefront models                   # which models are on disk, and how to get the missing ones
```

## Models

tapefront never downloads models. It looks for them in the Hugging Face cache (`~/.cache/huggingface/hub`, or
`$HF_HOME/hub`), where `hf download` puts them and where [Handy](https://handy.computer) already keeps its live
model. `tapefront models` prints the download command for each missing one:

```sh
hf download handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf  # live, 0.7 GB
hf download ggerganov/whisper.cpp ggml-large-v3.bin       # final, 3.1 GB, ~4 GB VRAM
hf download ggerganov/whisper.cpp ggml-large-v3-q5_0.bin  # final, 1.1 GB
hf download ggml-org/whisper-vad ggml-silero-v5.1.2.bin   # silence detection, 1 MB (the Nix package has it)
```

`hf` comes with `huggingface_hub` (`pipx install huggingface_hub`, or
`nix shell nixpkgs#python3Packages.huggingface-hub`). One whisper model is enough. To keep the VAD model
elsewhere, point `TAPEFRONT_VAD_MODEL` at it.

## How it works

One ffmpeg process records the PulseAudio `default` source and `@DEFAULT_MONITOR@`, mixes each down to mono and
writes them as the two channels of `audio.flac`. It follows the default devices, so a headset you plug in
mid-meeting gets picked up.

For the live transcript, a separate worker process runs a second ffmpeg at 16 kHz and streams each channel through
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) on the CPU, with the Nemotron streaming model
that Handy uses. If the worker crashes, the recording goes on.

The final pass splits the channels and runs `whisper-cli` on each, with Silero VAD to skip silences, then merges
the results by time. It tries the full large-v3 on the GPU first, then large-v3 q5_0 on the GPU (it fits when
something else holds most of the VRAM), then q5_0 on the CPU. A try that fails is skipped for the second channel
too. With `--model`, it tries that model on the GPU, then on the CPU. With `--device cpu`, or no GPU, it tries
only the CPU.

## Development

```sh
python3 -m unittest    # or nix build, which runs the tests too
```

## License

MIT, see [LICENSE](LICENSE).
