import datetime
import os
import tempfile
import json
import subprocess
import unittest
from unittest import mock

import tapefront as tf


class TimestampTest(unittest.TestCase):
    def test_format(self):
        self.assertEqual(tf.fmt_ts(0), "00:00:00")
        self.assertEqual(tf.fmt_ts(59.9), "00:00:59")
        self.assertEqual(tf.fmt_ts(3600 + 12 * 60 + 5), "01:12:05")
        self.assertEqual(tf.fmt_ts(-1), "00:00:00")

    def test_line_and_header(self):
        self.assertEqual(tf.format_line(725, "me", "hello"), "[00:12:05] me: hello")
        self.assertEqual(tf.format_line(725, None, "hello"), "[00:12:05] hello")
        self.assertEqual(tf.header(None, "2026-11-04 10:00:00"),
                         "# Meeting — started 2026-11-04 10:00:00 (local)")
        self.assertEqual(tf.header("Standup", "x"), "# Standup — started x (local)")


class LiveLinesTest(unittest.TestCase):
    def test_sentence_end_waits_for_following_text(self):
        b = tf.LiveLines()
        self.assertIsNone(b.update(" Hello", 1.0))
        self.assertIsNone(b.update(" Hello there friend.", 2.0))
        self.assertEqual(b.update(" Hello there friend. How", 2.5), (1.0, "Hello there friend."))
        self.assertEqual(b.update(" Hello there friend. How are you? Fine", 3.0), (2.5, "How are you?"))

    def test_complete_sentence_flushes_on_idle(self):
        b = tf.LiveLines(idle=1.5)
        self.assertIsNone(b.update("好的。", 1.0))
        self.assertEqual(b.update("好的。", 2.5), (1.0, "好的。"))
        self.assertIsNone(b.update("好的。", 5.0))

    def test_idle_flush_emits_whole_text(self):
        b = tf.LiveLines(idle=1.5)
        self.assertIsNone(b.update(" maybe yes because this is taking", 1.0))
        self.assertEqual(b.update(" maybe yes because this is taking", 2.5),
                         (1.0, "maybe yes because this is taking"))
        self.assertEqual(b.update(" maybe yes because this is taking forever. And", 3.0),
                         (3.0, "forever."))

    def test_revision_inside_pending_text(self):
        b = tf.LiveLines(idle=1.5)
        self.assertIsNone(b.update("We need fifteen", 1.0))
        self.assertIsNone(b.update("We need fifty people.", 1.5))
        self.assertEqual(b.update("We need fifty people.", 3.0), (1.0, "We need fifty people."))

    def test_revision_before_published_text_is_ignored(self):
        b = tf.LiveLines(idle=1.5)
        b.update("One two three", 1.0)
        self.assertEqual(b.update("One two three", 2.5), (1.0, "One two three"))
        self.assertIsNone(b.update("One", 3.0))  # shrank below pub: nothing re-emitted
        self.assertIsNone(b.update("One", 3.0, final=True))

    def test_shrink_then_regrow_does_not_repeat_published_text(self):
        b = tf.LiveLines(idle=1.5)
        b.update("One two three", 1.0)
        self.assertEqual(b.update("One two three", 2.5), (1.0, "One two three"))
        b.update("One", 3.0)
        b.update("One two three four.", 3.5)
        self.assertEqual(b.update("One two three four.", 5.0), (3.5, "four."))

    def test_revised_leading_space_keeps_first_char(self):
        b = tf.LiveLines(idle=1.5)
        b.update(" hello", 1.0)
        b.update("hello there", 1.5)
        self.assertEqual(b.update("hello there", 3.0), (1.0, "hello there"))

    def test_late_punctuation_after_idle_flush_is_dropped(self):
        b = tf.LiveLines(idle=1.5)
        b.update(" so slow and dead", 36.0)
        self.assertEqual(b.update(" so slow and dead", 37.5), (36.0, "so slow and dead"))
        self.assertIsNone(b.update(" so slow and dead. I", 38.0))
        self.assertEqual(b.update(" so slow and dead. I", 38.0, final=True), (38.0, "I"))

    def test_leading_punctuation_after_flush_is_dropped(self):
        b = tf.LiveLines(idle=1.5)
        b.update("All done.", 1.0)
        self.assertEqual(b.update("All done.", 2.5), (1.0, "All done."))
        self.assertIsNone(b.update("All done. , okay then.", 3.0))
        self.assertEqual(b.update("All done. , okay then.", 4.5), (3.0, "okay then."))

    def test_revision_into_written_text_resumes_at_word_boundary(self):
        b = tf.LiveLines(idle=1.5)
        b.update("We need fifteen", 1.0)
        self.assertEqual(b.update("We need fifteen", 2.5), (1.0, "We need fifteen"))
        b.update("We need fifty people.", 3.0)
        self.assertEqual(b.update("We need fifty people.", 4.5), (3.0, "people."))

    def test_revision_does_not_replay_later_sentences(self):
        b = tf.LiveLines(idle=1.5)
        text = "We need fifteen people. The budget is approved. Next topic"
        b.update(text, 1.0)
        b.update(text, 2.5)
        text = text.replace("fifteen", "fifty") + " is hiring."
        b.update(text, 3.0)
        self.assertEqual(b.update(text, 4.5), (3.0, "is hiring."))

    def test_revision_without_spaces_does_not_suppress_text(self):
        b = tf.LiveLines(idle=1.5)
        b.update("今天我们开会", 1.0)
        self.assertEqual(b.update("今天我们开会", 2.5), (1.0, "今天我们开会"))
        b.update("今天咱们开会讨论预算", 3.0)
        self.assertEqual(b.update("今天咱们开会讨论预算", 4.5), (3.0, "讨论预算"))
        b.update("今天咱们开会讨论预算。明天继续。", 5.0)
        self.assertEqual(b.update("今天咱们开会讨论预算。明天继续。", 6.5), (5.0, "明天继续。"))

    def test_decimal_after_idle_flush_keeps_its_point(self):
        b = tf.LiveLines(idle=1.5)
        b.update("It costs 3", 1.0)
        self.assertEqual(b.update("It costs 3", 2.5), (1.0, "It costs 3"))
        b.update("It costs 3.5 million", 3.0)
        self.assertEqual(b.update("It costs 3.5 million", 4.5), (3.0, ".5 million"))

    def test_max_age_without_spaces_keeps_last_char(self):
        b = tf.LiveLines(idle=1.5, max_age=15.0)
        b.update("今天我们", 0.0)
        self.assertEqual(b.update("今天我们讨论", 15.0), (0.0, "今天我们讨"))

    def test_decimal_point_does_not_split(self):
        b = tf.LiveLines()
        self.assertIsNone(b.update(" version 3.5 is", 1.0))
        self.assertIsNone(b.update(" version 3.5 is", 2.0))

    def test_max_age_cuts_growing_text(self):
        b = tf.LiveLines(idle=1.5, max_age=15.0)
        t, text, line = 0.0, "", None
        while line is None:
            t += 0.5
            text += " word"
            line = b.update(text, t)
        self.assertEqual(line[0], 0.5)
        self.assertAlmostEqual(t, 15.5)
        self.assertEqual(line[1], " ".join(["word"] * 30))  # last word kept pending

    def test_whitespace_only_is_not_a_line(self):
        b = tf.LiveLines()
        self.assertIsNone(b.update(" ", 1.0))
        self.assertIsNone(b.update(" ", 5.0))
        self.assertIsNone(b.update(" ", 5.0, final=True))
        self.assertEqual(b.update("  hi.", 6.0, final=True), (6.0, "hi."))

    def test_final_emits_fragment(self):
        b = tf.LiveLines()
        b.update(" trailing wor", 4.0)
        self.assertEqual(b.update(" trailing wor", 4.2, final=True), (4.0, "trailing wor"))


class FakeProc:
    """Popen stand-in: `outcome(cmd)` returns (exit code, stdout lines)."""
    def __init__(self, outcome, cmd, **kw):
        self.rc, self.stdout = outcome(cmd)
        kw["stderr"].write("" if self.rc == 0 else "ErrorOutOfDeviceMemory")

    def wait(self):
        return self.rc


def fake_popen(outcome):
    return lambda cmd, **kw: FakeProc(outcome, cmd, **kw)


class WhisperTest(unittest.TestCase):
    def test_parse_and_merge(self):
        me = tf.parse_whisper_json({"transcription": [
            {"offsets": {"from": 1500, "to": 3000}, "text": " Hi all."},
            {"offsets": {"from": 9000, "to": 9500}, "text": "  "},
            {"offsets": {"from": 20000, "to": 21000}, "text": " Bye."},
        ]}, "me")
        them = tf.parse_whisper_json({"transcription": [
            {"offsets": {"from": 1500, "to": 2000}, "text": " Hello."},
            {"offsets": {"from": 5000, "to": 8000}, "text": " Agenda first."},
        ]}, "them")
        self.assertEqual(me, [(1.5, "me", "Hi all."), (20.0, "me", "Bye.")])
        self.assertEqual(tf.merge_segments(me, them), [
            (1.5, "me", "Hi all."), (1.5, "them", "Hello."),
            (5.0, "them", "Agenda first."), (20.0, "me", "Bye."),
        ])

    def test_empty_json(self):
        self.assertEqual(tf.parse_whisper_json({}, "me"), [])

    def test_attempts_order(self):
        self.assertEqual(tf.whisper_attempts(["full", "q5"], 1),
                         [("full", 1), ("q5", 1), ("q5", None)])
        self.assertEqual(tf.whisper_attempts(["mine"], 0), [("mine", 0), ("mine", None)])
        self.assertEqual(tf.whisper_attempts(["full", "q5"], None), [("q5", None)])

    def test_gpu_devices(self):
        out = ("ggml_vulkan: Found 2 Vulkan devices:\n"
               "ggml_vulkan: 0 = Intel(R) Graphics (ARL) (Intel open-source Mesa driver) | uma: 1 | fp16: 1\n"
               "ggml_vulkan: 1 = NVIDIA RTX PRO 500 Laptop GPU (NVIDIA) | uma: 0 | fp16: 1\n"
               "error: failed to initialize whisper context\n")
        with mock.patch.object(tf.subprocess, "run",
                               return_value=subprocess.CompletedProcess([], 3, stdout="", stderr=out)):
            devices = tf.gpu_devices()
        self.assertEqual(devices, [(0, "Intel(R) Graphics (ARL) (Intel open-source Mesa driver)", True),
                                   (1, "NVIDIA RTX PRO 500 Laptop GPU (NVIDIA)", False)])
        self.assertEqual(tf.pick_device(devices), 1)
        self.assertEqual(tf.pick_device(devices[:1]), 0)
        self.assertIsNone(tf.pick_device([]))

    def test_device_arg(self):
        self.assertEqual([tf.device_arg(v) for v in ("auto", "cpu", "2")], ["auto", "cpu", 2])
        with self.assertRaises(Exception):
            tf.device_arg("nvidia")

    def test_segment_end(self):
        self.assertEqual(tf.segment_end("[00:00:01.500 --> 01:02:03.250]   Hi all.\n"), 3723.25)
        self.assertIsNone(tf.segment_end("whisper_print_timings: load time"))
        self.assertIsNone(tf.segment_end("\n"))

    def test_file_transcript_path(self):
        self.assertEqual(tf.file_transcript_path("/r/call.m4a"), "/r/call.m4a.transcript.md")

    def test_failed_attempts_are_dropped_for_the_next_channel(self):
        calls = []

        def outcome(cmd):
            calls.append(cmd)
            if "full" in cmd:  # the full model never fits
                return 1, []
            with open(cmd[cmd.index("-of") + 1] + ".json", "w") as f:
                json.dump({"transcription": []}, f)
            return 0, ["[00:00:00.000 --> 00:00:02.000]   Hi.\n"]

        attempts = tf.whisper_attempts(["full", "q5"], 1)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(tf.subprocess, "Popen", fake_popen(outcome)), \
                mock.patch.object(tf, "warn"), mock.patch("builtins.print"):
            tf.whisper(attempts, "vad", "a.wav", os.path.join(d, "me"))
            tf.whisper(attempts, "vad", "b.wav", os.path.join(d, "them"))
        self.assertEqual([c[c.index("-m") + 1] for c in calls], ["full", "q5", "q5"])
        self.assertEqual(attempts, [("q5", 1), ("q5", None)])

    def test_language_reaches_whisper(self):
        calls = []

        def outcome(cmd):
            calls.append(cmd)
            with open(cmd[cmd.index("-of") + 1] + ".json", "w") as f:
                json.dump({"transcription": []}, f)
            return 0, []

        with tempfile.TemporaryDirectory() as d, mock.patch.object(tf.subprocess, "Popen", fake_popen(outcome)), \
                mock.patch.object(tf, "final_models", return_value=["q5"]), mock.patch("builtins.print"), \
                mock.patch.dict(os.environ, {tf.VAD_MODEL_ENV: __file__}):
            wav = os.path.join(d, "call.wav")
            with open(wav, "wb"):
                pass
            for argv in ([wav], ["--lang", "en", wav]):
                with mock.patch.object(tf.subprocess, "run",
                                       return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")):
                    tf.main(argv)
        self.assertEqual([c[c.index("-l") + 1] for c in calls], ["auto", "en"])

    def test_only_one_channel(self):
        transcribed = []

        def outcome(cmd):
            base = cmd[cmd.index("-of") + 1]
            transcribed.append(os.path.basename(base))
            with open(base + ".json", "w") as f:
                json.dump({"transcription": [{"offsets": {"from": 0}, "text": "hi"}]}, f)
            return 0, []

        with tempfile.TemporaryDirectory() as d, mock.patch.object(tf.subprocess, "Popen", fake_popen(outcome)), \
                mock.patch.object(tf, "final_models", return_value=["q5"]), mock.patch("builtins.print"), \
                mock.patch.dict(os.environ, {tf.VAD_MODEL_ENV: __file__}), \
                mock.patch.object(tf.subprocess, "run",
                                  return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")):
            with open(os.path.join(d, "audio.flac"), "wb"):
                pass
            tf.main([d, "--only", "them"])
            with open(os.path.join(d, "transcript.md")) as f:
                body = f.read()
            with self.assertRaises(SystemExit):
                tf.main([os.path.join(d, "audio.flac"), "--only", "them"])
        self.assertEqual(transcribed, ["them"])
        self.assertIn("them: hi", body)
        self.assertNotIn("me:", body)

    def test_recording_flags_rejected_with_a_path(self):
        with mock.patch.object(tf, "final") as final, mock.patch.object(tf, "record") as record, \
                mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            tf.main(["call.m4a", "--title", "x"])
        final.assert_not_called()
        record.assert_not_called()

    def test_last_attempt_failing_exits(self):
        fail = fake_popen(lambda cmd: (1, []))
        with mock.patch.object(tf.subprocess, "Popen", fail), mock.patch.object(tf, "warn"), \
                mock.patch("builtins.print"), self.assertRaises(SystemExit) as e:
            tf.whisper(tf.whisper_attempts(["q5"], 1), "vad", "a.wav", "/nonexistent/x")
        self.assertIn("ErrorOutOfDeviceMemory", str(e.exception))

    def test_no_transcript_exits_without_retrying(self):
        # whisper-cli exits 0 without output for an unknown --lang
        calls = []
        with mock.patch.object(tf.subprocess, "Popen", fake_popen(lambda cmd: calls.append(cmd) or (0, []))), \
                mock.patch("builtins.print"), self.assertRaises(SystemExit) as e:
            tf.whisper(tf.whisper_attempts(["full", "q5"], 1), "vad", "a.wav", "/nonexistent/x", "xx")
        self.assertEqual(len(calls), 1)
        self.assertIn("wrote no transcript", str(e.exception))


class ModelLookupTest(unittest.TestCase):
    def put(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        return path

    def test_hf_cache(self):
        repo, name, _ = tf.FINAL_MODELS[1]
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"HF_HOME": d, tf.VAD_MODEL_ENV: ""}):
            self.assertIsNone(tf.find_model(tf.FINAL_MODELS[1]))
            hub = os.path.join(d, "hub", "models--" + repo.replace("/", "--"), "snapshots")
            cached = self.put(os.path.join(hub, "abc", name))
            self.assertEqual(tf.find_model(tf.FINAL_MODELS[1]), cached)
            vad = self.put(os.path.join(d, "hub", "models--ggml-org--whisper-vad", "snapshots", "x",
                                        tf.VAD_MODEL[1]))
            self.assertEqual(tf.find_model(tf.VAD_MODEL), vad)
            with mock.patch.dict(os.environ, {tf.VAD_MODEL_ENV: __file__}):
                self.assertEqual(tf.find_model(tf.VAD_MODEL), __file__)


class SessionDirTest(unittest.TestCase):
    def test_name(self):
        t = datetime.datetime(2026, 11, 4, 9, 5)
        self.assertEqual(tf.session_name(t, None), "2026-11-04_0905")
        self.assertEqual(tf.session_name(t, "Q4 Review: Budget!"), "2026-11-04_0905_q4-review-budget")
        self.assertEqual(tf.session_name(t, "???"), "2026-11-04_0905")

    def test_collision_and_latest(self):
        with tempfile.TemporaryDirectory() as root:
            a = tf.make_session_dir(root, "x")
            b = tf.make_session_dir(root, "x")
            c = tf.make_session_dir(root, "x")
            self.assertEqual([os.path.basename(p) for p in (a, b, c)], ["x", "x_2", "x_3"])
            tf.point_latest(root, a)
            tf.point_latest(root, c)
            latest = os.path.join(root, "latest")
            self.assertEqual(os.path.realpath(latest), os.path.realpath(c))
            self.assertEqual(sorted(os.listdir(root)), ["latest", "x", "x_2", "x_3"])


if __name__ == "__main__":
    unittest.main()
