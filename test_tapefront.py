import datetime
import os
import tempfile
import unittest

import tapefront as tf


class TimestampTest(unittest.TestCase):
    def test_format(self):
        self.assertEqual(tf.fmt_ts(0), "00:00:00")
        self.assertEqual(tf.fmt_ts(59.9), "00:00:59")
        self.assertEqual(tf.fmt_ts(3600 + 12 * 60 + 5), "01:12:05")
        self.assertEqual(tf.fmt_ts(-1), "00:00:00")

    def test_line_and_header(self):
        self.assertEqual(tf.format_line(725, "me", "hello"), "[00:12:05] me: hello")
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
