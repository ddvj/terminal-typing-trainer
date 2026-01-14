import time
import random
from collections import defaultdict
from wordfreq import top_n_list

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import DummyControl
from prompt_toolkit.widgets import TextArea, Frame, Label
from prompt_toolkit.keys import Keys
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

#config

WORD_COUNT = 15000
PASSAGE_LENGTH = 15
TARGET_PATTERNS = 3
MAX_CONSECUTIVE_ERRORS = 3

style = Style.from_dict({
    "target": "",
    "target.active": "underline",
    "input.pending": "",
    "input.correct": "#00ff00",
    "input.incorrect": "#ff0000",
    "stats": "reverse"
})

def load_words():
    words = top_n_list("en", WORD_COUNT)
    return [w for w in words if w.isalpha() and len(w) >= 3]

def extract_patterns(word):
    patterns = set(word)
    for i in range(len(word) - 2):
        patterns.add(word[i:i+3])
    if len(word) >= 4:
        patterns.add(word[-2:])
        patterns.add(word[-3:])
    return patterns

def preprocess(words):
    pattern_words = defaultdict(list)
    for word in words:
        for p in extract_patterns(word):
            pattern_words[p].append(word)
    return pattern_words

# stats

class Stats:
    def __init__(self):
        self.errors = defaultdict(int)
        self.counts = defaultdict(int)
        self.latencies = defaultdict(list)

    def record(self, expected, actual, latency):
        self.counts[expected] += 1
        self.latencies[expected].append(latency)
        if expected != actual:
            self.errors[expected] += 1

    def weakest_patterns(self, k):
        scored = []
        for p in self.counts:
            err = self.errors[p] / max(1, self.counts[p])
            lat = sum(self.latencies[p]) / max(1, len(self.latencies[p]))
            scored.append((err + lat, p))
        scored.sort(reverse=True)
        return [p for _, p in scored[:k]]
    

def generate_passage(stats, pattern_words, fallback_words):
    weak = stats.weakest_patterns(TARGET_PATTERNS)
    candidates = set()

    for p in weak:
        candidates.update(pattern_words.get(p, []))

    if not candidates:
        candidates = fallback_words

    words = random.sample(
        list(candidates),
        min(PASSAGE_LENGTH, len(candidates))
    )

    return " ".join(words)

# prompt tk stuff

def render_target(passage, pos):
    fragments =[]

    for i, ch in enumerate(passage):
        if i == pos:
            display = "_" if ch == " " else ch
            fragments.append(("class:target.active", display))
        else:
            fragments.append(("class:target", ch))

    return fragments
    
def render_input(typed_chars):
    fragments = []
    for ch, status, *_ in typed_chars:
        if status == "pending":
            fragments.append(("class:input.pending", ch))
        else:
            fragments.append((f"class:input.{status}", ch))
    fragments.append(("class:input.pending", "_"))
    return fragments

def compute_metrics(start_time, committed, correct):
    elapsed = max(time.time() - start_time, 1e-6)
    minutes = elapsed / 60
    wpm = (committed / 5) / minutes
    accuracy = (correct / max(1, committed)) * 100
    return wpm, accuracy

def session_averages(history):
    if history:
        total_wpm, total_acc = 0, 0
        for wpm, acc in history:
            total_wpm += wpm
            total_acc += acc
        return total_wpm / len(history), total_acc / len(history)
    else:
        return 0.0, 0.0

def format_stats(cur_wpm, cur_acc, prev_wpm, prev_acc, avg_wpm, avg_acc):
    def fmt(x, pct = False):
        if x is None:
            return "  -- "
        return f"{x:5.1f}%" if pct else f"{x:5.1f}"
    
    return (
        f"Curr: {fmt(cur_wpm)} WPM  {fmt(cur_acc, True)}  |  "
        f"Prev: {fmt(prev_wpm)} WPM  {fmt(prev_acc, True)}  |  "
        f"Avg: {fmt(avg_wpm)} WPM  {fmt(avg_acc, True)}"
    )

def run_typing_session(passage, stats, history):
    typed_chars = []

    typed_area = Label(text = "")
    target_area = Label(text = "")
    focus_sink = Window(
        content = DummyControl(),
        height = 0,
        width = 0
    )
    stats_area = TextArea(
        height = 1,
        focusable = False,
        style = "class:stats"
    )

    pos = 0
    consecutive_errors = 0
    last_time = time.time()

    start_time = None
    committed = 0
    correct_committed = 0
    
    target_area.text = FormattedText(
        render_target(passage, pos)
    )

    typed_area.text = FormattedText(
        render_input(typed_chars)
    )

    kb = KeyBindings()

    prev_wpm, prev_acc = history[-1] if history else (None, None)
    avg_wpm, avg_acc = session_averages(history) if history else (None, None)
    stats_area.text = format_stats(
        cur_wpm = None,
        cur_acc = None,
        prev_wpm = prev_wpm,
        prev_acc = prev_acc,
        avg_wpm = avg_wpm,
        avg_acc = avg_acc
    )
    @kb.add(Keys.Backspace)
    def _(event):
        nonlocal pos, consecutive_errors, committed, correct_committed

        if not typed_chars:
            return
    
        ch, status, was_committed, was_correct = typed_chars.pop()

        if was_committed:
            pos -= 1
            committed -= 1
            if was_correct:
                correct_committed -= 1

        consecutive_errors = 0

        target_area.text = FormattedText(
            render_target(passage, pos)
        )
        typed_area.text = FormattedText(
            render_input(typed_chars)
        )
    
    @kb.add(Keys.Any)
    def _(event):
        nonlocal pos, consecutive_errors, last_time, committed, correct_committed, start_time

        key = event.key_sequence[0].key
        if len(key) != 1 or pos >= len(passage):
            return

        expected = passage[pos]

        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS and key != expected:
            return

        now = time.time()
        latency = now - last_time
        last_time = now

        stats.record(expected, key, latency)

        if start_time is None:
            start_time = time.time()

        if key == expected:
            typed_chars.append((key, "correct", True, True))
            pos += 1
            consecutive_errors = 0
            committed += 1
            correct_committed += 1
        else:
            consecutive_errors += 1
            if consecutive_errors < MAX_CONSECUTIVE_ERRORS:
                typed_chars.append((key, "incorrect", True, False))
                pos += 1
                committed += 1
            else:
                #typed_chars.append((key, "incorrect", False, False))
                return

        if start_time is not None:
            wpm, acc = compute_metrics(start_time, committed, correct_committed)
        else:
            wpm, acc = 0.0, 100.0
        
        stats_area.text = format_stats(
            cur_wpm = wpm,
            cur_acc = acc,
            prev_wpm = prev_wpm,
            prev_acc = prev_acc,
            avg_wpm = avg_wpm,
            avg_acc = avg_acc
        )
        target_area.text = FormattedText(
            render_target(passage, pos)
        )
        typed_area.text = FormattedText(
            render_input(typed_chars)
        )
        
        if pos >= len(passage):
            history.append((wpm, acc))
            event.app.exit()

    app = Application(
        layout = Layout(
            HSplit([
                stats_area,
                Frame(target_area, title="Target"),
                Frame(typed_area, title="Your Input"),
                focus_sink
            ]), 
            focused_element = focus_sink
        ),
        key_bindings = kb,
        full_screen = True,
        cursor = None,
        style = style
    )

    app.run()


def main():
    words = load_words()
    pattern_words = preprocess(words)
    stats = Stats()
    session_history = []

    while True:
        passage = generate_passage(stats, pattern_words, words)
        run_typing_session(passage, stats, session_history)

if __name__ == "__main__":
    main()
