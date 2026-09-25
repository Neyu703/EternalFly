"""One-off script: stimulate the real cached connectome via brain_loader.py and report
the resulting readout rates and achieved simulation speed - used to calibrate
ReadingSessionConfig's valence/arousal ceilings (brain_loader.py's
POSITIVE_VALENCE_CEILING/NEGATIVE_VALENCE_CEILING/AROUSAL_CEILING) against what the
real network can actually reach, and to check whether real reflexes emerge from the
connectome itself (e.g. a sweet word driving the real feeding motor pool). Not unit-
tested itself - composes already-tested eternalfly functions, same convention as
scripts/build_connectome_cache.py. Replaces the old VADER-era calibrate_sentiment.py.

Run as `python -m scripts.audit_brain` from backend/.
"""

import time

from eternalfly.brain_loader import build_reading_session

STEPS_PER_FRAME = 150  # one frame per word at the default sim_ms_per_word
WORD_REPEAT_COUNT = 30  # enough repeats for slow (1.5s) channel decay to settle near steady state

# Words chosen from real verification against the model (see the C4/C5 commit
# messages) - clear, strong triggers for a handful of channels, used to measure
# achievable ceilings and check for emergent reflexes. Not every real channel has a
# word this strong yet - anchor/background coverage is an ongoing tuning target.
POSITIVE_TEXT_WORD = "wunderbar"  # clearly positive context anchor match
NEGATIVE_TEXT_WORD = "schrecklich"  # clearly negative context anchor match
SWEET_TRIGGER_WORD = "Honig"
HEAT_TRIGGER_WORD = "brennt"
# LC4/LPLC2 (the "looming" channel) is the real visual pathway into the Giant Fiber
# escape circuit (von Reyn et al. 2014; de Vries & Clandinin 2012) - a stronger,
# biologically direct escape trigger than heat, and the best available real stimulus
# for backing (MDN) / turn_left / turn_right (DNa02) too, since none of them have a
# dedicated semantic channel of their own (steering/backing are recruited by the same
# threat-response circuitry, not driven by a distinct sense).
LOOMING_TRIGGER_WORD = "bedrohung"  # verified against the real model: "Monster" itself only weakly
# matches the looming anchor centroid (drive ~0.08) despite being one of the anchor words itself -
# "bedrohung" drives it ~0.58, a real, far stronger trigger
NEUTRAL_WORD = "Tisch"


def _run_repeated_word(session, word: str, repeat_count: int) -> list:
    session.load_new_text([word] * repeat_count)
    session.precompute_upcoming_words(repeat_count)
    return [session.advance(STEPS_PER_FRAME) for _ in range(repeat_count)]


def _trailing_mean(values: list[float], trailing_count: int = 10) -> float:
    trailing = values[-trailing_count:]
    return sum(trailing) / len(trailing)


def main() -> None:
    print("building session against the real cached connectome...")
    build_start = time.perf_counter()
    session = build_reading_session([NEUTRAL_WORD])
    print(f"  build time: {time.perf_counter() - build_start:.1f}s")

    print(f"\nstimulating positive context ({POSITIVE_TEXT_WORD!r} x{WORD_REPEAT_COUNT})...")
    sim_start = time.perf_counter()
    positive_results = _run_repeated_word(session, POSITIVE_TEXT_WORD, WORD_REPEAT_COUNT)
    elapsed = time.perf_counter() - sim_start
    steps_per_second = (WORD_REPEAT_COUNT * STEPS_PER_FRAME) / elapsed
    print(f"  achieved {steps_per_second:.0f} simulation steps/s ({elapsed:.1f}s wall clock)")
    reward_rate = _trailing_mean([r.region_activity["reward"] for r in positive_results])
    print(f"  trailing reward_pam rate: {reward_rate:.5f}")
    print(f"  trailing rating: {_trailing_mean([r.rating_0_10 for r in positive_results]):.2f}/10")

    print(f"\nstimulating negative context ({NEGATIVE_TEXT_WORD!r} x{WORD_REPEAT_COUNT})...")
    negative_results = _run_repeated_word(session, NEGATIVE_TEXT_WORD, WORD_REPEAT_COUNT)
    punishment_rate = _trailing_mean([r.region_activity["punishment"] for r in negative_results])
    print(f"  trailing punishment_ppl1 rate: {punishment_rate:.5f}")
    print(f"  trailing rating: {_trailing_mean([r.rating_0_10 for r in negative_results]):.2f}/10")

    print(f"\nstimulating sweet ({SWEET_TRIGGER_WORD!r} x{WORD_REPEAT_COUNT})...")
    sweet_results = _run_repeated_word(session, SWEET_TRIGGER_WORD, WORD_REPEAT_COUNT)
    print(f"  trailing sense sweet: {_trailing_mean([r.senses['sweet'] for r in sweet_results]):.3f}")
    feeding_rate = _trailing_mean([r.behaviors["feeding"] for r in sweet_results])
    print(f"  trailing feeding behavior: {feeding_rate:.5f}")

    print(f"\nstimulating heat ({HEAT_TRIGGER_WORD!r} x{WORD_REPEAT_COUNT})...")
    heat_results = _run_repeated_word(session, HEAT_TRIGGER_WORD, WORD_REPEAT_COUNT)
    print(f"  trailing sense heat: {_trailing_mean([r.senses['heat'] for r in heat_results]):.3f}")

    print(f"\nstimulating looming threat ({LOOMING_TRIGGER_WORD!r} x{WORD_REPEAT_COUNT})...")
    looming_results = _run_repeated_word(session, LOOMING_TRIGGER_WORD, WORD_REPEAT_COUNT)
    print(f"  trailing sense looming: {_trailing_mean([r.senses['looming'] for r in looming_results]):.3f}")
    escape_rate = _trailing_mean([r.behaviors["escape"] for r in looming_results])
    backing_rate = _trailing_mean([r.behaviors["backing"] for r in looming_results])
    turn_left_rate = _trailing_mean([r.behaviors["turn_left"] for r in looming_results])
    turn_right_rate = _trailing_mean([r.behaviors["turn_right"] for r in looming_results])
    print(f"  trailing escape behavior: {escape_rate:.5f}")
    print(f"  trailing backing behavior: {backing_rate:.5f}")
    print(f"  trailing turn_left behavior: {turn_left_rate:.5f}")
    print(f"  trailing turn_right behavior: {turn_right_rate:.5f}")

    print(f"\nneutral baseline ({NEUTRAL_WORD!r} x{WORD_REPEAT_COUNT})...")
    neutral_results = _run_repeated_word(session, NEUTRAL_WORD, WORD_REPEAT_COUNT)
    arousal_rate = _trailing_mean([r.region_activity["arousal"] for r in neutral_results])
    print(f"  trailing arousal_oa rate (baseline): {arousal_rate:.5f}")

    print(f"\nstimulating heat again for arousal ceiling ({HEAT_TRIGGER_WORD!r} x{WORD_REPEAT_COUNT})...")
    arousal_stim_rate = _trailing_mean([r.region_activity["arousal"] for r in heat_results])
    print(f"  trailing arousal_oa rate (heat-stimulated): {arousal_stim_rate:.5f}")

    print("\n--- suggested ceilings (raw achieved rate under strong stimulation) ---")
    print(f"POSITIVE_VALENCE_CEILING ~= {reward_rate:.4f}")
    print(f"NEGATIVE_VALENCE_CEILING ~= {punishment_rate:.4f}")
    print(f"AROUSAL_CEILING ~= {max(arousal_stim_rate, arousal_rate):.4f}")
    print(f"BEHAVIOR_CEILINGS['escape'] ~= {escape_rate:.5f}")
    print(f"BEHAVIOR_CEILINGS['feeding'] ~= {feeding_rate:.5f}")
    print(f"BEHAVIOR_CEILINGS['backing'] ~= {backing_rate:.5f}")
    print(f"BEHAVIOR_CEILINGS['turn_left'] ~= {turn_left_rate:.5f}")
    print(f"BEHAVIOR_CEILINGS['turn_right'] ~= {turn_right_rate:.5f}")


if __name__ == "__main__":
    main()
