"""
lem_main.py
===========
Runs the Linguistic Event Matching pipeline and outputs matched_pairs.csv.

Usage:
    python3 lem_main.py --transcript your_transcript.csv --lookup linguistic_lookup.csv
"""

import argparse
import numpy as np
import pandas as pd

SIMILARITY_FACTORS = ["Grammatical_Class", "Is_Proper_Noun", "Phoneme_Manner", "Syllable_Count"]
BUFFER_DEFAULT = 5.0

# ── Buffer Filter ─────────────────────────────────────────────────────────────

def check_buffer(row, stutter_events, buffer):
    """
    A fluent word passes if no stuttered event falls within `buffer` seconds
    on either side of it.
    """
    too_close = stutter_events[
        (stutter_events["end_time"]   > row["start_time"] - buffer) &
        (stutter_events["start_time"] < row["end_time"]   + buffer)
    ]
    return too_close.empty

# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_similarity(stutter, candidate):
    return sum(1 for f in SIMILARITY_FACTORS if stutter[f] == candidate[f])

def compute_distance(stutter, candidate):
    return abs(stutter["Word_Frequency"] - candidate["Word_Frequency"])

# ── Greedy Matching ───────────────────────────────────────────────────────────

def run_matching(stutters, fluent_pool):
    claimed       = set()
    matched_pairs = []
    unmatched     = []

    # Sort stutters by scarcity: fewest perfect (4-pt) matches first
    def count_perfect(s_idx):
        stutter   = stutters.loc[s_idx]
        available = fluent_pool[~fluent_pool.index.isin(claimed)]
        return sum(compute_similarity(stutter, row) == 4
                   for _, row in available.iterrows())

    for s_idx in sorted(stutters.index, key=count_perfect):
        stutter   = stutters.loc[s_idx]
        available = fluent_pool[~fluent_pool.index.isin(claimed)].copy()

        if available.empty:
            unmatched.append(s_idx)
            continue

        available["_sim"] = available.apply(lambda r: compute_similarity(stutter, r), axis=1)
        available["_dst"] = available.apply(lambda r: compute_distance(stutter, r),   axis=1)

        best   = available["_sim"].max()
        top    = available[available["_sim"] == best]
        winner = top.loc[top["_dst"].idxmin()]
        claimed.add(winner.name)

        matched_pairs.append({
            "stutter_word":       stutter["word"],
            "stutter_start":      stutter["start_time"],
            "stutter_end":        stutter["end_time"],
            "stutter_gram_class": stutter["Grammatical_Class"],
            "stutter_proper":     stutter["Is_Proper_Noun"],
            "stutter_manner":     stutter["Phoneme_Manner"],
            "stutter_syllables":  stutter["Syllable_Count"],
            "stutter_freq":       stutter["Word_Frequency"],
            "twin_word":          winner["word"],
            "twin_start":         winner["start_time"],
            "twin_end":           winner["end_time"],
            "twin_gram_class":    winner["Grammatical_Class"],
            "twin_proper":        winner["Is_Proper_Noun"],
            "twin_manner":        winner["Phoneme_Manner"],
            "twin_syllables":     winner["Syllable_Count"],
            "twin_freq":          winner["Word_Frequency"],
            "similarity_score":   int(best),
            "distance_score":     round(float(winner["_dst"]), 4),
        })

    return matched_pairs, unmatched

# ── Main ──────────────────────────────────────────────────────────────────────

def main(transcript_path, lookup_path, buffer):
    transcript = pd.read_csv(transcript_path)
    lookup     = pd.read_csv(lookup_path)

    for col in ["word", "start_time", "end_time", "label"]:
        if col not in transcript.columns:
            raise ValueError(f"Transcript missing column: '{col}'")

    merged = transcript.merge(lookup, on="word", how="left")

    # Normalize labels
    merged["_label"] = merged["label"].astype(str).str.strip()
    print(f"Labels found: {merged['_label'].unique().tolist()}")

    stutters   = merged[merged["_label"] == "Stuttered"].copy()
    fluent_all = merged[merged["_label"] == "Fluent"].copy()
    print(f"Stuttered events : {len(stutters)}")
    print(f"Fluent candidates: {len(fluent_all)}")

    if stutters.empty or fluent_all.empty:
        print("❌ Could not find Stuttered/Fluent rows. Check your label column uses exactly 'Stuttered' and 'Fluent'.")
        return

    # Drop rows missing any required feature
    needed = SIMILARITY_FACTORS + ["Word_Frequency"]
    bad    = fluent_all[needed].isnull().any(axis=1)
    if bad.any():
        print(f"⚠️  Dropping {bad.sum()} fluent row(s) with missing features.")
        fluent_all = fluent_all[~bad]

    # Buffer filter (proximity to stutters only)
    print(f"\nApplying {buffer}s buffer filter...")
    stutter_times = stutters[["start_time", "end_time"]].reset_index(drop=True)
    fluent_all    = fluent_all.copy()
    fluent_all["_pass"] = fluent_all.apply(
        lambda r: check_buffer(r, stutter_times, buffer), axis=1
    )
    fluent_pool = fluent_all[fluent_all["_pass"]].drop(columns=["_pass"])
    print(f"Candidates passing buffer: {len(fluent_pool)} / {len(fluent_all)}")

    if fluent_pool.empty:
        print("❌ No candidates passed the buffer filter. Try --buffer 3.0")
        return

    # Run matching
    print("\nRunning Linguistic Event Matching...")
    matched_pairs, unmatched = run_matching(stutters, fluent_pool)

    # Summary
    if matched_pairs:
        scores = [p["similarity_score"] for p in matched_pairs]
        print(f"\n── Match Quality ───────────────────────────────")
        print(f"   Matched pairs      : {len(matched_pairs)}")
        print(f"   Mean similarity    : {np.mean(scores):.2f} / 4")
        for s, c in pd.Series(scores).value_counts().sort_index(ascending=False).items():
            print(f"   Score {s}/4 : {c} pair(s)")
        print(f"────────────────────────────────────────────────")

    # Save outputs
    if matched_pairs:
        pd.DataFrame(matched_pairs).to_csv("matched_pairs.csv", index=False)
        print(f"\n✅ matched_pairs.csv saved ({len(matched_pairs)} pairs)")
    if unmatched:
        df = stutters.loc[unmatched, ["word","start_time","end_time"]].copy()
        df["reason"] = "No fluent twin available"
        df.to_csv("unmatched_log.csv", index=False)
        print(f"⚠️  {len(unmatched)} unmatched stutter(s) saved to unmatched_log.csv")
    else:
        print("✅ All stuttered events matched.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--lookup",     required=True)
    parser.add_argument("--buffer",     type=float, default=BUFFER_DEFAULT)
    args = parser.parse_args()
    main(args.transcript, args.lookup, args.buffer)
