"""
lem_lookup.py
=============
Reads your transcript CSV and generates linguistic_lookup.csv.

Requirements (run once in Terminal):
    pip3 install pandas spacy wordfreq nltk
    pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
    open /Applications/Python\ 3.14/Install\ Certificates.command
    python3 -c "import nltk; nltk.download('cmudict')"

Usage:
    python3 lem_lookup.py --transcript your_transcript.csv
"""

import argparse
import re
import pandas as pd
import spacy
import nltk
from wordfreq import zipf_frequency

# ── Load CMU Dict ─────────────────────────────────────────────────────────────

print("Loading CMU Pronouncing Dictionary...")
cmu_dict = dict(nltk.corpus.cmudict.entries())

# ── Constants ─────────────────────────────────────────────────────────────────

PHONEME_MANNER = {
    "P": "Stop",      "B": "Stop",      "T": "Stop",      "D": "Stop",
    "K": "Stop",      "G": "Stop",
    "F": "Fricative", "V": "Fricative", "TH": "Fricative", "DH": "Fricative",
    "S": "Fricative", "Z": "Fricative", "SH": "Fricative", "ZH": "Fricative",
    "HH": "Fricative",
    "CH": "Affricate","JH": "Affricate",
    "M": "Nasal",     "N": "Nasal",     "NG": "Nasal",
    "L": "Liquid",    "R": "Liquid",
    "W": "Glide",     "Y": "Glide",
    "AA": "Vowel",    "AE": "Vowel",    "AH": "Vowel",    "AO": "Vowel",
    "AW": "Vowel",    "AY": "Vowel",    "EH": "Vowel",    "ER": "Vowel",
    "EY": "Vowel",    "IH": "Vowel",    "IY": "Vowel",    "OW": "Vowel",
    "OY": "Vowel",    "UH": "Vowel",    "UW": "Vowel",
}

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_stress(phoneme):
    return re.sub(r"\d", "", phoneme)

def get_syllable_count(phones):
    vowels = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
    return max(1, sum(1 for p in phones if strip_stress(p) in vowels))

def get_phoneme_manner(phones):
    if not phones:
        return "Unknown"
    return PHONEME_MANNER.get(strip_stress(phones[0]), "Unknown")

def lookup_cmu(word):
    phones = cmu_dict.get(word.lower())
    if not phones:
        return None, "Unknown"
    return get_syllable_count(phones), get_phoneme_manner(phones)

def get_grammatical_features(word, nlp):
    token = nlp(word)[0]
    gram_class = "Content" if token.pos_ in CONTENT_POS else "Function"
    is_proper  = token.pos_ == "PROPN"
    return gram_class, is_proper

# ── Main ──────────────────────────────────────────────────────────────────────

def generate_lookup(transcript_path, output_path="linguistic_lookup.csv"):
    print("Loading transcript...")
    df = pd.read_csv(transcript_path)

    required = {"word", "start_time", "end_time", "label"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Transcript CSV is missing columns: {missing}")

    words = df["word"].dropna().unique().tolist()
    print(f"Found {len(words)} unique words. Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    records, not_found = [], []

    for word in words:
        clean = re.sub(r"[^a-zA-Z'-]", "", str(word)).strip()
        if not clean:
            continue

        syllables, manner = lookup_cmu(clean)
        if syllables is None:
            not_found.append(clean)
            syllables = 1

        gram_class, is_proper = get_grammatical_features(clean, nlp)
        freq = zipf_frequency(clean.lower(), "en")

        records.append({
            "word":              word,
            "Grammatical_Class": gram_class,
            "Is_Proper_Noun":    is_proper,
            "Phoneme_Manner":    manner,
            "Syllable_Count":    syllables,
            "Word_Frequency":    freq,
        })

    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"\n✅ Lookup table saved to: {output_path}")

    if not_found:
        print(f"\n⚠️  {len(not_found)} word(s) not found in CMU Dict (syllable count defaulted to 1):")
        for w in not_found:
            print(f"   - {w}")
        print("   You can manually correct these in linguistic_lookup.csv if needed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output", default="linguistic_lookup.csv")
    args = parser.parse_args()
    generate_lookup(args.transcript, args.output)
