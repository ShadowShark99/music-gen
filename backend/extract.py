import os
import pickle
import pretty_midi
import numpy as np
import torch
from collections import defaultdict
from music21 import chord

SEQ_LENGTH = 32
MIDI_DIR = "midi"
DURATIONS = [.25,.5,1.0,2.0]
TIME_TOL = 0.05   # notes within this window = same chord

def chord_to_root_quality(pitches):
    c = chord.Chord(pitches)
    return c.root().midi, c.quality

def quantize_duration(d):
    return min(DURATIONS, key=lambda x: abs(x-d))

def midi_to_notes(path):
    midi = pretty_midi.PrettyMIDI(path)
    events = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        notes_by_time = defaultdict(list)

        for note in instrument.notes:
            t = round(note.start / TIME_TOL) * TIME_TOL
            notes_by_time[t].append(note)

        for t in sorted(notes_by_time): 
            chord_notes = notes_by_time[t]
            pitches = tuple(sorted(n.pitch for n in chord_notes))
            duration = quantize_duration(max(n.end for n in chord_notes) - t)
            root, quality = chord_to_root_quality(pitches)
            if quality not in {"major", "minor", "diminished", "dominant-seventh"}:
                continue
            events.append((root, quality, duration)) # calc duration of the whole chord

    return events

def build_dataset():
    all_events = []

    for file in os.listdir(MIDI_DIR):
        if file.endswith(".mid"):
            path = os.path.join(MIDI_DIR, file)
            all_events.extend(midi_to_notes(path))

    # Create vocab
    unique_events = sorted(set(all_events))
    event_to_idx = {e: i for i, e in enumerate(unique_events)}
    idx_to_event = {i: e for e, i in event_to_idx.items()}

    # Encode notes
    encoded = [event_to_idx[e] for e in all_events]

    # Create sequences
    X, y = [], []
    for i in range(len(encoded) - SEQ_LENGTH):
        X.append(encoded[i:i+SEQ_LENGTH])
        y.append(encoded[i+SEQ_LENGTH])

    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    # Save
    torch.save((X, y), "dataset.pt")
    with open("vocab.pkl", "wb") as f:
        pickle.dump((event_to_idx, idx_to_event), f)

    print(f"Dataset created: {X.shape[0]} sequences")

if __name__ == "__main__":
    build_dataset()
