import os
import pickle
import pretty_midi
import numpy as np
import torch

SEQ_LENGTH = 32
MIDI_DIR = "midi"
DURATIONS = [.25,.5,1.0,2.0]

def quantize_duration(d):
    return min(DURATIONS, key=lambda x: abs(x-d))

def midi_to_notes(path):
    midi = pretty_midi.PrettyMIDI(path)
    events = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            duration = quantize_duration(note.end - note.start)
            events.append((note.pitch, duration)) # note, dur data point

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
