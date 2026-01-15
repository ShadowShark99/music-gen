import os
import pickle
import pretty_midi
import numpy as np
import torch

SEQ_LENGTH = 32
MIDI_DIR = "midi"

def midi_to_notes(path):
    midi = pretty_midi.PrettyMIDI(path)
    notes = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append(note.pitch)

    return notes

def build_dataset():
    all_notes = []

    for file in os.listdir(MIDI_DIR):
        if file.endswith(".mid"):
            path = os.path.join(MIDI_DIR, file)
            all_notes.extend(midi_to_notes(path))

    # Create vocab
    unique_notes = sorted(set(all_notes))
    note_to_idx = {n: i for i, n in enumerate(unique_notes)}
    idx_to_note = {i: n for n, i in note_to_idx.items()}

    # Encode notes
    encoded = [note_to_idx[n] for n in all_notes]

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
        pickle.dump((note_to_idx, idx_to_note), f)

    print(f"Dataset created: {X.shape[0]} sequences")

if __name__ == "__main__":
    build_dataset()
