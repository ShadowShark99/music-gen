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
INSTRUMENTS = {
    "piano": 0,
    "guitar": 1,
    "bass": 2,
}
# token: (chord_root, chord_quality, instrument, pitch, duration, offset_in_chord)

def chord_to_root_quality(pitches):
    c = chord.Chord(pitches)
    return c.root().midi, c.quality

def quantize_duration(d):
    return min(DURATIONS, key=lambda x: abs(x-d))

def midi_to_chords(path):
    midi = pretty_midi.PrettyMIDI(path)
    notes_by_time = defaultdict(list)

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            t = round(note.start / TIME_TOL) * TIME_TOL
            notes_by_time[t].append(note)

    chord_track = []

    for t in sorted(notes_by_time):
        notes = notes_by_time[t]
        pitches = tuple(sorted(n.pitch for n in notes))

        duration = quantize_duration(
            max(n.end for n in notes) - t
        )

        root, quality = chord_to_root_quality(pitches)

        if quality not in {
            "major", "minor", "diminished", "augmented",
            "dominant-seventh", "major-seventh",
            "minor-seventh", "half-diminished"
        }:
            continue

        chord_track.append((t, root, quality, duration))

    return chord_track

# returns sequences for each instrument [0-127]
def midi_to_events(path):
    midi = pretty_midi.PrettyMIDI(path)
    events = defaultdict(list)

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        inst_id = instrument.program
        notes_by_time = defaultdict(list)

        for note in instrument.notes:
            t = round(note.start / TIME_TOL) * TIME_TOL
            notes_by_time[t].append(note)

        for t in sorted(notes_by_time):
            chord_notes = notes_by_time[t]
            pitches = tuple(sorted(n.pitch for n in chord_notes))
            duration = quantize_duration(
                max(n.end for n in chord_notes) - t
            )

            root, quality = chord_to_root_quality(pitches)

            if quality not in {
                "major", "minor", "diminished", "augmented",
                "dominant-seventh", "major-seventh",
                "minor-seventh", "half-diminished"
            }:
                continue

            events[inst_id].append(
                (t, root, quality, duration)
            )

    return events

def midi_to_instrument_events(path, chord_track):
    midi = pretty_midi.PrettyMIDI(path)
    events = defaultdict(list)

    chord_times = [t for t, *_ in chord_track]

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        inst_id = instrument.program

        for note in instrument.notes:
            start = note.start
            duration = quantize_duration(note.end - note.start)

            # Find closest chord
            chord_idx = min(
                range(len(chord_times)),
                key=lambda i: abs(chord_times[i] - start)
            )

            offset = start - chord_times[chord_idx]

            events[inst_id].append(
                (note.pitch, duration, offset)
            )

    return events

def build_dataset():
    all_events = []

    for file in os.listdir(MIDI_DIR):
        if file.endswith(".mid"):
            path = os.path.join(MIDI_DIR, file)
            all_events.extend(midi_to_chords(path))

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

def build_chord_dataset():
    all_chords = []
    songs = []

    for file in os.listdir(MIDI_DIR):
        if not file.endswith(".mid"):
            continue

        path = os.path.join(MIDI_DIR, file)
        chord_track = midi_to_chords(path)

        if len(chord_track) < SEQ_LENGTH + 1:
            continue

        song_chords = [(r, q, d) for _, r, q, d in chord_track]
        songs.append(song_chords)
        all_chords.extend(song_chords)

    # Build chord vocab
    unique_chords = sorted(set(all_chords))
    chord_to_idx = {c: i for i, c in enumerate(unique_chords)}
    idx_to_chord = {i: c for c, i in chord_to_idx.items()}

    # Encode sequences
    X, y = [], []
    for song in songs:
        encoded = [chord_to_idx[c] for c in song]
        for i in range(len(encoded) - SEQ_LENGTH):
            X.append(encoded[i:i+SEQ_LENGTH])
            y.append(encoded[i+SEQ_LENGTH])

    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    torch.save((X, y), "chord_dataset.pt")
    with open("chord_vocab.pkl", "wb") as f:
        pickle.dump((chord_to_idx, idx_to_chord), f)

    print(f"Chord dataset: {X.shape[0]} sequences")

def build_instrument_dataset():
    sequences = []
    all_tokens = []

    # Load chord vocab
    with open("chord_vocab.pkl", "rb") as f:
        chord_to_idx, _ = pickle.load(f)

    for file in os.listdir(MIDI_DIR):
        if not file.endswith(".mid"):
            continue

        path = os.path.join(MIDI_DIR, file)

        chord_track = midi_to_chords(path)
        if len(chord_track) < 2:
            continue

        instrument_events = midi_to_instrument_events(path, chord_track)

        for inst_id, events in instrument_events.items():
            if len(events) < SEQ_LENGTH + 1:
                continue

            seq = []
            for pitch, duration, offset in events:
                token = (
                    inst_id,
                    pitch,
                    duration,
                    round(offset / TIME_TOL)
                )
                seq.append(token)
                all_tokens.append(token)

            sequences.append(seq)

    # Build vocab
    unique_tokens = sorted(set(all_tokens))
    token_to_idx = {t: i for i, t in enumerate(unique_tokens)}
    idx_to_token = {i: t for t, i in token_to_idx.items()}

    # Build sequences
    X, y = [], []
    for seq in sequences:
        encoded = [token_to_idx[t] for t in seq]
        for i in range(len(encoded) - SEQ_LENGTH):
            X.append(encoded[i:i+SEQ_LENGTH])
            y.append(encoded[i+SEQ_LENGTH])

    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    torch.save((X, y), "instrument_dataset.pt")
    with open("instrument_vocab.pkl", "wb") as f:
        pickle.dump((token_to_idx, idx_to_token), f)

    print(f"Instrument dataset created: {X.shape[0]} sequences")


if __name__ == "__main__":
    build_chord_dataset()
    build_instrument_dataset()
