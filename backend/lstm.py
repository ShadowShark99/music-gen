import pickle
import torch
import torch.nn as nn
import pretty_midi
import os
import torch.nn.functional as F
import sys
from collections import defaultdict
os.makedirs("output", exist_ok=True)

INSTRUMENT_MAP = {
    # Piano
    0:  ("Acoustic Grand Piano", (21, 108)),
    1:  ("Bright Acoustic Piano", (21, 108)),
    2:  ("Electric Grand Piano", (28, 104)),
    3:  ("Honky-tonk Piano", (28, 104)),
    4:  ("Electric Piano 1", (28, 104)),
    5:  ("Electric Piano 2", (28, 104)),
    6:  ("Harpsichord", (41, 88)),
    7:  ("Clavinet", (36, 96)),

    # Chromatic Percussion
    8:  ("Celesta", (48, 96)),
    9:  ("Glockenspiel", (60, 96)),
    10: ("Music Box", (60, 96)),
    11: ("Vibraphone", (48, 96)),
    12: ("Marimba", (36, 96)),
    13: ("Xylophone", (60, 96)),
    14: ("Tubular Bells", (48, 84)),
    15: ("Dulcimer", (48, 84)),

    # Organ
    16: ("Drawbar Organ", (36, 96)),
    17: ("Percussive Organ", (36, 96)),
    18: ("Rock Organ", (36, 96)),
    19: ("Church Organ", (24, 96)),
    20: ("Reed Organ", (36, 96)),
    21: ("Accordion", (48, 84)),
    22: ("Harmonica", (60, 84)),
    23: ("Tango Accordion", (48, 84)),

    # Guitar
    24: ("Acoustic Guitar (nylon)", (40, 88)),
    25: ("Acoustic Guitar (steel)", (40, 88)),
    26: ("Electric Guitar (jazz)", (40, 88)),
    27: ("Electric Guitar (clean)", (40, 88)),
    28: ("Electric Guitar (muted)", (40, 84)),
    29: ("Overdriven Guitar", (40, 88)),
    30: ("Distortion Guitar", (40, 88)),
    31: ("Guitar Harmonics", (52, 96)),

    # Bass
    32: ("Acoustic Bass", (28, 52)),
    33: ("Electric Bass (finger)", (28, 52)),
    34: ("Electric Bass (pick)", (28, 52)),
    35: ("Fretless Bass", (28, 52)),
    36: ("Slap Bass 1", (28, 52)),
    37: ("Slap Bass 2", (28, 52)),
    38: ("Synth Bass 1", (24, 60)),
    39: ("Synth Bass 2", (24, 60)),

    # Strings
    40: ("Violin", (55, 103)),
    41: ("Viola", (48, 96)),
    42: ("Cello", (36, 84)),
    43: ("Contrabass", (28, 60)),
    44: ("Tremolo Strings", (36, 96)),
    45: ("Pizzicato Strings", (36, 96)),
    46: ("Orchestral Harp", (36, 96)),
    47: ("Timpani", (36, 60)),

    # Ensemble
    48: ("String Ensemble 1", (36, 96)),
    49: ("String Ensemble 2", (36, 96)),
    50: ("SynthStrings 1", (36, 96)),
    51: ("SynthStrings 2", (36, 96)),
    52: ("Choir Aahs", (48, 84)),
    53: ("Voice Oohs", (48, 84)),
    54: ("Synth Voice", (48, 84)),
    55: ("Orchestra Hit", (48, 72)),

    # Brass
    56: ("Trumpet", (58, 82)),
    57: ("Trombone", (40, 72)),
    58: ("Tuba", (28, 60)),
    59: ("Muted Trumpet", (58, 82)),
    60: ("French Horn", (36, 84)),
    61: ("Brass Section", (36, 84)),
    62: ("SynthBrass 1", (36, 96)),
    63: ("SynthBrass 2", (36, 96)),

    # Reed
    64: ("Soprano Sax", (60, 88)),
    65: ("Alto Sax", (56, 80)),
    66: ("Tenor Sax", (48, 76)),
    67: ("Baritone Sax", (36, 68)),
    68: ("Oboe", (58, 88)),
    69: ("English Horn", (52, 84)),
    70: ("Bassoon", (34, 70)),
    71: ("Clarinet", (50, 88)),

    # Pipe
    72: ("Piccolo", (74, 108)),
    73: ("Flute", (60, 96)),
    74: ("Recorder", (60, 84)),
    75: ("Pan Flute", (60, 84)),
    76: ("Blown Bottle", (60, 84)),
    77: ("Shakuhachi", (57, 84)),
    78: ("Whistle", (72, 96)),
    79: ("Ocarina", (60, 84)),

    # Synth Lead
    80: ("Lead 1 (square)", (48, 96)),
    81: ("Lead 2 (sawtooth)", (48, 96)),
    82: ("Lead 3 (calliope)", (48, 96)),
    83: ("Lead 4 (chiff)", (48, 96)),
    84: ("Lead 5 (charang)", (48, 96)),
    85: ("Lead 6 (voice)", (48, 96)),
    86: ("Lead 7 (fifths)", (48, 96)),
    87: ("Lead 8 (bass + lead)", (36, 96)),

    # Synth Pad
    88: ("Pad 1 (new age)", (36, 96)),
    89: ("Pad 2 (warm)", (36, 96)),
    90: ("Pad 3 (polysynth)", (36, 96)),
    91: ("Pad 4 (choir)", (36, 96)),
    92: ("Pad 5 (bowed)", (36, 96)),
    93: ("Pad 6 (metallic)", (36, 96)),
    94: ("Pad 7 (halo)", (36, 96)),
    95: ("Pad 8 (sweep)", (36, 96)),
}



SEQ_LENGTH = 32
EPOCHS = 30
CHORD_INTERVALS = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented": [0, 4, 8],
    "dominant-seventh": [0, 4, 7, 10],
    "major-seventh": [0, 4, 7, 11],
    "minor-seventh": [0, 3, 7, 10],
    "half-diminished": [0, 3, 6, 10],
}
BEATS_PER_MEASURE = 4

# instrument range
PIANO_RANGE = (36, 84)   # C2–C6
BASS_RANGE  = (24, 52)   # C1–E3
GUITAR_RANGE   = (48, 84)   # C4–C6

def clamp_pitch(p, l, h):
    while p < l:
        p += 12
    while p > h:
        p -= 12
    return p

def safe_note(pitch, start, duration, velocity=90):
    if duration is None:
        return None

    duration = max(float(duration), 0.05)
    start = max(float(start), 0.0)
    end = start + duration

    if end <= start:
        return None

    return pretty_midi.Note(
        velocity=int(velocity),
        pitch=int(pitch),
        start=start,
        end=end
    )


# makes music more random, less loops
def sample(logits, temperature=0.8):
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()

class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed=128, hidden=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])



def train_chord_model():
    X, y = torch.load("chord_dataset.pt")

    with open("chord_vocab.pkl", "rb") as f:
        chord_to_idx, idx_to_chord = pickle.load(f)

    model = MusicLSTM(len(chord_to_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        print(f"[Chord] Epoch {epoch+1}: {loss.item():.4f}")

    torch.save(model.state_dict(), "chord_model.pt")

def train_instrument_model():
    X, y = torch.load("instrument_dataset.pt")

    with open("instrument_vocab.pkl", "rb") as f:
        token_to_idx, idx_to_token = pickle.load(f)

    model = MusicLSTM(len(token_to_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        print(f"[Inst] Epoch {epoch+1}: {loss.item():.4f}")

    torch.save(model.state_dict(), "instrument_model.pt")





def load_model(vocab_size, path="model.pt"):
    model = MusicLSTM(vocab_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def generate_only(length=200):
    X, _ = torch.load("dataset.pt")
    with open("vocab.pkl", "rb") as f:
        event_to_idx, idx_to_event = pickle.load(f)
    model = load_model(len(event_to_idx))
    seed = X[0].tolist()
    generate(model, seed, idx_to_event)

# gen chord structure
def generate_chords(length=64):
    with open("chord_vocab.pkl", "rb") as f:
        _, idx_to_chord = pickle.load(f)

    model = MusicLSTM(len(idx_to_chord))
    model.load_state_dict(torch.load("chord_model.pt"))
    model.eval()

    X, _ = torch.load("chord_dataset.pt")
    seed = X[0].tolist()

    chords = seed.copy()
    for _ in range(length):
        x = torch.tensor(chords[-SEQ_LENGTH:]).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        chords.append(sample(logits))

    return [idx_to_chord[i] for i in chords]

def build_chord_times(chords):
    times = []
    t = 0.0
    for _, _, d in chords:
        times.append(t)
        t += d
    return times

def generate_instruments(chords, chord_times):
    with open("instrument_vocab.pkl", "rb") as f:
        _, idx_to_token = pickle.load(f)

    model = MusicLSTM(len(idx_to_token))
    model.load_state_dict(torch.load("instrument_model.pt"))
    model.eval()

    instruments = defaultdict(list)
    seed = list(range(SEQ_LENGTH))

    notes = seed.copy()
    for _ in range(300):
        x = torch.tensor(notes[-SEQ_LENGTH:]).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        notes.append(sample(logits))

    for idx in notes:
        chord_idx, inst_id, pitch, duration, offset = idx_to_token[idx]
        instruments[inst_id].append((chord_idx, pitch, duration, offset))

    return instruments

def render_midi(chords, chord_times, instruments, out_path="output/generated.mid"):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)

    for inst_id, notes in instruments.items():
        if inst_id not in INSTRUMENT_MAP:
            continue

        name, pitch_range = INSTRUMENT_MAP[inst_id]
        program = pretty_midi.instrument_name_to_program(name)
        instrument = pretty_midi.Instrument(program=program)

        for chord_idx, pitch, duration, offset in notes:
            if chord_idx >= len(chord_times):
                continue

            start = chord_times[chord_idx] + offset * 0.05
            end = start + duration

            pitch = clamp_pitch(pitch, *pitch_range)

            
            note = safe_note(pitch, start, duration)
            if note:
                instrument.notes.append(note)

            instrument.notes.append(note)

        midi.instruments.append(instrument)

    midi.write(out_path)
    print(f"🎶 MIDI written to {out_path}")

    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python lstm.py train_chords")
        print("  python lstm.py train_instruments")
        print("  python lstm.py generate")
        sys.exit(0)

    if sys.argv[1] == "train_chords":
        train_chord_model()

    elif sys.argv[1] == "train_instruments":
        train_instrument_model()

    elif sys.argv[1] == "generate":
        chords = generate_chords()
        chord_times = build_chord_times(chords)
        instruments = generate_instruments(chords, chord_times)
        # call your MIDI rendering here
        render_midi(chords, chord_times, instruments)

    else:
        print("Unknown command")

