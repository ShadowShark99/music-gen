import pickle
import torch
import torch.nn as nn
import pretty_midi
import os
import torch.nn.functional as F
import sys
os.makedirs("output", exist_ok=True)


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

def train():
    X, y = torch.load("dataset.pt")

    with open("vocab.pkl", "rb") as f:
        event_to_idx, idx_to_event = pickle.load(f)

    model = MusicLSTM(len(event_to_idx))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "model.pt")
    # generate model right after save
    # generate(model, X[0].tolist(), idx_to_event)

def generate(model, seed, idx_to_event, length=200):
    model.eval()
    notes = seed.copy()

    for _ in range(length):
        context = notes[-SEQ_LENGTH:]
        if len(context) < SEQ_LENGTH:
            context = [context[0]] * (SEQ_LENGTH - len(context)) + context

        x = torch.tensor(context).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        next_idx = sample(logits)
        notes.append(next_idx)

    # using preset instruments
    piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))
    bass = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Electric Bass (finger)"))
    guitar = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Guitar (steel)"))


    #. midi = pretty_midi.PrettyMIDI() # midi object
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0) # a track of the song

    time = 0.0 # write notes based on time, sequentially adds notes to the instrument
    beat_in_measure = 0.0

    for idx in notes:
        root, quality, duration = idx_to_event[idx]
        intervals = CHORD_INTERVALS.get(quality, [0])
        pitches = [root + i for i in intervals]

        if beat_in_measure + duration > BEATS_PER_MEASURE:
            time += BEATS_PER_MEASURE - beat_in_measure
            beat_in_measure = 0.0

        # bass is one note
        bass_pitch = clamp_pitch(root, *BASS_RANGE)
        bass.notes.append(pretty_midi.Note(
            velocity=90,
            pitch=bass_pitch,
            start=time,
            end=time + duration
        ))

        # piano
        for p in pitches:
            piano_pitch = clamp_pitch(p, *PIANO_RANGE)
            piano.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=piano_pitch,
                start=time,
                end=time + duration
            ))

        # guitar
        for p in pitches:
            guitar_pitch = clamp_pitch(p, *GUITAR_RANGE)
            guitar.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=guitar_pitch,
                start=time,
                end=time + duration * .9
            ))

        for pitch in pitches:
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=time,
                end=time + duration
            )
            instrument.notes.append(note)

        time += duration
        beat_in_measure += duration

    # midi.instruments.append(instrument)
    midi.instruments.extend([piano,bass,guitar])
    midi.write("output/generated.mid")
    print("Generated output/generated.mid")

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

if __name__ == "__main__":
    # train()
    if len(sys.argv) < 2:
        print("Usage: python lstm.py [train|generate]")
    if sys.argv[1] == "train":
        train()
    
    if sys.argv[1] == "generate":
        generate_only()
    
