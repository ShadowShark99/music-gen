import pickle
import torch
import torch.nn as nn
import pretty_midi
import os
os.makedirs("output", exist_ok=True)


SEQ_LENGTH = 32
EPOCHS = 30

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
        note_to_idx, idx_to_note = pickle.load(f)

    model = MusicLSTM(len(note_to_idx))
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
    generate(model, X[0].tolist(), idx_to_note)

def generate(model, seed, idx_to_note, length=200):
    model.eval()
    notes = seed.copy()

    for _ in range(length):
        x = torch.tensor(notes[-SEQ_LENGTH:]).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        next_idx = torch.argmax(logits, dim=1).item()
        notes.append(next_idx)

    midi = pretty_midi.PrettyMIDI() # midi object
    instrument = pretty_midi.Instrument(program=0) # a track of the song

    time = 0.0 # write notes based on time, sequentially adds notes to the instrument
    for idx in notes:
        pitch = idx_to_note[idx]
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=time,
            end=time + 0.5
        )
        instrument.notes.append(note)
        time += 0.5

    midi.instruments.append(instrument)
    midi.write("output/generated.mid")
    print("Generated output/generated.mid")

if __name__ == "__main__":
    train()
