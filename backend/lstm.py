import pickle
import torch
import torch.nn as nn
import pretty_midi
import os
import torch.nn.functional as F
os.makedirs("output", exist_ok=True)


SEQ_LENGTH = 32
EPOCHS = 30

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
    generate(model, X[0].tolist(), idx_to_event)

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

    midi = pretty_midi.PrettyMIDI() # midi object
    instrument = pretty_midi.Instrument(program=0) # a track of the song

    time = 0.0 # write notes based on time, sequentially adds notes to the instrument
    for idx in notes:
        pitch, duration = idx_to_event[idx]
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=time,
            end=time + duration
        )
        instrument.notes.append(note)
        time += duration

    midi.instruments.append(instrument)
    midi.write("output/generated.mid")
    print("Generated output/generated.mid")

if __name__ == "__main__":
    train()
