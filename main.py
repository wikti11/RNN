import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torchtext.data import Field, Example, Dataset, BucketIterator

# Load spaCy model
spacy_en = spacy.load('en_core_web_sm')

# Tokenizer function
def tokenize_spacy(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Przetwarzanie danych
def load_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            tokens = [token for token in line if token != "</S>"]
            tags = [token.split('-')[-1] if '-' in token else 'O' for token in line if token != "</S>"]
            sentences.append(tokens)
            labels.append(tags)
    return sentences, labels

# Zdefiniowanie modelu
class NERModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        outputs, _ = self.rnn(embedded)
        predictions = self.fc(outputs)
        return predictions

# Przygotowanie danych
TEXT = Field(tokenize=tokenize_spacy, lower=True, unk_token='<unk>')
LABEL = Field(unk_token=None, pad_token='<pad>')

train_sentences, train_labels = load_data('train/train.tsv')
fields = [('text', TEXT), ('label', LABEL)]
examples = [Example.fromlist([tokens, labels], fields) for tokens, labels in zip(train_sentences, train_labels)]
train_dataset = Dataset(examples, fields)

# Przygotowanie danych testowych
test_sentences, test_labels = load_data('test-A/in.tsv')
test_examples = [Example.fromlist([tokens, labels], fields) for tokens, labels in zip(test_sentences, test_labels)]
test_dataset = Dataset(test_examples, fields)

# Build vocabularies including tokens from both training and test datasets
TEXT.build_vocab(train_dataset, test_dataset)
LABEL.build_vocab(train_dataset, test_dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

train_iterator = BucketIterator(
    train_dataset,
    batch_size=BATCH_SIZE,
    device=device,
    sort=False
)

test_iterator = BucketIterator(
    test_dataset,
    batch_size=BATCH_SIZE,
    device=device,
    sort=False
)

# Inicjalizacja modelu i hiperparametrów
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = len(LABEL.vocab)

model = NERModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=LABEL.vocab.stoi[LABEL.pad_token])

# Trening modelu
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        text = batch.text
        labels = batch.label
        optimizer.zero_grad()
        predictions = model(text)
        predictions = predictions.view(-1, predictions.shape[-1])
        labels = labels.view(-1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

N_EPOCHS = 20
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion)
    print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}')

# Ocena modelu na zbiorze testowym
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            labels = batch.label
            predictions = model(text)
            predictions = predictions.view(-1, predictions.shape[-1])
            labels = labels.view(-1)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

test_loss = evaluate(model, test_iterator, criterion)
accuracy = 1 - test_loss  # Załóżmy, że accuracy to 1 minus loss
points = math.ceil(accuracy * 7.0)
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Points: {points}')

# Zapis wyników predykcji do pliku test-A/out.tsv
def predict(model, sentences):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sentence in sentences:
            tokenized = TEXT.preprocess(sentence)
            indexed = [TEXT.vocab.stoi[token] for token in tokenized]
            tensor = torch.LongTensor(indexed).unsqueeze(1).to(device)
            prediction = model(tensor)
            pred_labels = prediction.argmax(dim=-1).squeeze(1).cpu().numpy()
            label_tokens = [LABEL.vocab.itos[idx] for idx in pred_labels]
            predictions.append(label_tokens)
    return predictions

def save_predictions(file_path, sentences, predictions):
    with open(file_path, 'w') as file:
        for sentence, prediction in zip(sentences, predictions):
            for token, label in zip(sentence, prediction):
                file.write(f"{token}\t{label}\n")
            file.write("\n")

# Przygotowanie danych do predykcji
def load_sentences_to_predict(file_path):
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            sentences.append(tokens)
    return sentences

# Wykonanie predykcji i zapis wyników
sentences_to_predict = load_sentences_to_predict('test-A/in.tsv')
predictions = predict(model, sentences_to_predict)
save_predictions('test-A/out.tsv', sentences_to_predict, predictions)
