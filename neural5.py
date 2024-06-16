import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Example dataset (you can expand this)
data = [
    ("Hi", "Hello"),
    ("How are you?", "I am fine, thank you!"),
    ("What is your name?", "I am a chatbot."),
    # Add more pairs as needed
]

# Simplified tokenization for demonstration purposes
word2index = {"Hi": 0, "Hello": 1, "How": 2, "are": 3, "you?": 4, "I": 5, "am": 6, "fine,": 7, "thank": 8, "you!": 9, "What": 10, "is": 11, "your": 12, "name?": 13, "a": 14, "chatbot.": 15}
index2word = {v: k for k, v in word2index.items()}
SOS_token = 16
EOS_token = 17
word2index["SOS"] = SOS_token
word2index["EOS"] = EOS_token
vocab_size = len(word2index)

# Convert data to tensors
def sentence_to_tensor(sentence):
    return torch.tensor([word2index[word] for word in sentence.split(' ')], dtype=torch.long)

dataset = [(sentence_to_tensor(pair[0]), sentence_to_tensor(pair[1])) for pair in data]
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=10):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]])
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

hidden_size = 256
encoder = Encoder(vocab_size, hidden_size)
decoder = Decoder(hidden_size, vocab_size)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.1)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.1)
criterion = nn.NLLLoss()

n_iters = 1000
print_every = 100
for iter in range(1, n_iters + 1):
    for pair in dataset:
        input_tensor, target_tensor = pair
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    if iter % print_every == 0:
        print(f'Iteration {iter}, Loss: {loss:.4f}')
def evaluate(encoder, decoder, sentence, max_length=10):
    with torch.no_grad():
        input_tensor = sentence_to_tensor(sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return ' '.join(decoded_words)

while True:
    sentence = input("You: ")
    if sentence == 'q':
        break
    response = evaluate(encoder, decoder, sentence)
    print(f"Bot: {response}")
