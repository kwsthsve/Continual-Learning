import nltk
from nltk.translate.bleu_score import corpus_bleu
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from torch.nn import functional as F
import numpy as np


nltk.download('punkt_tab')


# Compute CIDEr score
def compute_cider(ground_truth, predictions):

    """
    Evaluate image captions using CIDEr score.

    :param ground_truth: Dictionary of image_id to list of dictionaries with {'caption' : reference}
    :param predictions: Dictionary of image_id to list of dictionary with {'caption' : generated caption}
    :return: CIDEr score
    """

    # Tokenize ground truth and predictions
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(ground_truth)
    res = tokenizer.tokenize(predictions)

    # Calculate CIDEr score
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)

    return cider_score


# Define model architecture for image classification task
class MNISTModel(nn.Module):

    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # [batch_size, 64, height/2, width/2]
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(14 * 14 * 64, 250)
        self.fc2 = nn.Linear(250, 10)

        # Define loss function and optimizer
        self.optimizer = torch.optim.Adadelta(self.parameters())
        self.criterion = nn.CrossEntropyLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 14 * 14 * 64)  # Flatten
        x = F.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def evaluation(self, loader):
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in loader:

                # Move data to GPU
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            acc = 100 * correct / total

        return acc

    def train_task(self, loader, task, num_epochs, other_task_loader=None):

        print(f'\nTraining model to Task {task + 1}\n')
        for epoch in range(num_epochs):

            for i, (images, labels) in enumerate(loader):

                # Move data to GPU
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Print training progress
                if (i + 1) == len(loader):
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Define the CNN encoder for extracting image features
class EncoderCNN(nn.Module):

        def __init__(self, fine_tune):
            super(EncoderCNN, self).__init__()
            self.ft = fine_tune

            encoder = models.resnet50(weights="IMAGENET1K_V2")  # encoder_size=2048
            for param in encoder.parameters():
                param.requires_grad = False

            # Remove the last layer
            modules = list(encoder.children())[:-1]
            self.resnet = nn.Sequential(*modules)

            if self.ft:
                self.fine_tune()

        def forward(self, images):
            features = self.resnet(images)
            features = features.view(features.shape[0], -1)
            return features  # (batch_size, 2048)

        def fine_tune(self):
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = self.ft


# Define the RNN decoder for generating captions
class DecoderRNN(nn.Module):

        def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
            super(DecoderRNN, self).__init__()
            self.fc1 = nn.Linear(2048, embed_size)
            self.embed = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
            self.fc2 = nn.Linear(hidden_size, vocab_size)
            self.dropout = nn.Dropout(0.5)

        def forward(self, features, captions, lengths):
            features = self.fc1(features) # (batch_size, embed_size)
            embeddings = self.dropout(self.embed(captions))  # (batch_size, max_caption_len, embed_size)
            embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)  # [(batch_size, 1, embed_size) ; (batch_size, max_caption_len, embed_size)]
            packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
            hiddens, _ = self.lstm(packed)

            # Perform operations on the full sequence output rather than just the last hidden state
            unpacked, _ = pad_packed_sequence(hiddens, batch_first=True)
            outputs = self.fc2(unpacked)
            return outputs


# Combine the CNN and RNN to create the Image Captioning model
class ImageCaptioningModel(nn.Module):

        def __init__(self, embed_size, hidden_size, vocab_size, num_layers, fine_tune, lr=3e-4):
            super(ImageCaptioningModel, self).__init__()
            self.embed_size = embed_size
            self.hidden_size = hidden_size
            self.vocab_size = vocab_size
            self.prev_vocab_size = 0
            self.num_layers = num_layers
            self.lr = lr

            self.cnn = EncoderCNN(fine_tune)
            self.rnn = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

            # Define loss function and optimizer
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            self.criterion = nn.CrossEntropyLoss()

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def forward(self, images, captions, lengths):
            features = self.cnn(images)
            outputs = self.rnn(features, captions, lengths)
            return outputs

        def expand_embeddings_and_linear(self, new_vocab_size):
            tmp_device = next(self.parameters()).device

            # Extend the embedding and linear layer of the decoder
            new_embed = nn.Embedding(new_vocab_size, self.embed_size).to(tmp_device)
            new_fc = nn.Linear(self.hidden_size, new_vocab_size).to(tmp_device)

            # Copy existing weights to the new layers
            with torch.no_grad():
                new_embed.weight[:self.vocab_size].copy_(self.rnn.embed.weight)
                new_fc.weight[:self.vocab_size].copy_(self.rnn.fc2.weight)
                new_fc.bias[:self.vocab_size].copy_(self.rnn.fc2.bias)

            # Replace the old layers with the new ones
            self.prev_vocab_size = self.vocab_size
            self.rnn.fc2 = new_fc
            self.vocab_size = new_vocab_size
            self.rnn.embed = new_embed

            # Define optimizer (again)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        def evaluation(self, loader, vocab, max_seq_len=35):
            self.eval()

            # For CIDEr score
            ground_truth = {}
            prediction = {}

            # For BLEU score
            references = []
            hypotheses = []

            # Generate caption
            with torch.no_grad():
                for example, (image, captions) in enumerate(loader):

                    # Move data to GPU
                    image = image.to(self.device)

                    result_caption = []

                    x = self.cnn(image)
                    x = self.rnn.fc1(x)
                    states = None

                    for _ in range(max_seq_len):
                        hiddens, states = self.rnn.lstm(x, states)
                        output = self.rnn.fc2(hiddens.squeeze(0))
                        pred = torch.argmax(output)
                        result_caption.append(pred.item())
                        x = self.rnn.embed(pred).unsqueeze(0)

                        if vocab.idx2word[pred.item()] == '<end>':
                            break

                    # Convert indices to words
                    ref_tmp = []
                    ground_truth[example] = []

                    for caption in captions[0]:
                        caption = caption.tolist()
                        ref = []
                        for j in range(len(caption)):
                            word = vocab.idx2word[caption[j]]
                            if word == '<end>':
                                break
                            ref.append(word)

                        ground_truth[example].append({str('caption'): ' '.join(ref[1:])})
                        ref_tmp.append(ref[1:])

                    references.append(ref_tmp)

                    hyp = []
                    for j in range(len(result_caption)):
                        word = vocab.idx2word[result_caption[j]]
                        if word == '<end>':
                            break
                        hyp.append(word)

                    prediction[example] = [{str('caption'): ' '.join(hyp[1:])}]
                    hypotheses.append(hyp[1:])

                    # plt.imshow(image.cpu().squeeze(0).permute(1, 2, 0))
                    # plt.show()


            # Compute CIDEr, BLEU-3 and BLEU-4 scores
            cider = compute_cider(ground_truth, prediction)
            bleu = corpus_bleu(references, hypotheses, weights=[[0.33, 0.33, 0.33, 0], [0.25, 0.25, 0.25, 0.25]])


            return np.array(list([bleu[0], bleu[1], cider]))

        def train_task(self, loader, task, num_epochs):
            self.train()

            print(f'\nTraining model to Task {task}\n')
            for epoch in range(num_epochs):

                for i, (images, captions, lengths) in enumerate(loader):

                    # Move data to GPU
                    images = images.to(self.device)
                    captions = captions.to(self.device)

                    targets_packed = pack_padded_sequence(captions, lengths, batch_first=True).data

                    # Forward pass
                    outputs = self(images, captions, lengths)
                    outputs_packed = pack_padded_sequence(outputs, lengths, batch_first=True).data

                    # Calculate loss
                    loss = self.criterion(outputs_packed, targets_packed)

                    # Backward pass and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Print training statistics
                    if (i + 1) == len(loader):
                        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
