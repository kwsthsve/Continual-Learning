import copy
import torch
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from torch.nn.utils.rnn import pack_padded_sequence


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


def temperature_scaled_softmax(logits, temperature=1.0):
    # Softmax in last dimension, which is the vocabulary
    softmax = torch.nn.Softmax(dim=-1)

    logits = logits / temperature
    return softmax(logits)


# Knowledge distillation on the encoder
class KdLearner(torch.nn.Module):

    def __init__(self, net, num_tasks, lamda=1.0):
        super(KdLearner, self).__init__()
        self.net = net
        self.lamda = lamda

        for param in self.net.parameters():
            param.requires_grad = True

        self.teacher = None
        self.teacher_vocab = 0

        self.tasks = num_tasks
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.teacher_criterion = torch.nn.MSELoss()
        self.student_optimizer = torch.optim.Adam(self.net.cnn.parameters(), lr=1e-5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Make matrix for bleu and cider score w.r.t. past tasks
        self.bleu = torch.zeros((self.tasks, self.tasks))
        self.bleu = self.bleu.to(self.device)

        self.cider = torch.zeros((self.tasks, self.tasks))
        self.cider = self.cider.to(self.device)

        print(f"\nRunning KD learner...")

    def expand(self, vocab_size):
        # Expand ImageCaptioningModel
        self.net.expand_embeddings_and_linear(vocab_size)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)

    def train_learner(self, loader, task, num_training_data, num_epochs):
        self.net.train()

        self.cur_task = task
        teacher_loss = 0.0

        if self.cur_task > 0:
            self.teacher = copy.deepcopy(self.net)
            self.teacher_vocab = self.net.prev_vocab_size

            for param in self.teacher.parameters():
                param.requires_grad = False

            self.teacher.eval()

        for ep in range(num_epochs):

            for i, (x, y, z) in enumerate(loader):

                # Move data to GPU
                x = x.to(self.device)
                y = y.to(self.device)

                # Compute MSELoss for the distillation objective
                if self.teacher:

                    teacher_pred = self.teacher(x, y, z)
                    teacher_pred_packed = pack_padded_sequence(teacher_pred, z, batch_first=True).data


                self.optimizer.zero_grad()
                self.student_optimizer.zero_grad()

                # Compute CrossEntropyLoss for the captioning objective
                pred = self.net(x, y, z)

                pred_packed = pack_padded_sequence(pred, z, batch_first=True).data
                targets_packed = pack_padded_sequence(y, z, batch_first=True).data

                loss = self.criterion(pred_packed, targets_packed)

                if self.teacher:
                    teacher_loss = self.teacher_criterion(pred_packed[:, :self.teacher_vocab], teacher_pred_packed[:, :self.teacher_vocab])

                distil_loss = loss + self.lamda * teacher_loss
                distil_loss.backward()

                # Print training statistics
                if (i + 1) == len(loader):
                    print(f'[Epoch {ep + 1}] Task {task + 1} Distillation loss: {distil_loss.item():.4f} Loss: {loss.item():.4f}')

                self.optimizer.step()

                if self.teacher:
                    self.student_optimizer.step()

    def evaluation(self, loader, task, vocab, max_seq_len=35):
        self.net.eval()

        # For CIDEr score
        ground_truth = {}
        prediction = {}

        # For BLEU-4 score
        references = []
        hypotheses = []

        # Generate caption
        with torch.no_grad():
            for example, (x, y) in enumerate(loader):

                # Move data to GPU
                image = x.to(self.device)

                result_caption = []

                x = self.net.cnn(image)
                x = self.net.rnn.fc1(x)
                states = None

                for _ in range(max_seq_len):
                    hiddens, states = self.net.rnn.lstm(x, states)
                    output = self.net.rnn.fc2(hiddens.squeeze(0))
                    pred = torch.argmax(output)
                    result_caption.append(pred.item())
                    x = self.net.rnn.embed(pred).unsqueeze(0)

                    if vocab.idx2word[pred.item()] == '<end>':
                        break

                # Convert indices to words
                ref_tmp = []
                ground_truth[example] = []

                for caption in y[0]:
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

        # Compute BLEU-4 and CIDEr
        self.bleu[self.cur_task][task] = corpus_bleu(references, hypotheses, weights=[0.25, 0.25, 0.25, 0.25])
        self.cider[self.cur_task][task] = compute_cider(ground_truth, prediction)


# LwF learner on the decoder
class LwFLearner(torch.nn.Module):

    def __init__(self, net, num_tasks, lamda=1.0):
        super(LwFLearner, self).__init__()
        self.net = net
        self.lamda = lamda

        self.teacher = None
        self.teacher_vocab = 0

        self.tasks = num_tasks
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Make matrix for bleu and cider score w.r.t. past tasks
        self.bleu = torch.zeros((self.tasks, self.tasks))
        self.bleu = self.bleu.to(self.device)

        self.cider = torch.zeros((self.tasks, self.tasks))
        self.cider = self.cider.to(self.device)

        print(f"\nRunning LwF learner...")

    def expand(self, vocab_size):
        # Expand ImageCaptioningModel
        self.net.expand_embeddings_and_linear(vocab_size)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)

    def train_learner(self, loader, task, num_training_data, num_epochs):
        self.net.train()

        self.cur_task = task
        teacher_loss = 0.0

        if self.cur_task > 0:
            self.teacher = copy.deepcopy(self.net.rnn)
            self.teacher_vocab = self.net.prev_vocab_size

            for param in self.teacher.parameters():
                param.requires_grad = False

            self.teacher.eval()

        for ep in range(num_epochs):

            for i, (x, y, z) in enumerate(loader):

                # Move data to GPU
                x = x.to(self.device)
                y = y.to(self.device)

                # Compute CrossEntropyLoss for the pseudo-labels
                if self.teacher:

                    teacher_x = self.net.cnn(x)
                    self.teacher.embed = torch.nn.Embedding.from_pretrained(self.net.rnn.embed.weight.clone(), freeze=True)

                    teacher_pred = self.teacher(teacher_x, y, z)
                    teacher_pred_packed = pack_padded_sequence(teacher_pred, z, batch_first=True).data


                self.optimizer.zero_grad()

                # Compute CrossEntropyLoss for the captioning objective
                pred = self.net(x, y, z)

                pred_packed = pack_padded_sequence(pred, z, batch_first=True).data
                targets_packed = pack_padded_sequence(y, z, batch_first=True).data

                loss = self.criterion(pred_packed, targets_packed)

                if self.teacher:
                    teacher_loss = self.criterion(temperature_scaled_softmax(pred_packed[:, :self.teacher_vocab], temperature=4.0), temperature_scaled_softmax(teacher_pred_packed[:, :self.teacher_vocab], temperature=4.0))

                loss = loss + self.lamda * teacher_loss
                loss.backward()

                # Print training statistics
                if (i + 1) == len(loader):
                    print(f'[Epoch {ep + 1}] Task {task + 1} Loss: {loss.item():.4f}')

                self.optimizer.step()

    def evaluation(self, loader, task, vocab, max_seq_len=35):
        self.net.eval()

        # For CIDEr score
        ground_truth = {}
        prediction = {}

        # For BLEU-4 score
        references = []
        hypotheses = []

        # Generate caption
        with torch.no_grad():
            for example, (x, y) in enumerate(loader):

                # Move data to GPU
                image = x.to(self.device)

                result_caption = []

                x = self.net.cnn(image)
                x = self.net.rnn.fc1(x)
                states = None

                for _ in range(max_seq_len):
                    hiddens, states = self.net.rnn.lstm(x, states)
                    output = self.net.rnn.fc2(hiddens.squeeze(0))
                    pred = torch.argmax(output)
                    result_caption.append(pred.item())
                    x = self.net.rnn.embed(pred).unsqueeze(0)

                    if vocab.idx2word[pred.item()] == '<end>':
                        break

                # Convert indices to words
                ref_tmp = []
                ground_truth[example] = []

                for caption in y[0]:
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

        # Compute BLEU-4 and CIDEr
        self.bleu[self.cur_task][task] = corpus_bleu(references, hypotheses, weights=[0.25, 0.25, 0.25, 0.25])
        self.cider[self.cur_task][task] = compute_cider(ground_truth, prediction)


# Freeze encoder or decoder
class PartialFreezeLearner(torch.nn.Module):

    def __init__(self, net, num_tasks, encoder=True):
        super(PartialFreezeLearner, self).__init__()
        self.net = net
        self.freeze_enc = encoder

        if self.freeze_enc:
            for param in self.net.cnn.parameters():
                param.requires_grad = False
        else:
            for param in self.net.cnn.parameters():
                param.requires_grad = True

            for param in self.net.rnn.parameters():
                param.requires_grad = False

        self.tasks = num_tasks
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Make matrix for bleu and cider score w.r.t. past tasks
        self.bleu = torch.zeros((self.tasks, self.tasks))
        self.bleu = self.bleu.to(self.device)

        self.cider = torch.zeros((self.tasks, self.tasks))
        self.cider = self.cider.to(self.device)

        print(f"\nRunning PartialFreeze learner with {'encoder' if self.freeze_enc else 'decoder'} frozen...")

    def expand(self, vocab_size):
        # Expand ImageCaptioningModel
        self.net.expand_embeddings_and_linear(vocab_size)

        if not self.freeze_enc:
            for param in self.net.rnn.parameters():
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)

    def train_learner(self, loader, task, num_training_data, num_epochs):
        self.net.train()

        self.cur_task = task

        for ep in range(num_epochs):

            for i, (x, y, z) in enumerate(loader):

                # Move data to GPU
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                # Compute CrossEntropyLoss for the captioning objective
                pred = self.net(x, y, z)

                pred_packed = pack_padded_sequence(pred, z, batch_first=True).data
                targets_packed = pack_padded_sequence(y, z, batch_first=True).data

                loss = self.criterion(pred_packed, targets_packed)
                loss.backward()

                # Print training statistics
                if (i + 1) == len(loader):
                    print(f'[Epoch {ep + 1}] Task {task + 1} Loss: {loss.item():.4f}')

                self.optimizer.step()

    def evaluation(self, loader, task, vocab, max_seq_len=35):
        self.net.eval()

        # For CIDEr score
        ground_truth = {}
        prediction = {}

        # For BLEU-4 score
        references = []
        hypotheses = []

        # Generate caption
        with torch.no_grad():
            for example, (x, y) in enumerate(loader):

                # Move data to GPU
                image = x.to(self.device)

                result_caption = []

                x = self.net.cnn(image)
                x = self.net.rnn.fc1(x)
                states = None

                for _ in range(max_seq_len):
                    hiddens, states = self.net.rnn.lstm(x, states)
                    output = self.net.rnn.fc2(hiddens.squeeze(0))
                    pred = torch.argmax(output)
                    result_caption.append(pred.item())
                    x = self.net.rnn.embed(pred).unsqueeze(0)

                    if vocab.idx2word[pred.item()] == '<end>':
                        break

                # Convert indices to words
                ref_tmp = []
                ground_truth[example] = []

                for caption in y[0]:
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

        # Compute BLEU-4 and CIDEr
        self.bleu[self.cur_task][task] = corpus_bleu(references, hypotheses, weights=[0.25, 0.25, 0.25, 0.25])
        self.cider[self.cur_task][task] = compute_cider(ground_truth, prediction)