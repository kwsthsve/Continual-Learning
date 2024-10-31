import numpy as np
import quadprog
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


# GEM algorithm for image classification model
class PlainGEMLearner(torch.nn.Module):

    def __init__(self, net, num_tasks, optimizer, criterion, mem_size, batch_size):
        super(PlainGEMLearner, self).__init__()
        self.net = net
        self.tasks = num_tasks
        self.optim = optimizer
        self.criterion = criterion
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initiallize Episodic Memory
        self.ep_mem = torch.FloatTensor(self.tasks, self.mem_size, 1, 28, 28)
        self.ep_labels = torch.LongTensor(self.tasks, self.mem_size)

        self.ep_mem = self.ep_mem.to(self.device)
        self.ep_labels = self.ep_labels.to(self.device)

        # Save each parameters' number of elements(numels)
        self.grad_numels = []
        for params in self.parameters():
            self.grad_numels.append(params.data.numel())

        # Make matrix for gradient w.r.t. past tasks
        self.G = torch.zeros((sum(self.grad_numels), self.tasks))
        self.G = self.G.to(self.device)

        # Make matrix for accuracy w.r.t. past tasks
        self.R = torch.zeros((self.tasks, self.tasks))
        self.R = self.R.to(self.device)

        print(f"\nRunning GEM learner with memory size = {self.mem_size}\n")

    def store_gradient(self, n_task):
        cnt = 0
        for param in self.parameters():
            if (param is not None) & param.requires_grad:
                stpt = 0 if cnt == 0 else sum(self.grad_numels[:cnt])
                endpt = sum(self.grad_numels[:cnt + 1])
                self.G[stpt:endpt, n_task].data.copy_(param.grad.data.view(-1))
                cnt += 1

    def project2cone2(self, margin=0.5, eps=1e-3):

        mem_grad_np = self.G[:, :self.cur_task].cpu().t().double().numpy()
        curtask_grad_np = self.G[:, self.cur_task].unsqueeze(1).cpu().contiguous().view(-1).double().numpy()

        t = mem_grad_np.shape[0]
        P = np.dot(mem_grad_np, mem_grad_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(mem_grad_np, curtask_grad_np) * (-1)
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, mem_grad_np) + curtask_grad_np

        return torch.Tensor(x).view(-1, )

    def overwrite_gradient(self, newgrad):
        cnt = 0
        for param in self.parameters():
            if (param is not None) & param.requires_grad:
                stpt = 0 if cnt == 0 else sum(self.grad_numels[:cnt])
                endpt = sum(self.grad_numels[:cnt + 1])
                param.grad.data.copy_(newgrad[stpt:endpt].contiguous().view(param.grad.data.size()))
                cnt += 1

    def train_learner(self, loader, task, num_training_data):
        self.cur_task = task

        running_loss = 0.0
        input_stack = torch.zeros((num_training_data, 1, 28, 28))
        label_stack = torch.zeros(num_training_data)

        input_stack = input_stack.to(self.device)
        label_stack = label_stack.to(self.device)

        for i, (x, y) in enumerate(loader):

            # Move data to GPU
            x = x.to(self.device)
            y = y.to(self.device)

            input_stack[i * self.batch_size: (i + 1) * self.batch_size] = x.clone()
            label_stack[i * self.batch_size: (i + 1) * self.batch_size] = y.clone()

            self.G.data.fill_(0.0)

            # Compute gradient w.r.t. past tasks with episodic memory
            if self.cur_task > 0:
                for k in range(0, self.cur_task):
                    self.zero_grad()

                    pred_ = self.net(self.ep_mem[k])
                    label_ = self.ep_labels[k]
                    loss_ = self.criterion(pred_, label_)
                    loss_.backward()

                    # Copy parameters from memory examples into Matrix "G"
                    self.store_gradient(k)

            self.zero_grad()

            # Compute gradient w.r.t. current continuum
            pred = self.net(x)
            loss = self.criterion(pred, y)
            loss.backward()

            running_loss += loss.item()
            if (i + 1) == len(loader):
                print(f'Task {task + 1} AVG. loss: {running_loss / len(loader):.3f}')

            if self.cur_task > 0:

                # Copy parameters from current examples into Matrix "G"
                self.store_gradient(self.cur_task)

                # Solve Quadratic Problem
                dotprod = torch.mm(self.G[:, self.cur_task].unsqueeze(0), self.G[:, :self.cur_task])

                # Projection if gradient violates constraints
                if (dotprod < 0).sum() != 0:
                    newgrad = self.project2cone2()

                    # Overwrite gradient into params
                    self.overwrite_gradient(newgrad)

            self.optim.step()

        # Choose random examples and keep them in memory
        perm = torch.randperm(num_training_data)
        perm = perm[:self.mem_size]
        self.ep_mem[self.cur_task] = input_stack[perm].clone().float()
        self.ep_labels[self.cur_task] = label_stack[perm].clone()

    def evaluation(self, loader, task):
        total = 0
        correct = 0
        self.net.eval()
        for i, (x, y) in enumerate(loader):
            # Move data to GPU
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.net(x)
            _, predicted = torch.max(output, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            self.R[self.cur_task][task] = 100 * correct / total


# Masked GEM algorithm for image captioning (encoder-decoder) full model
class MaskedFullArchitectureGEMLearner(torch.nn.Module):

    def __init__(self, net, num_tasks, mem_size, batch_size, mask):
        super(MaskedFullArchitectureGEMLearner, self).__init__()
        self.net = net
        self.tasks = num_tasks
        self.optim = self.net.optimizer
        self.criterion = self.net.criterion
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.mask = mask
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initiallize Episodic Memory
        self.ep_mem = torch.FloatTensor(self.tasks, self.mem_size,  3, 224, 224)
        self.ep_captions = torch.LongTensor(self.tasks, self.mem_size, 60)
        self.ep_lengths = {}

        # Create dictionary to track number of elements' changes
        self.numel_history = {}

        # Save each parameters' number of elements(numels)
        self.grad_numels = []
        for params in self.parameters():
            if params.requires_grad:
                self.grad_numels.append(params.data.numel())

        # Make matrix for gradient w.r.t. past tasks
        self.G = torch.zeros((sum(self.grad_numels), self.tasks))
        self.G = self.G.to(self.device)

        # Make matrix for masks w.r.t. past tasks' numels
        if self.mask:
            self.M = torch.zeros((sum(self.grad_numels), self.tasks - 1))
            self.M = self.M.to(self.device)

        # Make matrix for bleu and cider score w.r.t. past tasks
        self.bleu = torch.zeros((self.tasks, self.tasks))
        self.bleu = self.bleu.to(self.device)

        self.cider = torch.zeros((self.tasks, self.tasks))
        self.cider = self.cider.to(self.device)

        print(f"\nRunning GEM learner with memory size = {self.mem_size}\n")

    def store_gradient(self, n_task):
        cnt = 0
        for param in self.parameters():
            if (param is not None) & param.requires_grad:
                stpt = 0 if cnt == 0 else sum(self.grad_numels[:cnt])
                endpt = sum(self.grad_numels[:cnt + 1])
                self.G[stpt:endpt, n_task].data.copy_(param.grad.data.view(-1))
                cnt += 1

    def mask_gradient_matrix(self):

        # Mask the matrix 'G' only to the task's parameters, for every task when observing the last one (next will follow the projection)
        self.G[:, :self.cur_task] = self.G[:, :self.cur_task] * self.M[:, :self.cur_task]

    def project2cone2(self, margin=0.5, eps=1e-3):

        mem_grad_np = self.G[:, :self.cur_task].cpu().t().double().numpy()
        curtask_grad_np = self.G[:, self.cur_task].unsqueeze(1).cpu().contiguous().view(-1).double().numpy()

        t = mem_grad_np.shape[0]
        P = np.dot(mem_grad_np, mem_grad_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(mem_grad_np, curtask_grad_np) * (-1)
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, mem_grad_np) + curtask_grad_np

        return torch.Tensor(x).view(-1, )

    def overwrite_gradient(self, newgrad):
        cnt = 0
        for param in self.parameters():
            if (param is not None) & param.requires_grad:
                stpt = 0 if cnt == 0 else sum(self.grad_numels[:cnt])
                endpt = sum(self.grad_numels[:cnt + 1])
                param.grad.data.copy_(newgrad[stpt:endpt].contiguous().view(param.grad.data.size()))
                cnt += 1

    def expand_numels(self, vocab_size):

        # Expand ImageCaptioningModel
        self.net.expand_embeddings_and_linear(vocab_size)

        # Update each parameters' number of elements(numels)
        new_grad_numels = []
        for params in self.parameters():
            if params.requires_grad:
                new_grad_numels.append(params.data.numel())


        if len(new_grad_numels) == len(self.grad_numels):
            idx = []
            for i in range(len(new_grad_numels)):
                if new_grad_numels[i] != self.grad_numels[i]:
                    idx.append(i)

            for j in idx:
                print(f'Param {j}. Old numels = {self.grad_numels[j]}, New numels = {new_grad_numels[j]}')

        # Update the matrix for gradient w.r.t. past tasks
        new_G = torch.zeros((sum(new_grad_numels), self.tasks)).to(self.device)

        # Create mask for parameters of the previous task add current model's numels to history, update masks' matrix
        if self.mask:

            # Update mask's matrix w.r.t. to new numels
            new_M = torch.zeros((sum(new_grad_numels), self.tasks - 1)).to(self.device)

            self.numel_history[self.cur_task] = self.grad_numels

            # Iterate through the old and new lists
            for key, numels in self.numel_history.items():
                old_idx = 0
                new_idx = 0

                for i, (old_size, new_size) in enumerate(zip(numels, new_grad_numels)):
                    # For the unchanged part, fill the mask with ones
                    new_M[new_idx:new_idx + old_size, key] = 1

                    # Move the indices forward by the sizes
                    old_idx += old_size
                    new_idx += new_size

            self.M = new_M

        # Replace the old matrices with the new ones
        self.G = new_G
        self.grad_numels = new_grad_numels

        self.optim = self.net.optimizer

    def train_learner(self, loader, task, num_training_data, num_epochs):
        self.net.train()

        self.cur_task = task

        # Choose random examples and keep them in memory
        perm = torch.randperm(num_training_data)
        perm = perm[:self.mem_size]

        # # # ------------------------------------- ON THE FLY ALLOCATION OF INPUT STACK ------------------------------------- # # #

        # For memory efficiency, instead of initializing the input stack tensor (below)
        # save the random examples while training. The random indices are known.
        input_stack = {
            'images': [],
            'captions': [],
            'lengths': []
        }

        # Keep track of global index, because random indices are taken from num_training_data (on the fly allocation)
        global_index = 0

        # # # ------------------------------------- DIRECT ALLOCATION OF INPUT STACK ------------------------------------- # # #

        # Keep temporal tensors to CPU for GPU memory efficiency
        # input_stack = torch.zeros((num_training_data, 3, 224, 224))
        # caption_stack = torch.zeros(num_training_data, 60)
        # length_stack = []

        # # # --------------------------------------------------------------------------------------------------------------- # # #

        for ep in range(num_epochs):

            for i, (x, y, z) in enumerate(loader):

                # # # ------------------------------------- ON THE FLY ALLOCATION OF INPUT STACK ------------------------------------- # # #

                # In case of memory issues, fill the input stack on the fly to avoid allocation on tensor(num_training_data, 3, 224, 224)
                if ep == 0:
                    # Check if any indices we want to save are in this batch
                    batch_indices = set(range(global_index, global_index + self.batch_size))
                    indices_in_batch = batch_indices.intersection(perm.tolist())

                    if indices_in_batch:
                        for idx in indices_in_batch:
                            batch_idx = idx - global_index
                            input_stack['images'].append(x[batch_idx])
                            input_stack['captions'].append(y[batch_idx])
                            input_stack['lengths'].append(z[batch_idx])


                    global_index += self.batch_size

                # # # ------------------------------------- DIRECT ALLOCATION OF INPUT STACK ------------------------------------- # # #

                # Copy training examples before moving data to GPU
                # input_stack[i * self.batch_size: (i + 1) * self.batch_size] = x.clone()
                # caption_stack[i * self.batch_size: (i + 1) * self.batch_size] = y.clone()
                # length_stack.extend(z)

                # # # --------------------------------------------------------------------------------------------------------------- # # #

                self.G.data.fill_(0.0)

                # Compute gradient w.r.t. past tasks with episodic memory
                if self.cur_task > 0:

                    for k in range(0, self.cur_task):
                        self.zero_grad()

                        # Split episodic memory to batches
                        t_x = torch.split(self.ep_mem[k], self.batch_size, 0)
                        t_y = torch.split(self.ep_captions[k], self.batch_size, 0)
                        t_z = [self.ep_lengths[k][b: b + self.batch_size] for b in range(0, len(self.ep_lengths[k]), self.batch_size)]

                        for batch, (mem_images, mem_captions) in enumerate(zip(t_x, t_y)):

                            # Move data to GPU
                            mem_images = mem_images.to(self.device)
                            mem_captions = mem_captions.to(self.device)

                            pred_ = self.net(mem_images, mem_captions, t_z[batch])

                            pred_packed_ = pack_padded_sequence(pred_, t_z[batch], batch_first=True).data
                            targets_packed_ = pack_padded_sequence(mem_captions, t_z[batch], batch_first=True).data

                            loss_ = self.criterion(pred_packed_, targets_packed_)
                            loss_.backward()

                        # Copy parameters from memory examples into Matrix "G"
                        self.store_gradient(k)

                self.zero_grad()

                # Move data to GPU
                x = x.to(self.device)
                y = y.to(self.device)

                # Compute gradient w.r.t. current continuum
                pred = self.net(x, y, z)

                pred_packed = pack_padded_sequence(pred, z, batch_first=True).data
                targets_packed = pack_padded_sequence(y, z, batch_first=True).data

                loss = self.criterion(pred_packed, targets_packed)
                loss.backward()

                # Print training statistics
                if (i + 1) == len(loader):
                    print(f'[Epoch {ep + 1}] Task {task + 1} loss: {loss.item():.4f}')

                if self.cur_task > 0:

                    # Copy parameters from current examples into Matrix "G"
                    self.store_gradient(self.cur_task)

                    # Mask the gradients of previous tasks w.r.t. to current models parameters
                    if self.mask:
                        self.mask_gradient_matrix()

                    # Solve Quadratic Problem
                    dotprod = torch.mm(self.G[:, self.cur_task].unsqueeze(0), self.G[:, :self.cur_task])

                    # Projection if gradient violates constraints
                    if (dotprod < 0).sum() != 0:

                        newgrad = self.project2cone2()

                        # Overwrite gradient into params
                        self.overwrite_gradient(newgrad)

                self.optim.step()

        # # # ------------------------------------- DIRECT ALLOCATION OF INPUT STACK ------------------------------------- # # #

        # Sort lengths in descending order and apply same changes to perm
        # tmp_lengths = [length_stack[i] for i in perm.tolist()]
        # tmp_alignment = list(zip(tmp_lengths, perm.tolist()))
        # tmp_alignment = sorted(tmp_alignment, reverse=True)
        # sorted_lengths, perm = zip(*tmp_alignment)
        #
        # # Choose random indices from current task's training examples
        # self.ep_lengths[self.cur_task] = list(sorted_lengths)
        # perm = torch.IntTensor(perm)
        #
        # self.ep_mem[self.cur_task] = input_stack[perm].clone().float()
        # self.ep_captions[self.cur_task] = caption_stack[perm].clone()

        # # # ------------------------------------- ON THE FLY ALLOCATION OF INPUT STACK ------------------------------------- # # #
        # Get the keys in a fixed order
        input_stack_keys = list(input_stack.keys())

        # Zip the lists together with their indices
        zipped = list(zip(*[input_stack[key] for key in input_stack_keys], range(len(input_stack[input_stack_keys[0]]))))

        # Sort based on the third list (lengths) in descending order
        sorted_zipped = sorted(zipped, key=lambda x: x[2], reverse=True)

        # Unzip the sorted lists
        sorted_input_stack_list = list(zip(*sorted_zipped))

        # Create a new dictionary with sorted lists
        sorted_input_stack = {key: list(sorted_input_stack_list[i]) for i, key in enumerate(input_stack_keys)}

        for i in range(self.mem_size):
            self.ep_mem[self.cur_task][i] = sorted_input_stack['images'][i].clone().float()
            self.ep_captions[self.cur_task][i] = sorted_input_stack['captions'][i].clone()

        self.ep_lengths[self.cur_task] = sorted_input_stack['lengths']

        # # # --------------------------------------------------------------------------------------------------------------- # # #

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


# Masked GEM algorithm for image captioning decoder only
class MaskedDecoderOnlyGEMLearner(torch.nn.Module):

    def __init__(self, net, num_tasks, mem_size, batch_size, mask):
        super(MaskedDecoderOnlyGEMLearner, self).__init__()
        self.net = net
        self.tasks = num_tasks
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.mask = mask
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(123)

        # Initiallize Episodic Memory
        self.ep_mem = torch.FloatTensor(self.tasks, self.mem_size,  2048)
        self.ep_captions = torch.LongTensor(self.tasks, self.mem_size, 60)
        self.ep_lengths = {}

        # Create dictionary to track number of elements' changes
        self.numel_history = {}

        # Save each parameters' number of elements(numels)
        self.grad_numels = []
        for params in self.parameters():
            if params.requires_grad:
                self.grad_numels.append(params.data.numel())

        # Make matrix for gradient w.r.t. past tasks
        self.G = torch.zeros((sum(self.grad_numels), self.tasks))
        self.G = self.G.to(self.device)

        # Make matrix for masks w.r.t. past tasks' numels
        if self.mask:
            self.M = torch.zeros((sum(self.grad_numels), self.tasks - 1))
            self.M = self.M.to(self.device)

        # Make matrix for bleu and cider score w.r.t. past tasks
        self.bleu = torch.zeros((self.tasks, self.tasks))
        self.bleu = self.bleu.to(self.device)

        self.cider = torch.zeros((self.tasks, self.tasks))
        self.cider = self.cider.to(self.device)

        print(f"\nRunning GEM learner with memory size = {self.mem_size}\n")

    def store_gradient(self, n_task):
        cnt = 0
        for param in self.parameters():
            if (param is not None) & param.requires_grad:
                stpt = 0 if cnt == 0 else sum(self.grad_numels[:cnt])
                endpt = sum(self.grad_numels[:cnt + 1])
                self.G[stpt:endpt, n_task].data.copy_(param.grad.data.view(-1))
                cnt += 1

    def mask_gradient_matrix(self):
        # Mask the matrix 'G' only to the task's parameters, for every task when observing the last one (next will follow the projection)
        self.G[:, :self.cur_task] = self.G[:, :self.cur_task] * self.M[:, :self.cur_task]

    def project2cone2(self, margin=0.5, eps=1e-3):

        mem_grad_np = self.G[:, :self.cur_task].cpu().t().double().numpy()
        curtask_grad_np = self.G[:, self.cur_task].unsqueeze(1).cpu().contiguous().view(-1).double().numpy()

        t = mem_grad_np.shape[0]
        P = np.dot(mem_grad_np, mem_grad_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(mem_grad_np, curtask_grad_np) * (-1)
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, mem_grad_np) + curtask_grad_np

        return torch.Tensor(x).view(-1, )

    def overwrite_gradient(self, newgrad):
        cnt = 0
        for param in self.parameters():
            if (param is not None) & param.requires_grad:
                stpt = 0 if cnt == 0 else sum(self.grad_numels[:cnt])
                endpt = sum(self.grad_numels[:cnt + 1])
                param.grad.data.copy_(newgrad[stpt:endpt].contiguous().view(param.grad.data.size()))
                cnt += 1

    def expand_numels(self, vocab_size):

        # Expand ImageCaptioningModel
        self.net.expand_embeddings_and_linear(vocab_size)

        # Update each parameters' number of elements(numels)
        new_grad_numels = []
        for params in self.parameters():
            if params.requires_grad:
                new_grad_numels.append(params.data.numel())


        if len(new_grad_numels) == len(self.grad_numels):
            idx = []
            for i in range(len(new_grad_numels)):
                if new_grad_numels[i] != self.grad_numels[i]:
                    idx.append(i)

            for j in idx:
                print(f'Param {j}. Old numels = {self.grad_numels[j]}, New numels = {new_grad_numels[j]}')

        # Update the matrix for gradient w.r.t. past tasks
        new_G = torch.zeros((sum(new_grad_numels), self.tasks)).to(self.device)

        # Create mask for parameters of the previous task add current model's numels to history, update masks' matrix
        if self.mask:

            # Update mask's matrix w.r.t. to new numels
            new_M = torch.zeros((sum(new_grad_numels), self.tasks - 1)).to(self.device)

            self.numel_history[self.cur_task] = self.grad_numels

            # Iterate through the old and new lists
            for key, numels in self.numel_history.items():
                old_idx = 0
                new_idx = 0

                for i, (old_size, new_size) in enumerate(zip(numels, new_grad_numels)):

                    # For the unchanged part, fill the mask with ones
                    new_M[new_idx:new_idx + old_size, key] = 1

                    # Move the indices forward by the sizes
                    old_idx += old_size
                    new_idx += new_size

            self.M = new_M


        # Replace the old matrices with the new ones
        self.G = new_G
        self.grad_numels = new_grad_numels

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)

    def train_learner(self, loader, task, num_training_data, num_epochs):
        self.net.train()

        self.cur_task = task

        # Choose random examples and keep them in memory
        perm = torch.randperm(num_training_data)
        perm = perm[:self.mem_size]

        # Keep temporal tensors to CPU for GPU memory efficiency
        input_stack = torch.zeros((num_training_data, 2048))
        caption_stack = torch.zeros(num_training_data, 60)
        length_stack = []

        for ep in range(num_epochs):

            for i, (x, y, z) in enumerate(loader):

                # Copy training examples before moving data to GPU
                input_stack[i * self.batch_size: (i + 1) * self.batch_size] = self.net.cnn(x.to(self.device)).clone().cpu()
                caption_stack[i * self.batch_size: (i + 1) * self.batch_size] = y.clone()
                length_stack.extend(z)

                self.G.data.fill_(0.0)

                # Compute gradient w.r.t. past tasks with episodic memory
                if self.cur_task > 0:

                    for k in range(0, self.cur_task):
                        self.optimizer.zero_grad()

                        t_x = self.ep_mem[k].to(self.device)
                        t_y = self.ep_captions[k].to(self.device)
                        t_z = self.ep_lengths[k]

                        pred_ = self.net.rnn(t_x, t_y, t_z)

                        pred_packed_ = pack_padded_sequence(pred_, t_z, batch_first=True).data
                        targets_packed_ = pack_padded_sequence(t_y, t_z, batch_first=True).data

                        loss_ = self.criterion(pred_packed_, targets_packed_)
                        loss_.backward()

                        # Copy parameters from memory examples into Matrix "G"
                        self.store_gradient(k)

                self.optimizer.zero_grad()

                # Move data to GPU
                x = x.to(self.device)
                y = y.to(self.device)

                # Compute gradient w.r.t. current continuum
                pred = self.net(x, y, z)

                pred_packed = pack_padded_sequence(pred, z, batch_first=True).data
                targets_packed = pack_padded_sequence(y, z, batch_first=True).data

                loss = self.criterion(pred_packed, targets_packed)
                loss.backward()

                # Print training statistics
                if (i + 1) == len(loader):
                    print(f'[Epoch {ep + 1}] Task {task + 1} loss: {loss.item():.4f}')

                if self.cur_task > 0:

                    # Copy parameters from current examples into Matrix "G"
                    self.store_gradient(self.cur_task)

                    # Mask the gradients of previous tasks w.r.t. to current models parameters
                    if self.mask:
                        self.mask_gradient_matrix()

                    # Solve Quadratic Problem
                    dotprod = torch.mm(self.G[:, self.cur_task].unsqueeze(0), self.G[:, :self.cur_task])

                    # Projection if gradient violates constraints
                    if (dotprod < 0).sum() != 0:

                        newgrad = self.project2cone2()

                        # Overwrite gradient into params
                        self.overwrite_gradient(newgrad)

                self.optimizer.step()

        # Sort lengths in descending order and apply same changes to perm
        tmp_lengths = [length_stack[i] for i in perm.tolist()]
        tmp_alignment = list(zip(tmp_lengths, perm.tolist()))
        tmp_alignment = sorted(tmp_alignment, reverse=True)
        sorted_lengths, perm = zip(*tmp_alignment)

        # Choose random indices from current task's training examples
        self.ep_lengths[self.cur_task] = list(sorted_lengths)
        perm = torch.IntTensor(perm)

        self.ep_mem[self.cur_task] = input_stack[perm].clone().float()
        self.ep_captions[self.cur_task] = caption_stack[perm].clone()

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