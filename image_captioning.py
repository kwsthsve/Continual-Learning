import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from gem import MaskedFullArchitectureGEMLearner, MaskedDecoderOnlyGEMLearner
from methods import KdLearner, PartialFreezeLearner, LwFLearner
from coco2017_dataset import get_loader
from build_vocabulary import Vocabulary
from models import ImageCaptioningModel
import torchvision.transforms as transforms


# Data loader task specific
def task_loader(cat, transform, first=True):
    if first:

        train_loader = get_loader(root=f'./data/MSCOCO_annotations/task_{cat}_train_images/',
                                  json=f'./data/MSCOCO_annotations/task_{cat}_train_annotations.json',
                                  transform=transform,
                                  batch_size=64, shuffle=True,
                                  vocab=Vocabulary(vocab_file='./vocab_task_training.pkl',
                                                   annotations_file=f'./data/MSCOCO_annotations/task_{cat}_train_annotations.json'))

        test_loader = get_loader(root=f'./data/MSCOCO_annotations/task_{cat}_test_images/',
                                 json=f'./data/MSCOCO_annotations/task_{cat}_test_annotations.json',
                                 transform=transform,
                                 batch_size=1, shuffle=False,
                                 vocab=Vocabulary(vocab_file='./vocab_task_training.pkl', vocab_from_file=True),
                                 train=False)
    else:

        train_loader = get_loader(root=f'./data/MSCOCO_annotations/task_{cat}_train_images/',
                                  json=f'./data/MSCOCO_annotations/task_{cat}_train_annotations.json',
                                  transform=transform,
                                  batch_size=64, shuffle=True,
                                  vocab=Vocabulary(vocab_file='./vocab_task_training.pkl', vocab_from_file=True),
                                  first=first)

        test_loader = get_loader(root=f'./data/MSCOCO_annotations/task_{cat}_test_images/',
                                 json=f'./data/MSCOCO_annotations/task_{cat}_test_annotations.json',
                                 transform=transform,
                                 batch_size=1, shuffle=False,
                                 vocab=Vocabulary(vocab_file='./vocab_task_training.pkl', vocab_from_file=True),
                                 train=False)

    return train_loader, test_loader


# Compute BWT percentage
def compute_bwt_per_task_percentage(R):
    """
    Compute Backward Transfer (BWT) for each task as a percentage given a lower triangular accuracy matrix R.

    Args:
    R (torch.tensor): A 2D tensor of shape (N, N) where R[i,j] is the accuracy
                       on task j after training on task i. R should be lower triangular.

    Returns:
    list: A list of BWT percentages for each task (excluding the first task).
    """
    N = R.shape[0]
    bwt_per_task = [0.0]

    for k in range(2, N + 1):  # Start from the second task
        bwt_sum = 0.0
        for i in range(k - 1):  # Sum over all previous tasks
            bwt_sum += (R[i, i] - R[k - 1, i]) / R[i, i] * 100  # Convert to percentage
        bwt_per_task.append(bwt_sum / (k - 1))

    return bwt_per_task


def main():
    # Define model parameters
    embed = 256
    hidden = 512
    rnn_layers = 1
    num_epochs = 15
    n_tasks = 5

    task_ids = np.arange(n_tasks)

    task_to_supercat = {0: 'Living Things',
                        1: 'Vehicles and Outdoors',
                        2: 'Home and Furniture',
                        3: 'Food and Kitchen',
                        4: 'Personal Items'}

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Define transforms to preprocess the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # # # ------------------------------------- FULL DATASET TRAINING ------------------------------------- # # #

    # Load MSCOCO full dataset
    print('Loading MSCOCO dataset, preparing data loaders for the model and creating vocabulary...\n')

    train_loader = get_loader(root='./data/MSCOCO_annotations/MSCOCO2017_train_images/',
                                  json='./data/MSCOCO_annotations/filtered_train_annotations.json', transform=transform,
                                  batch_size=64, shuffle=True, vocab=Vocabulary(vocab_file='./vocab_full_dataset.pkl', annotations_file='./data/MSCOCO_annotations/filtered_train_annotations.json'))

    # # # ------------------------------------- FULL DATASET EVALUATION ------------------------------------- # # #

    test_loader = get_loader(root='./data/MSCOCO_annotations/MSCOCO2017_test_images/',
                                 json='./data/MSCOCO_annotations/filtered_test_annotations.json', transform=transform,
                                 batch_size=1, shuffle=False, vocab=Vocabulary(vocab_file='./vocab_full_dataset.pkl', vocab_from_file=True),
                                 train=False)

    # # # ------------------------------------- FULL DATASET TRAINING AND EVALUATION ------------------------------------- # # #

    # Initialize the model
    vocabulary = Vocabulary(vocab_from_file=True, vocab_file='./vocab_full_dataset.pkl')
    model = ImageCaptioningModel(embed, hidden, len(vocabulary), rnn_layers, fine_tune=True)

    # Load the model if there is a checkpoint
    model.load_state_dict(torch.load('./model_checkpoint_full_dataset_randomCap.pt'))

    model = model.to(device)
    print('Moved model to NVIDIA GeForce RTX 3060 Ti\n')
    print(f'Size of dataset is {len(train_loader.dataset)} examples.\n')
    print(f'Vocabulary length: {len(vocabulary)} tokens\n')

    start_time = time.time()

    # Train the model to full dataset
    # model.train_task(train_loader, 'full dataset', num_epochs=num_epochs)
    #
    # # Save the model
    # torch.save(model.state_dict(), './model_checkpoint_full_dataset_randomCap.pt')

    # Evaluate the model to full dataset
    full_dataset_scores = model.evaluation(test_loader, vocabulary)
    print(f'[BLEU-3, BLEU-4, CIDEr] score of the model on the full dataset images: [{full_dataset_scores[0]}, {full_dataset_scores[1]}, {full_dataset_scores[2]}]\n')

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time / 60:.2f} minutes')

    # # # ------------------------------------- TRAINING IN DISJOINT TASKS (CATASTROPHIC FORGETTING) ------------------------------------- # # #

    # Train and evaluate the model to tasks
    test = []
    avg_bleu_3 = []
    avg_bleu_4 = []
    avg_cider_cf = []

    start_time = time.time()

    for t, idx in enumerate(task_ids):

        # Load MSCOCO dataset divided in tasks
        print(f'Loading MSCOCO dataset task {task_to_supercat[idx]}, preparing data loaders for the model and creating vocabulary...\n')
        task_train_loader, task_test_loader = task_loader(task_to_supercat[idx], transform, first=(t == 0))
        test.append(task_test_loader)
        print(f'Size of dataset is {len(task_train_loader.dataset)} examples.\n')

        # Load vocabulary
        vocabulary = Vocabulary(vocab_file='./vocab_task_training.pkl', vocab_from_file=True)

        # Initialize the model if it is the first task else expand it
        if t == 0:
            print(f'Vocabulary length: {len(vocabulary)} tokens\n')
            model = ImageCaptioningModel(embed, hidden, len(vocabulary), rnn_layers, fine_tune=True)
            model = model.to(device)
            print('Moved model to NVIDIA GeForce RTX 3060 Ti\n')
        else:
            model.expand_embeddings_and_linear(len(vocabulary))
            print(f'Model expanded. New vocabulary length: {len(vocabulary)} tokens\n')

        # Train task
        model.train_task(task_train_loader, task_to_supercat[idx], num_epochs=num_epochs)

        bleu_3 = 0
        bleu_4 = 0
        cider = 0

        for i in range(t + 1):

            # Evaluate the model to test on seen tasks
            model_scores = model.evaluation(test[i], vocabulary)
            bleu_3 += model_scores[0]
            bleu_4 += model_scores[1]
            cider += model_scores[2]

            if i == t:
                print(f'[BLEU-3, BLEU-4, CIDEr] score of the model on the Task {task_to_supercat[idx]} images: [{model_scores[0]}, {model_scores[1]}, {model_scores[2]}]\n')


        avg_bleu_3.append(bleu_3 / (t + 1))
        print(f'Average BLEU-3 after {t + 1} task(s) = {avg_bleu_3[t]}')
        avg_bleu_4.append(bleu_4 / (t + 1))
        print(f'Average BLEU-4 after {t + 1} task(s) = {avg_bleu_4[t]}')
        avg_cider_cf.append(cider / (t + 1))
        print(f'Average CIDEr after {t + 1} task(s) = {avg_cider_cf[t]}')


    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time / 60:.2f} minutes')

    # Clean GPU memory before next iteration
    del model

    import gc
    gc.collect()
    torch.cuda.empty_cache()


    x = np.arange(1, n_tasks + 1, dtype=int)


    # Plot average bleu scores
    plt.title('Average BLEU-3 and BLEU-4 scores per tasks')
    plt.xticks(x, [task_to_supercat[t] for t in task_ids], rotation=45)
    plt.plot(x, avg_bleu_3, 'ro-', label='BLEU-3')
    plt.plot(x, avg_bleu_4, 'go-', label='BLEU-4')
    plt.xlabel('Tasks')
    plt.ylabel('Average BLEU score')
    plt.legend(loc=3)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.grid()
    plt.show()


    # Plot average cider score
    plt.title('Average CIDEr score per tasks')
    plt.xticks(x, [task_to_supercat[t] for t in task_ids], rotation=45)
    plt.plot(x, avg_cider_cf, 'ro-', label='CIDEr')
    plt.xlabel('Tasks')
    plt.ylabel('Average CIDEr score')
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.grid()
    plt.show()

    # # # ------------------------------------- APPLYING GEM ALGORITHM TO MODEL ------------------------------------- # # #

    memsize_list = [100, 300, 1000]

    avg_bleu = {}
    avg_cider = {}

    start_time = time.time()

    # Apply Gradient Episodic Memory algorithm
    for memsize in memsize_list:
        memsize_bleu = []
        memsize_cider = []
        test = []

        for t, idx in enumerate(task_ids):

            # Load MSCOCO dataset divided in tasks
            print(
                f'\nLoading MSCOCO dataset task {task_to_supercat[idx]}, preparing data loaders for the model and creating vocabulary...')
            task_train_loader, task_test_loader = task_loader(task_to_supercat[idx], transform, first=(t == 0))
            test.append(task_test_loader)

            print(f'Size of dataset is {len(task_train_loader.dataset)} examples.\n')

            # Load vocabulary
            vocabulary = Vocabulary(vocab_file='./vocab_task_training.pkl', vocab_from_file=True)

            # Initialize the model if it is the first task else expand it
            if t == 0:
                print(f'Vocabulary length: {len(vocabulary)} tokens\n')
                model = ImageCaptioningModel(embed, hidden, len(vocabulary), rnn_layers, fine_tune=False)
                model = model.to(device)
                print('Moved model to NVIDIA GeForce RTX 3060 Ti\n')

                gem = MaskedDecoderOnlyGEMLearner(model, n_tasks, memsize, 64, mask=True)

            else:
                gem.expand_numels(len(vocabulary))
                print(f'Model expanded. New vocabulary length: {len(vocabulary)} tokens\n')

            gem.train_learner(task_train_loader, t, len(task_train_loader.dataset), num_epochs)

            for i in range(t + 1):
                # Evaluate the model to test on seen tasks
                gem.evaluation(test[i], i, vocabulary)

                if i == t:
                    print(f'[BLEU-4, CIDEr] score of the model on the Task {task_to_supercat[idx]} images: [{gem.bleu[t][t]}, {gem.cider[t][t]}]\n')

            # Average bleu and cider score on seen tasks
            memsize_bleu.append(torch.sum(gem.bleu[t]).item() / (t + 1))
            print(f'Average BLEU-4 after {t + 1} task(s) = {torch.sum(gem.bleu[t]).item() / (t + 1)}')

            memsize_cider.append(torch.sum(gem.cider[t]).item() / (t + 1))
            print(f'Average CIDEr after {t + 1} task(s) = {torch.sum(gem.cider[t]).item() / (t + 1)}')

        avg_bleu[memsize] = memsize_bleu
        avg_cider[memsize] = memsize_cider

        # Clean GPU memory before next iteration
        del model
        del gem

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time / 60:.2f} minutes')

    # Plot average bleu score
    plt.title("Average BLEU-4 score while training on tasks")
    plt.xlabel("Tasks")
    plt.ylabel("Average BLEU-4")

    for memsize, bleu_list in avg_bleu.items():
        x = np.arange(1, n_tasks + 1, dtype=int)
        y = bleu_list
        plt.xticks(x, [task_to_supercat[t] for t in task_ids], rotation=45)
        plt.plot(x, y, 'o-', label=str(memsize))

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend(loc=3, title='memsize')
    plt.grid()
    plt.show()

    # Plot average cider score
    plt.title("Average CIDEr score while training on tasks")
    plt.xlabel("Tasks")
    plt.ylabel("Average CIDEr")

    for memsize, cider_list in avg_cider.items():
        x = np.arange(1, n_tasks + 1, dtype=int)
        y = cider_list
        plt.xticks(x, [task_to_supercat[t] for t in task_ids], rotation=45)
        plt.plot(x, y, 'o-', label=str(memsize))

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend(loc=3, title='memsize')
    plt.grid()
    plt.show()

    # # # ------------------------------------- BENCHMARKING ------------------------------------- # # #

    mem_size = 1000

    models = {0: 'distil_encoder', 1: 'GEM', 2: 'MaskedGEM', 3: 'freeze_encoder', 4: 'LwF'}
    avg_bleu = {}
    avg_cider = {}
    forgeting = {}

    start_time = time.time()

    # Apply Gradient Episodic Memory algorithm
    for m in models.keys():

        bleu = []
        cider = []
        test = []

        for t, idx in enumerate(task_ids):

            # Load MSCOCO dataset divided in tasks
            print(
                f'\nLoading MSCOCO dataset task {task_to_supercat[idx]}, preparing data loaders for the model and creating vocabulary...')
            task_train_loader, task_test_loader = task_loader(task_to_supercat[idx], transform, first=(t == 0))
            test.append(task_test_loader)

            print(f'Size of dataset is {len(task_train_loader.dataset)} examples.\n')

            # Load vocabulary
            vocabulary = Vocabulary(vocab_file='./vocab_task_training.pkl', vocab_from_file=True)

            # Initialize the model if it is the first task else expand it
            if t == 0:
                print(f'Vocabulary length: {len(vocabulary)} tokens\n')
                model = ImageCaptioningModel(embed, hidden, len(vocabulary), rnn_layers, fine_tune=False)
                model = model.to(device)
                print('Moved model to NVIDIA GeForce RTX 3060 Ti\n')

                if m == 1:
                    learner = MaskedDecoderOnlyGEMLearner(model, n_tasks, mem_size, 64, mask=False)
                elif m == 2:
                    learner = MaskedDecoderOnlyGEMLearner(model, n_tasks, mem_size, 64, mask=True)
                elif m == 0:
                    learner = KdLearner(model, n_tasks)
                elif m == 4:
                    learner = LwFLearner(model, n_tasks)
                else:
                    learner = PartialFreezeLearner(model, n_tasks)

            else:

                if m == 0 or m == 3 or m == 4:
                    learner.expand(len(vocabulary))
                else:
                    learner.expand_numels(len(vocabulary))

                print(f'Model expanded. New vocabulary length: {len(vocabulary)} tokens\n')

            learner.train_learner(task_train_loader, t, len(task_train_loader.dataset), num_epochs)

            for i in range(t + 1):
                # Evaluate the model to test on seen tasks
                learner.evaluation(test[i], i, vocabulary)

            # Average bleu and cider score on seen tasks
            bleu.append(torch.sum(learner.bleu[t]).item() / (t + 1))
            print(f'Average BLEU-4 after {t + 1} task(s) = {torch.sum(learner.bleu[t]).item() / (t + 1)}')

            cider.append(torch.sum(learner.cider[t]).item() / (t + 1))
            print(f'Average CIDEr after {t + 1} task(s) = {torch.sum(learner.cider[t]).item() / (t + 1)}')

        avg_bleu[models[m]] = bleu
        avg_cider[models[m]] = cider
        forgeting[models[m]] = compute_bwt_per_task_percentage(learner.bleu.cpu())

        print("\nBackward Transfer (BWT) per task (percentage):")
        for j, bwt in enumerate(forgeting[models[m]]):
            print(f"Task {task_to_supercat[j]}: {bwt:.2f}%")

        # Clean GPU memory before next iteration
        del model
        del learner

        gc.collect()
        torch.cuda.empty_cache()

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time / 60:.2f} minutes')

    # Plot average bleu score
    plt.title(f"Average BLEU-4 score while training on tasks")
    plt.xlabel("Tasks")
    plt.ylabel("Average BLEU-4")

    for model, bleu_list in avg_bleu.items():
        x = np.arange(1, n_tasks + 1, dtype=int)
        y = bleu_list
        plt.xticks(x, [task_to_supercat[t] for t in task_ids], rotation=45)
        plt.plot(x, y, 'o-', label=str(model))

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', title='model')
    plt.tight_layout()
    plt.grid()
    plt.show()

    # Plot average cider score
    plt.title(f"Average CIDEr score while training on tasks")
    plt.xlabel("Tasks")
    plt.ylabel("Average CIDEr")

    for model, cider_list in avg_cider.items():
        x = np.arange(1, n_tasks + 1, dtype=int)
        y = cider_list
        plt.xticks(x, [task_to_supercat[t] for t in task_ids], rotation=45)
        plt.plot(x, y, 'o-', label=str(model))

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', title='model')
    plt.tight_layout()
    plt.grid()
    plt.show()

    # Plot average cider score
    plt.title(f"BWT percentage while training on tasks")
    plt.xlabel("Tasks")
    plt.ylabel("BWT (%)")

    for model, bwt in forgeting.items():
        x = np.arange(1, n_tasks + 1, dtype=int)
        y = bwt
        plt.xticks(x, [task_to_supercat[t] for t in task_ids], rotation=45)
        plt.plot(x, y, 'o-', label=str(model))

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', title='model')
    plt.tight_layout()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
