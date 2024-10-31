import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchvision import datasets, transforms
from gem import PlainGEMLearner
from models import MNISTModel


# Split dataset in tasks
def filter_dataset(dataset, labels, train, batch_size=64):
    loader = {}
    sizes = []

    for n_t, task in enumerate(labels):
        indices = [i for i in range(len(dataset)) if dataset.targets[i] in task]
        subset = torch.utils.data.Subset(dataset, indices)

        if train:
            loader[n_t] = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=train)
            sizes.append(len(subset))
        else:
            loader[n_t] = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=train)
            sizes.append(len(subset))

    return loader, sizes


def main():

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Define training and testing transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST specific normalization values
    ])

    # Load MNIST dataset
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # 2-task setting or 5-task setting
    labels_2 = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    labels_5 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    # Create data loaders
    model_train_loader, train_sizes = filter_dataset(train_data, labels_5, True)
    model_test_loader, test_sizes = filter_dataset(test_data, labels_5, False)

    print(f'Number of training examples per task: {train_sizes}')
    print(f'Number of test examples per task: {test_sizes}\n')

# # # ------------------------------------- TRAINING IN DISJOINT TASKS (CATASTROPHIC FORGETTING) ------------------------------------- # # #

    # Create model instance
    model = MNISTModel()

    model = model.to(device)
    print('Moved model to NVIDIA GeForce RTX 3060 Ti\n')

    num_epochs = 10
    n_tasks = 5
    avg_acc = []

    start_time = time.time()

    for t in range(n_tasks):

        # Train the model to tasks
        model.train_task(model_train_loader[t], t, num_epochs=num_epochs)

        acc = 0

        for i in range(t + 1):

            # Evaluate the model to test on seen tasks
            acc += model.evaluation(model_test_loader[i])

        avg_acc.append(acc / (t + 1))

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time / 60:.2f} minutes')

    x = np.arange(1, n_tasks + 1, dtype=int)

    plt.title('Average accuracy when training in tasks')
    plt.xlabel('# Tasks')
    plt.ylabel('Average accuracy')
    plt.plot(x, np.round(avg_acc, decimals=2), 'go-')

    # Set integer ticks on the x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid()
    plt.show()

# # # ------------------------------------- APPLYING GEM ALGORITHM TO MODEL ------------------------------------- # # #

    # Create data loaders
    gem_train_loader, sizes = filter_dataset(train_data, labels_5, True)
    gem_test_loader, _ = filter_dataset(test_data, labels_5, False)

    memsize_list = [100, 300, 1000, 3000]
    avg_acc = {}

    start_time = time.time()

    # Apply Gradient Episodic Memory algorithm
    for mem_size in memsize_list:
        memsize_acc = []

        model = MNISTModel()
        model = model.to(device)

        criterion = model.criterion
        optimizer = model.optimizer

        gem = PlainGEMLearner(model, n_tasks, optimizer, criterion, mem_size, 64)

        for t in range(n_tasks):

            gem.train_learner(gem_train_loader[t], t, sizes[t])

            for i in range(t + 1):

                # Evaluate the model to test on seen tasks
                gem.evaluation(gem_test_loader[i], i)

            # Average accuracy on seen tasks
            memsize_acc.append(torch.sum(gem.R[t]).item() / (t + 1))

        avg_acc[mem_size] = memsize_acc

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time / 60:.2f} minutes')

    plt.title("Average Accuracy while training on disjoint tasks")
    plt.xlabel("# of tasks")
    plt.ylabel("Average Accuracy")

    for memsize, acc_list in avg_acc.items():
        x = np.arange(1, n_tasks + 1, dtype=int)
        y = acc_list
        plt.plot(x, y, 'o-', label=str(memsize))

    # Set integer ticks on the x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc=3, title='memsize')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
