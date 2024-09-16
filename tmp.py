import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import block_diag
import pickle
from sklearn.metrics.pairwise import cosine_similarity


def load_var(load_path):
    file = open(load_path, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable


def save_var(save_path, variable):
    file = open(save_path, 'wb')
    pickle.dump(variable, file)
    print("variable saved.")
    file.close()


def draw_adj(adjacency_matrix, labels, path):
    sample_idx = np.argsort(labels, kind='stable')
    adjacency_matrix = adjacency_matrix[sample_idx, :]
    adjacency_matrix = adjacency_matrix[:, sample_idx]

    # Step 2: Plotting the adjacency matrix
    # Using Seaborn to plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(adjacency_matrix, cmap='viridis', square=True, cbar=True)

    plt.axis('off')
    # Save the plot
    plt.savefig(path, dpi=300)


if __name__ == "__main__":
    file_path = "/home/roshanfekr/roshanfekr/data/data/cifar10/test/cifar10_test.pckl"
    imgs, labels, X = load_var(file_path)
    cluster_num = len(np.unique(labels))
    labels = np.asarray(labels)
    
    adjacency_matrix = cosine_similarity(X)
    A_min = np.min(adjacency_matrix)
    A_max = np.max(adjacency_matrix)

    # Scale the matrix between 0 and 1
    adjacency_matrix = (adjacency_matrix - A_min) / (A_max - A_min)
    
    sample_idx = np.argsort(labels, kind='stable')
    adjacency_matrix = adjacency_matrix[sample_idx, :]
    adjacency_matrix = adjacency_matrix[:, sample_idx]

    draw_adj(adjacency_matrix=adjacency_matrix, labels=labels, path='cifar10_CLIP_adj.png')
    
    
    embeddings, full_data, full_labels = load_var("cifar10_result/embeddings.pckl")
    X = embeddings
    labels = full_labels
    
    adjacency_matrix = cosine_similarity(X)
    A_min = np.min(adjacency_matrix)
    A_max = np.max(adjacency_matrix)

    # Scale the matrix between 0 and 1
    adjacency_matrix = (adjacency_matrix - A_min) / (A_max - A_min)
    
    sample_idx = np.argsort(labels, kind='stable')
    adjacency_matrix = adjacency_matrix[sample_idx, :]
    adjacency_matrix = adjacency_matrix[:, sample_idx]
    
    draw_adj(adjacency_matrix=adjacency_matrix, labels=labels, path='cifar10_SeNET_adj.png')
    
    print("the end")
