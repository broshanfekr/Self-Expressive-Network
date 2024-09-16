import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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


if __name__ == "__main__":
    custom_colors = [
        '#00FFFF',  # aqua
        '#00FF00',  # lime
        '#0000FF',  # blue
        '#FF0000',  # red
        '#D2691E',  # Chocolate
        '#FF00FF',  # fuchsia
        '#008000',  # green
        '#9A0EEA',  # violet
        '#FFFF00',  # yellow
        '#008080',  # teal
        '#650021',  # maroon
        '#01153E',  # navy
        '#F97306',  # orange
        '#808080',  # gray
        '#000000',  # black
    ]


    embeddings, full_data, full_labels = load_var("cifar10_result/embeddings.pckl")
    X = embeddings
    labels = full_labels


    # file_path = "/home/autcup/bero/codes/data/data/cifar10/test/cifar10_test.pckl"

    # imgs, labels, X = load_var(file_path)
    # cluster_num = len(np.unique(labels))
    # labels = np.asarray(labels)


    tsne = TSNE(n_components=2, random_state=42)
    transformed_data = tsne.fit_transform(X)
    ground_truth = labels
    corresponding_colors = [custom_colors[i] for i in ground_truth]
    
    
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)  # 6x4 inches, 300 DPI
    
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=corresponding_colors, s=1)
    ax.grid(False)
    # ax.set_title('Labels from clustering', fontsize = 14) 
    
    plt.axis('off')
    # Adding labels and title
    # ax.set_xlabel('X-axis', fontsize=14)
    # ax.set_ylabel('Y-axis', fontsize=14)
    # ax.set_title('Sample Plot', fontsize=16)
    # ax.legend(fontsize=12)
    
    # Set tick parameters
    # ax.tick_params(axis='both', which='major', labelsize=12)

    # Save the figure as a high-resolution PNG and PDF
    plt.tight_layout()
    plt.savefig('cifar10_SeNET_representation.png', format='png', dpi=300)
    plt.savefig('cifar10_SeNET_representation.pdf', format='pdf')  # PDF for vector graphics