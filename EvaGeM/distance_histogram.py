import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def distance_histogram(
    train_data,
    generated_data,
    res_save_dir=None,
    experiment_name="Default_experiment_name",
    reusable_histogram_embedding=True,
    dataset_name="Default_dataset_name",
    reusable_path=None,
):
    """Computes and saves the histogram of the shortest distance between any
    initial data point and other initial data points. Computes and saves the
    histogram of the shortest distance between any generated data point and the
    initial data points.

    Args:
        train_data (np.array): the training data.
        generated_data (np.array): the generated data.
        res_save_dir (str, optional): path of the directory where the results
            will be saved. If it is not provided, the results are printed and
            returned but not saved. Defaults to None.
        experiment_name (str, optional): name of the experiment. Used to save
            the results of different experiments in the same csv file.
            Defaults to "Default_experiment_name".
        reusable_histogram_embedding (bool, optional): Decides if the histogram
            of the real data is computed everytime (False) or if it can be
            saved and loaded (True). Useful if you want to plot the same base
            histograms for different tests of generated data. Defaults to True.
        dataset_name (str, optional): Name of the real dataset. Necessary when
            reusable computations are used. Defaults to "Default_dataset_name".
        reusable_path (str, optional): Path to use for reusable
            computations. Needs to be provided if reusable is True. Defaults
            to None.
    """
    # Reshape for the NearestNeighbors
    train_data_neighbors = train_data.reshape(train_data.shape[0], -1)
    generated_data_neighbors = generated_data.reshape(
        generated_data.shape[0], -1
    )

    # Train the NearestNeighbors model on the real data
    neighbors_base = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(
        train_data_neighbors
    )  # Computed anyways because we need it for the generated data

    # Deal with the reusability
    reusable = (
        reusable_histogram_embedding
        and dataset_name != "Default_dataset_name"
        and reusable_path is not None
    )
    if reusable:
        path_embedding = (
            reusable_path
            + "/precomputed/"
            + dataset_name
            + "/precomputedHistograms/"
        )
        if not os.path.exists(path_embedding):
            os.makedirs(path_embedding)

        if os.path.exists(path_embedding + "histogram_real.txt"):
            distances_real = np.loadtxt(path_embedding + "histogram_real.txt")

    if not reusable or not os.path.exists(
        path_embedding + "histogram_real.txt"
    ):  # Compute the histogram of the real data
        distances_real, _ = neighbors_base.kneighbors(train_data_neighbors)
        distances_real = distances_real[:, 1]
        # [:, 1] because the closest is the point itself

        if reusable:
            np.savetxt(path_embedding + "histogram_real.txt", distances_real)

    # Compute the histogram of the generated data
    distances_gen, _ = neighbors_base.kneighbors(generated_data_neighbors)
    distances_gen = distances_gen[:, 0]

    # Plot the histograms
    plt.figure(figsize=(5, 4), dpi=120)
    max_value = max(max(np.max(distances_real), np.max(distances_gen)), 6)
    plt.hist(
        distances_gen,
        density=True,
        range=(0, max_value),
        bins=100,
        label="Generated data",
        color="red",
        alpha=0.5,
    )
    plt.hist(
        distances_real,
        density=True,
        range=(0, max_value),
        bins=100,
        label="Initial data",
        color="blue",
        alpha=0.5,
    )
    plt.xlabel("Distance")
    plt.ylabel("Density of points")
    plt.title("Shortest distances to the real data")
    plt.legend()
    if res_save_dir is not None:
        plt.savefig(
            res_save_dir + "/" + experiment_name + "/histogram_distances.pdf"
        )
    plt.show()
