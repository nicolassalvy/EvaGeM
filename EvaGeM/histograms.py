import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def histograms(
    train_data,
    generated_data,
    test_data=None,
    res_save_dir=None,
    experiment_name="Default_experiment_name",
    compute_distance_histogram=True,
    compute_DCR=True,
    reusable_histogram_embedding=True,
    dataset_name="Default_dataset_name",
    reusable_path=None,
):
    """Computes and saves the histogram of the shortest distance between any
    initial data point and other initial data points. Computes and saves the
    histogram of the shortest distance between any generated data point and the
    initial data points.

    Args:
        train_data (np.array): The training data.
        generated_data (np.array): The generated data.
        test_data (np.array, optional): The test data, used as the holdout data
            for the DCR metric. Defaults to None.

        res_save_dir (str, optional): Path of the directory where the results
            will be saved. If it is not provided, the results are printed and
            returned but not saved. Defaults to None.
        experiment_name (str, optional): Name of the experiment. Used to save
            the results of different experiments in the same csv file.
            Defaults to "Default_experiment_name".

        compute_distance_histogram (bool, optional): Decides if the histogram
            of distances and its associated score should be computed. Defaults
            to True.
        compute_DCR (bool, optional): Decides if the Distance to Closest
            Records (DCR) histogram and associated score should be computed.
            Defaults to True.

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
    results = {
        "distance_score": None,
        "DCR_score": None,
    }
    if not compute_distance_histogram and not compute_DCR:
        print("No computation to do.")
        return results

    # Reshape for the NearestNeighbors
    train_data_neighbors = train_data.reshape(train_data.shape[0], -1)
    generated_data_neighbors = generated_data.reshape(
        generated_data.shape[0], -1
    )
    if compute_DCR and test_data is not None:
        test_data_neighbors = test_data.reshape(test_data.shape[0], -1)

    # Train the NearestNeighbors model on the real data
    neighbors_train = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(
        train_data_neighbors
    )  # Computed anyways because we need it for the generated data

    # Deal with the reusability
    if compute_distance_histogram:
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
                distances_real = np.loadtxt(
                    path_embedding + "histogram_real.txt"
                )

        if not reusable or not os.path.exists(
            path_embedding + "histogram_real.txt"
        ):  # Compute the histogram of the real data
            distances_real, _ = neighbors_train.kneighbors(
                train_data_neighbors
            )
            distances_real = distances_real[:, 1]
            # [:, 1] because the closest is the point itself

            if reusable:
                np.savetxt(
                    path_embedding + "histogram_real.txt", distances_real
                )

    # Compute the histogram of the generated data to train  # Always needed
    distances_gen_to_train, _ = neighbors_train.kneighbors(
        generated_data_neighbors
    )
    distances_gen_to_train = distances_gen_to_train[:, 0]

    # Compute the histogram of the generated data to test
    if test_data is not None and compute_DCR:
        neighbors_test = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
            test_data_neighbors
        )
        distances_test, _ = neighbors_test.kneighbors(generated_data_neighbors)
        distances_test = distances_test[:, 0]

    # Plot the histograms
    # Distance histogram
    if compute_distance_histogram:
        plt.figure(figsize=(5, 4), dpi=120)
        max_value = max(
            max(np.max(distances_real), np.max(distances_gen_to_train)), 6
        )
        plt.hist(
            distances_gen_to_train,
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
                res_save_dir
                + "/"
                + experiment_name
                + "/histogram_distances.pdf"
            )
        plt.show()

        distance_score = np.mean(distances_gen_to_train < distances_real)
        # Probability that a training point is closer to a generated point
        # than another training point
        # This should be close to #Generated / (#Generated + #Train)
        distance_target = generated_data.shape[0] / (
            generated_data.shape[0] + train_data.shape[0]
        )

        results["distance_score"] = 1 - np.abs(
            distance_score - distance_target
        ) / (1 - distance_target)

    if compute_DCR:
        plt.figure(figsize=(5, 4), dpi=120)
        max_value = max(
            max(np.max(distances_gen_to_train), np.max(distances_test)), 6
        )
        plt.hist(
            distances_gen_to_train,
            density=True,
            range=(0, max_value),
            bins=100,
            label="Train data",
            color="red",
            alpha=0.5,
        )
        plt.hist(
            distances_test,
            density=True,
            range=(0, max_value),
            bins=100,
            label="Test data",
            color="green",
            alpha=0.5,
        )
        plt.xlabel("Distance")
        plt.ylabel("Density of points")
        plt.legend()
        plt.title("DCR: Shortest distance to the generated data")
        if res_save_dir is not None:
            plt.savefig(
                res_save_dir + "/" + experiment_name + "/DCR_histogram.pdf"
            )

        plt.show()

        DCR_score = np.mean(distances_test > distances_gen_to_train)
        # Probability that a generated point is closer to the training data
        # than a holdout point
        # This should be close to #Train / (#Train + #Holdout)
        DCR_target = train_data.shape[0] / (
            train_data.shape[0] + test_data.shape[0]
        )
        print("DCR score:", DCR_score)
        print("DCR target:", DCR_target)

        results["DCR_score"] = 1 - np.abs(DCR_score - DCR_target) / (
            1 - DCR_target
        )

    return results
