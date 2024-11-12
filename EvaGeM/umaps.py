import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import umap
import joblib


def prepare_plots(
    train_data,
    generated_data,
    train_labels=None,
    generated_labels=None,
    UMAP_limit=2000,
):
    """
    Prepare the data for plots: limit the amount of data to be plotted while
        keeping the same proportion of data from each class as in the labels.

    Args:
        generated_data (np.array): Generated data.
        train_data (np.array): Training data.
            If train_data is a np.array, then the class of the initial
            data is not used.
        train_labels (np.array, optional): Training labels. Defaults to None.
        generated_labels (np.array, optional): Generated labels. Defaults to
            None.
        UMAP_limit (int, optional): Limit of the number of points to plot in
            the UMAPs (the total amount can go up to 2 times UMAP_limit, one
            time for the real data, and another for the generated data).
            Defaults to 2000.

    Returns:
        np.array: data and labels for plots.
    """

    def filter_data(data, labels, limit):
        if labels is None:
            index = np.random.permutation(np.arange(data.shape[0]))
            return data[index[:limit]], None

        length_data = data.shape[0]
        if length_data > limit:
            unique_labels = np.unique(labels)
            bincounts = np.array(
                list(
                    {
                        label: np.sum(labels == label)
                        for label in unique_labels
                    }.values()
                )
            )  # np.bincount only works for integers
            proportion = bincounts / length_data
            n_classes = len(proportion)

            n_points_per_class = (proportion * limit).astype(int)
            filtered_data = np.zeros(
                (n_points_per_class.sum(),) + data.shape[1:]
            )
            filtered_labels = np.zeros((n_points_per_class.sum(),))
            start = 0

            classes_labels = np.unique(labels)
            for i in range(n_classes):
                indices = np.where(labels == classes_labels[i])[0]
                n_points = n_points_per_class[i]
                filtered_data[start : start + n_points] = data[
                    indices[:n_points]
                ]
                filtered_labels[start : start + n_points,] = classes_labels[i]
                start += n_points
            return filtered_data, filtered_labels
        return data, labels

    gen_data_to_plot, gen_labels_to_plot = filter_data(
        data=generated_data, labels=generated_labels, limit=UMAP_limit
    )

    train_data_to_plot, train_labels_to_plot = filter_data(
        data=train_data, labels=train_labels, limit=UMAP_limit
    )

    return (
        train_data_to_plot,
        train_labels_to_plot,
        gen_data_to_plot,
        gen_labels_to_plot,
    )


def plot_umap(
    train_data_to_plot,
    gen_data_to_plot,
    train_labels_to_plot=None,
    gen_labels_to_plot=None,
    classes=None,
    scaler=None,
    res_save_dir=None,
    experiment_name=None,
    compute_UMAP_centroids=True,
    reusable_UMAP_embedding=True,
    dataset_name="Default_dataset_name",
    reusable_path=None,
    on_top=False,
    title="Default_title",
):
    """Plot umap. Saves the plot if res_save_dir is not None.

    Args:
        train_data_to_plot (np.array): Filtered training data.
        gen_data_to_plot (np.array): Filtered generated data.
        train_labels_to_plot (np.array, optional): Filtered training labels.
            Defaults to None.
        gen_labels_to_plot (np.array, optional): Filtered generated labels.
            Defaults to None.
        classes (np.array, optional): Names of the classes. Defaults to None.
        scaler (sklearn compatible scaler): The sklearn scaler used to scale
            the data. If provided, it is used to unscale the data before
            plotting UMAPs. Defaults to None.
        res_save_dir (str, optional): Path of the directory where the results
            will be saved. If it is not provided, the results are printed and
            returned but not saved. Defaults to None.
        experiment_name (str, optional): Name of the experiment. Used to save
            the results of different experiments in the same csv file.
            Defaults to "Default_experiment_name".
        compute_UMAP_centroids (bool, optional): Decides if another plot of the
            UMAPs computed should be shown with the Centroids of each class.
            Defaults to True.
        reusable_UMAP_embedding (bool, optional): Decides if the UMAP embedding
            of the real data is computed everytime (False) or if it can be
            saved and loaded (True). Useful if you want to test different
            generated datasets from the same real data. It is only used if a
            dataset_name is provided to avoid mistakes. Defaults to True.
        dataset_name (str, optional): Name of the real dataset. Necessary when
            reusable computations are used. Defaults to "Default_dataset_name".
        reusable_path (str, optional): Path to use for reusable
            computations. Needs to be provided if reusable is True. Defaults
            to None.
        on_top (bool, optional): Decides if the embedding is computed with the
            generated data (False) or if the generated data will be projected
            onto the embedding of the initial data (True). Defaults to False.
        title (str, optional): Title of the plot. Defaults to "Default_title".
    """
    # Unscale the data if a scaler is provided
    train_data_to_plot = train_data_to_plot.reshape(
        train_data_to_plot.shape[0], -1
    )
    gen_data_to_plot = gen_data_to_plot.reshape(gen_data_to_plot.shape[0], -1)
    if scaler is not None:
        train_data_to_plot = scaler.inverse_transform(train_data_to_plot)
        gen_data_to_plot = scaler.inverse_transform(gen_data_to_plot)

    # Deal with the reusability
    reusable = (
        reusable_UMAP_embedding
        and dataset_name != "Default_dataset_name"
        and reusable_path is not None
    )

    if reusable:
        path_embedding = (
            reusable_path
            + "/precomputed/"
            + dataset_name
            + "/precomputedUMAPs/"
        )
        if not os.path.exists(path_embedding):
            os.makedirs(path_embedding)

    # If on_top, load the UMAP embedding of the real data if it exists
    # and compute it if it does not. Save it if necessary.
    if on_top:
        if reusable and os.path.exists(
            path_embedding + "real_data_reducer.pkl"
        ):
            reducer = joblib.load(path_embedding + "real_data_reducer.pkl")
            real_embedding = joblib.load(
                path_embedding + "real_data_embedding.pkl"
            )
        else:
            reducer = umap.UMAP()
            real_embedding = reducer.fit_transform(train_data_to_plot)
            if reusable:
                joblib.dump(reducer, path_embedding + "real_data_reducer.pkl")
                joblib.dump(
                    real_embedding,
                    path_embedding + "real_data_embedding.pkl",
                )

        gen_embedding = reducer.transform(gen_data_to_plot)
        embedding = np.concatenate((real_embedding, gen_embedding))
    else:  # Otherwise, compute the UMAP embedding of all the data.
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(
            np.concatenate((train_data_to_plot, gen_data_to_plot))
        )

    # Plot the UMAP
    fig, ax_full = plt.subplots(figsize=(5, 5), dpi=120)
    gs = GridSpec(2, 1, height_ratios=[0.2, 0.8])
    ax = fig.add_subplot(gs[1])
    ax_legend = fig.add_subplot(gs[0])
    ax_legend.axis("off")
    ax_full.axis("off")
    ax.set_aspect("equal", "datalim")

    # Plot the real data with marker x
    real_data = ax.scatter(
        embedding[: train_data_to_plot.shape[0], 0],
        embedding[: train_data_to_plot.shape[0], 1],
        s=4,
        c="blue",
        marker="x",
        label="Initial data",
        alpha=0.5,
    )

    # Plot the generated data with marker o
    generated_data = ax.scatter(
        embedding[train_data_to_plot.shape[0] :, 0],
        embedding[train_data_to_plot.shape[0] :, 1],
        s=4,
        c="red",
        marker="o",
        label="Generated data",
        alpha=0.5,
    )

    legend_handles = [real_data, generated_data]
    legend_labels = ["Real data", "Generated data"]

    ax_legend.legend(
        legend_handles, legend_labels, loc="upper center", fontsize=12
    )
    plt.subplots_adjust(hspace=0)

    plt.title(title)

    if res_save_dir is not None:
        if not os.path.exists(
            os.path.join(res_save_dir, experiment_name, "UMAPs")
        ):
            os.makedirs(os.path.join(res_save_dir, experiment_name, "UMAPs"))
        plt.savefig(
            os.path.join(
                res_save_dir, experiment_name, "UMAPs", title + ".pdf"
            )
        )
    plt.show()

    # Centroids
    if compute_UMAP_centroids:
        if train_data_to_plot is not None:  # more than one real class
            classes_labels = np.unique(train_labels_to_plot)

            centroids_real = np.array(
                [
                    np.mean(
                        (embedding[: train_data_to_plot.shape[0]])[
                            train_labels_to_plot == classes_labels[i]
                        ],
                        axis=0,
                    )
                    for i in range(len(classes_labels))
                ]
            )

            centroids_gen = np.array(
                [
                    np.mean(
                        (embedding[train_data_to_plot.shape[0] :])[
                            gen_labels_to_plot == classes_labels[i]
                        ],
                        axis=0,
                    )
                    for i in range(len(classes_labels))
                ]
            )

        else:
            centroids_real = np.array(
                [np.mean(embedding[: train_data_to_plot.shape[0]], axis=0)]
            )

            centroids_gen = np.array(
                [np.mean(embedding[train_data_to_plot.shape[0] :], axis=0)]
            )

        n_iter = 1 if classes is None else 2
        for j in range(n_iter):  # once with the lines, and once with the names
            fig, ax_full = plt.subplots(figsize=(5, 5), dpi=120)
            gs = GridSpec(2, 1, height_ratios=[0.3, 0.7])
            ax = fig.add_subplot(gs[1])
            ax_legend = fig.add_subplot(gs[0])
            ax_legend.axis("off")
            ax_full.axis("off")

            # Plot the real data with marker x
            real_data = ax.scatter(
                embedding[: train_data_to_plot.shape[0], 0],
                embedding[: train_data_to_plot.shape[0], 1],
                s=4,
                c="grey",
                marker="x",
                label="Initial data",
                alpha=0.2,
            )

            # Plot the generated data with marker o
            generated_data = ax.scatter(
                embedding[train_data_to_plot.shape[0] :, 0],
                embedding[train_data_to_plot.shape[0] :, 1],
                s=4,
                c="grey",
                marker="o",
                label="Generated data",
                alpha=0.2,
            )

            initial_centroids = ax.scatter(
                centroids_real[:, 0],
                centroids_real[:, 1],
                s=40,
                c="blue",
                marker="x",
                label="Initial centroids",
            )
            generated_centroids = ax.scatter(
                centroids_gen[:, 0],
                centroids_gen[:, 1],
                s=40,
                c="red",
                marker="o",
                label="Generated centroids",
            )

            if j == 0:
                # Plot a line between each initial centroid and its generated
                # counterpart
                # and a line between each generated centroid and its closest
                # real centroid
                if (
                    train_data_to_plot is not None
                ):  # more than one initial class
                    for i in range(len(classes_labels)):
                        ax.plot(
                            [centroids_real[i, 0], centroids_gen[i, 0]],
                            [centroids_real[i, 1], centroids_gen[i, 1]],
                            c="black",
                            linestyle="dashed",
                        )
                        closest_centroid = np.argmin(
                            np.linalg.norm(
                                centroids_real - centroids_gen[i], axis=1
                            )
                        )
                        ax.plot(
                            [
                                centroids_gen[i, 0],
                                centroids_real[closest_centroid, 0],
                            ],
                            [
                                centroids_gen[i, 1],
                                centroids_real[closest_centroid, 1],
                            ],
                            c="red",
                            linestyle="dashed",
                        )
            else:
                for i in range(centroids_real.shape[0]):
                    ax.text(
                        centroids_real[i, 0],
                        centroids_real[i, 1],
                        classes[i],
                        fontsize=12,
                        horizontalalignment="center",
                        verticalalignment="center",
                    )

            ax.set_aspect("equal", "datalim")
            title_centroids = "Centroids"

            legend_handles = [
                real_data,
                generated_data,
                initial_centroids,
                generated_centroids,
            ]
            legend_labels = [
                "Real data",
                "Generated data",
                "Real Centroids",
                "Generated Centroids",
            ]

            ax_legend.legend(
                legend_handles, legend_labels, loc="upper center", fontsize=12
            )
            plt.subplots_adjust(hspace=0)

            if j == 0:
                plt.title(title + " " + title_centroids)

                if res_save_dir is not None:
                    plt.savefig(
                        os.path.join(
                            res_save_dir,
                            experiment_name,
                            "UMAPs",
                            title + title_centroids + ".pdf",
                        )
                    )

            else:
                plt.title(title + " " + title_centroids + " with names")

                if res_save_dir is not None:
                    plt.savefig(
                        os.path.join(
                            res_save_dir,
                            experiment_name,
                            "UMAPs",
                            title + title_centroids + "withnames.pdf",
                        )
                    )

            plt.show()


def umaps(
    train_data,
    generated_data,
    train_labels=None,
    generated_labels=None,
    classes=None,
    scaler=None,
    res_save_dir=None,
    experiment_name=None,
    compute_UMAP_top=True,
    compute_UMAP_together=True,
    compute_UMAP_centroids=True,
    reusable_UMAP_embedding=True,
    dataset_name="Default_dataset_name",
    reusable_path=None,
    UMAP_limit=2000,
):
    """Plot the umaps. Saves the plot if res_save_dir is not None. Filters the
        data to plot if there is too much.

    Args:
        train_data (np.array): The training data.
        generated_data (np.array): The generated data.

        train_labels (np.array, optional): Training labels. Defaults to None.
        generated_labels (np.array, optional): Generated labels. Defaults to
            None.

        classes (np.array, optional): Names of the classes. Defaults to None.
        scaler (sklearn compatible scaler): The sklearn scaler used to scale
            the data. If provided, it is used to unscale the data before
            plotting UMAPs. Defaults to None.

        res_save_dir (str, optional): Path of the directory where the results
            will be saved. If it is not provided, the results are printed and
            returned but not saved. Defaults to None.
        experiment_name (str, optional): Name of the experiment. Used to save
            the results of different experiments in the same csv file.
            Defaults to "Default_experiment_name".

        compute_UMAP_top (bool, optional): Decides if the plot of the generated
            data projected onto the UMAP of the real data should be computed.
            Defaults to True.
        compute_UMAP_together (bool, optional): Decides if the plot of the
            UMAP computed from the real data and the generated data together
            should be computed. Defaults to True.
        compute_UMAP_centroids (bool, optional): Decides if another plot of the
            UMAPs computed should be shown with the Centroids of each class.
            Defaults to True.

        reusable_UMAP_embedding (bool, optional): Decides if the UMAP embedding
            of the real data is computed everytime (False) or if it can be
            saved and loaded (True). Useful if you want to test different
            generated datasets from the same real data. It is only used if a
            dataset_name is provided to avoid mistakes. Defaults to True.
        dataset_name (str, optional): Name of the real dataset. Necessary when
            reusable computations are used. Defaults to "Default_dataset_name".
        reusable_path (str, optional): Path to use for reusable
            computations. Needs to be provided if reusable is True. Defaults
            to None.

        UMAP_limit (int, optional): Limit of the number of points to plot in
            the UMAPs (the total amount can go up to 2 times UMAP_limit, one
            time for the real data, and another for the generated data).
            Defaults to 2000.
    """
    if compute_UMAP_top or compute_UMAP_together:
        (
            train_data_to_plot,
            train_labels_to_plot,
            gen_data_to_plot,
            gen_labels_to_plot,
        ) = prepare_plots(
            train_data=train_data,
            generated_data=generated_data,
            train_labels=train_labels,
            generated_labels=generated_labels,
            UMAP_limit=UMAP_limit,
        )

    if compute_UMAP_top:
        plot_umap(
            train_data_to_plot=train_data_to_plot,
            gen_data_to_plot=gen_data_to_plot,
            train_labels_to_plot=train_labels_to_plot,
            gen_labels_to_plot=gen_labels_to_plot,
            classes=classes,
            scaler=scaler,
            res_save_dir=res_save_dir,
            experiment_name=experiment_name,
            compute_UMAP_centroids=compute_UMAP_centroids,
            reusable_UMAP_embedding=reusable_UMAP_embedding,
            dataset_name=dataset_name,
            reusable_path=reusable_path,
            on_top=True,
            title="Top",
        )

    if compute_UMAP_together:
        plot_umap(
            train_data_to_plot=train_data_to_plot,
            gen_data_to_plot=gen_data_to_plot,
            train_labels_to_plot=train_labels_to_plot,
            gen_labels_to_plot=gen_labels_to_plot,
            classes=classes,
            scaler=scaler,
            res_save_dir=res_save_dir,
            experiment_name=experiment_name,
            compute_UMAP_centroids=compute_UMAP_centroids,
            reusable_UMAP_embedding=reusable_UMAP_embedding,
            dataset_name=dataset_name,
            reusable_path=reusable_path,
            on_top=False,
            title="Together",
        )
