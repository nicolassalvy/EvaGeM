import numpy as np

import csv
import os

from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from .umaps import umaps

from .histograms import histograms
from .classification_based import classification_scores
from .distribution_based import (
    alpha_precision,
    beta_recall,
    authenticity,
    identifiability,
)


def evaluate_all(
    train_data,
    test_data,
    generated_data,
    train_labels=None,
    test_labels=None,
    generated_labels=None,
    classes=None,
    scaler=None,
    res_save_dir=None,
    experiment_name="Default_experiment_name",
    compute_UMAP_top=True,
    compute_UMAP_together=True,
    compute_UMAP_centroids=True,
    compute_distrib_based_alpha=True,
    compute_distrib_based_beta=True,
    compute_distrib_based_authenticity=False,
    compute_distrib_based_identifiability=True,
    compute_distrib_based_distance_histogram=True,
    compute_distrib_based_DCR=True,
    compute_classif_based_GAN_train=True,
    compute_classif_based_GAN_test=True,
    compute_classif_based_data_augmentation=True,
    compute_classif_based_discriminator=True,
    reusable_embeddings=True,
    reusable_classif_baseline=True,
    dataset_name="Default_dataset_name",
    reusable_path=None,
    UMAP_limit=2000,
    distrib_based_metrics_n_evaluated_points=50,
    distrib_identif_ref_leaked_proportion=0.05,
    distrib_based_n_jobs=2,
    classifiers=[MLPClassifier, XGBClassifier],
    classifier_params=[{"max_iter": 10}, {"n_estimators": 10}],
    classifiers_should_flatten=[True, True],
):
    """
    Evaluate the generated data.

    Args:
        train_data (np.array): The training data.
        test_data (np.array): The test data.
        generated_data (np.array): The generated data.

        train_labels (np.array, optional): Training labels. Defaults to None.
        test_labels (np.array, optional): Test labels. Defaults to None.
        generated_labels (np.array, optional): Generated labels. Defaults to
            None.
        classes (np.array, optional): Names of the classes. Defaults to None.

        scaler (sklearn compatible scaler, optional): The sklearn scaler used
            to scale the data. If provided, it is used to unscale the data
            before plotting UMAPs. Defaults to None.
        res_save_dir (str, optional): Path of the directory where the results
            will be saved. If it is not provided, the results are printed and
            returned but not saved. Defaults to None.
        experiment_name (str, optional): Name of the experiment. Used to save
            the results of different experiments in the same CSV file, and to
            save the plots in different directories. Defaults to
            "Default_experiment_name".

        compute_UMAP_top (bool, optional): Decides if the plot of the generated
            data projected onto the UMAP of the real data should be computed.
            Defaults to True.
        compute_UMAP_together (bool, optional): Decides if the plot of the
            UMAP computed from the real data and the generated data together
            should be computed. Defaults to True.
        compute_UMAP_centroids (bool, optional): Decides if another plot of the
            UMAPs computed should be shown with the centroids of each class.
            Defaults to True.

        compute_distrib_based_alpha (bool, optional): Decides if the
            distribution-based alpha-precision metric should be computed.
            Defaults to True.
        compute_distrib_based_beta (bool, optional): Decides if the
            distribution-based beta-recall metric should be computed.
            Defaults to True.
        compute_distrib_based_authenticity (bool, optional): Decides if the
            distribution-based authenticity metric should be computed. Defaults
            to False.
        compute_distrib_based_identifiability (bool, optional): Decides if the
            distribution-based identifiability metric should be computed.
            Defaults to True.
        compute_distrib_based_distance_histogram (bool, optional): Decides if
            the histogram of distances and its associated score should be
            computed. Defaults to True.
        compute_distrib_based_DCR (bool, optional): Decides if the
            distribution-based Distance to Closest Records (DCR) histogram and
            associated score should be computed. Defaults to True.
        compute_classif_based_GAN_train (bool, optional): Decides if
            GAN-train should be computed. Defaults to True.
        compute_classif_based_GAN_test (bool, optional): Decides if
            GAN-test should be computed. Defaults to True.
        compute_classif_based_data_augmentation (bool, optional): Decides if
            the model should also be evaluated as a data augmentation model.
            Defaults to True.
        compute_classif_based_discriminator (bool, optional): Decides
            if the discriminator score should be computed. Defaults to True.

        reusable_embeddings (bool, optional): Decides if the UMAP embedding and
            the histogram of distances of the real data are computed every time
            (False) or if they can be saved and loaded (True). Useful if you
            want to test different generated datasets from the same real data.
            It is only used if a dataset_name is provided to avoid mistakes.
            Defaults to True.
        reusable_classif_baseline (bool, optional): Decides if the
            classification baseline is computed every time (False) or if it can
            be saved and loaded (True). Useful if you want to test different
            generated datasets for the same real data. It is only used if a
            dataset_name is provided to avoid mistakes. Defaults to True.
        dataset_name (str, optional): Name of the real dataset. Necessary when
            reusable computations are used. Defaults to "Default_dataset_name".
        reusable_path (str, optional): Path to use for reusable computations.
            Needs to be provided if reusable is True. Defaults to None.

        UMAP_limit (int, optional): Limit of the number of points to plot in
            the UMAPs (the total amount can go up to 2 times UMAP_limit, one
            time for the real data, and another for the generated data).
            Defaults to 2000.
        distrib_based_metrics_n_evaluated_points (int, optional): Number of
            points to evaluate the distribution-based metrics on. Defaults to
            50.
        distrib_identif_ref_leaked_proportion (float, optional): The reference
            leaked proportion for the distribution-based metric
            identifiability. The proportion of points in the reference real
            data (eval_data) that can be considered as leaked in the
            train_data. There is no real leakage; it is the reference value of
            what can be considered normal. Defaults to 0.05.
        distrib_based_n_jobs (int, optional): Number of jobs to use for the
            distribution-based metrics. Defaults to 2.
        classifiers (list, optional): List of sklearn-compatible classifiers
            to instantiate for the classification-based metrics. Defaults to
            [MLPClassifier, XGBClassifier].
        classifier_params (list, optional): List of dictionaries with the
            parameters to instantiate the classifiers.
            Defaults to [{"max_iter": 10}, {"n_estimators": 10}].
        classifiers_should_flatten (list, optional): List of booleans that
            decides if the data should be flattened before training the
            classifier. Defaults to [True, True].

    Returns: dict: A dictionnary with the following keys:
        - Min (float): Min of the generated data.
        - Max (float): Max of the generated data.
        - DPA (float): Alpha-Precision metric.
        - DCB (float): Beta-Recall metric.
        - authenticity (float): Authenticity metric.
        - identifiability (float): Identifiability metric.
        - GAN_train (float): GAN train accuracy.
        - GAN_test (float): GAN test accuracy.
        - data_augmentation (float): Data augmentation accuracy.
        - discriminator (float): Discriminator accuracy.
    """
    results = {
        "alpha-precision": None,
        "beta-recall": None,
        "authenticity": None,
        "identifiability": None,
        "distance_score": None,
        "DCR_score": None,
        "classification_baseline": None,
        "GAN_train": None,
        "GAN_test": None,
        "data_augmentation": None,
        "discriminator": None,
    }

    results["Min"] = np.min(generated_data)
    results["Max"] = np.max(generated_data)
    print("Generated data: Min:", results["Min"], "Max:", results["Max"])

    if compute_UMAP_top or compute_UMAP_together:
        umaps(
            train_data=train_data,
            generated_data=generated_data,
            train_labels=train_labels,
            generated_labels=generated_labels,
            classes=classes,
            scaler=scaler,
            res_save_dir=res_save_dir,
            experiment_name=experiment_name,
            compute_UMAP_top=compute_UMAP_top,
            compute_UMAP_together=compute_UMAP_together,
            compute_UMAP_centroids=compute_UMAP_centroids,
            reusable_UMAP_embedding=reusable_embeddings,
            dataset_name=dataset_name,
            reusable_path=reusable_path,
            UMAP_limit=UMAP_limit,
        )

    if compute_distrib_based_distance_histogram or compute_distrib_based_DCR:
        res_hist = histograms(
            train_data=train_data,
            test_data=test_data,
            generated_data=generated_data,
            res_save_dir=res_save_dir,
            experiment_name=experiment_name,
            compute_distance_histogram=compute_distrib_based_distance_histogram,
            compute_DCR=compute_distrib_based_DCR,
            reusable_histogram_embedding=reusable_embeddings,
            dataset_name=dataset_name,
            reusable_path=reusable_path,
            n_jobs=distrib_based_n_jobs,
        )
        if compute_distrib_based_distance_histogram:
            results["distance_score"] = res_hist["distance_score"]
            print("Distance score:", np.round(results["distance_score"], 4))
        if compute_distrib_based_DCR:
            results["DCR_score"] = res_hist["DCR_score"]
            print("DCR score:", np.round(results["DCR_score"], 4))

    if compute_distrib_based_alpha:
        results["alpha-precision"] = alpha_precision(
            real_data=train_data,
            generated_data=generated_data,
            number_of_alphas=distrib_based_metrics_n_evaluated_points,
            n_jobs=distrib_based_n_jobs,
        )
        print("Alpha-Precision:", np.round(results["alpha-precision"], 4))

    if compute_distrib_based_beta:
        results["beta-recall"] = beta_recall(
            real_data=train_data,
            generated_data=generated_data,
            number_of_betas=distrib_based_metrics_n_evaluated_points,
            n_jobs=distrib_based_n_jobs,
        )
        print("Beta-Recall:", np.round(results["beta-recall"], 4))

    if compute_distrib_based_authenticity:
        results["authenticity"] = authenticity(
            real_data=train_data,
            generated_data=generated_data,
            n_jobs=distrib_based_n_jobs,
        )
        print("Authenticity:", np.round(results["authenticity"], 4))

    if compute_distrib_based_identifiability:
        results["identifiability"] = identifiability(
            real_data=train_data,
            reference_real_data=test_data,
            generated_data=generated_data,
            reference_leaked_proportion=distrib_identif_ref_leaked_proportion,
            n_jobs=distrib_based_n_jobs,
        )
        print("Identifiability:", np.round(results["identifiability"], 4))

    if (
        (
            compute_classif_based_GAN_train
            or compute_classif_based_GAN_test
            or compute_classif_based_data_augmentation
        )
        and (train_labels is not None)
    ) or compute_classif_based_discriminator:
        given_labels = train_labels is not None
        classification_results = classification_scores(
            train_data=train_data,
            test_data=test_data,
            generated_data=generated_data,
            train_labels=train_labels,
            test_labels=test_labels,
            generated_labels=generated_labels,
            compute_GAN_train=compute_classif_based_GAN_train and given_labels,
            compute_GAN_test=compute_classif_based_GAN_test and given_labels,
            compute_data_augmentation=compute_classif_based_data_augmentation
            and given_labels,
            compute_discriminator=compute_classif_based_discriminator,
            reusable_baseline=reusable_classif_baseline,
            dataset_name=dataset_name,
            reusable_path=reusable_path,
            classifiers=classifiers,
            classifier_params=classifier_params,
            classifiers_should_flatten=classifiers_should_flatten,
        )

        results["classification_baseline"] = classification_results["baseline"]
        results["GAN_train"] = classification_results["GAN_train"]
        results["GAN_test"] = classification_results["GAN_test"]
        results["data_augmentation"] = classification_results["DA"]
        results["discriminator"] = classification_results["discriminator"]

    if (
        res_save_dir is not None
        and experiment_name != "Default_experiment_name"
    ):
        save_path = res_save_dir + "/" + experiment_name
        if not os.path.isfile(save_path + "/res.csv"):
            with open(save_path + "/res.csv", "w") as f:
                wr = csv.writer(f)
                wr.writerow(
                    [
                        "experiment_name",
                        "alpha-precision",
                        "beta-recall",
                        "authenticity",
                        "identifiability",
                        "distance_score",
                        "DCR_score",
                        "classification_baseline",
                        "GAN_train",
                        "GAN_test",
                        "data_augmentation",
                        "discriminator",
                    ]
                )
        with open(save_path + "/res.csv", "a", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(
                [
                    experiment_name,
                    round_with_None(results["alpha-precision"], 4),
                    round_with_None(results["beta-recall"], 4),
                    round_with_None(results["authenticity"], 4),
                    round_with_None(results["identifiability"], 4),
                    round_with_None(results["distance_score"], 4),
                    round_with_None(results["DCR_score"], 4),
                    round_with_None(results["classification_baseline"], 4),
                    round_with_None(results["GAN_train"], 4),
                    round_with_None(results["GAN_test"], 4),
                    round_with_None(results["data_augmentation"], 4),
                    round_with_None(results["discriminator"], 4),
                ]
            )

    print("\nResults:")
    for key, value in results.items():
        if value is not None:
            print("   " + key + ":", np.round(value, 4))

    return results


def round_with_None(value, n):
    if value is None:
        return None
    return round(value, n)
