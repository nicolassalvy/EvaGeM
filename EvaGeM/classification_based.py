from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import os


def train_classifier(
    data,
    labels,
    test_data,
    test_labels,
    classifiers=[MLPClassifier, XGBClassifier],
    classifier_params=[{"max_iter": 10}, {"n_estimators": 10}],
    classifiers_should_flatten=[True, True],
):
    """Trains each classifier in the list and returns the average accuracy.

    Args:
        data (np.array): Data to train the model.
        labels (np.array): Labels of the data.
        test_data (np.array): Data to test the model.
        test_labels (np.array): Labels of the test data.
        classifiers (list, optional): List of sklearn compatible classifiers
            to instantiate. Defaults to [MLPClassifier, XGBClassifier].
        classifier_params (list, optional): List of dictionaries with the
            parameters to instantiate the classifiers.
            Defaults to [{"max_iter": 10}, {"n_estimators": 10}].
        classifiers_should_flatten (list, optional): List of booleans that
            decides if the data should be flattened before training the
            classifier. Defaults to [True, True].

    Returns:
        float: Accuracy of the model.
    """
    total_accuracy = 0
    for i in range(len(classifiers)):
        model = classifiers[i](**classifier_params[i])
        print("   Training the model:", model.__class__.__name__)

        data_ = (
            data.reshape(data.shape[0], -1)
            if classifiers_should_flatten[i]
            else data
        )
        test_data_ = (
            test_data.reshape(test_data.shape[0], -1)
            if classifiers_should_flatten[i]
            else test_data
        )

        # map the labels to [0, n_classes - 1]
        labels_list = np.unique(labels)
        labels_dict = {labels_list[i]: i for i in range(len(labels_list))}
        labels = np.array([labels_dict[label] for label in labels])
        test_labels = np.array([labels_dict[label] for label in test_labels])

        model.fit(data_, labels)
        y_pred = model.predict(test_data_)

        accuracy = accuracy_score(test_labels, y_pred)
        print("      Accuracy:", np.round(accuracy, 4))

        total_accuracy += accuracy

    return total_accuracy / len(classifiers)


def classification_scores(
    train_data,
    test_data,
    generated_data,
    train_labels=None,
    test_labels=None,
    generated_labels=None,
    compute_GAN_train=True,
    compute_GAN_test=True,
    compute_data_augmentation=True,
    compute_discriminator=True,
    reusable_baseline=True,
    dataset_name="Default_dataset_name",
    reusable_path=None,
    classifiers=[MLPClassifier, XGBClassifier],
    classifier_params=[{"max_iter": 10}, {"n_estimators": 10}],
    classifiers_should_flatten=[True, True],
):
    """Compute the classification based scores. train_data and generated_data
        should have the same number of samples.

    Args:
        train_data (np.array): the training data.
        test_data (np.array): the test data.
        generated_data (np.array): the generated data.

        train_labels (np.array, optional): training labels. Defaults to None.
        test_labels (np.array, optional): test labels. Defaults to None.
        generated_labels (np.array, optional): generated labels. Defaults to
            None.

        compute_GAN_train (bool, optional): Decides if GAN-train should be
            computed. Defaults to False.
        compute_GAN_test (bool, optional): Decides if GAN-test should be
            computed. Defaults to False.
        compute_data_augmentation (bool, optional): Decides if the model should
            also be evaluated as a data augmentation model. Defaults to False.
        compute_discriminator (bool, optional): Decides if the discriminator
            score should be computed. Defaults to False.

        reusable_baseline (bool, optional): Decides if the baseline is computed
            everytime (False) or if it can be saved and loaded (True). Useful
            if you want to test different generated datasets for the same real
            data. It is only used if a dataset_name is provided to avoid
            mistakes. Defaults to True.
        dataset_name (str, optional): Name of the real dataset. Necessary when
            reusable computations are used. Defaults to "Default_dataset_name".
        reusable_path (str, optional): Path to use for reusable computations.
            Needs to be provided if reusable is True. Defaults to None.

        classifiers (list, optional): List of sklearn compatible classifiers
            to instantiate. Defaults to [MLPClassifier, XGBClassifier].
        classifier_params (list, optional): List of dictionaries with the
            parameters to instantiate the classifiers.
            Defaults to [{"max_iter": 10}, {"n_estimators": 10}].
        classifiers_should_flatten (list, optional): List of booleans that
            decides if the data should be flattened before training the
            classifier. Defaults to [True, True].

    Returns:
        A dictionary with the following keys:
        - baseline: accuracy of the baseline classifier.
        - GAN_train: accuracy of the GAN-train classifier.
        - GAN_test: accuracy of the GAN-test classifier.
        - DA: accuracy of the data augmentation classifier.
        - discriminator: accuracy of the discriminator classifier.
    """
    classification_results = {
        "baseline": 0,
        "GAN_train": 0,
        "GAN_test": 0,
        "DA": 0,
        "discriminator": 0,
    }

    # Baseline
    if (
        compute_GAN_train or compute_GAN_test or compute_data_augmentation
    ) and train_labels is not None:
        print("Training the baseline...")
        reusable = (
            reusable_baseline
            and dataset_name != "Default_dataset_name"
            and reusable_path is not None
        )
        if reusable:  # TODO: the classifier objects could be saved?
            # for GAN-test we would not need to train another classifier
            path_baseline = (
                reusable_path
                + "/precomputed/"
                + dataset_name
                + "/classification/"
            )
            precomputed_path_pkl = path_baseline + "/baseline.pkl"

            if not os.path.exists(path_baseline):
                os.makedirs(path_baseline)

            if os.path.exists(precomputed_path_pkl):
                with open(precomputed_path_pkl, "rb") as f:
                    classification_results["baseline"] = joblib.load(f)

        if not reusable or not os.path.exists(precomputed_path_pkl):
            classification_results["baseline"] = train_classifier(
                data=train_data,
                labels=train_labels,
                test_data=test_data,
                test_labels=test_labels,
                classifiers=classifiers,
                classifier_params=classifier_params,
                classifiers_should_flatten=classifiers_should_flatten,
            )

            if reusable:
                with open(precomputed_path_pkl, "wb") as f:
                    joblib.dump(classification_results["baseline"], f)

        print(
            "Baseline accuracy:",
            np.round(classification_results["baseline"], 4),
        )

    # GAN train
    if compute_GAN_train and train_labels is not None:
        print("Training the GAN-train classifier...")
        classification_results["GAN_train"] = train_classifier(
            data=generated_data,
            labels=generated_labels,
            test_data=test_data,
            test_labels=test_labels,
            classifiers=classifiers,
            classifier_params=classifier_params,
            classifiers_should_flatten=classifiers_should_flatten,
        )
        print("GAN-train accuracy:", classification_results["GAN_train"])

    # GAN test
    if compute_GAN_test and train_labels is not None:
        print("Training the GAN-test classifier.")
        classification_results["GAN_test"] = train_classifier(
            data=train_data,
            labels=train_labels,
            test_data=generated_data,
            test_labels=generated_labels,
            classifiers=classifiers,
            classifier_params=classifier_params,
            classifiers_should_flatten=classifiers_should_flatten,
        )

        print(
            "GAN-test accuracy:",
            np.round(classification_results["GAN_test"], 4),
        )

    # Discriminator
    if compute_discriminator:
        print("Training the discriminator classifier...")
        min_len = min(train_data.shape[0], generated_data.shape[0])

        train_len = int(min_len * 0.8)
        indices_generated = np.arange(len(generated_data))
        indices_generated = np.random.permutation(indices_generated)
        indices_real = np.arange(len(train_data))
        indices_real = np.random.permutation(indices_real)

        generated_train = generated_data[indices_generated[:train_len]]
        generated_test = generated_data[indices_generated[train_len:min_len]]
        real_train = train_data[indices_real[:train_len]]
        real_test = train_data[indices_real[train_len:min_len]]

        discriminator_train = np.concatenate([real_train, generated_train])
        discriminator_train_labels = np.concatenate(
            [[0] * train_len, [1] * train_len]
        )
        discriminator_test = np.concatenate([real_test, generated_test])
        discriminator_test_labels = np.concatenate(
            [[0] * (min_len - train_len), [1] * (min_len - train_len)]
        )

        classification_results["discriminator"] = train_classifier(
            data=discriminator_train,
            labels=discriminator_train_labels,
            test_data=discriminator_test,
            test_labels=discriminator_test_labels,
            classifiers=classifiers,
            classifier_params=classifier_params,
            classifiers_should_flatten=classifiers_should_flatten,
        )

        print(
            "Discriminator accuracy:",
            np.round(classification_results["discriminator"], 4),
        )

    # Data augmentation
    if compute_data_augmentation and train_labels is not None:
        print("Training the data augmentation classifier...")
        classification_results["DA"] = train_classifier(
            data=np.concatenate([train_data, generated_data]),
            labels=np.concatenate((train_labels, generated_labels)),
            test_data=test_data,
            test_labels=test_labels,
            classifiers=classifiers,
            classifier_params=classifier_params,
            classifiers_should_flatten=classifiers_should_flatten,
        )

        print(
            "Data augmentation accuracy:",
            np.round(classification_results["DA"], 4),
        )

    return classification_results
