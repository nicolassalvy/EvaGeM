import os
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


# All n_jobs=-1 were removed because it does not work on some systems
# https://github.com/OpenMathLib/OpenBLAS/issues/3321#issuecomment-885186404


def OCSVM_scores(
    base_data,
    data_to_test,
    nus=np.linspace(1e-3, 1 - 1e-3, 50),
    n_jobs=2,
):
    """
        Compute the scores of data_to_test compared to base_data for different
        nu values. THE DATA SHOULD BE STANDARDIZED!

    Args:
        base_data (np.ndarray): The data to compare to.
        data_to_test (np.ndarray): The data to test.
        nus (np.ndarray, optional): The nu values to test. Defaults to
            np.linspace(1e-3, 1 - 1e-3, 50).
        n_jobs (int, optional): The number of jobs to run in parallel. Defaults
            to 2.

    Returns:
        np.array, np.array: The scores for each true alpha value,
            The true proportion of inliers in the base_data.
    """
    base_data_neighbours = base_data.reshape(base_data.shape[0], -1)
    data_to_test_neighbours = data_to_test.reshape(data_to_test.shape[0], -1)
    # We start by computing the gamma parameter of the Nystroem approximation.
    # The default in scikit-learn is
    # gamma = 1 / (np.prod(base_data.shape[1:]) * base_data.var())
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    # but gamma is related to the distances between the points. We compute it
    # based on the median of the 5th nearest neighbors distances.
    neighbors_base = NearestNeighbors(n_neighbors=5, n_jobs=n_jobs).fit(
        base_data_neighbours
    )
    distances_base, _ = neighbors_base.kneighbors(base_data_neighbours)
    distances_base_5 = distances_base[:, 4]
    distances_base_5_median = np.median(distances_base_5)
    gamma = 1 / (distances_base_5_median**2)

    transform = Nystroem(
        gamma=gamma,
        n_components=base_data_neighbours.shape[
            1
        ],  # number of coeffs for the approx, we just use everything
        n_jobs=n_jobs,
    ).fit(base_data_neighbours)
    transformed_base = transform.transform(base_data_neighbours)
    transformed_to_test = transform.transform(data_to_test_neighbours)

    scores_data_to_test = []
    # what is the ratio of inliers in the data to test
    # for different nu values (alphas)
    true_inliers = []  # how many are really inliers

    for nu in nus:
        clf = SGDOneClassSVM(
            nu=nu,
            shuffle=True,
            fit_intercept=True,
            tol=1e-3,
            # the iterations will stop when (loss > previous_loss - tol).
            # --> higher means worse
        )
        clf.fit(transformed_base)
        y_pred_train = clf.predict(transformed_base)
        true_inliers.append(
            y_pred_train[y_pred_train == 1].size / transformed_base.shape[0]
        )
        y_pred_to_test = clf.predict(transformed_to_test)
        scores_data_to_test.append(
            y_pred_to_test[y_pred_to_test == 1].size
            / transformed_to_test.shape[0]
        )

    # sort true_alphas and scores_data_to_test by true_alphas
    true_inliers, scores_data_to_test = zip(
        *sorted(zip(true_inliers, scores_data_to_test))
    )

    return np.array(scores_data_to_test), np.array(true_inliers)


def compute_integral(
    f_x,
    x,
):
    """
    Compute the integral of f_x - x with respect to x.

    Args:
        f_x (np.ndarray): The function values.
        x (np.ndarray): The x values.

    Returns:
        float: The integral.
    """
    if x[-1] < 1:
        x = np.concatenate((x, np.array([1])))
        f_x = np.concatenate((f_x, np.array([f_x[-1]])))
    if x[0] > 0:
        x = np.concatenate((np.array([0]), x))
        f_x = np.concatenate((np.array([f_x[0]]), f_x))
    return np.trapz(np.abs(f_x - x), x)


def alpha_precision(
    train_data,
    generated_data,
    res_save_dir=None,
    experiment_name="Default_experiment_name",
    plot_curve=True,
    number_of_alphas=50,
    n_jobs=2,
):
    """Compute the Alpha-Precision score. THE DATA SHOULD BE STANDARDIZED!

    Args:
        train_data (np.ndarray): The training data.
        generated_data (np.ndarray): The generated data.

        res_save_dir (str, optional): Path of the directory where the results
            will be saved. If it is not provided, the results are printed and
            returned but not saved. Defaults to None.
        experiment_name (str, optional): Name of the experiment. Used to save
            the results of different experiments in the same csv file.
            Defaults to "Default_experiment_name".

        plot_curve (bool, optional): If True, the curve of Alpha-Precision is
            plotted. Defaults to True.
        number_of_alphas (int, optional): The number of alpha values to test.
            Defaults to 50.

        n_jobs (int, optional): The number of jobs to run in parallel. Defaults
            to 2.

    Returns:
        float: The Alpha-Precision score.
    """
    alphas = np.linspace(
        1e-3, 1 - 1e-3, number_of_alphas
    )  # proportion of inliers
    nus = 1 - alphas  # how many can be outliers = 1 - proportion of inliers

    alpha_precision_curve, true_alphas = OCSVM_scores(
        base_data=train_data,
        data_to_test=generated_data,
        nus=nus,
        n_jobs=n_jobs,
    )

    Delta_Precision_Alpha = compute_integral(
        alpha_precision_curve, true_alphas
    )

    if plot_curve:
        plt.figure(figsize=(5, 4), dpi=120)
        plt.plot(true_alphas, alpha_precision_curve, marker="o")
        plt.xlabel("True proportion of training inliers")
        plt.ylabel("Proportion of generated inliers")
        plt.title("Alpha-Precision curve")
        plt.plot([0, 1], [0, 1], "--", color="black")
        if res_save_dir is not None:
            save_path = res_save_dir + "/" + experiment_name
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path + "/alpha_precision_curve.pdf")
        plt.show()

    IPAlpha = 1 - 2 * Delta_Precision_Alpha
    print("Alpha-Precision:", np.round(IPAlpha, 4))
    return IPAlpha


def beta_recall(
    train_data,
    generated_data,
    res_save_dir=None,
    experiment_name="Default_experiment_name",
    plot_curve=True,
    number_of_betas=50,
    n_jobs=2,
):
    """Compute the Beta-Recall score. THE DATA SHOULD BE STANDARDIZED!

    Args:
        train_data (np.ndarray): The real data.
        generated_data (np.ndarray): The generated data.

        res_save_dir (str, optional): Path of the directory where the results
            will be saved. If it is not provided, the results are printed and
            returned but not saved. Defaults to None.
        experiment_name (str, optional): Name of the experiment. Used to save
            the results of different experiments in the same csv file.
            Defaults to "Default_experiment_name".

        plot_curve (bool, optional): If True, the curve of the Beta-Recall is
            plotted. Defaults to True.
        number_of_betas (int, optional): The number of beta values to test.
            Defaults to 50.

        n_jobs (int, optional): The number of jobs to run in parallel. Defaults
            to 2.

    Returns:
        float: The Beta-Recall score.
    """

    betas = np.linspace(
        1e-3, 1 - 1e-3, number_of_betas
    )  # proportion of inliers
    nus = 1 - betas  # how many can be outliers = 1 - proportion of inliers

    beta_coverage_curve, true_betas = OCSVM_scores(
        base_data=generated_data,
        data_to_test=train_data,
        nus=nus,
        n_jobs=n_jobs,
    )

    Delta_Coverage_Beta = compute_integral(beta_coverage_curve, true_betas)

    if plot_curve:
        plt.figure(figsize=(5, 4), dpi=120)
        plt.plot(true_betas, beta_coverage_curve, marker="o")
        plt.xlabel("True proportion of generated inliers")
        plt.ylabel("Proportion of training inliers")
        plt.title("Beta-Recall curve")
        plt.plot([0, 1], [0, 1], "--", color="black")
        if res_save_dir is not None:
            save_path = res_save_dir + "/" + experiment_name
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path + "/beta_recall_curve.pdf")
        plt.show()

    IRBeta = 1 - 2 * Delta_Coverage_Beta
    print("Beta-Recall:", np.round(IRBeta, 4))
    return IRBeta


def authenticity(
    real_data,
    generated_data,
    n_jobs=2,
):
    """Compute the authenticity score.

    Args:
        real_data (np.ndarray): The real data.
        generated_data (np.ndarray): The generated data.
        n_jobs (int, optional): The number of jobs to run in parallel. Defaults
            to 2.

    Returns:
        float: The authenticity score.
    """
    real_data_neighbours = real_data.reshape(real_data.shape[0], -1)
    generated_data_neighbours = generated_data.reshape(
        generated_data.shape[0], -1
    )

    neighbors_real = NearestNeighbors(n_neighbors=2, n_jobs=n_jobs).fit(
        real_data_neighbours
    )
    # 2 to remove the closest which is the point itself
    distances_real, _ = neighbors_real.kneighbors(real_data_neighbours)
    distances_real = distances_real[:, 1]
    # [:, 1] because the closest is the point itself

    # for each generated point, we find the closest real point
    distances_generated_to_real, indices_real = neighbors_real.kneighbors(
        generated_data_neighbours
    )
    distances_generated_to_real = distances_generated_to_real[:, 0]
    distances_real_to_real = distances_real[indices_real[:, 0]]
    # [:, 0] because the closest is NOT the point itself --> not the same data

    authenticity = distances_generated_to_real > distances_real_to_real
    authenticity = authenticity.sum() / authenticity.size
    print("Authenticity:", np.round(authenticity, 4))
    return authenticity


def identifiability(
    real_data,
    reference_real_data,
    generated_data,
    reference_leaked_proportion=0.05,
    n_jobs=2,
    filter_leaky_generated=False,
):
    """
    Compute the identifiability score with the inlier_proportion. The
        identifiability score is the proportion of real points that have leaked
        in the generated_data.

    Args:
        real_data (np.array): The real data.
        reference_real_data (np.array): The reference real data used to compute
            the r value. (Proportion of Defaults to None.
        data_to_test (np.array): The data to test.
        reference_leaked_proportion (float, optional): The proportion of
            points in the reference_real_data that can be considered as leaked
            in the real_data. There is no real leakage, it is the reference
            value of what can be considered normal. Defaults to 0.05.
        n_jobs (int, optional): The number of jobs to run in parallel. Defaults
            to 2.
        filter_leaky_generated (bool, optional): If True, the function will
            also return a boolean array indicating which generated points are
            too close to the real data. Defaults to False.

    Returns:
        float: The identifiability score.
        optional np.array: The boolean array indicating which generated points
            are too close to the real data.
    """
    real_data_neighbours = real_data.reshape(real_data.shape[0], -1)
    reference_real_data_neighbours = reference_real_data.reshape(
        reference_real_data.shape[0], -1
    )
    generated_data_neighbours = generated_data.reshape(
        generated_data.shape[0], -1
    )

    neighbors_real = NearestNeighbors(n_neighbors=2, n_jobs=n_jobs).fit(
        real_data_neighbours
    )
    # 2 to remove the closest which is the point itself
    distances_real, _ = neighbors_real.kneighbors(real_data_neighbours)
    distances_real = distances_real[:, 1]
    # [:, 1] because the closest is the point itself

    neighbors_ref = NearestNeighbors(n_neighbors=1, n_jobs=n_jobs).fit(
        reference_real_data_neighbours
    )
    distances_real_to_ref, _ = neighbors_ref.kneighbors(real_data_neighbours)
    distances_real_to_ref = distances_real_to_ref[:, 0]

    dataset_count_ratio = (
        reference_real_data.shape[0] / real_data.shape[0]
    )  # to deal with different dataset sizes

    # remove the 0 distances
    distances_real[distances_real == 0] = distances_real[
        distances_real != 0
    ].min()
    distances_real_to_ref[distances_real_to_ref == 0] = distances_real_to_ref[
        distances_real_to_ref != 0
    ].max()
    r = np.quantile(
        distances_real_to_ref / distances_real,
        reference_leaked_proportion * dataset_count_ratio,
    )

    print(
        "The chosen r value for the identifiability score is", np.round(r, 3)
    )
    # r is such that
    # ident_ref = distances_real_to_gen < r * distances_base
    # ident_ref = ident_ref.sum() / ident_ref.size
    # is around reference_leaked_proportion

    # Compute the identifiability score with the found r value
    neighbors_gen = NearestNeighbors(n_neighbors=1, n_jobs=n_jobs).fit(
        generated_data_neighbours
    )
    distances_real_to_gen, _ = neighbors_gen.kneighbors(real_data_neighbours)
    distances_real_to_gen = distances_real_to_gen[:, 0]
    # [:, 0] because the closest is NOT the point itself --> not the same data

    # for each real_data point, is the closest point in generated_data or in
    # real_data? (Multiplied by the ratio from the reference dataset)
    identifiability = distances_real_to_gen < r * distances_real
    identifiability = identifiability.sum() / identifiability.size
    print("Identifiability:", np.round(identifiability, 4))

    if filter_leaky_generated:
        # We remove the generated points that are too close to the real data
        # that have leaked.
        distances_gen_to_real, real_indices = neighbors_real.kneighbors(
            generated_data_neighbours
        )
        # For each generated point, is it closer to its closest real point than
        # this real point is to its closest real point (multiplied by the
        # ratio)?
        leaky_generated = (
            distances_gen_to_real[:, 0]
            < r * distances_real[real_indices[:, 0]]
        )

        return identifiability, leaky_generated

    return identifiability
