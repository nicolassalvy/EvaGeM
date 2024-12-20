# EvaGeM (Evaluation of Generative Models)

EvaGeM is a package that provides a collection of metrics for the evaluation of generative models through the generated synthetic data they produce. It is model-agnostic and largely agnostic to the type of generated data!


## Install

We recommend installing `EvaGeM` in a virtual Python environment, such as a `conda` environment:

```
conda create -n EvaGeM python=3.10
conda activate EvaGeM
```

Then `git clone` this repository in your preferred path:
```
git clone https://github.com/nicolassalvy/EvaGeM.git
cd EvaGeM
pip install -e .
```

PyTorch is needed for the tutorial (both `torch` and `torchvision`) and is not covered by the dependencies, since the PyTorch version needed is dependent on your OS and device. To install PyTorch, use the command corresponding to your configuration from the [PyTorch website](https://pytorch.org/get-started/locally/).


## Using the package

Please refer to `tutorial.ipynb` for detailed instructions on using the package.

[Documentation](https://nicolassalvy.github.io/EvaGeM/) is available.

### General guidelines:
- **Standardize your data**: Your data should be standardized or preprocessed in the same way it was given to your generative model. Otherwise, the metrics might be biased.

- **Data splitting**: Split your real data into a training and test dataset. The generative model should have never seen the test dataset. There should be as much data in the train and generated datasets.


## Metrics

We evaluate 4 aspects of the generated data:
- **Quality**: How well the generated data approximates the desired output.
- **Coverage**: The extent to which the generated data covers the entire underlying data distribution.
- **Conditioning**: The model's ability to produce samples belonging to specific classes or attributes.
- **Generalisation**: The distinctiveness of the generated samples compared to the training data, discerning potential overfitting.

### List of Measures and their Evaluated aspects:

#### Distribution based

| Measure | Quality | Coverage | Generalisation | Conditioning |
|:------- |:-------:|:--------:|:--------------:|:--------------:|
| $\alpha$-Precision | :heavy_check_mark: | | | |
| $\beta$-Recall | | :heavy_check_mark: | | |
| Identifiability | | | :heavy_check_mark: | |
| Histogram of distances | | :heavy_check_mark: | :heavy_check_mark: | |
| DCR | | | :heavy_check_mark: | |

The [α-Precision](https://arxiv.org/abs/2102.08921) metric measures the fidelity to the real data and how typical each generated sample is (**do the generated samples come from the initial distribution?**), in $[0,1]$ (higher is better).

The [β-Recall](https://arxiv.org/abs/2102.08921) metric measures the **coverage**, in $[0,1]$ (higher is better).

They are usually used with the authenticity metric, that was introduced in the same paper to measure generalisation. However, as in [here](https://arxiv.org/abs/2310.09656) we found that the $\beta$-Recall and Authenticity metrics were usually very (negatively) correlated. 

Instead, we use a modified version of the [Identifiability](https://ieeexplore.ieee.org/abstract/document/9034117) metric. It measures the **proportion of real data leaked by the generated data**. Our version uses a reference real dataset for calibration, so that a given percentage (default 5%) of identifiability can be considered normal.

The **Histogram of distances** is a visualisation of the histogram of distances from each real data point to its closest real data point and the histogram of distances from each generated point to its closest real data point. If the generated data follows the same distribution as the initial data, those two histograms should match. The associated **distance score** is the proportion of training points that are closer to the generated set than to the training set. The ideal value is 0.5 (if there are as many generated and training samples). Note that this ideal value can be reached if half of the generated samples are near copies of the training data and the other half are of very poor quality (far from the training data). You can control for this by looking at the histogram.

Close to the histogram of distances, the histogram of **Distance to Closest Record** ([DCR](https://www.clearbox.ai/blog/2022-06-07-synthetic-data-for-privacy-preservation-part-2)) compares the distance of the generated data to the training set and to the holdout test set. The associated **DCR score** is the proportion of generated points that are closer to the training set than to the test set. The ideal value is the ratio of training points in the real dataset. Note that the DCR score can be good for generated data of very poor quality, because it is equally far from the train and test sets.


#### Classification based

| Measure | Quality | Coverage | Generalisation | Conditioning |
|:------- |:-------:|:--------:|:--------------:|:--------------:|
| GAN-train | :heavy_check_mark: | :heavy_check_mark: | | :heavy_check_mark: |
| GAN-test | | | :heavy_check_mark: | :heavy_check_mark: |
| Data Augmentation | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Discriminator | :heavy_check_mark: | | | |

The classification based metrics evaluate the synthetic data through the downstream task of classification. Except for the Discriminator metric, to use them your data needs to belong to different classes, as the classifiers are trained to classify them.

The [GAN-train](https://arxiv.org/abs/1807.09499) and [GAN-test](https://arxiv.org/abs/1807.09499) metrics are accuracy scores of classifiers. Their name starts with "GAN" because they were first introduced for GANs. The baseline for both metrics is a classifier trained on the real training dataset $D_{train}$ (used to train the generative model), and evaluated on the real test dataset $D_{test}$. The classifier architecture should perform well on this task.

The **GAN-train accuracy score** is obtained by **training a classifier on the generated dataset $D_{generated}$ and evaluating it on $D_{test}$**. This accuracy score should be as high as possible. This metric is also sometimes called Machine Learning Efficiency, Machine Learning Efficacy ([MLE](https://arxiv.org/abs/1907.00503)), or utility.

The **GAN-test accuracy score** is obtained by **evaluating the classifier trained on $D_{train}$ on $D_{generated}$**, the data that was generated from $D_{train}$. This should be close to the baseline. Scores significantly higher indicate potential overfitting or data leakage.

The **Data Augmentation accuracy score is obtained by evaluating a classifier trained on both $D_{train}$ and $D_{generated}$ on $D_{test}$**. This should be as high as possible.

The **Discriminator accuracy score is obtained by training a classifier to distinguish real from synthetic data** using $D_{train}$ and $D_{generated}$. This score should be as close to 0.5 as possible.


#### UMAPs

| Measure | Quality | Coverage | Generalisation | Conditioning |
|:------- |:-------:|:--------:|:--------------:|:--------------:|
| Generated on top | | :heavy_check_mark: | | |
| Together | | :heavy_check_mark: | | |
| Centroids | | :heavy_check_mark: | | :heavy_check_mark: |

These are different UMAP visualisations:
- **Generated on top** projects the generated data onto a UMAP computed from the real data.
- **Together** computes the UMAP from both the real and generated data together.
- **Centroids** shows the centroids of each class from both the real and generated data. A black dashed line connects each generated centroid to its corresponding initial centroid, while a red dashed line connects each generated centroid to its closest initial centroid.

