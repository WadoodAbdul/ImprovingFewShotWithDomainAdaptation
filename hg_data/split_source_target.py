from datasets.combine import concatenate_datasets
from typing import Tuple
from datasets.arrow_dataset import Dataset as hgDataset


def sample_source_target_dataset(
    dataset: hgDataset,
    label_column: str = "label",
    num_samples_source: int = 10,
    num_samples_target: int = 50,
    seed: int = 42,
) -> Tuple[hgDataset, hgDataset]:
    """Samples a Dataset to create an equal number of samples per class (when possible)."""
    shuffled_dataset = dataset.shuffle(seed=seed)
    num_labels = len(dataset.unique(label_column))
    source_samples = []
    target_samples = []
    for label in range(num_labels):
        data = shuffled_dataset.filter(
            lambda example: int(example[label_column]) == label
        )
        num_label_samples_source = min(len(data), num_samples_source)
        num_label_samples_target = max(
            min(len(data) - num_label_samples_source, num_samples_target), 0
        )
        source_samples.append(data.select(list(range(num_label_samples_source))))
        target_samples.append(
            data.select(
                list(
                    range(
                        num_label_samples_source,
                        num_label_samples_source + num_label_samples_target,
                    )
                )
            )
        )

    all_source_samples = concatenate_datasets(source_samples)
    all_target_samples = concatenate_datasets(target_samples)
    return all_source_samples.shuffle(seed=seed), all_target_samples.shuffle(seed=seed)
