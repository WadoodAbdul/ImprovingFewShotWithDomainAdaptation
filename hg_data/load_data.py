from datasets.load import load_dataset
from hg_data.dataloader import create_dataloader
from hg_data.split_source_target import sample_source_target_dataset


def load_and_split(
    dataset_name: str,
    tokenizer,
    batch_size: int,
    num_samples_source: int,
    num_sample_target: int,
    seed: int,
):
    dataset = load_dataset(dataset_name)
    source_dataset, target_dataset = sample_source_target_dataset(
        dataset["train"],  # type: ignore
        num_samples_source=num_samples_source,
        num_samples_target=num_sample_target,
        seed=seed,
    )
    val_dataset = dataset["validation"].shuffle(seed=seed).select(list(range(100)))  # type: ignore
    source_dataloader = create_dataloader(source_dataset, tokenizer, batch_size)
    target_dataloader = create_dataloader(target_dataset, tokenizer, batch_size)
    val_dataloader = create_dataloader(val_dataset, tokenizer, batch_size)  # type: ignore
    return source_dataloader, target_dataloader, val_dataloader
