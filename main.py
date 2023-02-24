from hg_data.load_data import load_and_split
from finetune.transformer_model import load_encoder_n_tokenizer, Classifier
from finetune.training import train_source
from domain_adaptation.adapt_cdan import train_target
import torch
from domain_adaptation.discriminator import AdversarialNetwork, RandomLayer


def main():
    tokenizer, encoder = load_encoder_n_tokenizer("roberta-base")
    classifier = Classifier(768, 5)
    source_dataloader, target_dataloader, val_dataloader = load_and_split(
        "SetFit/sst5", tokenizer, 16, 32, 32, 13
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_layer = RandomLayer([768, 5], output_dim=1024)
    critic = AdversarialNetwork(1024, 512)
    random_layer._to(device)
    critic.to(device)
    classifier.to(device)
    encoder.to(device)
    encoder, classifier = train_source(
        encoder, classifier, source_dataloader, val_dataloader, device
    )
    train_target(
        encoder,
        classifier,
        critic,
        random_layer,
        source_dataloader,
        target_dataloader,
        val_dataloader,
        device,
    )


if __name__ == "__main__":
    main()
