from itertools import count

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from finetune.eval import evaluate

lr = 1e-3
epochs = 5
beta1 = 0.5
beta2 = 0.9


def train_source(
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    source_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    device: torch.device,
):

    optimizer = optim.Adam(
        encoder.parameters(),
        lr=lr,
        # betas=(beta1, beta2),
    )
    cross_entropy_loss = nn.CrossEntropyLoss()

    true_labels = []

    for epoch in range(epochs):
        encoder.train()
        classifier.train()
        correct = 0
        total = 0
        total_loss = 0
        for step, source_batch in zip(
            count(),
            source_dataloader,
        ):
            true_labels += source_batch["labels"].numpy().tolist()
            source_batch = {
                k: v.type(torch.long).to(device) for k, v in source_batch.items()
            }
            labels_source = source_batch.pop("labels")

            optimizer.zero_grad()

            # source_feature = encoder(**source_batch, output_hidden_states=True)
            # source_feature = source_feature.hidden_states[0][:, 0, :]
            source_feature = encoder(
                **source_batch,
            )
            source_output = source_feature
            print("source_output.logits", source_output.logits)
            # source_output = classifier(source_feature[0][:, 0])

            classifier_loss = cross_entropy_loss(source_output.logits, labels_source)

            classifier_loss.backward()
            # print(list(encoder.parameters())[-4])

            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            total_loss += classifier_loss.item()

            # _, pred_cls = torch.max(source_output.data, 1)
            # correct += (pred_cls == labels_source).sum().item()
            # total += labels_source.size(0)
        # print(f"training accuracy = {correct/total}")
        print(f"training loss after {epoch} epochs is {total_loss}")
        print(evaluate(encoder, classifier, eval_dataloader, device))

    return encoder, classifier
