from itertools import count
from typing import Optional

# from test import eval_tgt

from domain_adaptation import loss_cdan

# import params
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from finetune.eval import evaluate

lr = 1e-4
epochs = 5
log_step = 10


def train_target(
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    critic: torch.nn.Module,
    random_layer: Optional[torch.nn.Module],
    source_dataloader: DataLoader,
    target_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    device: torch.device,
):
    print("training start using target data")

    softmax = nn.Softmax(dim=1)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=lr,
    )
    optimizer_critic = optim.Adam(
        critic.parameters(),
        lr=lr,
    )
    len_data_loader = min(len(source_dataloader), len(target_dataloader))
    true_labels = []

    for epoch in range(epochs):
        encoder.train()
        critic.train()
        classifier.train()
        correct = 0
        total = 0
        total_loss = 0
        for step, source_batch, target_batch in zip(
            count(), source_dataloader, target_dataloader
        ):
            true_labels += source_batch["labels"].numpy().tolist()
            source_batch = {
                k: v.type(torch.long).to(device) for k, v in source_batch.items()
            }
            labels_source = source_batch.pop("labels")

            target_batch = {
                k: v.type(torch.long).to(device) for k, v in target_batch.items()
            }
            _ = target_batch.pop("labels")

            optimizer_critic.zero_grad()
            optimizer.zero_grad()

            source_feature = encoder(
                **source_batch, output_hidden_states=True
            ).hidden_states[-1][:, 0, :]
            target_feature = encoder(
                **target_batch, output_hidden_states=True
            ).hidden_states[-1][:, 0, :]
            concat_feature = torch.cat((source_feature, target_feature), 0)

            source_output = classifier(source_feature)
            target_output = classifier(target_feature)
            concat_output = torch.cat((source_output, target_output), 0)

            softmax_out = softmax(concat_output)

            entropy = loss_cdan.Entropy(softmax_out)

            transfer_loss = loss_cdan.CDAN(
                [concat_feature, softmax_out],
                critic,
                device,
                entropy,
                random_layer,
            )

            classifier_loss = nn.CrossEntropyLoss()(source_output, labels_source)

            total_loss = classifier_loss - 0.1 * transfer_loss

            total_loss.backward(retain_graph=True)
            transfer_loss.backward()

            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            # if epoch > 3:
            optimizer.step()

            optimizer_critic.step()

            _, pred_cls = torch.max(source_output.data, 1)
            correct += (pred_cls == labels_source).sum().item()
            total += labels_source.size(0)

            if (step + 1) % log_step == 0:
                print(
                    "Epoch [{}/{}] Step [{}/{}]:"
                    "transfer_loss={:.3f} total_loss={:.3f} acc={:.3f}".format(
                        epoch + 1,
                        epochs,
                        step + 1,
                        int(len_data_loader / source_dataloader.batch_size),  # type: ignore
                        transfer_loss.item(),
                        total_loss.item(),
                        correct / total,
                    )
                )
        print(f"{epoch=}")
        print(evaluate(encoder, classifier, eval_dataloader, device))

    return encoder, classifier
