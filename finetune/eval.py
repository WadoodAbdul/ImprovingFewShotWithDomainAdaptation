import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification


# Define the evaluation function
def evaluate(
    encoder: torch.nn.Module,
    classifer: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    # Set the model to evaluation mode
    encoder.eval()
    classifer.eval()

    # Initialize empty lists to store predictions and labels
    all_preds = []
    all_labels = []

    # Loop over the batches in the dataloader
    for batch in dataloader:
        # Move the batch to the device
        batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}
        labels = batch.pop("labels")

        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            # outputs = encoder(**batch, output_hidden_states=True).hidden_states[-1][
            #     :, 0, :
            # ]
            # outputs = classifer(outputs)
            source_feature = encoder(
                **batch,
            )
            outputs = source_feature.logits
        print("prediction", outputs)
        # Convert logits to predictions
        preds = torch.argmax(outputs, dim=1)

        # Append the predictions and labels to the lists
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        print("preds=", preds.cpu().tolist())
        print("labels=", labels.cpu().tolist())

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return f"{accuracy=}, {precision=}, {recall=}, {f1=}"
