from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel


def load_encoder_n_tokenizer(model_name: str):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    return tokenizer, model


class Classifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)

    def forward(self, inputs):
        return self.fc2(inputs)
