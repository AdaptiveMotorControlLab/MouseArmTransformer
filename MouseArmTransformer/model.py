import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from tqdm import tqdm


# Architeture:
class SimpleTransformer(nn.Module):

    def __init__(self, num_joints=25):
        super(SimpleTransformer, self).__init__()
        self.num_joints = num_joints
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=2 * self.num_joints,
                                                                            nhead=(num_joints // 4)),
                                                 num_layers=2)
        self.fc = nn.Linear(2 * self.num_joints, 3 * self.num_joints)

    def forward(self, x):
        x = rearrange(x, "b t j c -> b t (j c)")
        #lstm_output, _ = self.lstm(x)  # LSTM output
        transformer_output = self.transformer(x)
        output = self.fc(transformer_output)  # Fully connected layer
        output = rearrange(output, 'b t (j c) -> b t j c', c=3)

        return output


def predict(model, camera1_data, seq_length):
    input_sequences = [camera1_data[max(0, i - seq_length + 1):i + 1] for i in range(camera1_data.shape[0])]
    input_sequences = torch.tensor(input_sequences, dtype=torch.float32)
    input_sequences = input_sequences.to('cuda')

    # Turn off gradients for prediction
    with torch.no_grad():
        # Forward pass through the model
        output, output_q = model(input_sequences)

    return output, output_q
