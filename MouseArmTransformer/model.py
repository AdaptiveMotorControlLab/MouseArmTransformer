import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from tqdm import tqdm


# Architeture:
class SimpleTransformer(nn.Module):
    """Transformer that lifts windows of 2D keypoints of one camera to 3D.

    With the default arguments this is the published architecture (the released weights load
    unchanged). The other arguments are opt-in options:

    Args:
        num_joints: Number of keypoints.
        d_model: Width of the transformer. Default ``2 * num_joints`` (the flattened input);
            another width needs ``input_proj=True``.
        nhead: Attention heads. Default ``num_joints // 4``.
        num_layers, dim_feedforward, dropout: Transformer encoder settings (PyTorch's
            defaults for the last two, as published).
        input_proj: Linear layer from the flattened 2D input to ``d_model``.
        pos_encoding: Learned positional encoding over the frames of a window (needs windows
            of at most ``max_seq_length`` frames).
        image_size: ``(width, height)`` in pixels. If given, inputs are centred on the image
            and divided by a third of its width inside the model, so the model takes raw
            pixels and the scaling travels with its weights.
        output_norm: If True, outputs are ``raw * output_scale + output_mean`` (a buffer of
            shape ``(num_joints, 3)``, set by the trainer), i.e. in the units of the training
            targets.
        output_scale: See ``output_norm``.
    """

    def __init__(self,
                 num_joints=25,
                 d_model=None,
                 nhead=None,
                 num_layers=2,
                 dim_feedforward=2048,
                 dropout=0.1,
                 input_proj=False,
                 pos_encoding=False,
                 max_seq_length=64,
                 image_size=None,
                 output_norm=False,
                 output_scale=1.0):
        super(SimpleTransformer, self).__init__()
        self.num_joints = num_joints
        in_dim = 2 * num_joints
        if d_model is None:
            d_model = in_dim
        if nhead is None:
            nhead = num_joints // 4
        if not input_proj and d_model != in_dim:
            raise ValueError(f"d_model={d_model} needs input_proj=True (input width is {in_dim}).")
        self._config = dict(num_joints=num_joints,
                            d_model=d_model,
                            nhead=nhead,
                            num_layers=num_layers,
                            dim_feedforward=dim_feedforward,
                            dropout=dropout,
                            input_proj=input_proj,
                            pos_encoding=pos_encoding,
                            max_seq_length=max_seq_length,
                            image_size=None if image_size is None else [float(v) for v in image_size],
                            output_norm=output_norm,
                            output_scale=float(output_scale))
        self.image_size = self._config["image_size"]
        self.output_scale = float(output_scale)

        # Optional layers are only created when used, so the default model's parameters are
        # exactly the published ones.
        if input_proj:
            self.proj = nn.Linear(in_dim, d_model)
        if pos_encoding:
            self.pos = nn.Parameter(0.02 * torch.randn(max_seq_length, d_model))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model,
                                                                            nhead=nhead,
                                                                            dim_feedforward=dim_feedforward,
                                                                            dropout=dropout,
                                                                            batch_first=True),
                                                 num_layers=num_layers)
        self.fc = nn.Linear(d_model, 3 * self.num_joints)
        if output_norm:
            self.register_buffer("output_mean", torch.zeros(num_joints, 3))

    def model_config(self):
        """Constructor arguments of this model (JSON-serialisable)."""
        return dict(self._config)

    def scale_input(self, x):
        """Centre pixel coordinates on the image and divide by a third of its width."""
        w, h = self.image_size
        offset = x.new_tensor([w / 2, h / 2])
        return (x - offset) / (w / 3)

    def forward(self, x):
        """Lift a batch of windows of 2D keypoints to 3D.

        x: (batch, time, joints, 2) in pixels. Returns (batch, time, joints, 3). Attention
        runs over the time steps of each window; windows in a batch are independent.
        """
        if self.image_size is not None:
            x = self.scale_input(x)
        x = rearrange(x, "b t j c -> b t (j c)")
        if hasattr(self, "proj"):
            x = self.proj(x)
        if hasattr(self, "pos"):
            x = x + self.pos[:x.shape[1]]
        #lstm_output, _ = self.lstm(x)  # LSTM output
        transformer_output = self.transformer(x)
        output = self.fc(transformer_output)  # Fully connected layer
        output = rearrange(output, 'b t (j c) -> b t j c', c=3)
        if hasattr(self, "output_mean"):
            output = output * self.output_scale + self.output_mean

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
