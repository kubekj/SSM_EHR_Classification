import torch
import torch.nn as nn


class DSSM(nn.Module):
    def __init__(self, input_size, hidden_size, static_input_size, num_classes,
                 num_layers=2, dropout_rate=0.2, bidirectional=True):
        super(DSSM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.temporal_encoder = TemporalEncoder(
            input_size, hidden_size, num_layers, dropout_rate, bidirectional
        )
        self.static_encoder = StaticEncoder(
            static_input_size, hidden_size, dropout_rate
        )
        self.state_transition = StateTransition(
            hidden_size * self.num_directions, dropout_rate
        )
        self.classifier = Classifier(
            hidden_size * (self.num_directions + 1), hidden_size, num_classes, dropout_rate
        )

    def forward(self, temporal_data, static_data, seq_lengths):
        batch_size = temporal_data.size(0)

        # Process temporal and static data
        temporal_repr = self.temporal_encoder(temporal_data, seq_lengths)  # Shape: [batch, seq_len, hidden]
        static_repr = self.static_encoder(static_data)  # Shape: [batch, hidden]

        # Get final temporal representation (use the last relevant timestep for each sequence)
        temporal_repr = temporal_repr[torch.arange(batch_size), seq_lengths - 1]  # Shape: [batch, hidden]

        # Apply state transition
        state = self.state_transition(temporal_repr)  # Shape: [batch, hidden]

        # Combine representations and classify
        combined = torch.cat([state, static_repr], dim=1)  # Shape: [batch, hidden * (num_directions + 1)]
        output = self.classifier(combined)  # Shape: [batch, num_classes]

        return output


class TemporalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, bidirectional):
        super(TemporalEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * (2 if bidirectional else 1),
            num_heads=4,
            dropout=dropout_rate
        )

    def forward(self, temporal_data, seq_lengths):
        # Pack sequence for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            temporal_data, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Process with LSTM
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Apply attention
        attention_mask = (torch.arange(lstm_output.size(1))[None, :].to(lstm_output.device)
                          >= seq_lengths[:, None].to(lstm_output.device))
        attention_output = self._apply_attention(lstm_output, attention_mask)

        return attention_output

    @staticmethod
    def _create_attention_mask(tensor, seq_lengths):
        return torch.arange(tensor.size(1))[None, :] >= seq_lengths[:, None]

    def _apply_attention(self, lstm_output, mask):
        # Prepare attention inputs
        query = lstm_output.permute(1, 0, 2)
        key = value = query

        # Apply attention
        attn_output, _ = self.attention(query, key, value, key_padding_mask=mask)
        return attn_output.permute(1, 0, 2)


class StaticEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(StaticEncoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, static_data):
        return self.network(static_data)


class StateTransition(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(StateTransition, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.network(x)


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate):
        super(Classifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)
