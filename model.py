import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

class PositionalEmbeddings(nn.Module):
    """
    PositionalEmbeddings layer.

    This layer generates positional embeddings based on input IDs.
    It uses an Embedding layer to map position IDs to position embeddings.

    Args:
        config (object): Configuration object containing parameters.
            - seq_len (int): Maximum sequence length.
            - hidden_size (int): Size of the hidden embeddings.
    """

    def __init__(self, config):
        """
        Initializes the PositionalEmbeddings layer.

        Args:
            config (object): Configuration object containing parameters.
                - seq_len (int): Maximum sequence length.
                - hidden_size (int): Size of the hidden embeddings.
        """
        super().__init__()

        self.seq_len: int = config.seq_len
        self.hidden_size: int = config.hidden_size
        self.positional_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=self.seq_len, embedding_dim=self.hidden_size
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate positional embeddings.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.

        Returns:
            torch.Tensor: Positional embeddings tensor of shape (batch_size, seq_length, hidden_size).
        """
        seq_length: int = input_ids.size(1)
        position_ids: torch.Tensor = torch.arange(seq_length, dtype=torch.int32, device=input_ids.device).unsqueeze(0)
        position_embeddings: torch.Tensor = self.positional_embeddings(position_ids)
        return position_embeddings



class Embeddings(nn.Module):

    """
    Embeddings layer.

    This layer combines token embeddings with positional embeddings and segment embeddings
    to create the final embeddings.

    Args:
        config (object): Configuration object containing parameters.
            - hidden_size (int): Size of the hidden embeddings.
            - vocab_size (int): Size of the vocabulary.
            - hidden_dropout_prob (float): Dropout probability for regularization.

    Attributes:
        token_embeddings (nn.Embedding): Token embedding layer.
        positional_embeddings (PositionalEmbeddings): Positional Embeddings layer.
        segment_embeddings (nn.Embedding): Segment embedding layer.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, config):
        """
        Initializes the Embeddings layer.

        Args:
            config (object): Configuration object containing parameters.
                - hidden_size (int): Size of the hidden embeddings.
                - vocab_size (int): Size of the vocabulary.
                - hidden_dropout_prob (float): Dropout probability for regularization.
        """
        super().__init__()

        self.hidden_size: int = config.hidden_size
        self.vocab_size: int = config.vocab_size
        self.hidden_dropout_prob: float = config.hidden_dropout_prob

        self.token_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_size
        )
        self.segment_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=3, embedding_dim=self.hidden_size
        )
        self.positional_embeddings: PositionalEmbeddings = PositionalEmbeddings(config)
        self.dropout: nn.Dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Forward pass of the Embeddings layer.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            segment_ids (torch.Tensor): Input tensor containing segment IDs.
            training (bool): Whether the model is in training mode.

        Returns:
            torch.Tensor: Final embeddings tensor.
        """
        pos_info: torch.Tensor = self.positional_embeddings(input_ids)
        seg_info: torch.Tensor = self.segment_embeddings(segment_ids)
        x: torch.Tensor = self.token_embeddings(input_ids)
        x: torch.Tensor = x + pos_info + seg_info
        if training:
            x: torch.Tensor = self.dropout(x)
        return x

    def forward_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the mask for the inputs.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.

        Returns:
            torch.Tensor: Computed mask tensor.
        """
        return input_ids != 0
    


class AttentionHead(nn.Module):
    """
    Attention head implementation.

    Args:
        hidden_size (int): Hidden size for the model (embedding dimension).
        head_dim (int): Dimensionality of the attention head.

    Attributes:
        query_weights (nn.Linear): Linear layer for query projection.
        key_weights (nn.Linear): Linear layer for key projection.
        value_weights (nn.Linear): Linear layer for value projection.
    """

    def __init__(self, hidden_size, head_dim):
        """
        Initializes the AttentionHead.

        Args:
            hidden_size (int): Hidden size for the model (embedding dimension).
            head_dim (int): Dimensionality of the attention head.
        """
        super().__init__()
        self.head_dim = head_dim
        self.query_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.key_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.value_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, training:bool = False) -> torch.Tensor:
        """
        Applies attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated value embeddings after applying attention mechanism.
        """
        query: torch.Tensor = self.query_weights(query)
        key: torch.Tensor = self.key_weights(key)
        value: torch.Tensor = self.value_weights(value)

        att_scores: torch.Tensor = torch.matmul(query, key.transpose(1, 2)) / self.head_dim ** 0.5

        if mask is not None:
            mask = mask.to(torch.int)
            att_scores: torch.Tensor = att_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        att_weights: torch.Tensor = F.softmax(att_scores, dim=-1)
        att_weights: torch.Tensor = self.dropout(att_weights) 
        n_value: torch.Tensor = torch.matmul(att_weights, value)

        return n_value


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer implementation.

    Args:
        config (object): Configuration object containing hyperparameters.
            - hidden_size (int): Hidden size for the model (embedding dimension).
            - num_heads (int): Number of attention heads.
            - head_dim (int): Dimensionality of each attention head.

    Attributes:
        hidden_size (int): Hidden size for the model (embedding dimension).
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        attention_heads (nn.ModuleList): List of AttentionHead layers.
        fc (nn.Linear): Fully connected layer for final projection.
    """

    def __init__(self, config):
        """
        Initializes the MultiHeadAttention layer.

        Args:
            config (object): Configuration object containing hyperparameters.
                - hidden_size (int): Hidden size for the model (embedding dimension).
                - num_heads (int): Number of attention heads.
                - head_dim (int): Dimensionality of each attention head.
        """
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_heads
        self.head_dim: int = config.hidden_size // config.num_heads
        self.attention_heads: nn.ModuleList = nn.ModuleList([AttentionHead(self.hidden_size, self.head_dim) for _ in range(self.num_heads)])
        self.fc: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, training:bool = False) -> torch.Tensor:
        """
        Applies multi-head attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated hidden state after applying multi-head attention mechanism.
        """
        attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask=mask, training = training) for attention_head in self.attention_heads]
        hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
        hidden_state: torch.Tensor = self.fc(hidden_state)
        return hidden_state
    

class FeedForward(nn.Module):
    """
    Feed-forward layer implementation.

    Args:
        config (object): Configuration object containing hyperparameters.
            - hidden_size (int): Hidden size for the model (embedding dimension).
            - hidden_dropout_prob (float): Dropout probability for regularization.

    Attributes:
        hidden_size (int): Hidden size for the model (embedding dimension).
        intermediate_fc_size (int): Intermediate size for the fully connected layers.
        hidden_dropout_prob (float): Dropout probability for regularization.
        fc1 (nn.Linear): First linear layer.
        fc2 (nn.Linear): Second linear layer.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, config):
        """
        Initializes the FeedForward layer.

        Args:
            config (object): Configuration object containing hyperparameters.
                - hidden_size (int): Hidden size for the model (embedding dimension).
                - hidden_dropout_prob (float): Dropout probability for regularization.
        """
        super().__init__()

        self.hidden_size: int = config.hidden_size
        self.intermediate_fc_size: int = self.hidden_size * 4
        self.hidden_dropout_prob: float = config.hidden_dropout_prob

        self.fc1: nn.Linear = nn.Linear(self.hidden_size, self.intermediate_fc_size)
        self.fc2: nn.Linear = nn.Linear(self.intermediate_fc_size, self.hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, hidden_state: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Applies feed-forward transformation to the input hidden state.

        Args:
            hidden_state (torch.Tensor): Hidden state tensor (batch_size, sequence_length, hidden_size).
            training (bool): Boolean indicating whether the model is in training mode or inference mode.

        Returns:
            torch.Tensor: Updated hidden state after applying feed-forward transformation.
        """
        hidden_state: torch.Tensor = self.fc1(hidden_state)
        hidden_state: torch.Tensor = F.gelu(hidden_state)
        hidden_state: torch.Tensor = self.dropout(hidden_state)
        hidden_state: torch.Tensor = self.fc2(hidden_state)
        
        return hidden_state
    

class Encoder(nn.Module):
    """
    Encoder layer implementation.

    Args:
        config (object): Configuration object containing hyperparameters.
            - hidden_size (int): Hidden size for the model (embedding dimension).
            - hidden_dropout_prob (float): Dropout probability for regularization.

    Attributes:
        hidden_size (int): Hidden size for the model (embedding dimension).
        hidden_dropout_prob (float): Dropout probability for regularization.
        multihead_attention (MultiHeadAttention): Multi-head attention layer.
        norm1 (nn.LayerNorm): Layer normalization layer.
        norm2 (nn.LayerNorm): Layer normalization layer.
        feed_forward (FeedForward): Feed-forward layer.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, config):
        """
        Initializes the Encoder layer.

        Args:
            config (object): Configuration object containing hyperparameters.
                - hidden_size (int): Hidden size for the model (embedding dimension).
                - hidden_dropout_prob (float): Dropout probability for regularization.
        """
        super().__init__()

        self.hidden_size: int = config.hidden_size
        self.hidden_dropout_prob: float = config.hidden_dropout_prob
        self.multihead_attention: MultiHeadAttention = MultiHeadAttention(config)
        self.norm1: nn.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.norm2: nn.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.feed_forward: FeedForward = FeedForward(config)
        self.dropout: nn.Dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, hidden_state: torch.Tensor, mask: torch.Tensor = None, training: bool = False) -> torch.Tensor:
        """
        Applies the encoder layer to the input hidden state.

        Args:
            hidden_state (torch.Tensor): Hidden state tensor (bs, len, dim).
            mask (torch.Tensor): Padding mask tensor (bs, len) or None.
            training (bool): Boolean flag indicating whether the layer is in training mode or not.

        Returns:
            torch.Tensor: Updated hidden state after applying the encoder layer.
        """
        x_norm1: torch.Tensor = self.norm1(hidden_state)
        attention_output: torch.Tensor = self.multihead_attention(x_norm1, x_norm1, x_norm1, mask, training)
        hidden_state: torch.Tensor = attention_output + hidden_state
        
        x_norm2: torch.Tensor = self.norm2(hidden_state)
        feed_forward_output: torch.Tensor = self.feed_forward(x_norm2)
        x_enc: torch.Tensor = feed_forward_output + hidden_state
        hidden_state: torch.Tensor = self.dropout(x_enc)
        
        return hidden_state
    
class BERT(nn.Module):
    """
    BERT model.

    Args:
        config (object): Configuration object containing hyperparameters.
            - num_blocks (int): Number of encoder blocks.
            - vocab_size (int): Size of the vocabulary.
            - d_model (int): Dimensionality of the model's hidden layers.
            - hidden_size (int): Size of the hidden embeddings.

    Attributes:
        num_blocks (int): Number of encoder blocks.
        vocab_size (int): Size of the vocabulary.
        final_dropout_prob (float): Dropout probability for the final layer.
        hidden_size (int): Size of the hidden embeddings.
        embed_layer (Embeddings): Embeddings layer.
        encoder (nn.ModuleList): List of encoder layers.
        mlm_prediction_layer (nn.Linear): Masked Language Model (MLM) prediction layer.
        nsp_classifier (nn.Linear): Next Sentence Prediction (NSP) classifier layer.
        softmax (nn.LogSoftmax): LogSoftmax layer for probability computation.
    """

    def __init__(self, config):
        """
        Initializes the BERT model.
        """
        super(BERT, self).__init__()

        self.num_blocks: int = config.num_blocks
        self.vocab_size: int = config.vocab_size
        self.hidden_size: int = config.hidden_size

        self.embed_layer: Embeddings = Embeddings(config)
        self.encoder: nn.ModuleList = nn.ModuleList([Encoder(config) for _ in range(self.num_blocks)])
        self.mlm_prediction_layer: nn.Linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.nsp_classifier: nn.Linear = nn.Linear(self.hidden_size, 2)
        self.softmax: nn.LogSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the BERT model.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            segment_ids (torch.Tensor): Input tensor containing segment IDs.
            training (bool): Whether the model is in training mode.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: MLM outputs and NSP outputs.
        """
        x_enc: torch.Tensor = self.embed_layer(input_ids, segment_ids, training)
        mask = self.embed_layer.forward_mask(input_ids)

        for encoder_layer in self.encoder:
            x_enc: torch.Tensor = encoder_layer(x_enc, mask, training=training)

        mlm_logits: torch.Tensor = self.mlm_prediction_layer(x_enc)
        nsp_logits: torch.Tensor = self.nsp_classifier(x_enc[:, 0, :])

        return self.softmax(mlm_logits), self.softmax(nsp_logits)