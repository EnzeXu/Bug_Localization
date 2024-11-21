import torch
import torch.nn as nn
from ..utils.pretrained import T5CODE_MODEL, T5TEXT_MODEL

class BLNT5(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        """
        Initializes the BLNT5 model.
        Args:
            hidden_dim (int): The number of hidden units in the MLP.
        """
        super(BLNT5, self).__init__()

        # Load T5 and CodeT5 models
        self.t5 = T5TEXT_MODEL
        self.code_t5 = T5CODE_MODEL

        # Freeze T5 and CodeT5 to avoid fine-tuning if not needed
        for param in self.t5.parameters():
            param.requires_grad = False
        for param in self.code_t5.parameters():
            param.requires_grad = False

        # Define the MLP layers
        t5_output_dim = self.t5.config.d_model  # Output dimension of T5
        code_t5_output_dim = self.code_t5.config.hidden_size  # Output dimension of CodeT5
        combined_dim = t5_output_dim + code_t5_output_dim  # Combined dimension of v1 and v2

        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, br_input_ids, br_attention_mask, method_input_ids, method_attention_mask):
        """
        Forward pass for the BLNT5 model.
        Args:
            br_input_ids (torch.Tensor): Tokenized input IDs for bug report.
            br_attention_mask (torch.Tensor): Attention mask for bug report.
            method_input_ids (torch.Tensor): Tokenized input IDs for method.
            method_attention_mask (torch.Tensor): Attention mask for method.
        Returns:
            torch.Tensor: The predicted relativity score (sigmoid output), between [0,1].
        """
        # Process bug report (br) through T5
        br_outputs = self.t5.encoder(input_ids=br_input_ids, attention_mask=br_attention_mask)
        br_hidden_states = br_outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_dim)  (2, 512, 512)
        # print("br_layer_outputs:", br_hidden_states , "shape:", br_hidden_states.shape )
        v1 = br_hidden_states.mean(dim=1)  # Mean pooling across the sequence (Shape: (batch_size, hidden_dim))    (2, 512)
        # print("v1_outputs:", v1 , "shape:", v1.shape )


        # Process method (m) through CodeT5
        method_outputs = self.code_t5.encoder(input_ids=method_input_ids, attention_mask=method_attention_mask)
        method_hidden_states = method_outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_dim)
        # print("method_layer_outputs:", method_hidden_states , "shape:", method_hidden_states .shape )   # (2, 512, 768)
        v2 = method_hidden_states.mean(dim=1)  # Mean pooling across the sequence (Shape: (batch_size, hidden_dim))
        # print("v2_outputs:", v2 , "shape:", v2.shape )  #  (2, 768)


        # Concatenate v1 and v2
        v = torch.cat((v1, v2), dim=1)  # Shape: (batch_size, combined_dim)   (2, 512+768=1280)
        # print("v_outputs:", v , "shape:", v.shape )

        # Pass through MLP
        logits = self.mlp(v)  # predictions: tensor( [ [0.5041], [0.5032] ], grad_fn=<SigmoidBackward0>) shape: torch.Size([2, 1] )

        # print("logits:", logits , "shape:", logits.shape )

        return logits
