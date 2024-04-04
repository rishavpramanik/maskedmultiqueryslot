from utils import assert_shape
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import Tensor
import random
import os
from scipy.optimize import linear_sum_assignment
class SlotAttention(nn.Module):
    def __init__(self, in_features, num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon=1e-8):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.slot_size, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros(
                (1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros(
                (1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )

    def forward(self, inputs: Tensor):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [batch_size, num_inputs, slot_size].
        k = self.project_k(inputs)
        assert_shape(k.size(), (batch_size, num_inputs, self.slot_size))
        # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)
        assert_shape(v.size(), (batch_size, num_inputs, self.slot_size))

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots_init = torch.randn((batch_size, self.num_slots, self.slot_size))
        slots_init = slots_init.type_as(inputs)
        slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            # Shape: [batch_size, num_slots, slot_size].
            q = self.project_q(slots)
            assert_shape(
                q.size(), (batch_size, self.num_slots, self.slot_size))

            attn_norm_factor = self.slot_size ** -0.5
            attn_logits = attn_norm_factor * torch.matmul(k, q.transpose(2, 1))
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn.size(), (batch_size, num_inputs, self.num_slots))

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates.size(), (batch_size,
                                          self.num_slots, self.slot_size))

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            slots = self.gru(
                updates.view(batch_size * self.num_slots, self.slot_size),
                slots_prev.view(batch_size * self.num_slots, self.slot_size),
            )
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            assert_shape(slots.size(), (batch_size,
                                        self.num_slots, self.slot_size))
            slots = slots + self.mlp(self.norm_mlp(slots))
            assert_shape(slots.size(), (batch_size,
                                        self.num_slots, self.slot_size))

        return slots

class MultiQuerySlot(nn.Module):
    def __init__(self, in_features, num_iterations, num_slots, slot_size, mlp_hidden_size, num_heads=8, epsilon=1e-8):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.heads = num_heads

        self.mean = nn.Parameter(torch.zeros(self.slot_size))
        self.log_variance = nn.Parameter(torch.zeros(self.slot_size))

        self.norm_inputs = nn.LayerNorm(self.in_features)
        self.norm_slots = nn.ModuleList([nn.LayerNorm(self.slot_size)
                           for _ in range(self.heads)])
        self.norm_mlp = nn.ModuleList([nn.LayerNorm(self.slot_size)
                         for _ in range(self.heads)])
        self.project_q = nn.ModuleList([nn.Linear(
            self.slot_size, self.slot_size, bias=False) for _ in range(self.heads)])
        self.gru = nn.ModuleList([nn.GRUCell(self.slot_size, self.slot_size)
                    for _ in range(self.heads)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(self.slot_size, self.mlp_hidden_size), nn.ReLU(
        ), nn.Linear(self.mlp_hidden_size, self.slot_size)) for _ in range(self.heads)])
        self.project_k = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.slot_size, self.slot_size, bias=False)

        # Slot update functions.
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros(
                (self.heads,1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros(
                (self.heads,1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )

    def match_and_sum_tensors(self,tensor1, tensor2):
        batch, channel1, feat1 = tensor1.size()
        _, channel2, feat2 = tensor2.size()

        tens=torch.clone(tensor1)

        #cosine_sim
        tensor1_normalized = F.normalize(tensor1.view(batch, channel1, -1), dim=2, p=2)
        tensor2_normalized = F.normalize(tensor2.view(batch, channel2, -1), dim=2, p=2)
        similarity_matrix = torch.bmm(tensor1_normalized, tensor2_normalized.transpose(1, 2))

        for b in range(batch):
            row,col=self.hungarian_algorithm(similarity_matrix[b])
            # if k>7:
            #     print(f"Matching:{row,col}")
            for r,c in zip(row,col):
                tens[b,r]+=tensor2[b,c]
        return tens
    def hungarian_algorithm(self,cost_matrix):
        """
        Solve the assignment problem using the Hungarian algorithm.

        Parameters:
        cost_matrix (tensor): A square cost matrix where each element represents the cost of assigning a worker to a task.

        Returns:
        row_indices (tensor): A tensor of row indices corresponding to the optimal assignments.
        col_indices (tensor): A tensor of column indices corresponding to the optimal assignments.
        total_cost (float): The total cost of the optimal assignment.
        """
        # Convert the cost matrix to a NumPy array
        cost_matrix_np = cost_matrix.cpu().detach().numpy()

        # Use scipy's linear_sum_assignment to solve the problem
        row_indices_np, col_indices_np = linear_sum_assignment(cost_matrix_np,maximize=True)

        # Convert the results back to PyTorch tensors
        row_indices = torch.tensor(row_indices_np,device=cost_matrix.device)
        col_indices = torch.tensor(col_indices_np,device=cost_matrix.device)

        return row_indices, col_indices

    def forward(self, inputs: Tensor):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [batch_size, num_inputs, slot_size].
        k = self.project_k(inputs)
        assert_shape(k.size(), (batch_size, num_inputs, self.slot_size))
        # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)
        assert_shape(v.size(), (batch_size, num_inputs, self.slot_size))
        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots_init = [torch.randn(
            (batch_size, self.num_slots, self.slot_size)) for _ in range(self.heads)]
        slots_init = [i.type_as(inputs) for i in slots_init]
        slots = [self.slots_mu[i] + self.slots_log_sigma[i].exp() * slots_init[i]
                 for i in range(self.heads)]
        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            for j in range(self.heads):
                slots[j] = self.norm_slots[j](slots[j])

            # Attention.
            # Shape: [batch_size, num_slots, slot_size].
                q = (self.project_q[j])(slots[j])
            # assert_shape(
            #     q.size(), (batch_size, self.num_slots, self.slot_size))

                attn_norm_factor = self.slot_size ** -0.5
                attn_logits = attn_norm_factor * \
                    torch.matmul(k, q.transpose(2, 1))
                attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
                assert_shape(attn.size(), (batch_size,
                                           num_inputs, self.num_slots))

            # Weighted mean.
                attn = attn + self.epsilon
                attn = attn / torch.sum(attn, dim=1, keepdim=True)
                updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].
                assert_shape(updates.size(), (batch_size,
                                              self.num_slots, self.slot_size))

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
                slots[j] = (self.gru[j])(
                    updates.view(batch_size * self.num_slots, self.slot_size),
                    slots_prev[j].view(batch_size * self.num_slots, self.slot_size))

                slots[j] = slots[j].view(
                    batch_size, self.num_slots, self.slot_size)

                slots[j] = slots[j] + self.mlp[j]((self.norm_mlp[j])(slots[j]))
        
        #Do the fusion
        sumed=torch.zeros_like(slots[0])
        if not self.training:
            k=random.choice(range(self.heads)) #type: ignore
            for i in range(self.heads):
                #tensor1 should be consistent
                if i==k:
                    continue

                sumed+=self.match_and_sum_tensors(slots[k],slots[i])

            sumed-=((self.heads-2)*slots[k])
            return sumed/self.heads
        return random.choice(slots)