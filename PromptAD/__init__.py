from .model import *

# def get_model_from_args(**kwargs)->WinClipAD:
#     model = WinClipAD(**kwargs)
#     return model

class TripletLoss(nn.Module):
    """
    Hard Negative Margin Loss for multi-abnormal prototypes.
    
    Original: loss = relu(d(a,p) - d(a,n) + margin)
    New (Hard Negative): 
        i* = argmax_i cos(anchor, negative_i)  # hardest negative
        loss = relu(cos(anchor, negative_i*) - cos(anchor, positive) + margin)
    
    This ensures we push away from the HARDEST abnormal prototype, not just the mean.
    """
    def __init__(self, margin=0.03):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor: [B, D] - image features
            positive: [1, D] or [B, D] - normal prototype
            negatives: [K, D] - multiple abnormal prototypes (K learned prototypes)
        """
        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        
        # Compute similarities
        sim_positive = torch.sum(anchor * positive, dim=-1)  # [B] or scalar if positive is [1,D]
        sim_negatives = torch.matmul(anchor, negatives.T)  # [B, K]
        
        # Find hardest negative for each sample
        sim_hardest_negative, _ = sim_negatives.max(dim=-1)  # [B]
        
        # Hard negative margin loss: push hardest negative away from positive
        loss = torch.relu(sim_hardest_negative - sim_positive + self.margin)
        
        return torch.mean(loss)