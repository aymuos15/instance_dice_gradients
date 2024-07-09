import torch
import torch.nn as nn
import torch.optim as optim

import cupy as cp
from cucim.skimage import measure as cucim_measure

from torch.autograd import gradcheck

# Function to check if tensor has grad
def check_tensor_grad(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.requires_grad:
            print("Tensor has gradient enabled")
        else:
            print("Tensor does not have gradient enabled")
    else:
        print("Not a PyTorch tensor")
    return tensor

# Define a simple model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)  # Simple convolution layer

    def forward(self, x):
        x = self.conv1(x)
        return x

class Instance_DiceLoss(nn.Module):
    def __init__(self, num_gt_lesions, smooth=1):
        super(Instance_DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_gt_lesions = num_gt_lesions
    
    def dice(self, pred, gt):
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        return 2 * intersection / (union + self.smooth)

    def forward(self, pred, gt):

        pred_label_cc = pred
        pred_label_cc = torch.sigmoid(pred_label_cc)
        print('Pred: ', check_tensor_grad(pred_label_cc))
        print('Shape: ', pred_label_cc.shape)
        print()

        gt_label_cc = gt

        num_gt_lesions = torch.unique(gt_label_cc[gt_label_cc != 0]).size(0)

        lesion_dice_scores = 0
        tp = torch.tensor([])
        fn = torch.tensor([])

        for gtcomp in range(1, num_gt_lesions + 1):
            gt_tmp = (gt_label_cc == gtcomp)
            intersecting_cc = torch.unique(pred_label_cc[gt_tmp])
            print('Intersecting_cc: ', check_tensor_grad(intersecting_cc))
            print('Shape: ', intersecting_cc.shape)
            print()

            intersecting_cc = intersecting_cc[intersecting_cc != 0]
            print('Intersecting_cc 2: ', check_tensor_grad(intersecting_cc))
            print('Shape: ', intersecting_cc.shape)
            print()

            if len(intersecting_cc) > 0:
                # pred_tmp = torch.zeros_like(pred_label_cc, dtype=torch.bool)
                pred_tmp = torch.zeros_like(pred_label_cc, dtype=torch.float32, requires_grad=True)
                print('Pred tmp: ', check_tensor_grad(pred_tmp))
                print('Shape: ', pred_tmp.shape)
                print()

                # pred_tmp[torch.isin(pred_label_cc, intersecting_cc)] = True
                pred_tmp = torch.where(torch.isin(pred_label_cc, intersecting_cc), torch.tensor(1., device=pred.device), pred_tmp)
                print('Pred tmp 2: ', check_tensor_grad(pred_tmp))
                print('Shape: ', pred_tmp.shape)
                print()

                dice_score = self.dice(pred_tmp, gt_tmp)
                lesion_dice_scores += dice_score
                tp = torch.cat([tp, intersecting_cc])
            else:
                fn = torch.cat([fn, torch.tensor([gtcomp])])
        
        mask = (pred_label_cc != 0) & (~torch.isin(pred_label_cc, tp))
        fp = torch.unique(pred_label_cc[mask], sorted=True)
        fp = fp[fp != 0]

        print('%'*50)

        return lesion_dice_scores / (num_gt_lesions + len(fp))

def get_connected_components(img, connectivity=None):
    img_cupy = cp.asarray(img.cpu().numpy())
    labeled_img, num_features = cucim_measure.label(img_cupy, connectivity=connectivity, return_num=True)
    labeled_img_torch = torch.tensor(labeled_img, device=img.device, dtype=torch.float32)
    return labeled_img_torch, num_features

# Example input and target
targets = torch.randint(0, 2, (1, 1, 2, 2)).float()  # Example target
targets_cc_label, num_gt_lesions = get_connected_components(targets)

inputs = torch.randn(1, 1, 2, 2, requires_grad=False)  # Example input (logits)
inputs_cc_label, num_pred_lesions = get_connected_components(inputs)
inputs_cc_label.requires_grad = True

# Instantiate model, loss, and optimizer
model = SimpleCNN()
criterion = Instance_DiceLoss(num_gt_lesions=num_gt_lesions)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Forward pass
outputs = model(inputs)
loss = criterion(outputs, targets_cc_label)
print(f'Loss: {loss.item()}')

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print('Backward pass and optimization completed.')

# Function for gradcheck
def gradcheck_dice_loss():
    model = SimpleCNN().double()
    criterion = Instance_DiceLoss(num_gt_lesions=1)

    inputs = torch.randn(1, 1, 2, 2, dtype=torch.double, requires_grad=True)
    targets = torch.randint(0, 2, (1, 1, 2, 2), dtype=torch.double)
    targets_cc_label, num_gt_lesions = get_connected_components(targets)

    outputs = model(inputs)

    # gradcheck requires a function that outputs the loss and inputs as a tuple
    return gradcheck(criterion, (outputs, targets_cc_label), eps=1e-6, atol=1e-4)

# Perform gradcheck
test = gradcheck_dice_loss()
print("Gradcheck passed:", test)