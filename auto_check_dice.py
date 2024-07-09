import torch
import torch.nn as nn
import torch.optim as optim
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

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)  # Simple convolution layer

    def forward(self, x):
        x = self.conv1(x)
        return x

# Dice loss function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to get probabilities
        print('Inputs: ', check_tensor_grad(inputs))
        print('Shape: ', inputs.shape)
        print()

        # Compute the intersection and sums over the entire 4D tensors
        intersection = (inputs * targets).sum(dim=(2, 3))  # Sum over height and width
        print('Intersection: ', check_tensor_grad(intersection))
        print('Shape: ', intersection.shape)
        print()

        inputs_sum = inputs.sum(dim=(2, 3))
        print('inputs sum: ',check_tensor_grad(inputs_sum))
        print('Shape: ', inputs_sum.shape)
        print()

        targets_sum = targets.sum(dim=(2, 3)) #! Only this does not have gradients --> Because ground truth.
        print('targets sum: ',check_tensor_grad(targets_sum))
        print('Shape: ', targets_sum.shape)
        print()
        
        # Compute Dice coefficient for each sample in the batch
        dice_coefficient = (2. * intersection + self.smooth) / (inputs_sum + targets_sum + self.smooth)
        print('Coeff: ', check_tensor_grad(dice_coefficient))
        print('Shape: ', dice_coefficient.shape)
        print()
        
        # Average the Dice coefficient over the batch
        dice_loss = 1 - dice_coefficient.mean()
        print('Final: ', check_tensor_grad(dice_loss))
        print()
        print('%'*50)
        
        return dice_loss

# Instantiate model, loss, and optimizer
model = SimpleCNN()
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example input and target
inputs = torch.randn(1, 1, 256, 256)  # Example input (logits)
targets = torch.randint(0, 2, (1, 1, 256, 256)).float()  # Example target

# Forward pass
outputs = model(inputs)
loss = criterion(outputs, targets)
print(f'Loss: {loss.item()}')

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print('Backward pass and optimization completed.')

# Function for gradcheck
def gradcheck_dice_loss():
    model = SimpleCNN().double()
    criterion = DiceLoss()

    inputs = torch.randn(1, 1, 2, 2, dtype=torch.double, requires_grad=True)
    targets = torch.randint(0, 2, (1, 1, 2, 2), dtype=torch.double)

    outputs = model(inputs)

    # gradcheck requires a function that outputs the loss and inputs as a tuple
    def func(inputs):
        loss = criterion(inputs, targets)
        return loss, inputs

    # Check gradients
    return gradcheck(func, (outputs,), eps=1e-6, atol=1e-4)

# Perform gradcheck
test = gradcheck_dice_loss()
print("Gradcheck passed:", test)
