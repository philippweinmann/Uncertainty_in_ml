# %%
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import sys
import optuna
from PIL import Image
from IPython.display import display
import numpy as np

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("selected device: ", device)
# %%
transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])


# Download and load the training data
train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load the test data
test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# %%
class NonBayesianReLUNet(nn.Module):
    def __init__(self):
        super(NonBayesianReLUNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
# %%
def train_loop(dataloader, model, loss_fn = nn.CrossEntropyLoss(), learning_rate = 0.001):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and  dropout layers
    # Unnecessary in this situation but added for best practices
    model.to(device)
    model.train()

    # reinitialized every 100 losses
    average_loss = 0
    for batch_number, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        average_loss += loss
        if (batch_number % 100 == 0):
            # Update progress
            progress = f"{batch_number}/{size}, {100 * batch_number // size}% done"

            average_loss, current = average_loss / 100, (batch_number + 1) * len(X)
            progress_loss = f"loss: {average_loss:>7f}  [{current:>5d}/{size:>5d}]"
            sys.stdout.write('\r' + progress + " " + progress_loss)
            sys.stdout.flush()
    print(f"100% done")
    return loss.item()

# %%
def test_loop(dataloader, model, loss_fn=nn.CrossEntropyLoss()):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss
# %%

def objective(trial):
    learning_rate_sug = 10 ** -trial.suggest_float('learning_rate_sug', 1, 5)
    model = NonBayesianReLUNet()
    model.to(device)
    train_loop(dataloader=train_dataloader, model=model, learning_rate=learning_rate_sug)
    error = test_loop(dataloader=test_dataloader, model=model)

    return error

def find_best_parameters():
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    # jobs = -1 to use all cores
    study.optimize(objective, n_trials=10, n_jobs=-1, show_progress_bar=True)

    print("best abs_mean_error: ", study.best_value)
    print("best parameters: ", study.best_params)

    optuna.visualization.plot_slice(study)
    optuna.visualization.plot_optimization_history(study)

# %%
# from best the optuna hyperparameter search:
best_learning_rate = 10 ** -(1.911)

def train_model(model, learning_rate):
    model.to(device)

    train_loop(dataloader=train_dataloader, model=model, learning_rate=learning_rate)
    error = test_loop(dataloader=test_dataloader, model=model)
    return model, error
# %%
def forward_np_img(img : np.array, model):
    torch_img = torch.tensor(img)
    batch_img = torch_img.float().unsqueeze(0).unsqueeze(0).to(device)
    output = model(batch_img)
    return output



def display_image_or_direction(img):
    g_img = Image.fromarray(img, 'L')
    g_img = g_img.resize((400,400))
    display(g_img)

# %%
non_bayesian_reLU_model, bnr_error = train_model(model=NonBayesianReLUNet(), learning_rate=best_learning_rate)

# %%
def evaluate_overconfidence(random_grayscale_image, random_direction, model):
    
    display_image_or_direction(random_grayscale_image)
    output = forward_np_img(random_grayscale_image, model)

    print("raw output: ", output)
    print("interpretable output: ", torch.exp(output) * 100)
    scaling_factor = 10000
    display_image_or_direction(random_direction)
    random_scaled_direction = scaling_factor * random_direction
    modified_image = random_scaled_direction * random_grayscale_image
    scaled_output = forward_np_img(modified_image, model)

    print("raw output: ", scaled_output)
    print("interpretable output: ", torch.exp(scaled_output) * 100)
    # with this input, the neural network has a nearly 100% confidence in its output.
    # Notice how the predicted number (where it predicts it at 100%) changed each time the code is rerun due to the random direction
# %%
random_grayscale_image = np.random.rand(28,28)
random_direction = np.random.rand(28,28)
evaluate_overconfidence(random_grayscale_image,random_direction, non_bayesian_reLU_model)
