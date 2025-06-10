import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Simulated fraud dataset
n_features = 10
n_samples = 10000
X = np.random.rand(n_samples, n_features).astype(np.float32)
y = np.random.randint(0, 2, size=(n_samples, 1)).astype(np.float32)

dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, z, labels):
        x = torch.cat((z, labels), dim=1)
        return self.model(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + label_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        x = torch.cat((x, labels), dim=1)
        return self.model(x)


noise_dim = 20
G = Generator(noise_dim, 1, n_features)
D = Discriminator(n_features, 1)

loss_fn = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002)
opt_D = optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(100):
    for real_x, real_y in dataloader:
        bs = real_x.size(0)

        # Labels
        real_label = torch.ones(bs, 1)
        fake_label = torch.zeros(bs, 1)

        # === Train Discriminator ===
        z = torch.randn(bs, noise_dim)
        fake_x = G(z, real_y)
        d_real = D(real_x, real_y)
        d_fake = D(fake_x.detach(), real_y)

        loss_d = loss_fn(d_real, real_label) + loss_fn(d_fake, fake_label)
        opt_D.zero_grad()
        loss_d.backward()
        opt_D.step()

        # === Train Generator ===
        z = torch.randn(bs, noise_dim)
        gen_x = G(z, real_y)
        d_gen = D(gen_x, real_y)
        loss_g = loss_fn(d_gen, real_label)
        opt_G.zero_grad()
        loss_g.backward()
        opt_G.step()

    print(
        f"Epoch {epoch:02d} | D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f}"
    )


# Generate synthetic samples conditioned on fraud label
z = torch.randn(1000, noise_dim)
fraud_labels = torch.ones(1000, 1)
synthetic_fraud = G(z, fraud_labels).detach().numpy()

print("Synthetic Fraud Samples Generated")
real_fraud = X[np.where(y == 1)[0]]  # Original fraud samples
print("Real Fraud Samples")
print(real_fraud[:5])  # Print original fraud samples
print("Synthetic Fraud Samples")
print(synthetic_fraud[:5])  # Print first 5 samples
