import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=20, input_shape=(1, 80, 172)):
        super(Encoder, self).__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        h1 = (h + 2*1 - 4) // 2 + 1
        h2 = (h1 + 2*1 - 4) // 2 + 1
        w1 = (w + 2*1 - 4) // 2 + 1
        w2 = (w1 + 2*1 - 4) // 2 + 1
        self._flattened_dim = 64 * h2 * w2
        self.fc_mu     = nn.Linear(self._flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flattened_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, input_shape=(1, 80, 172)):
        super(Decoder, self).__init__()
        c, h, w = input_shape
        h1 = (h + 2*1 - 4) // 2 + 1
        h2 = (h1 + 2*1 - 4) // 2 + 1
        w1 = (w + 2*1 - 4) // 2 + 1
        w2 = (w1 + 2*1 - 4) // 2 + 1
        self._flattened_dim = 64 * h2 * w2
        self.fc     = nn.Linear(latent_dim, self._flattened_dim)
        self.convT1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.convT2 = nn.ConvTranspose2d(32, c,  kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        batch_size = z.size(0)
        x = F.relu(self.fc(z))
        x = x.view(batch_size, 64, (self._flattened_dim // 64) // ((self._flattened_dim // 64) // (self._flattened_dim // (64 * ((self._flattened_dim // 64) // 64)))) , -1)  # 此处根据实际计算替换为 (h2, w2)
        x = F.relu(self.convT1(x))
        x = torch.sigmoid(self.convT2(x))
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=20, input_shape=(1, 80, 172)):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, input_shape)
        self.decoder = Decoder(latent_dim, input_shape)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

if __name__ == "__main__":
    from data import mel_dataset
    from torch.utils.data import DataLoader

    input_shape = (1, 80, 172)
    model = VAE(latent_dim=20, input_shape=input_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = mel_dataset('wav.scp', sample_rate=22050, duration=2.0, n_mels=80, n_fft=1024, hop_length=256)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(1, 3):
        model.train()
        total_loss = 0
        for utt_ids, mel in loader:
            mel = mel.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(mel)
            loss = vae_loss(recon, mel, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}\tLoss per sample: {total_loss/len(dataset):.4f}")
    torch.save(model.state_dict(), 'vae_mel.pth')
