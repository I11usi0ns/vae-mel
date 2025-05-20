import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import your VAE and dataset
from model import VAE
from data import mel_dataset


def compare_reconstruction(
    model_path='vae_mel.pth',
    scp_path='wav.scp',
    sr=22050,
    duration=2.0,
    n_mels=80,
    n_fft=1024,
    hop_length=256,
    latent_dim=20
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = mel_dataset(
        scp_path,
        sr=sr,
        duration=duration,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    utt_id, mel = next(iter(loader)) 
    mel = mel.to(device)

    with torch.no_grad():
        recon, mu, logvar = model(mel.squeeze(1))  
        recon = recon.unsqueeze(1)

    orig = mel.squeeze().cpu().numpy()
    recn = recon.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Mel-spectrogram')
    plt.imshow(orig, aspect='auto', origin='lower')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Mel-spectrogram')
    plt.imshow(recn, aspect='auto', origin='lower')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    compare_reconstruction()
