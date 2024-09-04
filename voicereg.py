import torch
import torchaudio
from speechbrain import EncoderClassifier

# Load the pre-trained model
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir")

# Function to extract embeddings
def get_embedding(audio_file):
    # Load the audio file
    signal, fs = torchaudio.load(audio_file)
    # Ensure the audio has the correct shape
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    # Get the embedding
    embedding = classifier.encode_batch(signal)
    return embedding.squeeze().detach().cpu().numpy()

# Load known and new audio samples
known_audio = "filename.wav"
new_audio = "output.wav"

# Get embeddings for both audio samples
known_embedding = get_embedding(known_audio)
new_embedding = get_embedding(new_audio)

# Calculate similarity (cosine similarity)
cosine_similarity = torch.nn.functional.cosine_similarity(
    torch.tensor(known_embedding).unsqueeze(0),
    torch.tensor(new_embedding).unsqueeze(0)
)

# Define a threshold for determining if the same person is speaking
threshold = 0.8  # This value might need tuning based on your use case

if cosine_similarity.item() > threshold:
    print("Same person")
else:
    print("Different person")
