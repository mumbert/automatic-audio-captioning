"""
This is an example using CLAPCAP for audio captioning.
Code adapted from: https://github.com/microsoft/CLAP/blob/main/examples/audio_captioning.py
"""
from msclap import CLAP
import sys

# Load and initialize CLAP
clap_model = CLAP(version = 'clapcap', use_cuda=False)

def clap_inference(audio_files = []):

    # Generate captions for the recording
    captions = clap_model.generate_caption(audio_files, 
                                           resample=True, 
                                           beam_size=5, 
                                           entry_length=67, 
                                           temperature=0.01)

    # Print the result
    for i in range(len(audio_files)):
        print(f"Audio file: {audio_files[i]} \n")
        print(f"Generated caption: {captions[i]} \n")

if __name__ == "__main__":

    audio_files = [sys.argv[1]]
    clap_inference(audio_files)