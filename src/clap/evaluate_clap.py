from torch.utils.data.dataloader import DataLoader
from aac_datasets import Clotho
from aac_datasets.utils.collate import BasicCollate
from aac_metrics import Evaluate
from msclap import CLAP
from tqdm import tqdm
import os

# params
subset="val"
batch_size=2

# dataset and subset folders
datafolder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
subset2folder={"val": "validation", "dcase_aac_test": "test"}
subsetfolder = os.path.join(datafolder, "CLOTHO_v2.1/clotho_audio_files/", subset2folder[subset])

print(f"Loading Clotho subset: {subset}")
dataset = Clotho(root=datafolder, subset=subset, download=True) # dcase_aac_test
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=BasicCollate())

# Load and initialize CLAP
print("Loading CLAP model")
clap_model = CLAP(version = 'clapcap', use_cuda=False)

# Get captions
candidates = []
mult_references = []
max_elements = 2
for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    if i+1 > max_elements:
        break

    # Generate captions for the recording
    print(f"{batch['fname']}")
    audio_files = [os.path.join(subsetfolder, fname) for fname in batch['fname']]
    captions = clap_model.generate_caption(audio_files, 
                                           resample=True, 
                                           beam_size=5, 
                                           entry_length=67, 
                                           temperature=0.01)
    candidates.extend(captions)
    mult_references.extend(batch['captions'])
    print(f"i: {i+1}/{max_elements}: ")
    print(batch['captions'])
    print(captions)

# Evaluate captions
evaluate = Evaluate(metrics=["spider", "fense", "vocab"])
corpus_scores, _ = evaluate(candidates, mult_references)
print(corpus_scores)

vocab_size = corpus_scores["vocab.cands"]
spider_score = corpus_scores["spider"]
fense_score = corpus_scores["fense"]
