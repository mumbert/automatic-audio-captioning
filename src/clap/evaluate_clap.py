from torch.utils.data.dataloader import DataLoader
from aac_datasets import Clotho
from aac_datasets.utils.collate import BasicCollate
from aac_metrics import Evaluate
from msclap import CLAP
from tqdm import tqdm
import os

##### Proposal to generalize baseline evaluation that could also be useful for trained models

# Read configuration YAML file
# config = utils.read_config()

# Get Input Data
# dataloader = utils.prepare_dataloader(config["data"])

# Prepare Model to Evaluate
# model = utils.prepare_model(config["model"])

# Get Output Candidate Captions and Groundtruth
# candidates, mult_references = model(config["predict"])

# Results
# corpus_scores = utils.evaluate(config["evaluate"])

#####

# params
subset="eval" # 'dev', 'val', 'eval', 'dcase_aac_test', 'dcase_aac_analysis', 'dcase_t2a_audio', 'dcase_t2a_captions'
batch_size=4

# dataset and subset folders
datafolder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# datafolder = "/".join(__file__.split("/")[:-4])
subset2folder={"val": "validation", "dcase_aac_test": "test", "eval": "evaluation"}
subsetfolder = os.path.join(datafolder, "CLOTHO_v2.1/clotho_audio_files/", subset2folder[subset])

print(f"Downloading Clotho subset: {subset} in folder {datafolder}")
dataset = Clotho(root=datafolder, subset=subset, download=True) # dcase_aac_test

print("Preparing dataloader")
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=BasicCollate())

# Load and initialize CLAP
print("Loading CLAP model")
clap_model = CLAP(version = 'clapcap', use_cuda=False)

# Get captions
candidates = []
mult_references = []
max_elements = float('Inf')
for i, batch in tqdm(enumerate(dataloader), total=min(max_elements, len(dataloader))):
    if i+1 > max_elements:
        break

    # Generate captions for the recording
    # print(f"{batch['fname']}")
    audio_files = [os.path.join(subsetfolder, fname) for fname in batch['fname']]
    captions = clap_model.generate_caption(audio_files, 
                                           resample=True, 
                                           beam_size=5, 
                                           entry_length=67, 
                                           temperature=0.01)
    candidates.extend(captions)
    mult_references.extend(batch['captions'])
    # print(f"i: {i+1}/{max_elements}: ")
    # print(batch['captions'])
    # print(captions)

# Evaluate captions
evaluate = Evaluate(metrics=["spider", "fense", "vocab"])
corpus_scores, _ = evaluate(candidates, mult_references)
print(corpus_scores)

vocab_size = corpus_scores["vocab.cands"]
spider_score = corpus_scores["spider"]
fense_score = corpus_scores["fense"]

print(vocab_size)
print(spider_score)
print(fense_score)