from torch.utils.data.dataloader import DataLoader
from aac_datasets import Clotho
from aac_datasets.utils.collate import BasicCollate
from aac_metrics import Evaluate
from msclap import CLAP
from tqdm import tqdm
import os
import pandas as pd

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
folder_append=os.path.join(dir_path, "..")
sys.path.append(folder_append)
from common import utils

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
output_root = "/".join(__file__.split("/")[:-3])
output_folder = os.path.join(output_root, "results", "clap", subset)
os.makedirs(output_folder, exist_ok=True)

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
audio_files_all = []
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
    audio_files_all.extend(audio_files)
    # print(f"i: {i+1}/{max_elements}: ")
    # print(batch['captions'])
    # print(captions)

# Saving pickle files with the data
output_file_candidate = os.path.join(output_folder, "candidates.pkl")
output_file_references = os.path.join(output_folder, "mult_references.pkl")
output_file_all_audios = os.path.join(output_folder, "audio_files_all.pkl")
utils.save_pickle(filename=output_file_candidate, data=candidates)
utils.save_pickle(filename=output_file_references, data=mult_references)
utils.save_pickle(filename=output_file_all_audios, data=audio_files_all)

# Evaluate captions
print("Evaluating")
evaluate = Evaluate(metrics=["spider", "fense", "vocab"])
corpus_scores, _ = evaluate(candidates, mult_references)
print(corpus_scores)

vocab_size = corpus_scores["vocab.cands"]
spider_score = corpus_scores["spider"]
fense_score = corpus_scores["fense"]

print(vocab_size)
print(spider_score)
print(fense_score)

print("Evaluating only FENSE for ranking")
evaluate = Evaluate(metrics=["fense"])

res = []
for f, c, mult_r in tqdm(zip(audio_files_all, candidates, mult_references)):
    corpus_scores, _ = evaluate([c], [mult_r])
    sbert_sim = float(corpus_scores["sbert_sim"])
    fer = float(corpus_scores["fer"])
    fense = float(corpus_scores["fense"])
    refs = {f"reference_{i}":r for i, r in enumerate(mult_r)}
    res.append({"file": f, "sbert_sim": sbert_sim, "fer": fer, "fense": fense, "candidate": c} | refs)

output_file_res = os.path.join(output_folder, "res.pkl")
utils.save_pickle(filename=output_file_res, data=res)

output_file_csv = os.path.join(output_folder, "res.csv")
df = pd.DataFrame(res)
df.sort_values(by=["fense"]).to_csv(output_file_csv, index=False)
print(output_file_csv)