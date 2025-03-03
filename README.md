<!-- Using the following project as a reference: https://github.com/othneildrew/Best-README-Template/tree/main -->

<a id="readme-top"></a>
<br />
<div align="center">
  <h1 align="center">Automatic Audio Captioning</h1>

  <p align="center">
    Automatic audio captioning project for the AI with DL postgraduate course at the UPC.
    <br />
    <br />
    <a>
        <img src="doc/images/task6_aac_system_example.png" alt="Logo" width="200" height="200">
  </a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Introduction and state-of-the-art">About The Project</a>
         <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
       <li><a href="#DCASE 2024 and CLAP demo deployment">Usage</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#demo">Demo</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
        <li><a href="#References">Contributing</a></li>
  </ol>
</details>

## Introduction and state-of-the-art

### Introduction

Automatic Audio Captioning (AAC) is the task of generating natural language descriptions for audio recordings. It involves analyzing complex acoustic signals and translating them into human-readable text, often describing specific sound events, their interactions, or environmental context. This process combines signal processing, deep learning, and natural language processing techniques.

Recent advancements in deep learning have significantly improved AAC performance. Traditional approaches relied on handcrafted audio features such as Mel-Frequency Cepstral Coefficients (MFCCs) and statistical models, but modern systems predominantly use neural networks. Convolutional Neural Networks (CNNs) are commonly employed for feature extraction, transforming raw waveforms or spectrogram representations into high-level embeddings (see [Figure 1](#fig-aac-pipeline)). These embeddings are then processed by sequence models such as Recurrent Neural Networks (RNNs), Transformer architectures, or attention mechanisms to generate coherent textual descriptions.

<p align="center">
  <img src="doc/images/1_aac_pipeline.png" alt="AAC Pipeline" width="600" style="max-width: 100%; height: auto;">
</p>
<p align="center"><a id="fig-aac-pipeline"></a><em>Figure 1: Overview of a typical AAC pipeline</em></p>

### State-of-the-art

Recent advances in AAC have progressively enhanced the ability to generate natural language descriptions for audio content. Early work in the field established the feasibility of the task by employing encoder–decoder frameworks based on RNNs with attention mechanisms. For example, [Drossos et al. (2017)](#drossos2017) introduced one of the first AAC systems using bi-directional gated recurrent units (GRUs) to capture temporal dynamics and an alignment model to focus the decoder on relevant audio features. While this approach demonstrated promising results, it also highlighted challenges in modeling long-range dependencies and the fine-grained acoustic details required for accurate captioning.

Subsequent research shifted towards leveraging CNNs and their hybrid forms (e.g., CRNNs) to improve feature extraction. CNN-based encoders proved effective at capturing local patterns and robust audio representations from spectrogram inputs. However, due to the inherent limitations of fixed receptive fields in CNNs, these methods often struggled to capture global contextual information across longer audio sequences.

A major turning point in AAC research has been the introduction of Transformer architectures. Transformer-based models, such as the Audio Captioning Transformer (ACT) presented in the 2021 DCASE work [(Mei et al., 2021)](#Mei2021), employ self-attention mechanisms that allow the model to directly model long-range dependencies. This approach enables a more holistic understanding of the audio input, resulting in improved coherence and contextual accuracy in the generated captions.

The most recent advances have integrated ideas from the computer vision domain to further enhance audio encoding efficiency. The CoNeTTE system [(Labbé et al., 2023)](#Labbe2023)  exemplifies this trend by adapting a ConvNeXt architecture—originally designed for image classification—for audio feature extraction. The incorporation of dataset-specific task embeddings to address biases arising from pre-training on large-scale audio tagging datasets makes CoNeTTE achieve competitive performance while significantly reducing the number of parameters compared to previous models.

Despite these advances, several challenges persist. Current systems still face limitations due to the scarcity of high-quality annotated datasets, potential biases inherited from pre-trained models, and difficulties in capturing the complex temporal and contextual relationships present in natural audio signals. Future research is likely to focus on developing more robust, data-efficient models and on further refining multi-modal approaches to close the gap between machine-generated and human-level descriptions.


For more information on the topic, please check the [topic description README](doc/README_topic_description.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## About the project

### Motivation

AAC represents a cutting‐edge intersection of audio signal processing and natural language generation with significant real-world impact. The practical applications of AAC are manifold—from enhancing accessibility for the hearing-impaired and augmenting multimedia retrieval systems to serving as a critical component in intelligent monitoring and human–machine interaction. In parallel, the emergence of challenges such as [DCASE 2024](#dcase2024) has highlighted the growing research momentum and industry relevance of AAC. The DCASE challenge not only benchmarks state-of-the-art methods but also catalyzes innovation by providing a structured platform for evaluating model performance under diverse conditions. This project is driven by the dual purposes of exploring AAC’s promising practical implementations and taking into account the insights gained from DCASE 2024 to refine and advance current methodologies.

### Objectives

Thus, the **objectives of this project** include investigating and experimenting with AAC by systematically analyzing state-of-the-art methods, replicating established baselines, and evaluating modifications to model architectures. The specific key goals are as follows:



1. **Understand state-of-the-art AAC systems**  
   A comprehensive review of recent developments in AAC will be conducted, focusing on model architectures, training strategies, and evaluation metrics. This includes analyzing encoder-decoder frameworks, transformer-based approaches, and techniques used to improve caption generation quality.

2. **DCASE 2024 challenge baseline replication**  
   The DCASE Challenge 2024 baseline model will be deployed and trained from scratch to assess the feasibility of reproducing reported benchmark metrics. This process will validate the reproducibility of existing AAC models and serve as a reference point for subsequent experiments.
 
3. **Modify the decoder architecture and adjust training strategies**  
   Modifications to the decoder architecture will be introduced to analyze their impact on performance. Particular attention will be given to model explainability, with a focus on interpreting attention weights and understanding how the model processes audio representations. Alternative training strategies will also be explored to optimize performance and generalization.

4. **Gain insights into audio captioning and deep learning**  
   Through experimentation and analysis, broader insights into AAC model behavior, limitations, and potential improvements will be gathered. This includes understanding the trade-offs between model complexity, explainability, and performance and identifying directions for future research in audio captioning.

This structured approach ensures a methodical evaluation of AAC systems, contributing both to theoretical understanding and practical advancements in the field.


### Schedule

The project is scheduled to run from December 1st to March 17th.  [Figure 2](#fig-schedule) provides an overview of the main objectives and corresponding tasks.


<p align="center">
  <img src="doc/images/schedule.png" alt="AAC Pipeline" width="700" style="max-width: 100%; height: auto;">
</p>
<p align="center"><a id="fig-schedule"></a><em>Figure 2: Project schedule</em></p>

The first goal, state-of-the-art review, was scheduled for completion by the end of January. Key tasks include conducting a literature review on architecture, datasets, and common evaluation metrics, as well as deploying the existing trained DCASE and CLAP baselines.

The second goal involves training the DCASE model from scratch using a virtual machine and comparing its performance against benchmarked metrics. This phase was also expected to be completed by the end of January.

The third goal focuses on evaluating the current model architecture, modifying it, retraining, and testing it to assess performance improvements. Additionally, for explainability, tasks include generating attention maps, analyzing model weights, and identifying the frames the model prioritizes when generating captions. These activities take place throughout February, with final explainability analysis conducted in early March.

Lastly, the fourth goal involves synthesizing all gathered insights and conducting the final discussion and conclusions. This phase is scheduled for the last weeks of the project in March.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Getting started

### Prerequisites

This project is founded on cloud-based infrastructure, specifically Google Cloud, to handle the extensive computational requirements associated with the large dataset used. Due to the substantial size of the dataset and the complexity of model training, the project utilizes Google Cloud's Virtual Machines (VMs) with specialized GPU support for efficient processing.


#### Hardware
The machine configuration is as follows:

- **Machine Type:** g2-standard-4  
This machine type is equipped with 4 vCPUs and 16 GB of memory, offering an appropriate balance of resources for handling data preprocessing and model training tasks.

- **GPU:** 1 x NVIDIA L4 GPU
The NVIDIA L4 GPU was chosen for its optimized performance in deep learning tasks, ensuring fast training and inference times for large models and datasets.

- **Architecture:** x86-64
The x86-64 architecture ensures compatibility with most modern computational frameworks and libraries used in machine learning and deep learning tasks.

#### Software setup

For software, the project uses Google Cloud's environment alongside Python-based frameworks and libraries. Below are the steps to set up the project environment:

1. **Java Installation**

```
sudo apt update
sudo apt install default-jdk
```
2. **DSCASE 2024 code modification**
A slight modification in the codebase is needed to avoid potential issues with multiprocessing in PyTorch. Specifically, add the following line after line 39 in the `cnext.py` script:

```
torch.multiprocessing.set_start_method('spawn')
```

3. **Creating the virtual environment**
The project utilizes a conda environment for managing dependencies. To create the environment, run:
```
conda create -n env_dcase24 python=3.11
conda activate env_dcase24
```

4. **Downgrading pip**
After activating the virtual environment, downgrade `pip` to a specific version for compatibility:



```
python -m pip install --force-reinstall pip==24.0
```

5. **Install project dependencies**
The required Python packages are listed in `requirements.txt`, which should be installed by running:
```
pip install -r requirements.txt
```

The ```requirements.txt``` file includes essential dependencies for AAC model training and evaluation. These include:
```
aac-datasets==0.5.0
aac-metrics==0.5.3
black==24.2.0
codecarbon==2.3.4
deepspeed==0.13.1
flake8==7.0.0
hydra-colorlog==1.2.0
hydra-core==1.3.2
ipykernel==6.29.3
ipython==8.22.1
lightning==2.2.0
nltk==3.8.1
numpy==1.26.4
pre-commit==3.6.2
pytest==8.0.2
PyYAML==6.0.1
tensorboard==2.16.2
tokenizers==0.15.2
torch==2.2.1
torchlibrosa==0.1.0
torchoutil[extras]==0.3.0
```
6. **Baseline deployment**
Final deployment of baseline can be conducting by following 
[The DCASE 2024 Task 6 Baseline Repository](https://github.com/Labbeti/dcase2024-task6-baseline) instructions. For further assistance, the following [instructions](doc/README_baselines.md) can also be consulted.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

What script needs to be run to install the project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## DCASE 2024 and CLAP demo deployment

To enhance the accessibility and user engagement of the AAC project, the DCASE 2024 baseline model and CLAP demo have been deployed as an [interactive web application on Hugging Face Spaces](https://huggingface.co/spaces/mumbert/automatic-audio-captioning-demo). This platform allows users to seamlessly experience the capabilities of the models through a user-friendly interface (see [Figure 3](#fig-example-demo)).

The application provides two primary modes for audio input:

- **Microphone Input**: Users can record audio directly using their device's microphone.
- **File Upload**: Users can upload audio files for processing.






<p align="center">
  <img src="doc/images/example_demo_huggingface.png" alt="AAC Pipeline" width="600" style="max-width: 100%; height: auto;">
</p>
<p align="center"><a id="fig-example-demo"></a><em>Figure 3: Example of the DCASE demo deployment employing microphone input for audio recording.</em></p>

For more information on the demo, please check the following [demo README](doc/README_demo.md).


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

In this section we can provide some simple commands to test the installation works, but we might want to put multiple commands on a separate readme file. We can also have separate notebooks if we have time on a separate folder for different configurations, training processes, etc.

For more information on using this project, please check the following [usage README](doc/README_usage.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Roadmap

We can list here while working on the project some roadmap items or we can even leave here what could eventually be done in the future.

- [x] github project: create project and push initial README file
- [ ] Test baseline
- [ ] Review DCASE dataset
- [ ] Review DCASE metrics

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Attention maps


<p align="center">
  <audio controls>
    <source src="doc/audios/140815_drezyna_3.wav" type="audio/wav">
    Your browser does not support the audio element.
  </audio>
</p>


model: ```a train approaches and passes by on the tracks```


<p align="center">
  <img src="doc/images/attentionmap.png" alt="AAC Pipeline" width="600" style="max-width: 100%; height: auto;">
</p>
<p align="center"><a id="fig-example-demo"></a><em>Figure 4: Attention map example</em></p>


<p align="center">
  <img src="doc/images/attention_audio.png" alt="AAC Pipeline" width="600" style="max-width: 100%; height: auto;">
</p>
<p align="center"><a id="fig-example-demo"></a><em>Figure 5: Audio</em></p>


<p align="center">
  <img src="doc/images/token1.png" alt="AAC Pipeline" width="600" style="max-width: 100%; height: auto;">
</p>
<p align="center"><a id="fig-example-demo"></a><em>Figure 6: token etc</em></p>

train




<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Thanks to the following contributors:
- [Martí Umbert](https://github.com/mumbert)
- [Victor Cuevas](https://github.com/victorcuevasv)
- [Roger Vergés](https://github.com/eirasroger)
- [Roger Calaf](https://github.com/Rcalaf)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## References



<a id="dcase2024"></a>DCASE. (2024). Detection and Classification of Acoustic Scenes and Events (DCASE) Challenge 2024. DCASE. https://dcase.community/challenge2024/

<a id="drossos2017"></a>Drossos, K., Adavanne, S., & Virtanen, T. (2017). Automated audio captioning with recurrent neural networks. *2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA).* [https://doi.org/10.48550/arXiv.1706.10006](https://doi.org/10.48550/arXiv.1706.10006)



<a id="Labbe2023"></a>Labbé, É., Pellegrini, T., & Pinquier, J. (2023). CoNeTTE: An efficient audio captioning system leveraging multiple datasets with task embedding. https://doi.org/10.48550/arXiv.2309.00454


<a id="Mei2021"></a>Mei, X., Liu, X., Plumbley, M. D., & Wang, W. (2021). Audio captioning transformer. Detection and Classification of Acoustic Scenes and Events (DCASE 2021). https://doi.org/10.48550/arXiv.2107.09817


<p align="right">(<a href="#readme-top">back to top</a>)</p>