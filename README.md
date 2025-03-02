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

AAC represents a cutting‐edge intersection of audio signal processing and natural language generation with significant real-world impact. The practical applications of AAC are manifold—from enhancing accessibility for the hearing-impaired and augmenting multimedia retrieval systems to serving as a critical component in intelligent monitoring and human–machine interaction. In parallel, the emergence of challenges such as [DCASE 2024](#dcase2024) has highlighted the growing research momentum and industry relevance of AAC. The DCASE challenge not only benchmarks state-of-the-art methods but also catalyzes innovation by providing a structured platform for evaluating model performance under diverse conditions. This project is driven by the dual purposes of exploring AAC’s promising practical implementations and taking into account the insights gained from DCASE 2024 to refine and advance current methodologies.



Thus, the **objectives of this project** are to investigate and experiment with AAC by systematically analyzing state-of-the-art methods, replicating established baselines, and evaluating modifications to model architectures. The key objectives are as follows:



1. **Understand state-of-the-art AAC systems**  
   A comprehensive review of recent developments in AAC will be conducted, focusing on model architectures, training strategies, and evaluation metrics. This includes analyzing encoder-decoder frameworks, transformer-based approaches, and techniques used to improve caption generation quality.

2. **DCASE 2024 challenge baseline replication**  
   The DCASE Challenge 2024 baseline model will be deployed and trained from scratch to assess the feasibility of reproducing reported benchmark metrics. This process will validate the reproducibility of existing AAC models and serve as a reference point for subsequent experiments.

3. **Modify the decoder architecture and adjust training strategies**  
   Modifications to the decoder architecture will be introduced to analyze their impact on performance. Particular attention will be given to model explainability, with a focus on interpreting attention weights and understanding how the model processes audio representations. Alternative training strategies will also be explored to optimize performance and generalization.

4. **Gain insights into audio captioning and deep learning**  
   Through experimentation and analysis, broader insights into AAC model behavior, limitations, and potential improvements will be gathered. This includes understanding the trade-offs between model complexity, explainability, and performance and identifying directions for future research in audio captioning.

This structured approach ensures a methodical evaluation of AAC systems, contributing both to theoretical understanding and practical advancements in the field.


## Getting started

### Prerequisites

Detail what is needed to run the project. We can detail what we have used either google cloud, AWS, or locally.

### Installation

What script needs to be run to install the project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

In this section we can provide some simple commands to test the installation works, but we might want to put multiple commands on a separate readme file. We can also have separate notebooks if we have time on a separate folder for different configurations, training processes, etc.

For more information on using this project, please check the following [usage README](doc/README_usage.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Demo

Placeholder for the demo we might want to create, be it notebooks, fastAPI, etc.

For more information on the demo, please check the following [demo README](doc/README_demo.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

We can list here while working on the project some roadmap items or we can even leave here what could eventually be done in the future.

- [x] github project: create project and push initial README file
- [ ] Test baseline
- [ ] Review DCASE dataset
- [ ] Review DCASE metrics

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Thanks to the following contributors:
- [Martí Umbert](https://github.com/mumbert)
- [Victor Cuevas](https://github.com/contributor2)
- [Roger Vergés](https://github.com/eirasroger)
- [Roger Calaf](https://github.com/Rcalaf)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## References



<a id="dcase2024"></a>DCASE. (2024). Detection and Classification of Acoustic Scenes and Events (DCASE) Challenge 2024. DCASE. https://dcase.community/challenge2024/

<a id="drossos2017"></a>Drossos, K., Adavanne, S., & Virtanen, T. (2017). Automated audio captioning with recurrent neural networks. *2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA).* [https://doi.org/10.48550/arXiv.1706.10006](https://doi.org/10.48550/arXiv.1706.10006)



<a id="Labbe2023"></a>Labbé, É., Pellegrini, T., & Pinquier, J. (2023). CoNeTTE: An efficient audio captioning system leveraging multiple datasets with task embedding. https://doi.org/10.48550/arXiv.2309.00454


<a id="Mei2021"></a>Mei, X., Liu, X., Plumbley, M. D., & Wang, W. (2021). Audio captioning transformer. Detection and Classification of Acoustic Scenes and Events (DCASE 2021). https://doi.org/10.48550/arXiv.2107.09817

