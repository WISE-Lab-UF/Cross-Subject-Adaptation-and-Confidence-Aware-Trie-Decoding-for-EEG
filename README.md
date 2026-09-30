# Cross-Subject Adaptation and Confidence-Aware Trie Decoding for EEG

Code and evaluation resources accompanying the manuscript
**"Cross-Subject Adaptation and Confidence-Aware Trie Decoding for
Low-Latency EEG-Based Imagined Handwriting Recognition."**

## Repository Contents

| File | Purpose |
| -------- | -------- |
| `eedgenet.py` | EEdGeNet architecture and 512-dimensional frozen representation |
| `classifier.py` | Target-specific shrinkage Linear Discriminant Analysis classifier using labeled calibration embeddings |
| `decoder.py` | Confidence-aware trie decoder and optimized implementation |
| `decoder_config.json` | Decoder parameters used in the experiments |
| `seeds.json` | Random seeds used for evaluation |
| `words_15.csv` | 15-word evaluation set |
| `words_50.csv` | 50-word evaluation set |
| `words_12500.csv` | 12,500-word lexical-diversity set |
| `posterior_streams_*.npz` | Saved character-posterior streams for decoder evaluation |
| `word_length_distribution.csv` | Word-length statistics |
| `zipf_frequency_distribution.csv` | Word-frequency statistics |
| `jetson_tx2_environment.txt` | NVIDIA Jetson TX2 software environment |
                                      
-----------------------------------------------------------------------

## Reproducing the Experiments

Install the required Python packages:

``` bash
pip install -r requirements.txt
```

The main manuscript components are associated with the following files:

| Manuscript Experiment | Files |
| -------- | -------- |
| EEdGeNet and 512-dimensional representation | `eedgenet.py` |
| Target-specific classifier | `classifier.py` |
| Fixed-candidate and adaptive word | `decoder.py`, `posterior_streams_*.npz`, `words_*.csv` |
| Decoder configuration and reproducibility | `decoder_config.json`, `seeds.json` |
| Decoder implementation optimization | `decoder.py` | 
| NVIDIA Jetson TX2 deployment | `decoder.py`, `jetson_tx2_environment.txt` |
   
-----------------------------------------------------------------------

### Manuscript Table Mapping

| Manuscript Table | Analysis | Relevant Repository Files | Reproducibility |
|---|---|---|---|
| Table 1 | Lexical characteristics of the reference-word sets | `words_15.csv`, `words_50.csv`, `words_12500.csv`, `word_length_distribution.csv`, `zipf_frequency_distribution.csv` | Reproducible directly from the released files |
| Table 2 | Target-specific classifier comparison | `classifier.py` | Classifier implementation is provided; reported values require the participant-level EEG-derived representations |
| Table 3 | Development-participant word-decoder comparison | `decoder.py`, `posterior_streams_50_words.npz`, `words_50.csv`, `decoder_config.json`, `seeds.json` | Released posterior streams support the word-decoding evaluation |
| Table 4 | Fixed-candidate, adaptive-decoding, implementation-optimization, and edge-device evaluation | `decoder.py`, `posterior_streams_*.npz`, `words_*.csv`, `decoder_config.json`, `seeds.json`, `jetson_tx2_environment.txt` | Decoder evaluation uses the released posterior streams; NVIDIA Jetson TX2 latency and energy measurements require the corresponding hardware |
| Table 5 | Leave-one-subject-out cross-subject evaluation | `eedgenet.py`, `classifier.py`, `decoder.py` | Requires the underlying participant EEG data and derived participant-level representations |

-----------------------------------------------------------------------

## Data Availability

The repository provides the code, evaluation lists, posterior streams,
configurations, and random seeds required for the word-decoding
experiments. Experiments involving EEG preprocessing, source-model
training, cross-subject adaptation, and representation controls require
the underlying participant EEG data, which are available from the
corresponding author upon reasonable request. NVIDIA Jetson TX2 latency
and energy measurements additionally require the corresponding
edge-device and power-measurement setup.
