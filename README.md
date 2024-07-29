# EEGNLP

```livescript
A repository that aggregates open-source EEG2Text works.
```



#### Data

- ###### Option 1

  Download ZuCo 1.0 and ZuCo 2.0 and process them according to the instructions in the [EEG-To-Text README](https://github.com/MikeWangWZHL/EEG-To-Text/blob/main/README.md). Place the 'datasets' directory at the same root level as 'EEGNLP'.

- ###### Option 2

  Download the preprocessed 'datasets' directory from xxx.

#### Models

|                          Reference                           | Paper                                                        | Command                                                      |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Wang and Ji, 2022]<br />https://github.com/MikeWangWZHL/EEG-To-Text <br />[Revised by NeuSpeech]<br />https://github.com/NeuSpeech/EEG-To-Text | Open Vocabulary Electroencephalography-To-Text Decoding and Zero-shot Sentiment Classification (UIUC) | **Decode**: <br />`bash ./scripts/train_decoding.sh`<br />`bash ./scripts/eval_decoding.sh`(with '-tf' argument enabling teacher-forcing)<br />**Sentimental**: <br />`bash ./scripts/train_eval_zeroshot_pipeline.sh`<br />`bash ./scripts/eval_sentiment_zeroshot_pipeline.sh` |
| [Hollenstein et al., 2023]<br />https://github.com/norahollenstein/zuco-benchmark | The ZuCo benchmark on cross-subject reading task classification with EEG and eye-tracking data （ETH, ZuCo author） | `cd src `<br />(leave-1-subject-out) `python3 validation.py`<br />(assigned test subjects)`python3 benchmark_baseline.py` |
| [Han et al., 2023]<br />https://github.com/Jason-Qiu/EEG_Language_Alignment | Can Brain Signals Reveal Inner Alignment with Human Languages?(CMU) | `bash train.sh`<br /><font color="orange">[Note]: Incomplete codes for 'bert': uses a uniform training interface with 'transformers', but did not implement a customized class to encapsulate 'BertModel'.</font> |
| [Feng et al., 2023]<br />https://github.com/gonzrubio/EEG-to-Text-Curriculum-Semantic-aware-Contrastive-Learning | Semantic-aware Contrastive Learning for Electroencephalography-to-Text Generation with Curriculum Learning (HIT) | `cd src`<br />`python3 train.py`                             |
| [Amrani et al., 2023]<br />https://github.com/hamzaamrani/EEG-to-Text-Decoding | Deep Representation Learning for Open Vocabulary Electroencephalography-to-Text Decoding (U-Milano) | `bash ./scripts/train_decoding_raw.sh`<br />`bash ./scripts/eval_decoding_raw.sh` |

