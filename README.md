# Plant Disease Classification

Image classification for diagnosing plant disease from leaf photographs, built as part of a wider pipeline intended to give growers both a diagnosis and treatment guidance.

This repository holds the model development work: the training notebooks and trained weights. The serving layer and knowledge base described below were built separately and are not in this repository.

---

## Problem

A grower photographs an affected leaf and needs two things: what the disease is, and what to do about it. The binding constraint is cost. Inference has to be cheap and fast enough to be usable at scale, which rules out the largest architectures regardless of how well they score.

## Approach

Four candidate models were built and evaluated, and the best performer was selected.

The selected model is based on **MobileNetV2**, chosen because serving cost mattered: depthwise separable convolutions give far fewer parameters and lower inference cost than a heavier backbone, at a modest accuracy trade-off.

## Results

The selected model reached **95% to 98% accuracy** across test sets.

## Beyond the model

The model is only part of the system. The wider pipeline included:

- **A FastAPI service** with async MongoDB pipelines for inference and data handling.
- **Confidence gating.** The model returns a probability distribution, not a fact. Any prediction below a defined certainty threshold was flagged as uncertain rather than returned as a diagnosis. For someone about to spend money on a treatment, a confident wrong answer is considerably more damaging than an honest "not sure".
- **A ChromaDB knowledge base** pairing each diagnosis with treatment guidance, using exact identifier lookup with semantic search as a fallback. Once the classifier has identified a disease, the correct entry is already known, so an exact lookup is both faster and cannot drift to a neighbouring entry. Semantic search is the safety net, not the main path.

## Known limitations

Stated plainly, because they are real and were identified during development rather than afterwards.

- **The notebooks here are development artefacts, not production code.** They were written to train and compare models, and have not been cleaned up or refactored for reuse.
- **The knowledge base needs replacing with verified agronomic sources** such as ICAR or FAO before any grower-facing deployment. Treatment advice must be attributable.
- **No model versioning.** There is no mechanism to identify which model version produced a given prediction, or to roll back.
- **No prediction logging**, so real-world drift is invisible. The test set is assumed to hold, which is an assumption rather than a measurement.
- **Services are not decoupled.** Model serving and the API are coupled, so they cannot be scaled or versioned independently.

## Repository contents

| File | Description |
|---|---|
| `Plant_Disease_V3.ipynb` | Model development notebook, version 3 |
| `Plant_Disease_V4.ipynb` | Model development notebook, version 4 |
| `plant_disease_model_v3.h5` | Trained Keras model weights |

## Stack

Python, TensorFlow/Keras, MobileNetV2, OpenCV, FastAPI, MongoDB, ChromaDB

## Author

Siddharth Sharma, MSc Artificial Intelligence, Heriot-Watt University
[LinkedIn](https://www.linkedin.com/in/siddharthsharma8000)
