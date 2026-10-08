# Plant Disease Classification

Image classification for diagnosing plant disease from leaf photographs, built as part of a wider pipeline intended to give growers both a diagnosis and treatment guidance.

This repository holds the model development work: the training notebooks and the V3 weights. The serving layer was built separately and is not in this repository.

---

## Problem

A grower photographs an affected leaf and needs two things: what the disease is, and what to do about it. The binding constraint is cost. Inference has to be cheap and fast enough to be usable at scale, which rules out the largest architectures regardless of how well they score.

## Approach

Four candidate models were built across three architectures. **MobileNetV2** was taken forward: pretrained on ImageNet, with the base frozen and a small classification head trained on top (about 142 thousand trainable parameters out of 2.4 million), keeping training feasible and inference cheap.

Data: the New Plant Diseases Dataset (Augmented), a Kaggle version of PlantVillage. 38 classes across 14 crops, 70,295 training images and 17,572 validation images.

## Results

| Version | Model | Framework | Input | Validation accuracy |
|---|---|---|---|---|
| V1 | ResNet9, trained from scratch | PyTorch | 256 x 256 | Did not complete: ran out of GPU memory (6 GB) |
| V2 | ResNet18, fine-tuned | fastai | 224 x 224 | 99.8%, on a random split (see note) |
| V3 | MobileNetV2, frozen, Dense 1024 head | TensorFlow | 256 x 256 | 94.5% |
| V4 | MobileNetV2, frozen, Dense 100 x 2 head | TensorFlow | 224 x 224 | **95.5%** |

V2's figure is not comparable with V3 and V4. It used a random 80/20 split over the combined folders, and this dataset contains rotated and flipped copies of the same leaf (visible in the filenames), so copies can fall on both sides of the split and inflate the score. V3 and V4 use the dataset's own train and validation folders. Those may carry a milder version of the same issue, because the dataset was augmented before it was split, so a leaf-level split would be the honest next step.

V4 also corrected two issues in V3: a mismatch between training size (256) and inference size (224), and augmentation applied to the validation data.

On leaf photos from outside the dataset, prediction confidence dropped sharply, mostly to between 55% and 77%. PlantVillage images are taken in controlled conditions on plain backgrounds, and real photographs are not.

## Serving

A FastAPI service with async MongoDB pipelines served the model in the first prototype release. It is not in this repository.

## Designed next phase

That drop shaped the next phase, which was designed but not built, because the contract ended before the client took it forward:

- **Confidence gating** (designed). Predictions below a certainty threshold would be flagged as uncertain rather than returned as a diagnosis. For someone about to spend money on a treatment, a confident wrong answer is far more damaging than an honest "not sure".
- **A treatment knowledge base** (designed), pairing each diagnosis with guidance through exact identifier lookup, with semantic search only as a fallback.

## Known limitations

Stated plainly, because they are real.

- **The notebooks here are development artefacts, not production code.** They were written to train and compare models, and have not been cleaned up or refactored for reuse.
- **No separate test set.** Results are on the validation split. The model has not been evaluated on field photographs beyond a handful of samples.
- **Per-class results were not computed.** A 95% average can hide one class that fails badly.
- **The V3 weights here and the V4 class order differ.** V3 used directory listing order; V4 sorts class names. Use the matching label order for whichever weights you load.
- **No model versioning.** There is no mechanism to identify which model version produced a given prediction, or to roll back.
- **No prediction logging**, so real-world drift is invisible. The validation result is assumed to hold, which is an assumption rather than a measurement.
- **Services are not decoupled.** Model serving and the API are coupled, so they cannot be scaled or versioned independently.

## Repository contents

| File | Description |
|---|---|
| `Plant_Disease_V1.ipynb` | ResNet9 from scratch, PyTorch |
| `Plant_Disease_V2.ipynb` | ResNet18 fine-tune, fastai |
| `Plant_Disease_V3.ipynb` | MobileNetV2, version 3 |
| `Plant_Disease_V4.ipynb` | MobileNetV2, version 4 |
| `plant_disease_model_v3.h5` | Trained V3 weights (94.5%). V4 weights are not included. |

## Stack

Python, TensorFlow/Keras, PyTorch, fastai, MobileNetV2, ResNet, OpenCV, FastAPI, MongoDB

## Author

Siddharth Sharma, MSc Artificial Intelligence, Heriot-Watt University
[LinkedIn](https://www.linkedin.com/in/siddharthsharma8000)
