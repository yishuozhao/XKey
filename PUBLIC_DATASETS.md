# Public Dataset Preparation

The demo uses a small processed ECG subset in `sample_data/`. For larger ECG
experiments, download public ECG datasets from their official sources, such as
PhysioNet, and keep raw data outside this repository.

Suggested ECG preparation steps:

1. Download records under the dataset's license and usage terms.
2. Select paired or synchronized ECG channels when available.
3. Apply ECG filtering, R-peak based windowing, beat alignment, and resampling.
4. Save only derived arrays that are permitted by the dataset license.

Do not commit self-collected biometric signals, participant metadata, consent or
IRB documents, raw facial videos, or restricted dataset files.
