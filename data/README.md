# Data Preparation

Multimodal Delphi handles data differently compared to the original version. The data directory should be structured as follows:

```bash
# example file tree of multimodal Delphi data corpus
ukb_data/
├── data.bin
├── time.bin
├── p2i.csv
├── tokenizer.yaml
├── expansion_packs/
│   └── surgical_ops/
│       ├── data.bin
│       ├── time.bin
│       ├── p2i.csv
│       └── tokenizer.yaml
└── biomarkers/
    └── polygenic_risk_scores/
        ├── data.npy
        └── p2i.csv
```

## Event Records

Event records represent disease occurrences with sex and lifestyle tokens that the original Delphi was trained on. These should be stored in two binary files:

### Required Files

**`data.bin`**
- Contains a contiguous sequence of all event tokens (`np.uint32`)
- **Important:** Token indexing must start at 1

**`time.bin`**
- Contains a contiguous sequence of timestamps measured in days (`np.uint32`)
- Timestamps correspond to when event tokens occurred
- Must map one-to-one with `data.bin` entries and be exactly the same length

**`p2i.csv`**
- Maps each participant ID (`pid`) to:
  - `start_pos`: starting index in the data files
  - `seq_len`: sequence length for that participant

**`tokenizer.yaml`**
- Dictionary mapping event names to token indices
- Must include these required mappings:
  ```yaml
  padding: 0
  no_event: 1
  female: 2
  male: 3
  ```
- Token indexing must be consistent with `data.bin` and start at 1

## Multimodal Extensions

Multimodal data can be incorporated in two ways: as biomarkers or expansion packs.

### Biomarkers

A biomarker represents a measurement taken at a specific time point in an individual's health trajectory. Examples include HbA1c levels, polygenic risk scores, and diet quality indices.

#### Required Files

**`data.npy`**
- One-dimensional, contiguous array containing biomarker measurements
- Data type varies depending on the specific biomarker

**`p2i.csv`**
- Maps each participant ID (`pid`) to:
  - `start_pos`: starting index in `data.npy`
  - `seq_len`: number of measurements (e.g., 36 for polygenic risk scores covering 36 diseases)
  - `time`: timestamp in days when the biomarker was measured

#### Registering a New Biomarker

Delphi includes a modality embedding layer that learns embeddings for each biomarker. To introduce a new biomarker, you must first register it in the system.

**Steps to register a new biomarker:**

1. **Update the Modality enum** - Navigate to `delphi/multimodal.py` and locate the `Modality` class:

    ```python
    class Modality(Enum):
        # 0 for padding; 1 for event tokens
        PRS = 2
        FAMILY_HX = 3
        BLOOD_ALL = 4
        WBC = 5
        LIPID = 6
        LFT = 7
        RENAL = 8
        HBA1C = 9
        CRP = 10
        URATE = 11
        CYSC = 12
        APO = 13
        VITD = 14
        DHT = 15
        SHBG = 16
        IGF1 = 17
        NAK = 18
        CREAT = 19
        ALBU = 20
        MEDS = 21
        DIET = 22
        MET = 23
        TELOMERE = 24
        # SLEEP_SCORE = 25
    ```

2. **Add your biomarker** - Append your new biomarker to the end of the enum with the next sequential index. For example, to add `SLEEP_SCORE`:
   - Use the next available index (current highest + 1)
   - Follow the existing naming convention (uppercase with underscores)

3. **Create matching directory** - The data directory name must match the enum name in lowercase. For `SLEEP_SCORE`, create a directory named `sleep_score` under `biomarkers`.

**Example:**
If adding a sleep quality biomarker, add `SLEEP_SCORE = 25` to the enum and create a `sleep_score/` directory containing the required files.

### Expansion Packs

An expansion pack contains additional timestamped events that enrich an individual's health trajectory. Examples include surgical operations and medical prescriptions that extend beyond Delphi's original disease occurrence vocabulary.

#### Required Files

**`data.npy`**
- One-dimensional, contiguous array of all tokens (`np.uint32`)
- **Important:** Token indexing must start at 1

**`time.npy`**
- One-dimensional, contiguous array of timestamps in days (`np.uint32`)
- Corresponds to when event tokens occurred

**`p2i.csv`**
- Maps each participant ID (`pid`) to:
  - `start_pos`: starting index in both `data.npy` and `time.npy`
  - `seq_len`: sequence length for that participant

**`tokenizer.yaml`**
- Dictionary mapping event names (surgical operations, medications, etc.) to token indices
- Token indexing must be consistent with `data.npy` and start at 1
