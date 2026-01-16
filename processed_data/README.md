# Processed Data Directory

This directory contains the preprocessed training and validation datasets.

## Structure

```
processed_data/
├── dataset_dict.json          # Dataset configuration
├── train/                     # Training split
│   ├── dataset_info.json      # Dataset metadata
│   ├── state.json            # Dataset state
│   └── data-*.arrow          # Training data (excluded from git)
└── validation/               # Validation split
    ├── dataset_info.json      # Dataset metadata
    ├── state.json            # Dataset state
    └── data-*.arrow          # Validation data (excluded from git)
```

## Generation

Run the following command to generate this data:

```bash
python prepare_data.py
```

This will:
1. Download HateXplain and ToxiGen datasets
2. Filter and process examples
3. Format with special control tokens: `<CATEGORY><TARGET> content`
4. Split into train (90%) and validation (10%)

## Dataset Statistics

- **Training examples**: ~18,000-20,000
- **Validation examples**: ~2,000-2,500
- **Format**: `<HATE_SPEECH><RACE> [toxic content example]`
- **Categories**: HATE_SPEECH, OFFENSIVE, HARASSMENT, IMPLICIT_HATE
- **Targets**: RACE, GENDER, RELIGION, DISABILITY, SEXUALITY, AGE, NATIONALITY, GENERAL

## Note

The actual data files (`.arrow` files) are excluded from git due to size.
Run `prepare_data.py` to regenerate them locally.
