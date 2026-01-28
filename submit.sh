#!/bin/bash

PROJECT_NAME="ubc-ocean"

# Initialize the kernel
# kaggle kernels init

# Update your custom datasets
kaggle datasets init -p src
sed -i "s/INSERT_TITLE_HERE/$PROJECT_NAME-src/g" src/dataset-metadata.json
sed -i "s/INSERT_SLUG_HERE/$PROJECT_NAME-src/g" src/dataset-metadata.json
kaggle datasets version -m "latest" -d -p src -r skip
kaggle datasets create -p src -r skip

kaggle datasets init -p checkpoints
sed -i "s/INSERT_TITLE_HERE/$PROJECT_NAME-checkpoints/g" checkpoints/dataset-metadata.json
sed -i "s/INSERT_SLUG_HERE/$PROJECT_NAME-checkpoints/g" checkpoints/dataset-metadata.json
kaggle datasets version -m "latest" -d -p checkpoints -r skip
kaggle datasets create -p checkpoints -r skip

kaggle datasets init -p models
sed -i "s/INSERT_TITLE_HERE/$PROJECT_NAME-models/g" models/dataset-metadata.json
sed -i "s/INSERT_SLUG_HERE/$PROJECT_NAME-models/g" models/dataset-metadata.json
kaggle datasets version -m "latest" -d -p models -r skip
kaggle datasets create -p models -r skip

# Submit the kernel
kaggle kernels push