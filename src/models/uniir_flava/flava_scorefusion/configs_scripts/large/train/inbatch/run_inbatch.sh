# Train CLIPScoreFusion model on MBEIR dataset

# set -e  # Exit immediately if a command exits with a non-zero status

# Initialize Conda
source /opt/conda/etc/profile.d/conda.sh # <--- Change this to the path of your conda.sh

UNIIRHOME="/home/ma4496/"
# Path to the codebase and config file
SRC="$UNIIRHOME/UniIR/src"  # Absolute path to codebse /UniIR/src # <--- Change this to the path of your UniIR/src

# Path to common dir
COMMON_DIR="$SRC/common"

# Path to MBEIR data and UniIR directory where we store the checkpoints, embeddings, etc.
UNIIR_DIR="$UNIIRHOME/UniIR1/UniIR" # <--- Change this to the UniIR directory
MBEIR_DATA_DIR="$UNIIRHOME/M-BEIR/" #--- Change this to the MBEIR data directory you download from HF page

# Path to config dir
MODEL="uniir_flava/flava_scorefusion"  # <--- Change this to the model you want to run
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="train"  # <--- Change this to the mode you want to run
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs_scripts/$SIZE/$MODE/$EXP_NAME"
conda activate uniir # <--- Change this to the name of your conda environment

# Set CUDA devices and PYTHONPATH
export CUDA_VISIBLE_DEVICES=0 # <--- Change this to the CUDA devices you want to us
NPROC=1
export PYTHONPATH=$SRC
echo "PYTHONPATH: $PYTHONPATH"
echo  "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Update config
CONFIG_PATH="$CONFIG_DIR/inbatch.yaml"
cd $COMMON_DIR
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct True

# Change to model directory
cd $MODEL_DIR
SCRIPT_NAME="train.py"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

# Activate conda environment
# conda activate clip

# Run training command
python -m torch.distributed.run --nproc_per_node=$NPROC $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --uniir_dir "$UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"