cd ..
export PYTHONPATH=.:$PYTHONPATH
MODEL=seminas
OUTPUT_DIR=outputs/$MODEL
DATASET_DIR=home/dzzp/workspace/dataset/
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=1 python3 train_seminas.py --data=$DATASET_DIR --output_dir=$OUTPUT_DIR | tee $OUTPUT_DIR/log.txt
