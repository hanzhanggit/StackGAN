#
# Extract text embeddings from the encoder
#
FLOWER_ENCODER=lm_sje_flowers_c10_hybrid_0.00070_1_10_trainvalids.txt_iter16400.t7 \
CAPTION_PATH=Data/flowers/example_captions \
GPU=0 \

export CUDA_VISIBLE_DEVICES=${GPU}

net_txt=models/text_encoder/${FLOWER_ENCODER} \
queries=${CAPTION_PATH}.txt \
filenames=${CAPTION_PATH}.t7 \
gpu=${GPU} \
th demo/get_embedding.lua

#
# Generate image from text embeddings
#
python demo/demo.py \
--cfg demo/cfg/flowers-demo.yml \
--gpu ${GPU} \
--caption_path ${CAPTION_PATH}.t7
