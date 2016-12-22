#
# Extract text embeddings from the encoder
#
CUB_ENCODER=lm_sje_nc4_cub_hybrid_gru18_a1_c512_0.00070_1_10_trainvalids.txt_iter30000.t7 \
CAPTION_PATH=Data/birds/example_captions \
GPU=0 \

export CUDA_VISIBLE_DEVICES=${GPU}

net_txt=models/text_encoder/${CUB_ENCODER} \
queries=${CAPTION_PATH}.txt \
filenames=${CAPTION_PATH}.t7 \
th demo/get_embedding.lua

#
# Generate image from text embeddings
#
python demo/demo.py \
--cfg demo/cfg/birds-demo.yml \
--gpu ${GPU} \
--caption_path ${CAPTION_PATH}.t7
