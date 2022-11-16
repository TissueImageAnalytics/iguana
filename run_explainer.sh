python run_explainer.py \
--gpu=0 \
--model_name="pna" \
--model_path="/root/lsf_workspace/iguana_data/weights/iguana_fold1.tar" \
--feature \
--node_exp_method="gnnexplainer" \
--feat_exp_method="gnnexplainer" \
--data_dir="/root/lsf_workspace/proc_slides/cobi/colchester/graphs/data" \
--output_dir="output_test/" \
--stats_dir="/root/lsf_workspace/proc_slides/cobi/uhcw/graphs/stats"

