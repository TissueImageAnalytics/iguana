python run_infer.py \
--gpu=0 \
--model_path="/root/lsf_workspace/iguana_data/weights/iguana_fold1.tar" \
--model_name="pna" \
--data_dir="/root/lsf_workspace/proc_slides/cobi/colchester/graphs/data" \
--data_info="/root/lsf_workspace/proc_slides/cobi/colchester/graphs/colchester_info.csv" \
--stats_dir="/root/lsf_workspace/proc_slides/cobi/uhcw/graphs/stats" \
--output_dir="output_test/" \
--batch_size=1