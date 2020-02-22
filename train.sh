#!/bin/bash
# python PATH
# export PYTHONPATH="${PYTHONPATH}:${HOME}/github"

# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"
# replace comma(,) with empty
#gpus=${gpus//,/}	
# the number of characters
#num_gpus=${#gpus}
#echo "the number of gpus is ${num_gpus}"

# choose the method
echo ""
echo "0  --  DANN"
echo "1  --  ALDA"
echo -n "choose the method: "
read method_choose

case ${method_choose} in
	0 )
		method="DANN"
		;;
	1 ) 
		method="ALDA_hard"
		;;
	* )
		echo "The choice of method is illegal!"
		exit 1 
		;;
esac

# choose the loss_type
echo ""
echo "all -- ALDA with full losses"
echo "nocorrect -- ALDA without the target loss"
echo -n "choose the loss_type: "
read loss_type

# choose the threshold
echo ""
echo "0.9 -- the optimum for office"
echo -n "choose the threshold: "
read threshold

echo ""
echo "0 -- default"
echo -n "run_id: "
read run_id

echo "${method}=${trade_off}_loss_type=${loss_type}_thresh=${threshold}_${run_id}"

for num in 01 02 03 04 05 06
do
	case ${num} in
		01 )
			s_dset_path="./data/office/amazon_list.txt"
			t_dset_path="./data/office/webcam_list.txt"
			output_dir="A2W"
			;;
		02 )
			s_dset_path="./data/office/webcam_list.txt"
			t_dset_path="./data/office/amazon_list.txt"
			output_dir="W2A"
			;;		
		03 )
			s_dset_path="./data/office/amazon_list.txt"
			t_dset_path="./data/office/dslr_list.txt"
			output_dir="A2D"
			;;
		04 )
			s_dset_path="./data/office/dslr_list.txt"
			t_dset_path="./data/office/amazon_list.txt"
			output_dir="D2A"
			;;
		05 )
			s_dset_path="./data/office/dslr_list.txt"
			t_dset_path="./data/office/webcam_list.txt"
			output_dir="D2W"
			;;		
		06 )
			s_dset_path="./data/office/webcam_list.txt"
			t_dset_path="./data/office/dslr_list.txt"
			output_dir="W2D"
			;;							
	esac

	# create PID
	output_dir="${output_dir}_${method}"
	final_log="${method}"

	case ${loss_type} in
		0 )
			output_dir="${output_dir}"
			;;	
		* )
			output_dir="${output_dir}=${loss_type}"
			final_log="${final_log}=${loss_type}"
			;;	
	esac

	output_dir="${output_dir}_thresh=${threshold}"
	final_log="${final_log}_thresh=${threshold}"

	case ${run_id} in
		0 )
			output_dir="${output_dir}"
			;;	
		* )
			output_dir="${output_dir}_${run_id}"
			final_log="${final_log}_${run_id}"
			;;	
	esac

	echo "Begin in ${output_dir}"
	echo "log in ${final_log}_log.txt"

	# train the model
	python train.py ${method} \
					--gpu_id ${gpus} \
					--net ResNet50 \
					--dset office \
					--test_interval 500 \
					--s_dset_path ${s_dset_path} \
					--t_dset_path ${t_dset_path} \
					--trade_off 1 \
					--batch_size 36 \
					--output_dir ${output_dir} \
					--final_log "${final_log}_log.txt" \
					--loss_type ${loss_type} \
					--threshold ${threshold} \
					--cos_dist False \
					--source_detach False
	echo "Finish in ${output_dir}"

done

echo "Training Finished!!!"