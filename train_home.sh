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
		method="ALDA"
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

echo "home_${method}=loss_type=${loss_type}_thresh=${threshold}_${run_id}"

for num in 01 02 03 04 05 06 07 08 09 10 11 12
do
	case ${num} in
		01 )
			s_dset_path="./data/office-home/Art.txt"
			t_dset_path="./data/office-home/Clipart.txt"
			output_dir="A2C"
			;;
		02 )
			s_dset_path="./data/office-home/Art.txt"
			t_dset_path="./data/office-home/Product.txt"
			output_dir="A2P"
			;;		
		03 )
			s_dset_path="./data/office-home/Art.txt"
			t_dset_path="./data/office-home/Real_World.txt"
			output_dir="A2R"
			;;
		04 )
			s_dset_path="./data/office-home/Clipart.txt"
			t_dset_path="./data/office-home/Art.txt"
			output_dir="C2A"
			;;
		05 )
			s_dset_path="./data/office-home/Clipart.txt"
			t_dset_path="./data/office-home/Product.txt"
			output_dir="C2P"
			;;		
		06 )
			s_dset_path="./data/office-home/Clipart.txt"
			t_dset_path="./data/office-home/Real_World.txt"
			output_dir="C2R"
			;;		
		07 )
			s_dset_path="./data/office-home/Product.txt"
			t_dset_path="./data/office-home/Art.txt"
			output_dir="P2A"
			;;
		08 )
			s_dset_path="./data/office-home/Product.txt"
			t_dset_path="./data/office-home/Clipart.txt"
			output_dir="P2C"
			;;		
		09 )
			s_dset_path="./data/office-home/Product.txt"
			t_dset_path="./data/office-home/Real_World.txt"
			output_dir="P2R"
			;;
		10 )
			s_dset_path="./data/office-home/Real_World.txt"
			t_dset_path="./data/office-home/Art.txt"
			output_dir="R2A"
			;;
		11 )
			s_dset_path="./data/office-home/Real_World.txt"
			t_dset_path="./data/office-home/Clipart.txt"
			output_dir="R2C"
			;;		
		12 )
			s_dset_path="./data/office-home/Real_World.txt"
			t_dset_path="./data/office-home/Product.txt"
			output_dir="R2P"
			;;								
	esac

	output_dir="home_${output_dir}_${method}"
	final_log="home_${method}"

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
					--dset office-home \
					--test_interval 500 \
					--s_dset_path ${s_dset_path} \
					--t_dset_path ${t_dset_path} \
					--batch_size 36 \
					--output_dir ${output_dir} \
					--final_log "${final_log}_log.txt" \
					--loss_type ${loss_type} \
					--threshold ${threshold} \
					--stop_step 10000
	echo "Finish in ${output_dir}"

done

echo "Training Finished!!!"