# Create a conda environement
	conda create --name nft_env
	conda activate nft_env 

# To install the dependncies-
	pip install -r requirements.txt

# To train a backdoor model with "blend" attack-

	python train_backdoor_cifar.py --poison-type blend --poison-rate 0.10 --output-dir your/folder/to/save --gpuid 0 

# To train a benign model-

 	python train_backdoor_cifar.py --poison-type benign --output-dir your/folder/to/save --gpuid 0 


# To remove backdoor-
	
	python Remove_Backdoor.py --poison-type blend --val-ratio 0.01 --output-dir your/folder/to/save --gpuid 0 



