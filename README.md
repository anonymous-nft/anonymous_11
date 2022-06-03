To train a backdoor model with "blend" attack with poison ratio of "10%"-

	python train_backdoor_cifar.py --poison-type blend --posion-rate 0.10 --output-dir your/folder/to/save --gpuid 0 

To train a benign model-

  python train_backdoor_cifar.py --poison-type benign --output-dir your/folder/to/save --gpuid 0 


To remove backdoor-
	
	python Remove_Backdoor.py --poison-type blend --val-frac 0.01 --output-dir your/folder/to/save --gpuid 0 



