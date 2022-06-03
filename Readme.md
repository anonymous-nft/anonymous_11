To train a backdoor model with "blend" attack with poison ratio of "10%"-

	python train_backdoor_cifar.py --poison-type blend --posion-rate 0.10 --output-dir your/folder/to/save --gpuid 0 

To train a benign model-

    python train_backdoor_cifar.py --poison-type benign --output-dir your/folder/to/save --gpuid 0 


To remove backdoor-
	
	python Remove_Backdoor.py --poison-type blend --val-frac 0.01 --output-dir your/folder/to/save --gpuid 0 




2. Change the training code structure. Up and down. Run at least for four attacks (Blend, SIG, CLB, Trojan, Badnets). Comment the code properly. 

3. List the packages that needs to be installed.


3.1. For Dynamic, the posion test file contains 10000 test data (all labeled as the target label). It should be 9000 posion, 1000 clean. And we should include only this 9000 as the poison data. 
 

4. You have 3 hourse to do all this. 

