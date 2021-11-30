# Influence-guided Data Augmentation for Neural Tensor Completion (ACM CIKM 2021)

Overview
---------------
**Influence-guided Data Augmentation for Neural Tensor Completion**  
[Sejoon Oh](https://sejoonoh.github.io/), [Sungchul Kim](https://sites.google.com/site/subright/), [Ryan A. Rossi](http://ryanrossi.com/), and [Srijan Kumar](https://www.cc.gatech.edu/~srijan/)  
*[ACM International Conference on Information and Knowledge Management (CIKM)](https://www.cikm2021.org/), 2021*  

[[Link to the paper PDF](https://arxiv.org/pdf/2108.10248.pdf)]

[[Link to video](https://youtu.be/QbFAYk_Sbm0)]

**DAIN** is an influence-guided data augmentation framework for enhancing the accuracy of neural tensor completion. You can predict missing values of tensors (or multi-dimensional data) more accurately with DAIN. It finds high-quality data augmentation by combining important entities from each dimension via influence functions. To apply DAIN to your model, you can adjust your model to DAIN by saving model gradients (please see the src/model.py and src/main.py to see how to do this).  

If you make use of this code, paper, or the datasets in your work, please cite the following paper:
```
 @inproceedings{oh2021influence,
	title={Influence-guided Data Augmentation for Neural Tensor Completion},
	author={Oh, Sejoon and Kim, Sungchul and Rossi, Ryan and Kumar, Srijan},
	booktitle={Proceedings of the 30th ACM International Conference on Information and Knowledge Management},
	year={2021},
	organization={ACM}
 }
```

Usage
---------------

The detailed execution procedure of **DAIN** is given as follows.

1) Install all required libraries by "pip install -r requirements.txt" (Python 3.6 or higher version is required).
2) "src/main.py" is the main source code for DAIN, and you can execute it with arguments.
3) "python src/main.py [arguments]" will execute DAIN with arguments, and specific information of the arguments are as follows.

````
--path:	path of an input tensor (e.g., data/synthetic_10K.tensor)
--epochs: number of epochs for training (default: 50)
--batch_size: size of mini-batches used for training (default: 1024)
--layers: layer structure for the MLP model (default: 150x1024x1024x128)
--lr: learning rate for training (default: 0.001)
--verbose: show training progress per every X epochs (default: 5)
--gpu: GPU number will be used for experiments (default: 0)
--output: name of the output log file (e.g., demo.txt)
--train_ratio: ratio of training data (default: 0.9)
````

Demo
---------------
To run the demo, please follow the following procedure. **DAIN** demo will be executed with a synthetic tensor with 10,000 nonzeros.

	1. Check permissions of files (if not, use the command "chmod 777 *")
	2. Execute "./demo.sh"
	3. Check "output/demo.txt" for the demo result of DAIN
  
  
Datasets
---------------
The datasets used in the paper are available at [this link](https://drive.google.com/file/d/1i-zZPzOG_uId-891ueo5yB32A2Kv271L/view).  
The input data format must be tab-separated, integer-type for indices, float-type for values (see the data/synthetic_10K.tensor file).
