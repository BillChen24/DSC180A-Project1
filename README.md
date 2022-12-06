# DSC180A-Project1

In the project we build a Random Forest Classifier and multiple Optimal Transportation Transformers.

In running the code below, you will get a printed report for Applying the Random Forest Transfoer on data transformed by different OT transformer.


```
mkdir result # use if result folder is not in this project directory
python run.py test
```
Our Original Data set is found on CIFAR 10 website https://www.cs.toronto.edu/~kriz/cifar.html.
However, for the purpose of the experiment, we add some other variation of the same dataset. Please access the true data set from the google drive:
https://drive.google.com/drive/folders/1_KPDpP_jgTCV3B7ZfI-ktNwB1_1kcdSC?usp=sharing

Once the Whole data file is downloaded, place the data folder inside folder of this project.
Run 
```
python run.py all
```
To generate the report on the Whole dataset

After all, run following command lines to remove the data and output files from runing through this project.
```
rm -r result
rm -r data
```
