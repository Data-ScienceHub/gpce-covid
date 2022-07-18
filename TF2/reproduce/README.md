# Introduction

This folder reproduces the old experiments with TFT tensorflow 2. The experiments are covid prediction based on

* Top 500 county
* Rurality cut
  * Stratum 1: MADgroup [1,2) with 3.501 < MedianRUCA <= 4.5. 70 counties.
  * Stratum 2: MADgroup [1,2) with 4.501 < MedianRUCA <= 5.0 . 79 counties.
  * Stratum 3: MADgroup [2,3) with 3.5 < MedianRUCA <= 5.0 . 65 counties.
  
All experiments are run for 60 epoch, no early stopping.

To run these

* clone the repository.
* if you are running on colab you should already have all the libraries.
* if you are using Rivanna or local system
  * setup the python virtual env using the [environment.yml](../v0/script/../environment.yml) in anaconda. 
  * activate the virtual environement if you want to run scripts.
  * or in colab session add the created environment as kernel.
* You can use the notebook of each rurality folder to run them
* Or use the [train.py](../v0/script/train.py) script to run it from terminal.
