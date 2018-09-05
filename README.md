# PumpItUp
Pump It Up: Data Mining the Water Table Competition

This repository implements a stacked classifier within the Scikit-Learn framework.  The StackClassifier class can use any of the classifiers in Scikit-Learn and blends the predictions of the individual classifiers.  The train model script uses data from the DataDriven competition:

https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/

A data pipeline transforms the raw inputs, create features, and trains the stacked classifier.  This model placed in the top 5% of the original competition.
