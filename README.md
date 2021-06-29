# border-graph


## Process
1. **train_munis.py**: Train the multiscale RAM model 
2. **extract_locations.py**: Use the trained weights from the RAM model to extract the glimpse locations for every image
3. **extract_features.py**: Use the glimpse locations from extract_locations to extract the FC glimpse layer for each image
4. **prep_features.py**: Convert the features from each of the 4 glimpses extracted in extract_features into a dictionary with the values being a combined list of all features
5. **construct_graph.py**: Construct a dictionary with keys = municipality ID and values = {'x':..., 'label':..., 'neighbors':...}
6. **train.py**: Train the borderGraph model using both imagery & census features


## Results

|  kfold   |  Final Validation MAE  |  Final Validation R2  |  Final All MAE  |  Final All R2  |
|----------|------------------------|-----------------------|-----------------|----------------|
|  kfold1  |      925.6048965	    |     0.8378669189	    |    831.006557	  |  0.8515901437  |
|  Kfold2  |	  1084.615131	    |     0.4999673727      |    989.885597   |  0.7803235397  |

