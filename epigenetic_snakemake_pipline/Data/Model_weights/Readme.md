# Readme file for OriGENE weights folder
   
This folder contains hdf5 files with different sets of weights for the OriGENE network. Although the model definition is the same, depending on the input shape (number of basepairs: 5000/40000, number of channels(1,2 or 6)) the architecture will vary slightly:

Here we have as examples the weights of the models trained on the first fold of the data. Attached are:

- OriGENE_Liver_I_fold_1.hdf5. Shape: (None,40000,2)
- OriGENE_Lung_I_fold_1.hdf5. Shape: (None,40000,2)
- OriGENE_NT2D1_I_fold_1.hdf5. Shape: (None,40000,6)

- OriGENE_Lung_I_40kbp_singletrack_1 (1).hdf5. Shape: (None,40000,1)
- OriGENE_Lung_I_cross_tissue_5kbp_fold_1.hdf5. Shape (None,5000,1)



These files can be loaded and used to predict new sequences once the OriGENE network has been defined (In the main ipynb file the OriGENE model is called Model_OriGENE). A fragment of the code could look like:

```python

Input_shape = xtrn_CD_prediction.shape[1:] # Will vary depending on the length of the sequences and number of channels. This will affect the model one can load.
model = Model_OriGENE(Input_shape)
model.load_weights('OriGENE_Lung_I_40kbp_singletrack_1.hdf5')
metrics_names = model.metrics_names

y_pred = model.predict(xval).flatten()
y_true = yval.flatten()

```
