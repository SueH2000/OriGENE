# Model weights

This folder contains example HDF5 weight files for trained OriGENE models.

The exact model shape depends on the input configuration, especially:

- sequence length, such as 5,000 bp or 40,000 bp,
- number of input channels, such as 1, 2, or 6.

Example files included here:

- `OriGENE_Liver_I_fold_1.hdf5`
- `OriGENE_Lung_I_fold_1.hdf5`
- `OriGENE_NT2D1_I_fold_1.hdf5`
- `OriGENE_Lung_I_40kbp_singletrack_1 (1).hdf5`
- `OriGENE_Lung_I_cross_tissue_5kbp_fold_1.hdf5`

## How to use these files

1. Define the OriGENE model in the notebook.
2. Make sure the input shape matches the weight file you want to load.
3. Load the weights before prediction or evaluation.

Example:

```python
Input_shape = xtrn_CD_prediction.shape[1:]
model = Model_OriGENE(Input_shape)
model.load_weights("Data/Model_weights/OriGENE_Lung_I_fold_1.hdf5")

y_pred = model.predict(xval).flatten()
y_true = yval.flatten()
```
