# Example: inference using AF2 parameters

Predict structure for `1ezi` from MSA files in a3m format

```
python ../../alphadock/inference.py --a3m_file bfd.mgnify30.metaeuk30.smag30.a3m --a3m_file uniref.a3m ../../download_af2_parameters/params_model_3.pth
```    

This will produce `prediction_000000.pdb`, which is close to the X-ray structure in `1ezi.cif`
