# Example: training on a toy dataset

The dataset `train_set.json` includes 3 protein chains. Start training a new model by running:

```
mkdir out
python ../../alphadock/train.py train_set.json --data_dir train_data --out_dir out
```    

Change the model configuration by supplying `--config_update_json`. The format is the 
same as `config = {..}` in `alphafold/config.py`. For example, the following will change
extra MSA size from 1024 to 512 and reduce the size of 1D representation and the number 
of Evoformer and Structure blocks to simplify the model:

```
python ../../alphadock/train.py train_set.json --data_dir train_data --out_dir out --config_update_json config_update.json
```    

If you stop and resume the training, the last saved epoch in `out/` will be used by default.
You can change it by using `--model_pth`.    

Type `python ../../alphadock/train.py --help` to 
view all available options. 
