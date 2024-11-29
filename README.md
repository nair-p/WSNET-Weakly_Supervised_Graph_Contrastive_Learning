To display the characteristics of the real-world graph datasets, run,
```
python data_characteristics.py
```

To run WSGCL on the real-world graphs, run
```
python main_real_world.py --data_path data/cor_ws/all_data_full.pkl --results_path results/cora_ws/wsgcl_results.pkl --config_file config.json
```

To run a different dataset, replace "cora_ws" with one of ['liar_new', 'asw_real', 'asw_synth'] and change to the appropriate data pickle file. Also change the results path. The same config file can be used.

To run WSGCL on the synthetic graphs, run
```
python main_synthetic.py --data_path data/Citeseer/all_data.pkl --results_path results/citeseer --config_file config.json
```

Choose a dataset from ['Citeseer', 'Coauthor', 'Amazon', 'Disease', 'Wisconsin'].

The LFs used for obtaining the weak labels are provided in the readme file inside each of the corresponding data folders. 

For example, more information on the weak labels for LIAR-WS dataset can be found in `data/liar_new/LF_README.md`.