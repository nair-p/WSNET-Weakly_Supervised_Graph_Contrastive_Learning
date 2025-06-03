To display the characteristics of the real-world graph datasets, run,
```
python data_characteristics.py
```

To run WSNET on the real-world graphs, run
```
python main_real_world.py --data_path data/liar_new/liar_new_data_graph.pkl --results_path results/liar_new/wsgcl_results.pkl --config_file config.json
```

To run a different dataset, choose one of `['liar_new', 'asw_real', 'asw_synth']` and change to the appropriate data pickle file. Also change the results path. The same config file can be used. `Cora-WS` had to be omitted due to large file size.

Before running WSNET on the synthetic graphs, create the graph by running:
```
python dataset.py Citeseer
```
Choose any dataset from `['Citeseer', 'Coauthor', 'Amazon', 'Disease', 'Wisconsin']`.

One the data has been created, run,

```
python main_synthetic.py --data_path data/Citeseer/all_data.pkl --results_path results/citeseer --config_file config.json
```

The LFs used for obtaining the weak labels are provided in the readme file inside each of the corresponding data fo
For example, more information on the weak labels for LIAR-WS dataset can be found in `data/liar_new/LF_README.md`.
