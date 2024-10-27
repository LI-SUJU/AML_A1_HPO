### Plots
Kindly check all plots in ./plots after running python files
### Random search
- LCDB:
```bash
python example_run_experiment.py --max_anchor_size 4000
```
- dataset-6:
```bash
python example_run_experiment.py --dataset dataset-6 --max_anchor_size 8000
```
- dataset-11:
```bash
python example_run_experiment.py --dataset dataset-11 --max_anchor_size 4000
```
- dataset-1457:
```bash
python example_run_experiment.py --dataset dataset-1457 --max_anchor_size 2000
```

### SMBO
- LCDB:
```bash
python run_experiment_smbo.py
```
- dataset-6:
```bash
python run_experiment_smbo.py --dataset dataset-6
```
- dataset-11:
```bash
python run_experiment_smbo.py --dataset dataset-11
```
- dataset-1457:
```bash
python run_experiment_smbo.py --dataset dataset-1457
```

### Successive halving
In order to generate the plots shown in our report, be mindful about the arguements:
- LCDB:
```bash
python succesive_halving.py
```
- dataset-6:
```bash
python succesive_halving.py --dataset dataset-6 --budget 1280000
```
- dataset-11:
```bash
python succesive_halving.py --dataset dataset-11 --budget 80000
```
- dataset-1457:
```bash
python succesive_halving.py --dataset dataset-1457 --budget 40000
```
