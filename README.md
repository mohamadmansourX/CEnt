# TODO LIST


- [x] Prepare CEM Notebook for images for Julia 
- [x] Train g on f predictions rather than actual labels
- [x] Factuals shouls be sampled s.t. predicted by model as gt
- [x] k = 500 per label
- [ ] time plot distributions


## Recource Methods

- [x] Cote [ANN + Linear]: Done
- [x] Cote [Forest ]: Done
- [x] Dice: Done
- [x] Growing Spheres: Done
- [x] Clue: Done
- [x] CCHVAE: Done
- [x] CRUDS: Done
- [x] Focus: Done
- [x] CEM: Done
- [x] Revisewachter: Done
- [x] Face: Done
- [ ] Feature Tweak: Done
- [ ] Actionable Recourse: Requires some more work to be done
- [ ] Causal Recourse: Using all the RAMs and crashing the session


Over 1000 Gridsearch
{'criterion': 'gini', 'max_depth': 6, 'max_features': 0.8, 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'best'}
{'criterion': 'gini', 'max_depth': 9, 'max_features': 0.8, 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'best'}

```
rm -rf outputs/

rm -rf ~/carla/models/*
```