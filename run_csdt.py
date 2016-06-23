import numpy as np
from costcla.datasets.base import load_bankmarketing
from costcla.datasets.base import load_creditscoring1
from costcla.datasets.base import load_creditscoring2
from costcla.datasets.base import load_creditgerman
from costcla.utils.dataset_folds import create_all_folds
from costcla.models import CostSensitiveDecisionTreeClassifier
from costcla.metrics import savings_score
from costcla.metrics import cost_loss

num_folds = 10
train_ratio = 0.8
min_samples_split = 10
#data = load_creditscoring1()
data = load_creditscoring2()
#data = load_creditgerman()
#data = load_bankmarketing()
ds = create_all_folds(data, num_folds, train_ratio, 0)

csdt = CostSensitiveDecisionTreeClassifier(min_samples_split=min_samples_split, min_gain=0)#pruned=False)#, min_samples_leaf=0, min_gain=0.0001) # min_samples_leaf=0


cost = 0
savings = 0
size = 0
for key, fold in ds.folds.items():
    tree = csdt.fit(fold.x_train, fold.y_train, fold.cost_mat_train)
    y_pred = tree.predict(fold.x_test)
    curr_cost = cost_loss(fold.y_test, y_pred, fold.cost_mat_test)
    curr_savings = savings_score(fold.y_test, y_pred, fold.cost_mat_test)
    cost += curr_cost
    savings += curr_savings
    size += tree.tree_.n_nodes
    
    print (key, curr_cost, curr_savings, tree.tree_.n_nodes)

print ("Summary:", cost/len(ds.folds), savings/len(ds.folds), size/len(ds.folds))
  

