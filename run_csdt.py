import numpy as np
from costcla.datasets.base import load_bankmarketing
from costcla.datasets.base import load_creditscoring1
from costcla.datasets.base import load_creditscoring2
from costcla.datasets.base import load_creditgerman
from costcla.utils.dataset_folds import create_all_folds
from costcla.models import CostSensitiveDecisionTreeClassifier
from costcla.metrics import savings_score
from costcla.metrics import cost_loss

def printTree(tree, prefix, feature_names):
    if tree['split'] == -1:
        print(': ' + str(tree['y_pred']), end='')
    else:
        print('\n' + prefix + feature_names[tree['split'][0]] + ' <= ' + str(tree['split'][1]), end='')
        printTree(tree['sl'], prefix + '| ', feature_names)
        print('\n' + prefix + feature_names[tree['split'][0]] + ' > ' + str(tree['split'][1]), end='')
        printTree(tree['sr'], prefix + '| ', feature_names)

num_folds = 10
train_ratio = 0.8
#data = load_creditscoring1()
#data = load_creditscoring2()
#data = load_creditgerman()
data = load_bankmarketing()
ds = create_all_folds(data, num_folds, train_ratio, 0)

csdt = CostSensitiveDecisionTreeClassifier(
    criterion='direct_cost',
    criterion_weight=False,
    num_pct=20000,
    max_features=None,
    max_depth=None,
    min_samples_split=30,
    min_samples_leaf=1,
    min_gain=0.01,
    pruned=False)


cost = 0
savings = 0
size = 0
for key, fold in ds.folds.items():
    tree = csdt.fit(fold.x_train, fold.y_train, fold.cost_mat_train)
    print('Fold: ' + str(key))
    printTree(tree.tree_.tree, '', ds.feature_names)
    print('\n')
    y_pred = tree.predict(fold.x_test)
    curr_cost = cost_loss(fold.y_test, y_pred, fold.cost_mat_test)
    curr_savings = savings_score(fold.y_test, y_pred, fold.cost_mat_test)
    cost += curr_cost
    savings += curr_savings
    size += tree.tree_.n_nodes
    
    print (key, curr_cost, curr_savings, tree.tree_.n_nodes)

print ("Summary:", cost/len(ds.folds), savings/len(ds.folds), size/len(ds.folds))
