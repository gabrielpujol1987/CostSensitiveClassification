from costcla.datasets.base import load_bankmarketing
from costcla.datasets.base import load_creditscoring1
from costcla.datasets.base import load_creditscoring2
from costcla.datasets.base import load_creditgerman
from costcla.datasets.base import load_kdd98
from costcla.datasets.base import load_skin
from costcla.utils.dataset_folds import create_all_folds
from costcla.create_cplusplus import create_c_files

num_folds = 10
train_ratio = 0.8
dataset_name = "skin"
#data = load_creditscoring1()
#data = load_creditscoring2(nominal_attributes=True)
#data = load_creditgerman(nominal_attributes=True)
#data = load_bankmarketing(nominal_attributes=True)
#data = load_kdd98(as_benefit=True)
data = load_skin()

ds = create_all_folds(data, num_folds, train_ratio, 0)
create_c_files(ds, dataset_name, "C:\\Users\\Daniel\\Documents\\DataSets\\Python\\skin\\Folds_10\\")
