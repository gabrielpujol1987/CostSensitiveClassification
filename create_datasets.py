from costcla.datasets.base import load_bankmarketing
from costcla.datasets.base import load_creditscoring1
from costcla.datasets.base import load_creditscoring2
from costcla.datasets.base import load_creditgerman
from costcla.datasets.base import load_kdd98
from costcla.datasets.base import load_skin
from costcla.datasets.base import load_diabetes
from costcla.utils.dataset_folds import create_all_folds
from costcla.create_cplusplus import create_c_files

num_folds = 10
train_ratio = 0.8
dataset_name = "tmp"
#data = load_creditscoring1(useCost=False)
#data = load_creditscoring2(nominal_attributes=True, useCost=False)
#data = load_creditgerman(nominal_attributes=True, useCost=False)
#data = load_bankmarketing(nominal_attributes=True, useCost=False)
data = load_kdd98(as_benefit=False, useCost=False)
#data = load_skin()
#data = load_diabetes(dataset_name)

ds = create_all_folds(data, num_folds, train_ratio, 0)
create_c_files(ds, dataset_name, "C:\\Users\\Daniel\\Documents\\DataSets\\Python\\tmp\\")
