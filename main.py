# external import
import yaml
import h5py
import glob

# internal import
import utils.DataPreprocessing as datapre

# load param configuration
with open('config/param.yml', 'r') as yml_file:
    cfg = yaml.safe_load(yml_file)

# load config params
sDatafile = cfg['sDatafile']

x_train = []
y_train = []
x_test = []

if glob.glob(sDatafile):
    f = h5py.File(sDatafile, 'r')
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']

else:
    x_train, y_train, x_test = datapre.fPreprocessData(cfg)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)