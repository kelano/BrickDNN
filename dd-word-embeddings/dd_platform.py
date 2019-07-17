import sys


MATPLOTLIB_USE = 'TkAgg' if sys.platform == 'darwin' else 'Agg'
DATA_LOC = '/Users/kelleng/hover-workspace/bastion/user/dd-data' if sys.platform == 'darwin' else '/home/ec2-user/dd-data'
# DATA_LOC = '/Users/kelleng/data/dd' if sys.platform == 'darwin' else '/home/ec2-user/dd-data'
