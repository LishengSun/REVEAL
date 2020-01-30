import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../env/')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents/')))

import MNIST_env
import segment_env
import DQN