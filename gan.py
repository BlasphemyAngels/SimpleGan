# !/usr/bin/python3
# _*_coding: utf-8_*_

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

mu, sigma = -1, 1
xs = np.linspace(-5, 5, 1000)
plt.plot(xs, norm.pdf(xs, loc=mu, scale=sigma))
