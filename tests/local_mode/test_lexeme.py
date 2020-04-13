import syft as sy
import torch
import syfertext
import numpy as np

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)