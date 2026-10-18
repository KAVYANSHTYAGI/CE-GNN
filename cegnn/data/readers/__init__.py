"""Dataset adapters. Importing this package registers every built-in dataset."""
from .base import DatasetAdapter, SampleRef, available_datasets, get_adapter, register  # noqa: F401
from . import ncaltech101, ncars, dvsgesture, cifar10dvs, dvslip  # noqa: F401  (registration side effect)
