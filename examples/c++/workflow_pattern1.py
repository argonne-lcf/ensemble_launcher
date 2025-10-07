from ensemble_launcher import EnsembleLauncher, write_results_to_json
from ensemble_launcher.config import SystemConfig
import os
import logging

##configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


##ensemble definition
ensemble_info = {
    "cpu_ensemble" : {
        "cmd_template": "./serial_example {sleep_time} true",
        "sleep_time": [1]*1024,
        "relation":"one-to-one",
        "nnodes":1,
        "ppn":1,
    },
    "gpu_ensemble": {
        "cmd_template": "./serial_example {sleep_time} true",
        "sleep_time": [1]*1024,
        "relation":"one-to-one",
        "nnodes":1,
        "ppn":1,
        "ngpus_per_process":1
    }
}

##create the system config
cpus = list(range(104))
cpus.pop(52) #can't use these cores on Aurora
cpus.pop(0) #can't use these cores on Aurora
sys_config = SystemConfig(
    name="Aurora",
    cpus = cpus,
    gpus = list(range(12))
)

##set some environment variables
os.environ["ZE_FLAT_DEVICE_HIERARCHY"]="FLAT"

##create the EnsembleLauncher class
el = EnsembleLauncher(
    ensemble_file=ensemble_info,
    system_config=sys_config,
)

results = el.run() ##blocking call
write_results_to_json(results)
