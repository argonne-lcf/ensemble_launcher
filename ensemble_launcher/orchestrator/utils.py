load_str = "import base64, json, socket; "\
           "from datetime import datetime; "\
           "from ensemble_launcher.orchestrator.worker import Worker; "\
           "from ensemble_launcher.orchestrator.master import Master; "\
           "hostname = socket.gethostname(); "\
           "start = datetime.now(); "\
           "all_children_dict = json.loads(base64.b64decode(json_str_b64).decode('utf-8')); "\
           "common_keys = common_keys_str.split(','); "\
           "child_dict = {key: all_children_dict[key][hostname] if key not in common_keys and isinstance(all_children_dict[key], dict) and hostname in all_children_dict[key] else all_children_dict[key] for key in all_children_dict.keys()}; "\
           "after_deserialization = datetime.now(); "\
           "child_obj = Worker.fromdict(child_dict) if child_dict['type'] == 'Worker' else Master.fromdict(child_dict); "\
           "child_obj.run(); "\
           "print(child_obj.node_id, start, after_deserialization);"

async_load_str = "import base64, json, socket, asyncio; "\
           "from datetime import datetime; "\
           "import os; "\
           "from ensemble_launcher.orchestrator.async_worker import AsyncWorker; "\
           "from ensemble_launcher.orchestrator.async_master import AsyncMaster; "\
           "from ensemble_launcher.orchestrator.async_workstealing_worker import AsyncWorkStealingWorker; "\
           "from ensemble_launcher.orchestrator.async_workstealing_master import AsyncWorkStealingMaster; "\
           "hostname = socket.gethostname(); "\
           "start = datetime.now(); "\
           "all_children_dict = json.loads(base64.b64decode(json_str_b64).decode('utf-8')); "\
           "common_keys = common_keys_str.split(','); "\
           "pals_id=os.environ.get('PALS_LOCAL_RANKID',None); "\
           "mpi_id=os.environ.get('MPI_LOCALRANKID',pals_id); "\
           "idx = 0 if mpi_id is None else int(mpi_id); "\
           "child_dict = {key: all_children_dict[key][hostname][idx] if key not in common_keys and isinstance(all_children_dict[key], dict) and hostname in all_children_dict[key] else all_children_dict[key] for key in all_children_dict.keys()}; "\
           "after_deserialization = datetime.now(); "\
           "class_map = {'AsyncWorker': AsyncWorker, 'AsyncMaster': AsyncMaster, 'AsyncWorkStealingWorker': AsyncWorkStealingWorker, 'AsyncWorkStealingMaster': AsyncWorkStealingMaster}; "\
           "child_obj = class_map[child_dict['type']].fromdict(child_dict); "\
           "asyncio.run(child_obj.run()); "\
           "print(child_obj.node_id, start, after_deserialization);"

# Simpler load_str for sequential single-child launches (no per-host dictionary needed)
simple_load_str = "import base64, json; "\
           "from datetime import datetime; "\
           "from ensemble_launcher.orchestrator.worker import Worker; "\
           "from ensemble_launcher.orchestrator.master import Master; "\
           "start = datetime.now(); "\
           "child_dict = json.loads(base64.b64decode(json_str_b64).decode('utf-8')); "\
           "after_deserialization = datetime.now(); "\
           "child_obj = Worker.fromdict(child_dict) if child_dict['type'] == 'Worker' else Master.fromdict(child_dict); "\
           "child_obj.run(); "\
           "print(child_obj.node_id, start, after_deserialization);"

async_simple_load_str = "import base64, json, asyncio; "\
           "from datetime import datetime; "\
           "from ensemble_launcher.orchestrator.async_worker import AsyncWorker; "\
           "from ensemble_launcher.orchestrator.async_master import AsyncMaster; "\
           "from ensemble_launcher.orchestrator.async_workstealing_worker import AsyncWorkStealingWorker; "\
           "from ensemble_launcher.orchestrator.async_workstealing_master import AsyncWorkStealingMaster; "\
           "start = datetime.now(); "\
           "child_dict = json.loads(base64.b64decode(json_str_b64).decode('utf-8')); "\
           "after_deserialization = datetime.now(); "\
           "class_map = {'AsyncWorker': AsyncWorker, 'AsyncMaster': AsyncMaster, 'AsyncWorkStealingWorker': AsyncWorkStealingWorker, 'AsyncWorkStealingMaster': AsyncWorkStealingMaster}; "\
           "child_obj = class_map[child_dict['type']].fromdict(child_dict); "\
           "asyncio.run(child_obj.run()); "\
           "print(child_obj.node_id, start, after_deserialization);"

# File-based variants: read JSON from a temp file instead of inlining base64.
# Use these when the serialised payload is too large for a command-line argument.
async_load_str_file = "import json, socket, asyncio; "\
           "from datetime import datetime; "\
           "import os; "\
           "from ensemble_launcher.orchestrator.async_worker import AsyncWorker; "\
           "from ensemble_launcher.orchestrator.async_master import AsyncMaster; "\
           "from ensemble_launcher.orchestrator.async_workstealing_worker import AsyncWorkStealingWorker; "\
           "from ensemble_launcher.orchestrator.async_workstealing_master import AsyncWorkStealingMaster; "\
           "hostname = socket.gethostname(); "\
           "start = datetime.now(); "\
           "all_children_dict = json.load(open(json_file_path)); "\
           "common_keys = common_keys_str.split(','); "\
           "pals_id=os.environ.get('PALS_LOCAL_RANKID',None); "\
           "mpi_id=os.environ.get('MPI_LOCALRANKID',pals_id); "\
           "idx = 0 if mpi_id is None else int(mpi_id); "\
           "child_dict = {key: all_children_dict[key][hostname][idx] if key not in common_keys and isinstance(all_children_dict[key], dict) and hostname in all_children_dict[key] else all_children_dict[key] for key in all_children_dict.keys()}; "\
           "after_deserialization = datetime.now(); "\
           "class_map = {'AsyncWorker': AsyncWorker, 'AsyncMaster': AsyncMaster, 'AsyncWorkStealingWorker': AsyncWorkStealingWorker, 'AsyncWorkStealingMaster': AsyncWorkStealingMaster}; "\
           "child_obj = class_map[child_dict['type']].fromdict(child_dict); "\
           "asyncio.run(child_obj.run()); "\
           "print(child_obj.node_id, start, after_deserialization);"

async_simple_load_str_file = "import json, asyncio; "\
           "from datetime import datetime; "\
           "from ensemble_launcher.orchestrator.async_worker import AsyncWorker; "\
           "from ensemble_launcher.orchestrator.async_master import AsyncMaster; "\
           "from ensemble_launcher.orchestrator.async_workstealing_worker import AsyncWorkStealingWorker; "\
           "from ensemble_launcher.orchestrator.async_workstealing_master import AsyncWorkStealingMaster; "\
           "start = datetime.now(); "\
           "child_dict = json.load(open(json_file_path)); "\
           "after_deserialization = datetime.now(); "\
           "class_map = {'AsyncWorker': AsyncWorker, 'AsyncMaster': AsyncMaster, 'AsyncWorkStealingWorker': AsyncWorkStealingWorker, 'AsyncWorkStealingMaster': AsyncWorkStealingMaster}; "\
           "child_obj = class_map[child_dict['type']].fromdict(child_dict); "\
           "asyncio.run(child_obj.run()); "\
           "print(child_obj.node_id, start, after_deserialization);"