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