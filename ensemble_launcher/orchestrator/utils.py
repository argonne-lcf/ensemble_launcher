load_str="from datetime import datetime;"\
         "import socket, os, json;"\
         "from ensemble_launcher.orchestrator.worker import Worker;"\
         "from ensemble_launcher.orchestrator.master import Master;"\
         "hostname = socket.gethostname();"\
         "start=datetime.now();"\
         "fname = os.path.join(dirname,f\'{hostname}_child_obj.json\');"\
         "f = open(fname, 'r');"\
         "child_dict = json.load(f);"\
         "f.close();"\
         "after_deserialization = datetime.now();"\
         "child_obj = Worker.fromdict(child_dict) if child_dict['type'] == 'Worker' else Master.fromdict(child_dict);"\
         "child_obj.run();" \
         "print(child_obj.node_id, start, after_deserialization);"