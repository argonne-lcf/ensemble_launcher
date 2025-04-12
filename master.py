from worker import *
from Node import *

class master(Node):
    def __init__(self,master_id:str,comm_config:dict={"comm_layer":"multiprocessing",
                                        "parents":{},
                                        "children":{}}):
        super().__init__(master_id,comm_config)        