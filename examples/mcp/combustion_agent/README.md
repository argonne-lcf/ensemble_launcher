This example shows how to run a simple local agent on Aurora supercomputer from the login node.
The example Gemini-2.5 pro and langgraph. General steps are as follows.

```bash
qsub submit.sh
```

Wait for the job to start and find the head node using 

```bash
qstat -f <job id>
```

Replace the <username> and <head node> in the agent.py.

```bash
python3 agent.py
```
