from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

_MPI_FLAVOR_DEFAULTS: Dict[str, Dict] = {
    # Intel MPI / Cray MPICH (default — matches existing EL behaviour)
    "intel": {
        "launcher": "mpirun",
        "nprocesses_flag": "-np",
        "processes_per_node_flag": "-ppn",
        "hosts_flag": "--hosts",
        "hostfile_flag": "--hostfile",
        "rankfile_flag": "-rankfile",
        "cpu_bind_flag": "--cpu-bind",
        "cpu_bind_method": "list",
    },
    # Cray PALS (mpiexec on Frontier/Aurora)
    "cray-pals": {
        "launcher": "mpiexec",
        "nprocesses_flag": "-n",
        "processes_per_node_flag": "--ppn",
        "hosts_flag": "--hosts",
        "hostfile_flag": "--hostfile",
        "rankfile_flag": None,
        "cpu_bind_flag": "--cpu-bind",
        "cpu_bind_method": "list",
    },
    # Open MPI (mpirun/mpiexec)
    "openmpi": {
        "launcher": "mpirun",
        "nprocesses_flag": "-np",
        "processes_per_node_flag": "--npernode",
        "hosts_flag": "-host",
        "hostfile_flag": "-hostfile",
        "rankfile_flag": "--rankfile",
        "cpu_bind_flag": "--bind-to",
        "cpu_bind_method": "bind-to",
    },
    # MPICH (hydra launcher)
    "mpich": {
        "launcher": "mpiexec",
        "nprocesses_flag": "-n",
        "processes_per_node_flag": "--ppn",
        "hosts_flag": "-hosts",
        "hostfile_flag": "--hostfile",
        "rankfile_flag": None,
        "cpu_bind_flag": "--cpu-bind",
        "cpu_bind_method": "list",
    },
    # SLURM srun — no explicit host flags, SLURM allocation controls placement
    "srun": {
        "launcher": "srun",
        "nprocesses_flag": "-n",
        "processes_per_node_flag": "--ntasks-per-node",
        "hosts_flag": None,
        "hostfile_flag": "--nodefile",
        "rankfile_flag": None,
        "cpu_bind_flag": "--cpu-bind",
        "cpu_bind_method": "list",
    },
    # Cray aprun (legacy XC systems)
    "aprun": {
        "launcher": "aprun",
        "nprocesses_flag": "-n",
        "processes_per_node_flag": "-N",
        "hosts_flag": None,
        "hostfile_flag": None,
        "rankfile_flag": None,
        "cpu_bind_flag": "-cc",
        "cpu_bind_method": "list",
    },
    # IBM Spectrum MPI jsrun (Summit/Frontier-like)
    "jsrun": {
        "launcher": "jsrun",
        "nprocesses_flag": "-n",
        "processes_per_node_flag": "-r",  # resource sets per host
        "hosts_flag": None,
        "hostfile_flag": None,
        "rankfile_flag": None,
        "cpu_bind_flag": "--bind",
        "cpu_bind_method": "none",
    },
}


class MPIConfig(BaseModel):
    """MPI launcher configuration.

    Captures the differences between MPI implementations (Intel MPI, Open MPI,
    MPICH, Cray PALS, srun, aprun, jsrun …).  Set ``flavor`` to auto-populate
    sensible defaults, then override individual fields as needed.
    """

    model_config = ConfigDict(extra="allow")

    # ------------------------------------------------------------------ #
    # Convenience shortcut — sets all flag defaults for known launchers.  #
    # Individual fields below always take precedence over flavor defaults. #
    # ------------------------------------------------------------------ #
    flavor: Optional[
        Literal["intel", "cray-pals", "openmpi", "mpich", "srun", "aprun", "jsrun"]
    ] = None

    # ------------------------------------------------------------------ #
    # Launcher binary                                                      #
    # ------------------------------------------------------------------ #
    launcher: str = "mpirun"  # mpirun | mpiexec | srun | aprun | jsrun …

    # ------------------------------------------------------------------ #
    # Process / rank flags                                                 #
    # ------------------------------------------------------------------ #
    nprocesses_flag: str = "-np"  # -np (Intel/OpenMPI) | -n (srun/aprun)
    processes_per_node_flag: Optional[str] = "-ppn"
    # -ppn | --npernode | --ntasks-per-node | -N | None to omit entirely

    # ------------------------------------------------------------------ #
    # Host / node specification                                            #
    # ------------------------------------------------------------------ #
    hosts_flag: Optional[str] = "--hosts"
    # Flag used for inline comma-separated host list.
    # Set to None for launchers that don't accept host lists (srun, aprun, jsrun).

    hostfile_flag: Optional[str] = "--hostfile"
    # Flag for a file listing hosts (one per line).
    # Set to None for launchers that don't support it.

    hostfile_threshold: int = 256
    # Switch from inline --hosts to --hostfile above this node count.

    rankfile_flag: Optional[str] = "--rankfile"
    # Flag for a rankfile that maps each MPI rank to a specific host+CPU slot.
    # OpenMPI: "--rankfile"  (rank N slot host:core)
    # Intel MPI: "-rankfile"
    # Set to None for launchers that don't support rankfiles (srun, aprun, jsrun).
    # When set, EL will generate and pass a rankfile instead of --hosts/--cpu-bind.

    use_rankfile: bool = False
    # If True, generate a rankfile for fine-grained rank→CPU placement and pass it
    # via rankfile_flag.  Overrides cpu_bind_method when active.

    # ------------------------------------------------------------------ #
    # CPU affinity                                                         #
    # ------------------------------------------------------------------ #
    cpu_bind_flag: str = "--cpu-bind"
    # Flag prefix passed to the launcher.
    # Intel/MPICH/Cray: "--cpu-bind"   →  --cpu-bind list:0:1:2
    # OpenMPI:          "--bind-to"    →  --bind-to core
    # aprun:            "-cc"          →  -cc list:0:1:2
    # jsrun:            "--bind"
    # Set to "" to disable affinity flags entirely.

    cpu_bind_method: Literal["list", "bind-to", "none"] = "list"
    # "list"    → <cpu_bind_flag> list:<core0>:<core1>:…   (Intel/MPICH/Cray/aprun)
    # "bind-to" → --bind-to core  (OpenMPI; map-by is handled separately)
    # "none"    → no CPU-binding flags at all

    openmpi_map_by: str = "slot"
    # OpenMPI --map-by value used when cpu_bind_method == "bind-to".
    # Common values: "slot", "node", "socket", "core", "hwthread",
    #                "slot:PE=<n>" to pin N hardware threads per rank.

    # ------------------------------------------------------------------ #
    # Misc / catch-all                                                     #
    # ------------------------------------------------------------------ #
    extra_launcher_flags: List[str] = Field(default_factory=list)
    # Arbitrary flags always appended right after the launcher binary,
    # before resource flags.  Examples:
    #   ["--oversubscribe"]          – OpenMPI, allow >1 rank per core
    #   ["--allow-run-as-root"]      – OpenMPI in containers
    #   ["--mca", "btl", "^openib"]  – disable legacy OpenIB transport

    @model_validator(mode="before")
    @classmethod
    def apply_flavor_defaults(cls, values):
        """Populate fields from the named flavor unless the caller already set them."""
        flavor = values.get("flavor")
        if flavor and flavor in _MPI_FLAVOR_DEFAULTS:
            for key, default in _MPI_FLAVOR_DEFAULTS[flavor].items():
                if key not in values:
                    values[key] = default
        return values
