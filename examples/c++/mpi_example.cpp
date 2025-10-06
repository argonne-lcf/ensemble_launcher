#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <cstdlib>
#include <sched.h>
#include <mpi.h>

// util-linux-2.13-pre7/schedutils/taskset.c
void get_cores(char *str)
{ 
  cpu_set_t mask;
  sched_getaffinity(0, sizeof(cpu_set_t), &mask);

  char *ptr = str;
  int i, j, entry_made = 0;
  for (i = 0; i < CPU_SETSIZE; i++) {
    if (CPU_ISSET(i, &mask)) {
      int run = 0;
      entry_made = 1;
      for (j = i + 1; j < CPU_SETSIZE; j++) {
        if (CPU_ISSET(j, &mask)) run++;
        else break;
      }
      if (!run)
        sprintf(ptr, "%d,", i);
      else if (run == 1) {
        sprintf(ptr, "%d,%d,", i, i + 1);
        i++;
      } else {
        sprintf(ptr, "%d-%d,", i, i + run);
        i += run;
      }
      while (*ptr != 0) ptr++;
    }
  }
  ptr -= entry_made;
  *ptr = 0;
}

int main(int argc, char *argv[])
{
  /*
  A simple MPI application which sleeps for a given time.
  In debug mode, it will print information about the node, CPU and GPU each rank is bound to.
  */
  // Initialize MPI
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm world = MPI_COMM_WORLD;
  MPI_Comm_rank(world, &rank);
  MPI_Comm_size(world, &size);
  //int rankl = int(std::getenv("PALS_LOCAL_RANKID"));

  // Parse command line arguments
  if (argc > 3) {
    printf("Usages:\n");
    printf("%s\n", argv[0]);
    printf("%s <sleep_time>\n", argv[0]);
    printf("%s <sleep_time> <debug true/false>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }
  int sleep_time;
  bool debug;
  if (argc == 1) {
    sleep_time = 10;
    debug = false;
  } else if (argc == 2) {
    sleep_time = atoi(argv[1]);
    debug = false;
  } else {
    sleep_time = atoi(argv[1]);
    debug = strcmp(argv[2], "true") == 0;
  }

  // Print debug information
  if(debug) {
    char hostname[16];
    gethostname(hostname, 16);
    char list_cores[7*CPU_SETSIZE];
    get_cores(list_cores);
    const char* gpu_affinity = std::getenv("ZE_AFFINITY_MASK") ? std::getenv("ZE_AFFINITY_MASK") : std::getenv("CUDA_VISIBLE_DEVICES");
    printf("Hello from rank %d/%d on node %s, CPU cores %s and GPU %s\n", 
            rank, size, hostname, list_cores, gpu_affinity);
    fflush();
  }

  // Sleep for the given time
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(sleep_time);
  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Finalize();
  return 0;
}
