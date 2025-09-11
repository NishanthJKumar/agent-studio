# profiles/slurm/submit.py
import sys, shlex, subprocess, json

# Snakemake passes a JSON on stdin with job properties
job = json.loads(sys.stdin.read())

res = job.get("resources", {})
threads = job.get("threads", 1)

time  = res.get("time", "02:00:00")
mem   = res.get("mem", "8G")
gpus  = int(res.get("gpus", 0))
qos   = res.get("qos", "")
acct  = res.get("account", "")
part  = res.get("partition", "")

gpu_flag = [] if gpus == 0 else ["--gres", f"gpu:{gpus}"]
qos_flag = [] if not qos else ["--qos", qos]
acct_flag = [] if not acct else ["--account", acct]
part_flag = [] if not part else ["--partition", part]

cmd = [
  "sbatch",
  "--parsable",
  "--cpus-per-task", str(threads),
  "--time", time,
  "--mem", mem,
  *gpu_flag, *qos_flag, *acct_flag, *part_flag,
]

# Snakemake gives us the jobscript path as the last CLI arg
cmd.append(sys.argv[-1])

print(subprocess.check_output(cmd, text=True).strip())
