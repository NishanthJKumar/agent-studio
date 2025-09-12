#!/usr/bin/env python3
import os, sys, socket, time, json, subprocess, shlex, signal, atexit, pathlib, argparse

# ---------- helpers ----------
def pick_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

class Runner:
    """Wrap subprocess calls and waits; print-only when dry-run=True."""
    def __init__(self, dry=False):
        self.dry = dry
        self.bg_procs = []

    def run(self, cmd, **kw):
        print(f"[run]{' (dry)' if self.dry else ''} {cmd}", flush=True)
        if self.dry:
            return subprocess.CompletedProcess(cmd, 0)
        return subprocess.run(cmd, shell=True, check=True, **kw)

    def start_bg(self, cmd):
        print(f"[bg]{' (dry)' if self.dry else ''} {cmd}", flush=True)
        if self.dry:
            class _Dummy: pid = 0
            return _Dummy()
        p = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        self.bg_procs.append(p)
        return p

    def wait_http_ok(self, url, total=120, every=2, must_contain=None):
        print(f"[wait]{' (dry)' if self.dry else ''} http {url} total={total}s every={every}s require={must_contain}", flush=True)
        if self.dry:
            return True
        waited = 0
        while waited < total:
            try:
                out = subprocess.run(
                    ["curl", "-sS", "-m", str(every), "-w", "%{http_code}", "-o", "/tmp/_body", url],
                    capture_output=True, text=True
                )
                if out.returncode == 0 and out.stdout.strip() == "200":
                    if must_contain is None:
                        return True
                    try:
                        body = open("/tmp/_body").read()
                    except Exception:
                        body = ""
                    if must_contain in body:
                        return True
            except Exception:
                pass
            time.sleep(every); waited += every
        return False

    def cleanup(self):
        if self.dry:
            return
        for p in self.bg_procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass

def _main(dry=False, dry_touch=False,
          task_path=None, round_idx=None, out_success=None, log_path=None,
          cfg=None, params=None, wc=None):
    R = Runner(dry=dry)

    # pull static stuff
    actor  = cfg["models"]["actor"]
    critic = cfg["models"]["critic"]
    hp     = cfg["hyperparams"]
    paths  = cfg["paths"]

    agent     = hp["agent"]
    prompting = hp["prompting_approach"]
    exp_eps   = hp["exp_episodes"]
    plan_score= hp["critique_approach"]
    plan_gen  = hp["plan_generation_approach"]

    conda_sh  = paths["conda_sh"]
    conda_env = paths["conda_env"]

    # per-job dynamic bits
    data_dir    = params["data_dir"]
    prev_plans  = params.get("prev_plans")
    prev_ckpt   = params.get("prev_ckpt")

    # ports & container names
    env_port = pick_port(); vnc_port = pick_port()
    api_ws   = pick_port(); api_sock = pick_port()
    srv_sock = pick_port(); hf_port  = pick_port()

    server_name = f"agent-studio-server-{os.getpid()}"
    client_name = f"agent-studio-client-{os.getpid()}"

    def _atexit_cleanup():
        # remove enroot containers even in dry? safer to skip in dry.
        if not dry:
            for name in (server_name, client_name):
                subprocess.run(["enroot", "remove", "-f", name], check=False)
        R.cleanup()
    atexit.register(_atexit_cleanup)

    # ensure dirs
    print(f"[mkdir]{' (dry)' if dry else ''} {data_dir}")
    if not dry:
        os.makedirs(data_dir, exist_ok=True)
    print(f"[mkdir]{' (dry)' if dry else ''} {os.path.dirname(out_success)}")
    if not dry:
        os.makedirs(os.path.dirname(out_success), exist_ok=True)

    # 1) Start env server
    R.run(f"enroot remove -f {server_name} || true")
    R.run(f"enroot create -n {server_name} agent-studio-server.sqsh")
    p_server = R.start_bg(
        " ".join([
            "enroot start",
            f"--env VNC_PASSWORD=123456",
            f"--env ENV_SERVER_PORT={env_port}",
            f"--env VNC_PORT={vnc_port}",
            f"--env SERVER_SOCKET={srv_sock}",
            f"--env API_WEB_SOCKET={api_ws}",
            f"--env API_SOCKET={api_sock}",
            "--mount /dev/shm:/dev/shm",
            f"--mount {shlex.quote(os.getcwd())}/agent_studio:/home/ubuntu/agent_studio/agent_studio:rbind,ro",
            f"--mount {shlex.quote(os.getcwd())}/eval_online_benchmarks/files:/home/ubuntu/agent_studio/data:rbind,ro",
            "--root --rw", server_name
        ])
    )

    if not R.wait_http_ok(f"http://127.0.0.1:{env_port}/health", total=180, every=2, must_contain="OK"):
        raise SystemExit("env server did not become healthy")

    # 2) Optional HF server if needed
    if plan_gen != "diversity":
        act = f'. {shlex.quote(conda_sh)} && conda activate {shlex.quote(conda_env)}'
        ckpt_dir = os.path.realpath(prev_ckpt) if prev_ckpt and os.path.exists(prev_ckpt) else ""
        cmd = (
            f'{act} && python scripts/huggingface_model_server.py '
            f'--model {shlex.quote(critic)} --port {hf_port} '
            f'{f"--model_weights_path {shlex.quote(ckpt_dir)}" if ckpt_dir else ""}'
        )
        p_hf = R.start_bg(cmd)
        if not R.wait_http_ok(f"http://127.0.0.1:{hf_port}/ready", total=300, every=5, must_contain='"status":"ready"'):
            raise SystemExit("HF server did not become ready")

    # 3) Run client once
    R.run(f"enroot remove -f {client_name} || true")
    R.run(f"enroot create -n {client_name} agent-studio-client.sqsh")

    client_cmd = f"""
      set -e
      cd /home/ubuntu/agent_studio
      python agent_studio/apps/online_exploration.py \
        --task_configs_path {shlex.quote(task_path)} \
        --agent {shlex.quote(agent)} \
        --prompting_approach {shlex.quote(prompting)} \
        --model {shlex.quote(actor)} \
        --vnc_port {vnc_port} \
        --env_server_port {env_port} \
        {'--model_server http://0.0.0.0:'+str(hf_port) if plan_gen != 'diversity' else ''} \
        --remote \
        --exp_episodes {exp_eps} \
        --plan_scoring_approach {shlex.quote(plan_score)} \
        --finetuning_data_path {shlex.quote(data_dir)} \
        {('--previous_plans_data_path ' + shlex.quote(prev_plans)) if prev_plans else ''} \
        --plan_proposing_approach {shlex.quote(plan_gen)} \
        --save_finetuning_data
    """
    R.run(" ".join([
        "enroot start",
        f"--env VNC_PASSWORD=123456",
        f"--env ENV_SERVER_PORT={env_port}",
        f"--env VNC_PORT={vnc_port}",
        f"--env SERVER_SOCKET={srv_sock}",
        f"--env API_WEB_SOCKET={api_ws}",
        f"--env API_SOCKET={api_sock}",
        f"--mount {shlex.quote(os.getcwd())}:/home/ubuntu/agent_studio",
        "--root --rw", client_name, "-c", shlex.quote(client_cmd)
    ]))

    # 4) success marker & log
    if dry and not dry_touch:
        print(f"[touch skipped (dry)] {out_success}")
    else:
        print(f"[touch]{' (dry)' if dry else ''} {out_success}")
        if not dry or dry_touch:
            pathlib.Path(out_success).touch()
    if log_path:
        print(f"[log append]{' (dry)' if dry else ''} {log_path}")
        if not dry or dry_touch:
            pathlib.Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as lf:
                lf.write(f"[done] task={task_path} round={round_idx}\n")

def main_snake():
    # Called via Snakemake's `script:`
    smk = globals()["snakemake"]
    cfg     = smk.config
    params  = dict(smk.params)
    wc      = smk.wildcards
    out_ok  = str(smk.output[0])
    log     = str(smk.log[0])

    # dry-run sources:
    # 1) param dryrun in rule (preferred), 2) env COLLECT_DRYRUN=1, 3) getattr(snakemake, 'dryrun', False)
    dry = bool(params.get("dryrun")) or os.getenv("COLLECT_DRYRUN") == "1" or bool(getattr(smk, "dryrun", False))
    # optionally “pretend success” by touching outputs even in dry (useful for end-to-end plumbing tests)
    dry_touch = bool(params.get("dry_touch")) or os.getenv("COLLECT_DRY_TOUCH") == "1"

    _main(
        dry=dry,
        dry_touch=dry_touch,
        task_path=params["task_path"],
        round_idx=int(wc.r),
        out_success=out_ok,
        log_path=log,
        cfg=cfg,
        params=params,
        wc=wc,
    )

if __name__ == "__main__":
    # Standalone CLI (optional) — handy for testing outside Snakemake
    if "snakemake" in globals():
        main_snake()
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--task-path", required=True)
        ap.add_argument("--round", type=int, required=True)
        ap.add_argument("--output-success", required=True)
        ap.add_argument("--log-file", required=True)
        ap.add_argument("--config-json", required=True)
        ap.add_argument("--dry-run", action="store_true")
        ap.add_argument("--dry-touch", action="store_true")
        args = ap.parse_args()
        cfg = json.loads(pathlib.Path(args.config_json).read_text())
        params = {
            "data_dir": os.path.join(os.path.dirname(args.output_success), "..", "raw"),
            "prev_plans": None,
            "prev_ckpt": None,
            "task_path": args.task_path,
        }
        _main(
            dry=args.dry_run, dry_touch=args.dry_touch,
            task_path=args.task_path, round_idx=args.round,
            out_success=args.output_success, log_path=args.log_file,
            cfg=cfg, params=params, wc=None,
        )
