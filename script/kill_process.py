import argparse

import psutil

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kill", default="a,c,hm,hp,m,n,s,w")
args = parser.parse_args()

dict_pids = {p.info["pid"]: p.info["name"] for p in psutil.process_iter(attrs=["pid", "name"])}
kill_name = {
    "a": "all.sh",
    "c": "cutmix.sh",
    "hm": "hidden_mix.sh",
    "hp": "hidden_pm.sh",
    "m": "mixup.sh",
    "n": "ntr.sh",
    "s": "scarf.sh",
    "w": "wo-da.sh",
}

select_p_names = args.kill.split(",")
kill_process = [p_name for k, p_name in kill_name.items() if k in select_p_names]
print(f"kill process are {kill_process}")

print(dict_pids)
kill_process_pid = [pid for pid, p_name in dict_pids.items() if p_name in kill_process]
print(kill_process_pid)

for pid in kill_process_pid:
    p = psutil.Process(pid)
    p.terminate()
