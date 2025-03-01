from plots.plot_utils import *
import json


writer = write(name="run",path=f"./run",h=24,max_tasks=24)

for setting in ["adam_wbn", "sgdm_wobn"]:
    for exp in os.listdir(f"./data/{setting}"):
    
        if os.path.exists(os.path.join(f"./data/{setting}",exp,"settings.json")):
            continue

        if exp.endswith("_div8"):
            continue
        
        hyperparams = list(os.listdir(os.path.join(f"./data/{setting}",exp)))

        for row,hyperparam in enumerate(hyperparams):
            if not os.path.exists(os.path.join(f"./data/{setting}",exp,hyperparam,"test_stats")):
                continue

            path = os.path.join(f"./data/{setting}",exp,hyperparam)
            stats = get_stats(path,"test_stats")
            acc_vals = list(stats["acc"].values())
            acc_keys = list(stats["acc"].keys())
            #num_data_points = min(int(0.25*len(acc_vals)),50)
            #num_data_points = int(0.25*len(acc_vals))
            #last_vals = np.mean(np.stack(acc_vals[-num_data_points:],axis=0),axis=-1)
            #last_keys = np.asanyarray(acc_keys[-num_data_points:])
            #m,b = np.polyfit(last_keys, last_vals, 1)

            #acc_vals = np.mean(np.stack(acc_vals,axis=0),axis=-1)
            
            #if m*1e8>1 and acc_vals.max() <= last_vals.max():
            num_data_points = int(0.25*len(acc_vals))
            last_vals = np.stack(acc_vals[-num_data_points:],axis=0)
            acc_vals = np.stack(acc_vals,axis=0)
            m = 0
            if np.mean(np.max(last_vals,axis=0)) >= np.mean(np.max(acc_vals,axis=0)):
                print(f"{'\033[91m'} Checked {os.path.join("./data",setting,exp,hyperparam)} with m={m*1e8} {'\033[91m'}")
                with open(os.path.join(f"./data/{setting}",exp,hyperparam,"settings.json"),"r") as f:
                    js = json.load(f)
                curr_steps = js["num_steps"]
                save_states = js["save_args"]["save_states_every"]
                writer.write(f"python code/main.py {os.path.join("./data",setting,exp,hyperparam)}/ --overwrite-num-steps {curr_steps + 200000} --overwrite-save-state {curr_steps + 200000} --overwrite-save-grad {-1} \n")
            else:
                print(f"{'\033[92m'} Checked {os.path.join("./data",setting,exp,hyperparam)} with m={m*1e8} {'\033[92m'}")
