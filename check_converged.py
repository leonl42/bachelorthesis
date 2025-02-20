from plots.plot_utils import *
import json


writer = write(name="run",path=f"./run",h=11,max_tasks=25)

for setting in ["adam_wbn","sgdm_wobn", "adam_wobn"]:
    for exp in os.listdir(f"./data/{setting}"):
    
        if os.path.exists(os.path.join(f"./data/{setting}",exp,"settings.json")):
            continue

        hyperparams = list(os.listdir(os.path.join(f"./data/{setting}",exp)))

        for row,hyperparam in enumerate(hyperparams):
            if not os.path.exists(os.path.join(f"./data/{setting}",exp,hyperparam,"test_stats")):
                continue

            path = os.path.join(f"./data/{setting}",exp,hyperparam)
            stats = get_stats(path,"test_stats")
            acc_vals = list(stats["acc"].values())
            acc_keys = list(stats["acc"].keys())
            last_vals = np.mean(np.stack(acc_vals[-int(0.25*len(acc_vals)):],axis=0),axis=-1)
            last_keys = np.asanyarray(acc_keys[-int(0.25*len(acc_vals)):])
            m,b = np.polyfit(last_keys, last_vals, 1)
        
            print(f"Checked {os.path.join("./data",setting,exp,hyperparam)} with m={m*1e8}")
            if m*1e8>4:
                with open(os.path.join(f"./data/{setting}",exp,hyperparam,"settings.json"),"r") as f:
                    js = json.load(f)
                curr_steps = js["num_steps"]
                save_states = js["save_args"]["save_states_every"]
                writer.write(f"python code/main.py {os.path.join("./data",setting,exp,hyperparam)}/ --overwrite-num-steps {curr_steps + 200000} --overwrite-save-state {curr_steps + 200000} --overwrite-save-grad {-1} \n")

