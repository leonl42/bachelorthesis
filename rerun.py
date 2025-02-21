import os 
import json
run = []
for exp in ["sgdm_wobn","adam_wobn"]:
    for subexp in ["wd","norm","cnorm","cnormu","gcstdu"]:
        for hyp in os.listdir(os.path.join("./data",exp,subexp)):
            with open(os.path.join("./data",exp,subexp,hyp,"settings.json"),"r") as f:
                js = json.load(f)

            if "apply_wd_to" in js["optimizer"].keys(): 
                if js["optimizer"]["apply_wd_to"] != "conv&kernel|out&kernel":
                    print(js["optimizer"]["apply_wd_to"])
            if "apply_norm_to" in js["norm"].keys(): 
                if js["norm"]["apply_norm_to"] != "conv&kernel|out&kernel":
                    print(js["norm"]["apply_norm_to"])