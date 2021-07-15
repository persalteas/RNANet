#/usr/bin/python3
import json
import os
import numpy as np

runDir = os.getcwd()

def get_best(i):
    weights = [ float(x.strip("[]")) for x in i["weights"] ]
    means = [ float(x.strip("[]")) for x in i["means"] ]
    s = sorted(zip(weights, means), reverse=True)
    return s[0][1]

def get_k(lw, bp):
    if lw == "cWW":
        if bp in ["GC", "CG"]:
            return 3.9
        if bp in ["AU", "UA"]:
            return 3.3
        if bp in ["GU", "UG"]:
            return 3.15
        return 2.4
    if lw == "tWW":
        return 2.4
    return 0.8

if __name__ == "__main__":
    print("processing HRNA jsons...")

    lws = []
    for c in "ct":
        for nt1 in "WHS":
            for nt2 in "WHS":
                lws.append(c+nt1+nt2)

    bps = []
    for nt1 in "ACGU":
        for nt2 in "ACGU":
            bps.append(nt1+nt2)
    
    fullresults = dict()
    fullresults["A"] = dict()
    fullresults["C"] = dict()
    fullresults["G"] = dict()
    fullresults["U"] = dict()
    counts = dict()
    for lw in lws:
        counts[lw] = 0
    for bp in bps:
        fullresults[bp[0]][bp[1]] = []

        # open json file
        with open(runDir + f"/results/geometry/json/hirerna_{bp}_basepairs.json", "rb") as f:
            data = json.load(f)
        
        # consider each BP type
        for lw in lws:
            this = dict()

            # gather params
            distance = 0
            a1 = 0
            a2 = 0
            for i in data:
                if i["measure"] == f"Distance between {lw} {bp} tips":
                    distance = np.round(get_best(i), 2)
                if i["measure"] == f"{lw}_{bp}_alpha_1":
                    a1 = np.round(np.pi/180.0*get_best(i), 2)
                if i["measure"] == f"{lw}_{bp}_alpha_2":
                    a2 = np.round(np.pi/180.0*get_best(i), 2)

            if distance == 0 and a1 == 0 and a2 == 0:
                # not found
                continue
            
            counts[lw] += 1

            # create entry
            this["rho"] = distance
            this["a1"] = a1
            this["a2"] = a2
            this["k"] = get_k(lw, bp)
            this["canonical"] = 1.0 if lw=="cWW" and bp in ["GC", "CG", "GU", "UG", "AU", "UA"] else 0.0
            this["LW"] = lw

            # store entry
            fullresults[bp[0]][bp[1]].append(this)

    with open(runDir + "/results/geometry/json/hirerna_basepairs_processed.json", "w") as f:
        json.dump(fullresults, f, indent=4)
