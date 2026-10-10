import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from env_utils import make_env
from q_network import QNetwork

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='results/checkpoint_final.pt');p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed',type=int,default=1000);p.add_argument('--output',default='results');args=p.parse_args()
    if args.episodes < 1: raise ValueError('episodes debe ser positivo')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    network=checkpoint.get('network', {'hidden_layers':[128,128]})
    model=QNetwork(checkpoint.get('observation_dim',7),checkpoint.get('n_actions',3),network['hidden_layers'][0])
    model.load_state_dict(checkpoint['model']);model.eval()
    env=make_env();rows=[]
    for ep in range(args.episodes):
        state,info=env.reset(seed=args.seed+ep);total=0.;distance=float('-inf');steps=0;term=trunc=False
        while not (term or trunc):
            with torch.no_grad():action=int(model(torch.as_tensor(state).unsqueeze(0)).argmax(1).item())
            state,reward,term,trunc,info=env.step(action);total+=float(reward);steps+=1;distance=max(distance,float(state[0])*1000.)
        rows.append(dict(episode=ep+1,reward=total,max_distance=distance,steps=steps,terminated=term,truncated=trunc,success=bool(term and distance>=1000)))
    env.close()
    with (out/'evaluation.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rewards=np.array([r['reward'] for r in rows]);distances=np.array([r['max_distance'] for r in rows])
    metrics=dict(episodes=len(rows),reward_mean=float(rewards.mean()),reward_median=float(np.median(rewards)),reward_std=float(rewards.std()),distance_mean=float(distances.mean()),success_rate=float(np.mean([r['success'] for r in rows])),truncation_rate=float(np.mean([r['truncated'] for r in rows])))
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
