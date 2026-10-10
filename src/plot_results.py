import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--logs',default='results/logs.csv');p.add_argument('--output',default='assets/training_curve.png');args=p.parse_args()
    with open(args.logs,encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError('No hay episodios en logs')
    y=np.array([float(r['reward']) for r in rows]);x=np.arange(1,len(y)+1);window=min(20,len(y));avg=np.convolve(y,np.ones(window)/window,mode='valid')
    plt.figure(figsize=(9,5));plt.plot(x,y,alpha=.4,label='Recompensa episodio');plt.plot(x[window-1:],avg,label=f'Media movil ({window})');plt.xlabel('Episodio');plt.ylabel('Recompensa');plt.title('Entrenamiento DQN');plt.legend();plt.tight_layout()
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);plt.savefig(out,dpi=160);print(out)
if __name__=='__main__':main()
