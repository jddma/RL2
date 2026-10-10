import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from env_utils import preprocess_observation
from replay_buffer import ReplayBuffer
from q_network import QNetwork
from dqn import train_dqn

def test_preprocessing():
    obs=dict(chassis_position=np.array([500.,350.]),chassis_angle=np.array([180.]),wheels_speed=np.array([0.,0.]),on_ground=np.array([1,0]))
    state=preprocess_observation(obs)
    assert state.shape==(7,) and state.dtype==np.float32
    assert np.allclose(state,[.5,.5,.5,0,0,1,0])

def test_network_and_buffer():
    net=QNetwork(); assert tuple(net(torch.zeros(64,7)).shape)==(64,3)
    buffer=ReplayBuffer(100)
    for _ in range(70):buffer.add(np.zeros(7),1,1.,np.ones(7),False)
    assert buffer.sample(64)[0].shape==(64,7)

class FakeActionSpace:
    n=3
    def sample(self):return 1
class FakeEnv:
    action_space=FakeActionSpace()
    def reset(self,**kwargs):self.t=0;return np.zeros(7,dtype=np.float32),{}
    def step(self,action):
        self.t+=1
        return np.ones(7,dtype=np.float32),1.,self.t>=4,False,{}

def test_dqn_smoke():
    net=QNetwork();target=QNetwork();buf=ReplayBuffer(100)
    out=train_dqn(FakeEnv(),net,target,buf,torch.optim.Adam(net.parameters(),lr=.001),num_episodes=2,learning_starts=2,batch_size=2,seed=42)
    assert len(out['history'])==2 and out['history'][0]['episode_length']==4
