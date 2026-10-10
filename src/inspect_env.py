from env_utils import make_env

def main():
    env=make_env()
    obs,info=env.reset(seed=42)
    print('Observacion procesada:',obs,'shape:',obs.shape,'dtype:',obs.dtype)
    print('Acciones:',env.action_space)
    for a in [0,1,1,2,1]:
        obs,reward,terminated,truncated,info=env.step(a)
        print('Accion',a,'reward',reward,'terminated',terminated,'truncated',truncated,'shape',obs.shape)
        if terminated or truncated:break
    env.close()
if __name__=='__main__':main()
