import os
import time
import sys
import comet_ml
import hydra, json

from hw3.roble.agents.ddpg_agent import DDPGAgent
from hw3.roble.agents.td3_agent import TD3Agent
from hw3.roble.agents.sac_agent import SACAgent
from hw3.roble.infrastructure.rl_trainer import RL_Trainer
from omegaconf import DictConfig, OmegaConf
from hw3.roble.infrastructure.dqn_utils import get_env_kwargs

class OffPolicyTrainer(object):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        self._params['alg']['batch_size_initial'] = self._params['alg']['batch_size']

        if self._params['alg']['rl_alg'] == 'dqn':
            agent = DQNAgent
        elif self._params['alg']['rl_alg'] == 'ddpg':
            agent = DDPGAgent
        elif self._params['alg']['rl_alg'] == 'td3':
            agent = TD3Agent    
        elif self._params['alg']['rl_alg'] == 'sac':
            agent = SACAgent
        elif self._params['alg']['rl_alg'] == 'pg':
            agent = PGAgent
        else:
            print("Pick a rl_alg first")
            sys.exit()
        print(self._params)
        print(self._params['alg']['train_batch_size'])

        ################
        ## RL TRAINER
        ################

        self._rl_trainer = RL_Trainer(self._params , agent_class =  agent)

    def run_training_loop(self):

        result = self._rl_trainer.run_training_loop(
            self._params['alg']['n_iter'],
            collect_policy = self._rl_trainer._agent._actor,
            eval_policy = self._rl_trainer._agent._actor,
            )
        return result
    
    def set_comet_logger(self, logger):
        
        self._rl_trainer.set_comet_logger(logger)
        
        

def my_app(cfg: DictConfig): 
    print(OmegaConf.to_yaml(cfg))
    import os
    print("Command Dir:", os.getcwd())
    params = vars(cfg)
    # params.extend(env_args)
    for key, value in cfg.items():
        params[key] = value

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################


    logdir_prefix = 'hw3_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    exp_name = logdir_prefix + cfg.env.exp_name + '_' + cfg.env.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, exp_name)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.logging.logdir = logdir
        cfg.logging.exp_name = exp_name

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################
    # cfg = OmegaConf.merge(cfg, params)
    print ("cfg.env:", cfg.env.env_name)
    trainer = OffPolicyTrainer(cfg)
    
    # For Comet to start tracking a training run,
    # just add these two lines at the top of
    # your training script:
    
    experiment = comet_ml.Experiment(
    api_key=your key,
    project_name=project name
    )
    experiment.add_tag("hw3")
    experiment.set_name(exp_name)
    experiment.set_filename(fname="cometML_test")
    
    trainer.set_comet_logger(experiment)
    
    # Metrics from this training run will now be
    # available in the Comet UI

    
    data = trainer.run_training_loop()
    print("Results: ", data)
    return data
