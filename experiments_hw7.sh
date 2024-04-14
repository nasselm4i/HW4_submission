
echo "Starting Training Pupper Sim2Real PPO ..."


echo "(Q2) Training Pupper Sim2Real - Vanilla - History Length : 0 ..."
python run_hw7_sim2real.py meta.add_to_runname="Q2.1" sim2real.history_len=0 

echo "(Q3) Training Pupper Sim2Real - History Length : 4 ..."
python run_hw7_sim2real.py meta.add_to_runname="Q3h4" sim2real.history_len=4

echo "(Q3) Training Pupper Sim2Real - History Length : 4 || Gaussian Noise : 0.01 ..."
python run_hw7_sim2real.py meta.add_to_runname="Q3g0.01" sim2real.history_len=4 sim2real.gaussian_obs_scale=0.01 sim2real.gaussian_act_scale=0.01

echo "(Q3) Training Pupper Sim2Real - History Length : 4 || Gaussian Noise : 0.1 ..."
python run_hw7_sim2real.py meta.add_to_runname="Q3g0.1" sim2real.history_len=4 sim2real.gaussian_obs_scale=0.1 sim2real.gaussian_act_scale=0.1

echo "(Q3) Training Pupper Sim2Real - History Length : 4 || Gaussian Noise : 1 ..."
python run_hw7_sim2real.py meta.add_to_runname="Q3g1" sim2real.history_len=4 sim2real.gaussian_obs_scale=1 sim2real.gaussian_act_scale=1

# ##########################################################################

echo "Starting Training Pupper Sim2Real SAC ..."


echo "(Q3) Training Pupper Sim2Real - Vanilla - History Length : 0 ..."
python run_hw7_sim2real.py meta.add_to_runname="Q2.2" sim2real.history_len=0 meta.sac_instead=True meta.track=True

echo "(Q3) Training Pupper Sim2Real - History Length : 4 ..."
python run_hw7_sim2real.py meta.add_to_runname="Q3h4" sim2real.history_len=4 meta.sac_instead=True

echo "(Q3) Training Pupper Sim2Real - History Length : 4 || Gaussian Noise : 0.01 ..."
python run_hw7_sim2real.py meta.add_to_runname="Q3g0.01" sim2real.history_len=4 sim2real.gaussian_obs_scale=0.01 sim2real.gaussian_act_scale=0.01 meta.sac_instead=True

echo "(Q3) Training Pupper Sim2Real - History Length : 4 || Gaussian Noise : 0.1 ..."
python run_hw7_sim2real.py meta.add_to_runname="Q3g0.1" sim2real.history_len=4 sim2real.gaussian_obs_scale=0.1 sim2real.gaussian_act_scale=0.1 meta.sac_instead=True

echo "(Q3) Training Pupper Sim2Real - History Length : 4 || Gaussian Noise : 1 ..."
python run_hw7_sim2real.py meta.add_to_runname="Q3g1" sim2real.history_len=4 sim2real.gaussian_obs_scale=1 sim2real.gaussian_act_scale=1 meta.sac_instead=True

echo "Successfully trained!"