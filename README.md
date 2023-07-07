# How to run

First fetch submodules:
```
git submodule init 
git submodule update
```

In order to execute anything first build the docker image:

```
cd docker
docker build -t dope_demo .
```


Adapt the `docker_run`, but make it point to ws_ope, such that the scripts can find everything.

Start-up docker 

```
./docker_run
```


# OPE 

Already computed runs are in `logdir`.

To run the OPE script (choose algo from [iw, dr, mb, dual_dice, fqe]):

```
cd /app/ws

python -m policy_eval.train_eval_trifinger --logtostderr --trifinger --env_name=trifinger-cube-push-real-mixed-v0 --trifinger_policy_class=trifinger_rl_example.example.TorchPushPolicy --target_policy_std=0.2 --nobootstrap --noise_scale=0.0 --num_updates=500000 --discount=0.995 --target_policy_noisy --algo=fqe
```

# MC rollouts

Precomputed MC returns are already in /data.
To run monte carlo rollouts with the same parameters as above:

```
cd /app/ws

python -m trifinger_mc --trifinger_policy_class=trifinger_rl_example.example.TorchPushPolicy --discount=0.995 --target_policy_std=0.2 --target_policy_noisy
```

# Computing Real-World MC Returns

Since we have an expert dataset I simply computed the discounted returns from this. This is done in the `calc_return_dataset` notebook.

# Jupyter Notebooks

They can be run by executing `./jupyter_run` in docker (sourcing eval first is necessary, but should be sourced already). Connect the notebooks with the 'eval' kernel.

`trifinger_demo` is essentially the same as the script, just with some explanations and annotations. Also it only supports FQE.

`trifinger_plot_dope` plots results taken from data and logdir. 
