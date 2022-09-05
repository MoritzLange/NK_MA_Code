#!/bin/bash
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=fsdp --seed=0
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=fsdp --seed=1
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=fsdp --seed=2
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=fsdp --seed=3
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=fsdp --seed=4
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=fsp --seed=0
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=fsp --seed=1
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=fsp --seed=2
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=fsp --seed=3
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=fsp --seed=4
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=rwp --seed=0
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=rwp --seed=1
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=rwp --seed=2
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=rwp --seed=3
python main.py --env=Pendulum-v1 --max_timesteps=12000 --wandb_name=TD3_v1 --wandb_entity=noah_ma --learning_rate=1e-3 --eval_freq=50 --start_timesteps=1000 --pretrain_steps=1000 --total_units=20 --aux_task=rwp --seed=4