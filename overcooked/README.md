# Installation Guide for Overcooked Example

To run the Overcooked example, follow these steps:

### (1) Go to:

```
../overcooked/src/overcooked_demo/server
```

### (2) Set Up the Environment Configuration

Create a `.env` file in the current directory with the following contents:

```bash
GITHUB_TOKEN=<YOUR-GITHUB-TOKEN>
REPO_URL=<GITHUB-URL-OF-THIS-REPO>
FLASK_SECRET_KEY=<FLASK-SECRET-KEY>
```

For example:

```bash
GITHUB_TOKEN=ghp_example1234567890
REPO_URL=github.com/HDDLGym/HDDLGym.git
FLASK_SECRET_KEY=myflasksecretkey123
```

### (3) Build the Docker Containers

Run the following command to build the Docker containers:

```bash
sudo docker compose build
```

### (4) Start the Docker Containers

Launch the application by starting the Docker containers:

```bash
sudo docker compose up
```

### (5) Access the Visualization

Once the containers are up and running, open your browser and navigate to:

```
http://localhost/experiment
```

### Troubleshooting

- **Permission Issues:** If you encounter permission errors, try running the commands with `sudo`.
- **Port Conflicts:** Ensure that port 80 is not being used by other services. You can modify the `docker-compose.yml` file to change the port if needed.
- **Missing Docker:** Make sure Docker and Docker Compose are installed and running on your machine.

### Prerequisites

Ensure the following are installed on your system:
- Ubuntu 20.04
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

By following these steps, you should be able to successfully run and visualize the Overcooked example.

# Overcooked-AI 🧑‍🍳🤖

Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/).

See [Overcooked-repo](https://github.com/HumanCompatibleAI/overcooked_ai) for the original repository.



## Code Structure Overview 🗺

`overcooked_ai_py` contains:

`mdp/`:
- `overcooked_mdp.py`: main Overcooked game logic
- `overcooked_env.py`: environment classes built on top of the Overcooked mdp
- `layout_generator.py`: functions to generate random layouts programmatically

`agents/`:
- `agent.py`: location of agent classes
- `benchmarking.py`: sample trajectories of agents (both trained and planners) and load various models

`planning/`:
- `planners.py`: near-optimal agent planning logic
- `search.py`: A* search and shortest path logic

`human_aware_rl` contains:

`ppo/`:
- `ppo_rllib.py`: Primary module where code for training a PPO agent resides. This includes an rllib compatible wrapper on `OvercookedEnv`, utilities for converting rllib `Policy` classes to Overcooked `Agent`s, as well as utility functions and callbacks
- `ppo_rllib_client.py` Driver code for configuing and launching the training of an agent. More details about usage below
- `ppo_rllib_from_params_client.py`: train one agent with PPO in Overcooked with variable-MDPs 
- `ppo_rllib_test.py` Reproducibility tests for local sanity checks
- `run_experiments.sh` Script for training agents on 5 classical layouts
- `trained_example/` Pretrained model for testing purposes

`rllib/`:
- `rllib.py`: rllib agent and training utils that utilize Overcooked APIs
- `utils.py`: utils for the above
- `tests.py`: preliminary tests for the above

`imitation/`:
- `behavior_cloning_tf2.py`:  Module for training, saving, and loading a BC model
- `behavior_cloning_tf2_test.py`: Contains basic reproducibility tests as well as unit tests for the various components of the bc module.

`human/`:
- `process_data.py` script to process human data in specific formats to be used by DRL algorithms
- `data_processing_utils.py` utils for the above

`utils.py`: utils for the repo

`overcooked_demo` contains:

`server/`:
- `app.py`: The Flask app 
- `game.py`: The main logic of the game. State transitions are handled by overcooked.Gridworld object embedded in the game environment
- `move_agents.py`: A script that simplifies copying checkpoints to [agents](src/overcooked_demo/server/static/assets/agents/) directory. Instruction of how to use can be found inside the file or by running `python move_agents.py -h`

`up.sh`: Shell script to spin up the Docker server that hosts the game 
