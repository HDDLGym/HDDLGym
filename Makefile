NVCC_RESULT := $(shell which nvcc 2> NULL; rm NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus all
else
GPUS=
endif


# Set flag for docker run command
MYUSER=myuser
BASE_FLAGS=-it --rm -v ${PWD}:/home/$(MYUSER) --shm-size 20G
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)

DOCKER_IMAGE_NAME = hddlgym
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE) -a hddlgym_container
USE_CUDA = $(if $(GPUS),true,false)
ID = $(shell id -u)

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=$(USE_CUDA) --build-arg MYUSER=$(MYUSER) --build-arg UID=$(ID) --tag $(IMAGE) --label mysuser=$(MYUSER) --progress=plain .

run:
	$(DOCKER_RUN) /bin/bash

run-hddl-gpu0:
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u main_train.py --dvc cuda:0 --problem-name overcooked_3agents_collab --monitor-training False --max-e-steps 25 --use-central-planner True"

run-hddl-gpu1: 
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u main_train.py --dvc cuda:1 --problem-name transport_1agent_no_collab --monitor-training False --max-e-steps 25"

run-transport-1agent-collab:
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u main_train.py --dvc cuda:0 --problem-name transport_1agent_collab --monitor-training False --max-e-steps 25"

run-transport-2agents-no-collab:
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u main_train.py --dvc cuda:1 --problem-name transport_2agents_no_collab --monitor-training False --max-e-steps 50"

run-transport-2agents-collab:
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u main_train.py --dvc cuda:0 --problem-name transport_2agents_collab --monitor-training False --max-e-steps 50"

run-transport-3agents-no-collab:
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u main_train.py --dvc cuda:1 --problem-name transport_3agents_no_collab --monitor-training False --max-e-steps 60 --use-central-planner True"

run-satellite-2obs-2sat-1mod:
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u main_train.py --dvc cuda:1 --problem-name satellite_2obs_2sat_1mod --monitor-training False --max-e-steps 50 --use-central-planner True"

run-satellite-3obs-3sat-1mod:
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u main_train.py --dvc cuda:1 --problem-name satellite_3obs_3sat_1mod --monitor-training False --max-e-steps 60 --use-central-planner True"

run-rover-1agent:
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u main_train.py --dvc cuda:0 --problem-name rover_1agent --monitor-training False --max-e-steps 50"

run-rover-2agents:
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u main_train.py --dvc cuda:0 --problem-name rover_2agents --monitor-training False --max-e-steps 70 --use-central-planner True"

run-rover-3agents:
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u main_train.py --dvc cuda:0 --problem-name rover_3agents --monitor-training False --max-e-steps 90 --use-central-planner True"

run-evaluation-transport-3agents-no-collab:
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u main_train.py --dvc cuda:0 --problem-name transport_3agents_no_collab --monitor-training False --max-e-steps 100 --use-central-planner True --run-training False --Loadmodel True --Model 1143000"

run-evaluation-transport-2agents-collab:
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u main_train.py --dvc cuda:1 --problem-name transport_2agents_collab --monitor-training False --max-e-steps 100 --use-central-planner True --run-training False --Loadmodel True --Model 4328000"

run-pretrain-eval-all:
	$(DOCKER_RUN) /bin/bash -c "cd src && python3 -u pretrain_analysis.py"

uest:
	$(DOCKER_RUN) /bin/bash -c "pytest ./tests/"

workflow-test:
	# without -it flag
	docker run --rm -v ${PWD}:/home/workdir --shm-size 20G $(IMAGE) /bin/bash -c "pytest ./tests/"
