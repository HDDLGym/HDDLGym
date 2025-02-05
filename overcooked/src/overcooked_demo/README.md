# Installation
To run the overcooked example. In this folder.

(1) Add an .env here file with contents: 
```
GITHUB_TOKEN=<YOUR-GITHUB-TOKEN>
REPO_URL=<GITHUB-URL-OF-THIS-REPO>
```

For example: REPO_URL = 'github.com/HDDLGym/HDDLGym.git''

(2) Run:
```bash
sudo docker compose build
```

(3) Start the Docker containers and view the visualization in your browser at `http://localhost/experiment`:
```bash
sudo docker compose up
```
# Installation Guide for Overcooked Example

To run the Overcooked example, follow these steps in this folder:

### (1) Set Up the Environment Configuration

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

### (2) Build the Docker Containers

Run the following command to build the Docker containers:

```bash
sudo docker compose build
```

### (3) Start the Docker Containers

Launch the application by starting the Docker containers:

```bash
sudo docker compose up
```

### (4) Access the Visualization

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
