# Installation
To run the overcooked example. In this folder.

(1) Add an .env here file with contents: 
```
GITHUB_TOKEN=<YOUR-GITHUB-TOKEN>
REPO_URL=<GITHUB-URL-OF-THIS-REPO>
```

(2) Run:
```bash
sudo docker compose build
```

(3) Start the Docker containers and view the visualization in your browser at `http://localhost/experiment`:
```bash
sudo docker compose up
```
