# HDDL Policy Rollout Visualization

This website provides a **visual representation** of the rollout of the **HDDL policy**, generated through the execution of the `run_hddl_policy.py` script.

## 🚀 Workflow Overview

### 1️⃣ Model Training
- Run the following command to train the model:
  ```bash
  python3 main_train.py --problem-name <PROBLEM-NAME>
  ```
- This script generates a trained model that will be used for the policy rollout.

### 2️⃣ Policy Rollout
- Execute the policy rollout with:
  ```bash
  python3 run_hddl_policy.py --problem-name <PROBLEM-NAME> --Model <MODEL-ID>
  ```
- This script:
  - Loads the trained model from `main_train.py`.
  - Generates the results of the optimal policy rollout.

### 3️⃣ Visualizing the Rollout
- Navigate to the website directory (this directory):
  ```bash
  cd website
  ```
- Start a local HTTP server to view the rollout visualization:
  ```bash
  python3 -m http.server 8000
  ```
- Open your web browser and visit:
  [http://localhost:8000](http://localhost:8000)

---

## 📋 Summary

- **`main_train.py`**: Trains and saves the model.
- **`run_hddl_policy.py`**: Loads the model and generates the policy rollout.
- **Website Visualization**: Displays an interactive visualization of the rollout.

---

## 💡 Quick Commands
```bash
# Train the model
python3 main_train.py --problem-name <PROBLEM-NAME>

# Run the policy rollout
python3 run_hddl_policy.py --problem-name <PROBLEM-NAME> --Model <MODEL-ID>

# Navigate to the website folder
cd website

# Launch the local server
python3 -m http.server 8000
```
