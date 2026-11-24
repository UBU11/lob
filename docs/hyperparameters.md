# Training Hyperparameters

Summary of hyperparameters employed for C51 Rainbow DQN and Double DQN:

| Parameter | C51 Rainbow DQN | Double DQN Baseline | Description |
| :--- | :--- | :--- | :--- |
| **Learning Rate** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | Adam optimizer learning rate |
| **Batch Size** | 64 | 64 | Minibatch sample size |
| **Discount Factor ($\gamma$)** | 0.995 | 0.995 | Return discount rate |
| **Replay Capacity** | 50,000 | 50,000 | Experience buffer capacity |
| **Atoms ($N$)** | 51 | N/A | Categorical distribution support bins |
| **Support Range** | $[-25.0, 10.0]$ | N/A | Basis points support bounds |
| **PER Exponent ($\alpha$)** | 0.6 | N/A | Priority sampling exponent |
| **PER Importance Sampling ($\beta$)** | $0.4 \to 1.0$ | N/A | Linear annealing of bias correction |
| **Target Update Period** | 200 steps | 200 steps | Hard target network synchronization |
