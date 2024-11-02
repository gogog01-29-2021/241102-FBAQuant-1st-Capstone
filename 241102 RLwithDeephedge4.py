import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import optuna
import os
from datetime import datetime
import seaborn as sns
import scipy.stats as stats

# Set stock parameters
M, N, S0, K = 300, 30, 100, 95
episodes = 10
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# Reinforcement Learning Parameters
gamma, epsilon, epsilon_min, epsilon_decay = 0.7, 0.5, 0.1, 0.9

# Generate Stock Prices with Adjustable Volatility
def generate_stock_prices(volatility="normal"):
    T, dt, r = 30 / 365, 30 / 365 / N, 0.0
    sigma = 0.2 * (1.5 if volatility == "high" else 1.0)
    S, rv = np.empty([M, N + 1]), np.random.normal(r * dt, sigma * np.sqrt(dt), [M, N])
    for i in range(M):
        S[i, 0] = S0
        for j in range(N): S[i, j + 1] = S[i, j] * (1 + rv[i, j])
    return S

# Prepare Data
x, y = make_regression(n_samples=100, n_features=1, noise=0.1)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Reinforcement Learning Environment
class RLEnvironment:
    def __init__(self, x_train, y_train, x_val, y_val, method="q_learning", lr=0.01):
        self.x_train, self.y_train, self.x_val, self.y_val = x_train, y_train, x_val, y_val
        self.method, self.lr, self.episodes = method, lr, episodes
        self.q_model, self.pg_model, self.ppo_model = self.create_models()

    def create_models(self):
        def build_model():
            model = tf.keras.Sequential([
                tf.keras.Input(shape=(self.x_train.shape[1],)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            return model
        q_model = build_model()
        q_model.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss='mse')
        pg_model, ppo_model = build_model(), build_model()
        return q_model, pg_model, ppo_model

    def train(self):
        rewards, r2_scores = [], []
        for episode in range(self.episodes):
            state = self.reset()
            total_reward, done = 0, False
            while not done:
                action = self.get_action(state)
                next_state, reward, done, r2_score = self.step(action)
                self.update_model(state, action, reward, next_state)
                state, total_reward = next_state, total_reward + reward
                if done:
                    rewards.append(total_reward)
                    r2_scores.append(r2_score)
        return r2_scores, rewards

    def get_action(self, state):
        if self.method == "q_learning":
            if np.random.rand() > epsilon:
                return np.argmax(self.q_model(tf.expand_dims(state, axis=0))[0].numpy())
            else:
                return 0
        elif self.method == "policy_gradient":
            return self.policy_gradient_action(state)
        elif self.method == "ppo":
            return self.ppo_action(state)
        return 0

    def policy_gradient_action(self, state):
        return np.random.choice([0, 1])

    def ppo_action(self, state):
        return np.random.choice([0, 1])

    def step(self, action):
        params = self.lr + action * 0.001  # Adjusted parameter
        model = self.q_model if self.method == "q_learning" else (self.pg_model if self.method == "policy_gradient" else self.ppo_model)
        model.fit(self.x_train, self.y_train, epochs=1, verbose=0)
        r2_score = self.evaluate_model(model)
        reward, done = r2_score, r2_score < 0.01
        return params, reward, done, r2_score

    def evaluate_model(self, model):
        predictions = model.predict(self.x_val)
        ss_res = np.sum((self.y_val - predictions) ** 2)
        ss_tot = np.sum((self.y_val - np.mean(self.y_val)) ** 2)
        return 1 - (ss_res / ss_tot)

    def update_model(self, state, action, reward, next_state):
        if self.method == "q_learning":
            q_values = self.q_model(tf.expand_dims(state, axis=0))
            next_q = self.q_model(tf.expand_dims(next_state, axis=0))
            target = reward + gamma * tf.reduce_max(next_q)
            q_values = q_values.numpy(); q_values[0][action] = target
            self.q_model.fit(tf.expand_dims(state, axis=0), q_values, verbose=0)

    def reset(self):
        return np.random.uniform(0.01, 0.05, (1,))

# Define Objective for Optuna
def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)  # Updated to suggest_float with log=True
    env = RLEnvironment(x_train, y_train, x_val, y_val, method="q_learning", lr=lr)
    r2_scores, _ = env.train()
    return -np.mean(r2_scores)

# Run Optuna Study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
best_params = study.best_params

# Training with Selected Parameters
rl_strategy, vol_type = "ppo", "high"
env = RLEnvironment(x_train, y_train, x_val, y_val, method=rl_strategy, lr=best_params['learning_rate'])
r2_scores, rewards = env.train()

# Plotting Results
plt.plot(r2_scores, label="R² Score per Episode")
plt.xlabel("Episode")
plt.ylabel("R² Score")
plt.legend()
plt.show()

# Training Report
average_reward, final_reward = np.mean(rewards), rewards[-1]
average_r2, final_r2 = np.mean(r2_scores), r2_scores[-1]
print("\n### Training Report ###")
print(f"Volatility Type: {vol_type}, RL Strategy: {rl_strategy}")
print(f"Best Learning Rate: {best_params['learning_rate']:.5f}")
print(f"Average Reward: {average_reward:.4f}, Final Reward: {final_reward:.4f}")
print(f"Average R² Score: {average_r2:.4f}, Final R² Score: {final_r2:.4f}")

# KDE Plot for Stock Volatility Distribution
rv_d1 = generate_stock_prices(vol_type).reshape(M * (N + 1), 1)
sns.kdeplot(data=rv_d1, color="red", fill=True)
plt.show()

# Shapiro-Wilk Test for Normality
test_stat, p_val = stats.shapiro(rv_d1)
print(f"Shapiro-Wilk Test: Test statistic = {test_stat}, p-value = {p_val}")
