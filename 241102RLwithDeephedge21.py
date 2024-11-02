import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import optuna
from sklearn.metrics import r2_score
import seaborn as sns
import scipy.stats as stats
import os
from datetime import datetime

# Set initial parameters
r = 0.00
sig = 0.2
T = 30 / 365
M = 300
N = 30
dt = T / N
rdt = r * dt
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# User selection for volatility and strategy
volatility_type = input("Select volatility type (normal/high): ").strip().lower()
strategy_choice = input("Select RL strategy (q_learning/policy_gradient/ppo): ").strip().lower()

# Generate stock prices with user-selected volatility
def generate_stock_prices(volatility='normal'):
    global sigsdt
    sigma_factor = 1.5 if volatility == 'high' else 1.0
    sigsdt = sig * sigma_factor * np.sqrt(dt)
    S = np.empty([M, N + 1])
    rv = np.random.normal(r * dt, sigsdt, [M, N])
    np.random.seed(20220617)
    for i in range(M):
        S[i, 0] = 100
        for j in range(N):
            S[i, j + 1] = S[i, j] * (1 + rv[i, j])
    return S

# Visualization and statistical tests
S = generate_stock_prices(volatility_type)
for i in range(M):
    plt.plot(S[i, :])
plt.show()

rv_d1 = S.flatten()
sns.kdeplot(data=rv_d1, color="red", fill=True)
plt.show()

test_stat, p_val = stats.shapiro(rv_d1)
print(f"Test statistics: {test_stat}, p-value: {p_val}")

# Option Pricing
K = 95
Savg = np.cumsum(S, axis=1) / (np.arange(S.shape[1]) + 1)
call_payoff = np.where(Savg - K < 0, 0, Savg - K)
put_payoff = np.where(K - Savg < 0, 0, K - Savg)

# Reinforcement Learning Environment Class
class R2Environment:
    def __init__(self, x_train, y_train, x_val, y_val, episodes=10, strategy='q_learning', learning_rate=0.01):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.strategy = strategy
        self.current_episode = 0
        self.state = self.reset()
        
        # Build models specific to each strategy
        if strategy == 'q_learning':
            self.model = self._build_q_model()
        elif strategy == 'policy_gradient' or strategy == 'ppo':
            self.model = self._build_pg_model()

    def reset(self):
        self.state = np.random.uniform(low=0.01, high=0.05, size=(1,))
        self.current_episode = 0
        return self.state

    def step(self, action):
        new_parameters = self.state + action
        model = self.model
        model.fit(self.x_train, self.y_train, epochs=1, verbose=0)
        r2_value = self._evaluate_model(model)
        reward = r2_value
        self.current_episode += 1
        done = self.current_episode >= self.episodes or r2_value < 0.01
        return new_parameters, reward, done

    def _build_q_model(self):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.x_train.shape[1],)),
            tf.keras.layers.Dense(1, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def _build_pg_model(self):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.x_train.shape[1],)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(2, activation="softmax")  # Output two probabilities for actions 0 and 1
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
        return model

    def _evaluate_model(self, model):
        predictions = model.predict(self.x_val)
        ss_res = np.sum((self.y_val - predictions) ** 2)
        ss_tot = np.sum((self.y_val - np.mean(self.y_val)) ** 2)
        r2_value = 1 - (ss_res / ss_tot)
        return r2_value

    def get_action(self, state):
        if self.strategy == 'q_learning':
            return np.argmax(self.model.predict(tf.expand_dims(state, axis=0))[0]) if np.random.rand() > epsilon else np.random.randint(2)
        elif self.strategy == 'policy_gradient':
            # Predict probabilities for two actions
            action_probs = tf.nn.softmax(self.model.predict(tf.expand_dims(state, axis=0))[0])
            action_probs = action_probs.numpy()  # Convert to numpy array for np.random.choice
            return np.random.choice([0, 1], p=action_probs)  # Choose action based on probabilities
        elif self.strategy == 'ppo':
            return np.argmax(self.model.predict(tf.expand_dims(state, axis=0))[0])
    
    def update_q_values(self, state, action, reward, next_state):
        if self.strategy == 'q_learning':
            q_values = self.model(tf.expand_dims(state, axis=0))
            next_q_values = self.model(tf.expand_dims(next_state, axis=0))
            target = reward + gamma * tf.reduce_max(next_q_values)
            q_values = q_values.numpy()
            q_values[0][action] = target
            self.model.fit(tf.expand_dims(state, axis=0), q_values, verbose=0)

# Data Preparation for RL
x, y = make_regression(n_samples=100, n_features=1, noise=0.1)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Training with Optuna Optimization
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    strategy = trial.suggest_categorical("strategy", ["q_learning", "policy_gradient", "ppo"])
    env = R2Environment(x_train, y_train, x_val, y_val, episodes=10, strategy=strategy, learning_rate=learning_rate)
    total_r2_score = []
    for episode in range(env.episodes):
        state = env.reset()
        done = False
        while not done:
            action = env.get_action(state)
            next_state, reward, done = env.step(action)
            env.update_q_values(state, action, reward, next_state)
            state = next_state
        total_r2_score.append(reward)
    return -np.mean(total_r2_score)

# Run Optuna Study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
print(f"Best Parameters: {study.best_params}")

# Training and Plotting
best_params = study.best_params
env = R2Environment(x_train, y_train, x_val, y_val, episodes=10, strategy=best_params["strategy"], learning_rate=best_params["learning_rate"])
r2_scores = []
for episode in range(env.episodes):
    state = env.reset()
    done = False
    while not done:
        action = env.get_action(state)
        next_state, reward, done = env.step(action)
        env.update_q_values(state, action, reward, next_state)
        state = next_state
    r2_scores.append(reward)

plt.plot(r2_scores, label="R² Score")
plt.xlabel("Episode")
plt.ylabel("R² Score")
plt.legend()
plt.show()

print("\n### Training Report ###")
print(f"Strategy: {best_params['strategy']}, Learning Rate: {best_params['learning_rate']}")
print(f"Final R² Score: {r2_scores[-1]:.4f}")
print(f"Average R² Score: {np.mean(r2_scores):.4f}")
