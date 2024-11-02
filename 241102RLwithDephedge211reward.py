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
strategy_choice = input("Select RL strategy (q_learning/policy_gradient/ppo/online_rl): ").strip().lower()

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

# Define the ANN Objective for Initial Tuning
def basic_model_objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    model.fit(x_train, y_train, epochs=10, verbose=0)
    
    # Evaluate and get the R² score
    predictions = model.predict(x_val).flatten()
    r2 = r2_score(y_val, predictions)
    return -r2

# Run Optuna on the ANN model for tuning
x, y = make_regression(n_samples=100, n_features=1, noise=0.1)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

study = optuna.create_study(direction="maximize")
study.optimize(basic_model_objective, n_trials=10)
best_params = study.best_params
print(f"Best Parameters from ANN tuning: {best_params}")

# Reinforcement Learning Environment Class
class R2Environment:
    def __init__(self, x_train, y_train, x_val, y_val, episodes=10, strategy='q_learning', learning_rate=0.01):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.episodes = episodes
        self.learning_rate = learning_rate  # Use tuned learning rate from ANN
        self.strategy = strategy
        self.current_episode = 0
        self.state = self.reset()
        
        if strategy == 'q_learning':
            self.model = self._build_q_model()
        elif strategy in ['policy_gradient', 'ppo']:
            self.model = self._build_pg_model()
        elif strategy == 'online_rl':
            self.model = self._build_online_rl_model()

    def reset(self):
        self.state = np.random.uniform(low=0.01, high=0.05, size=(1,))
        self.current_episode = 0
        return self.state

    def step(self, action):
        y_train_encoded = tf.keras.utils.to_categorical(self.y_train, num_classes=2) if self.strategy == 'policy_gradient' else self.y_train
        new_parameters = self.state + action
        model = self.model
        model.fit(self.x_train, y_train_encoded, epochs=1, verbose=0)
        r2_value = self._evaluate_model(model)
        reward = r2_value
        self.current_episode += 1
        done = self.current_episode >= self.episodes or r2_value < -1.0  # Stop if R² is less than -1
        return new_parameters, reward, done

    def _build_q_model(self):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.x_train.shape[1],)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(2)  # Output two values for Q-learning actions (0 and 1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def _evaluate_model(self, model):
        predictions = model.predict(self.x_val).flatten()
        if predictions.shape[0] != self.y_val.shape[0]:
            predictions = predictions[:self.y_val.shape[0]]
        ss_res = np.sum((self.y_val - predictions) ** 2)
        ss_tot = np.sum((self.y_val - np.mean(self.y_val)) ** 2)
        r2_value = 1 - (ss_res / ss_tot)
        return r2_value

# Use tuned learning rate for RL training
env = R2Environment(x_train, y_train, x_val, y_val, episodes=10, strategy=strategy_choice, learning_rate=best_params['learning_rate'])

# Optuna tuning for RL environment
def rl_objective(trial):
    env.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
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

# Run RL optimization
study = optuna.create_study(direction="minimize")
study.optimize(rl_objective, n_trials=10)
print(f"Best Parameters from RL tuning: {study.best_params}")

# Plot Optuna optimization history
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.show()
