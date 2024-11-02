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
        elif strategy in ['policy_gradient', 'ppo']:
            self.model = self._build_pg_model()
        elif strategy == 'online_rl':
            self.model = self._build_online_rl_model()

    def reset(self):
        self.state = np.random.uniform(low=0.01, high=0.05, size=(1,))
        self.current_episode = 0
        return self.state

    def step(self, action):
        # One-hot encode y_train if using policy gradient
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

    def _build_pg_model(self):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.x_train.shape[1],)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(2, activation="softmax")  # Output two probabilities for actions 0 and 1
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
        return model

    def _build_online_rl_model(self):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.x_train.shape[1],)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def _evaluate_model(self, model):
        predictions = model.predict(self.x_val).flatten()
        # Adjust prediction shape if needed to match y_val
        if predictions.shape[0] != self.y_val.shape[0]:
            predictions = predictions[:self.y_val.shape[0]]  # Trim or reshape to match y_val length
        ss_res = np.sum((self.y_val - predictions) ** 2)
        ss_tot = np.sum((self.y_val - np.mean(self.y_val)) ** 2)
        r2_value = 1 - (ss_res / ss_tot)
        return r2_value

    def get_action(self, state):
        if self.strategy == 'q_learning':
            return np.argmax(self.model.predict(tf.expand_dims(state, axis=0))[0]) if np.random.rand() > 0.1 else np.random.randint(2)
        elif self.strategy == 'policy_gradient':
            action_probs = tf.nn.softmax(self.model.predict(tf.expand_dims(state, axis=0))[0])
            action_probs = action_probs.numpy()  # Convert to numpy array for np.random.choice
            return np.random.choice([0, 1], p=action_probs)
        elif self.strategy == 'ppo' or self.strategy == 'online_rl':
            return np.argmax(self.model.predict(tf.expand_dims(state, axis=0))[0])
    
    def update_q_values(self, state, action, reward, next_state):
        if self.strategy == 'q_learning':
            q_values = self.model(tf.expand_dims(state, axis=0))
            next_q_values = self.model(tf.expand_dims(next_state, axis=0))
            target = reward + 0.7 * tf.reduce_max(next_q_values)
            q_values = q_values.numpy()
            if q_values.shape[1] == 2:
                q_values[0][action] = target  # Update only the chosen action's value
            else:
                q_values[0] = target  # Single-output model case
            self.model.fit(tf.expand_dims(state, axis=0), q_values, verbose=0)

# Data Preparation for RL
x, y = make_regression(n_samples=100, n_features=1, noise=0.1)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Training with Optuna Optimization
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    env = R2Environment(x_train, y_train, x_val, y_val, episodes=10, strategy=strategy_choice, learning_rate=learning_rate)
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

# Save the best model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(model_dir, f"optuna_optimized_model_{timestamp}.h5")
study.best_trial.user_attrs["model"].save(model_path)
print(f"Best model saved to {model_path}")

# Plot Optuna optimization progress
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.show()

# Plot the R² scores across episodes after optimization
best_params = study.best_params
env = R2Environment(x_train, y_train, x_val, y_val, episodes=10, strategy=strategy_choice, learning_rate=best_params["learning_rate"])
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
print(f"Strategy: {strategy_choice}, Optimized Learning Rate: {best_params['learning_rate']}")
print(f"Final R² Score: {r2_scores[-1]:.4f}")
print(f"Average R² Score: {np.mean(r2_scores):.4f}")
