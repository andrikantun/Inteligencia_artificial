# ==========================================
# EJERCICIO 1: Q-LEARNING - CLIFF WALKING
# ==========================================

# Importamos las bibliotecas necesarias
import random
import csv
import matplotlib.pyplot as plt


# ==========================================
# 1. CONFIGURACIÓN DEL ENTORNO
# ==========================================

ROWS = 4
COLS = 12

START = (3, 0)
GOAL = (3, 11)

# Casillas que forman el acantilado
CLIFF = [(3, col) for col in range(1, 11)]

# Acciones disponibles y sus movimientos
ACTIONS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1)
}


# ==========================================
# 2. CLASE DEL ENTORNO
# ==========================================

class CliffWalking:

    def __init__(self):
        self.state = START

    def reset(self):
        """Regresa al estado inicial."""
        self.state = START
        return self.state

    def step(self, action):
        """Ejecuta una acción y devuelve:
        siguiente estado, recompensa y finalización.
        """

        row, col = self.state

        # Obtener el movimiento correspondiente
        move_row, move_col = ACTIONS[action]

        new_row = row + move_row
        new_col = col + move_col

        # Evitar salir de los límites del mapa
        new_row = max(0, min(ROWS - 1, new_row))
        new_col = max(0, min(COLS - 1, new_col))

        next_state = (new_row, new_col)

        # Comprobar si el agente cayó al acantilado
        if next_state in CLIFF:
            self.state = START
            return START, -100, False

        # Comprobar si llegó a la meta
        if next_state == GOAL:
            self.state = GOAL
            return GOAL, -1, True

        # Movimiento ordinario
        self.state = next_state
        return next_state, -1, False

# ==========================================
# 3. TABLA Q Y FUNCIONES DEL AGENTE
# ==========================================

# Crear los estados que puede visitar el agente
STATES = [
    (row, col)
    for row in range(ROWS)
    for col in range(COLS)
    if (row, col) != GOAL and (row, col) not in CLIFF
]


def empty_q():
    """Crea la tabla Q inicializada en cero."""

    return {
        state: {action: 0.0 for action in ACTIONS}
        for state in STATES
    }


def argmax_action(q, state):
    """Selecciona la acción con mayor valor Q.

    En caso de empate, respeta el orden:
    N, E, S, W.
    """

    best_value = max(q[state].values())

    for action in ACTIONS:
        if q[state][action] == best_value:
            return action


def epsilon_greedy(q, state, epsilon, rng):
    """Selecciona una acción usando epsilon-greedy."""

    if rng.random() < epsilon:
        return rng.choice(list(ACTIONS))

    return argmax_action(q, state)


def td_target_q(reward, next_state, done, q, gamma):
    """Calcula el objetivo TD de Q-learning."""

    if done:
        return reward

    return reward + gamma * max(q[next_state].values())


def apply_update(q, state, action, target, alpha):
    """Actualiza el valor Q mediante la regla de Q-learning."""

    old_value = q[state][action]

    q[state][action] = old_value + alpha * (
        target - old_value
    )


# ==========================================
# 4. ENTRENAMIENTO DEL AGENTE
# ==========================================

def train(env, episodes, alpha, gamma, epsilon, seed):

    rng = random.Random(seed)

    q = empty_q()

    episode_rewards = []

    for episode in range(episodes):

        state = env.reset()
        total_reward = 0

        for step in range(200):

            # Elegir una acción mediante epsilon-greedy
            action = epsilon_greedy(
                q, state, epsilon, rng
            )

            # Ejecutar la acción
            next_state, reward, done = env.step(action)

            # Calcular el objetivo TD
            target = td_target_q(
                reward, next_state, done, q, gamma
            )

            # Actualizar la tabla Q
            apply_update(
                q, state, action, target, alpha
            )

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

    return q, episode_rewards


# ==========================================
# 5. EVALUACIÓN GREEDY
# ==========================================

def evaluate_greedy(env, q):

    state = env.reset()

    path = [state]
    actions_taken = []
    total_reward = 0

    for step in range(200):

        # Elegir la mejor acción sin exploración
        action = argmax_action(q, state)

        next_state, reward, done = env.step(action)

        actions_taken.append(action)
        path.append(next_state)

        total_reward += reward
        state = next_state

        if done:
            break

    return path, actions_taken, total_reward

# ==========================================
# 3. PRUEBAS MANUALES DEL ENTORNO
# ==========================================

# ==========================================
# 6. EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":

    # Parámetros del ejercicio
    ALPHA = 0.5
    GAMMA = 1.0
    EPSILON = 0.1
    EPISODES = 500
    SEED = 42

    # Crear entorno
    env = CliffWalking()

    # Entrenar agente
    q, rewards = train(
        env,
        EPISODES,
        ALPHA,
        GAMMA,
        EPSILON,
        SEED
    )

    # Evaluar política greedy
    path, actions, total_reward = evaluate_greedy(
        env, q
    )

    # Mostrar resultados
    print("\n===== RESULTADOS DEL ENTRENAMIENTO =====")

    print("Semilla:", SEED)
    print("Episodios:", EPISODES)
    print("Alpha:", ALPHA)
    print("Gamma:", GAMMA)
    print("Epsilon:", EPSILON)

    print("\n===== POLÍTICA GREEDY =====")

    print("Camino:", actions)
    print("Pasos:", len(actions))
    print("Retorno:", total_reward)

        # ==========================================
    # 7. GUARDAR RECOMPENSAS EN CSV
    # ==========================================

    window = 10

    moving_average = []

    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        average = sum(rewards[start:i + 1]) / len(rewards[start:i + 1])
        moving_average.append(average)

    with open("resultados.csv", "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "episodio",
            "recompensa",
            "media_movil_10"
        ])

        for i in range(EPISODES):
            writer.writerow([
                i + 1,
                rewards[i],
                moving_average[i]
            ])

    print("\nCSV guardado: resultados.csv")

        # ==========================================
    # 8. GRAFICAR RECOMPENSAS
    # ==========================================

    plt.figure(figsize=(12, 6))

    plt.plot(
        range(1, EPISODES + 1),
        rewards,
        alpha=0.35,
        label="Recompensa por episodio"
    )

    plt.plot(
        range(1, EPISODES + 1),
        moving_average,
        linewidth=2,
        label="Media móvil (10 episodios)"
    )

    plt.xlabel("Episodios")
    plt.ylabel("Recompensa acumulada")
    plt.title("Q-learning - Cliff Walking")

    plt.legend()
    plt.grid(True)

    plt.savefig("recompensa.png", dpi=150, bbox_inches="tight")

    plt.close()

    print("Gráfica guardada: recompensa.png")