# %%

import numpy as np
from itertools import product
from generative_model import GenerativeModel
from t_maze_world import TMazeWorld
from active_inference_agent import ActiveInferenceAgent

T = 3
NUM_TRIALS = 10

ALPHA = 64
BETA = 4
PARAM_A = 0.5
PARAM_C = 1.0


def get_policies_for_step(t, num_actions=4, T=T):
    remaining_steps = T - t
    if remaining_steps <= 0:
        return [[]]
    policy_tuples = product(range(num_actions), repeat=remaining_steps)

    return [list(p) for p in policy_tuples]

def o_to_location_and_stimulus(o):
    location_number = o // 4
    location_dict = {0: "Start Pos.", 1: "Left Arm  ", 2: "Right Arm ", 3: "Cue       "}
    stimulus_number = o % 4
    stimulus_dict = {0: "CS", 1: "CS", 2: "US", 3: "NS"}
    return location_dict[location_number], stimulus_dict[stimulus_number]

def a_to_action(a):
    action_dict = {0: "('Go to start    ')", 1: "('Go to left arm ')",
                   2: "('Go to right arm')", 3: "('Go to cue      ')"}
    return action_dict[a]

# %%

print("Erstelle Simulations-Objekte...")
model = GenerativeModel(a=PARAM_A, c=PARAM_C)
world = TMazeWorld(model=model)
agent = ActiveInferenceAgent(model=model, alpha=ALPHA, beta=BETA, T=T)
print("Erstellung der Simulations-Objekte abgeschlossen.")

print(f"Starte Simulation für {NUM_TRIALS} Trials...")
print("-" * 30)
trial_results = []
for trial in range(NUM_TRIALS):
    print(f"--- Starte Trial {trial + 1}/{NUM_TRIALS} ---")
    o_t = world.reset()
    agent.reset()

    for t in range(T):
        policies_t = get_policies_for_step(t)
        a_t = agent.infer_action(o_t, policies_t, t)
        
        o_t_index = np.argmax(o_t)
        print(f"  t={t}: Agent sieht {o_to_location_and_stimulus(o_t_index)} "
              f"und wählt Aktion {a_to_action(a_t)}")
        if a_t is None:
            print("  t={t}: T erreicht. Keine Aktion mehr.")
            break
            
        o_t = world.step(a_t)

print("-" * 30)
print("Simulation beendet.")
# %%
