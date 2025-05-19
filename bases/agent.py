
import bases
import random
import json


class RandomAgent:
    def choose_action(self, game):
        return random.choice(game.available_moves())


class QLearningAgent:
    def __init__(self, grid_size: int, learning_rate=0.01, discount_factor=0.5,
                 exploration_rate=0.01, training_mode=True):
        self.q_table = {}
        if training_mode is False:
            self.learning_rate = bases.cfg['learning_rate']
            self.discount_factor = bases.cfg['discount_factor']
            self.exploration_rate = bases.cfg['exploration_rate']
        else:
            self.learning_rate = learning_rate
            self.discount_factor = discount_factor
            self.exploration_rate = exploration_rate

        self.grid_size = grid_size
        self.training_mode = training_mode

    def choose_action(self, game) -> list:
        available_moves = game.available_moves()
        state = self.symmetrical_states(game.get_detailed_state())

        # Exploration is only allowed in training mode
        if self.training_mode and random.random() < self.exploration_rate:
            return random.choice(available_moves)

        # Exploitation: always select the best-known action based on Q-values
        q_values = [(move, self.q_table.get((state, move), 0))
                    for move in available_moves]
        max_q = max(q_values, key=lambda item: item[1])[1]
        best_actions = [move for move, q in q_values if q == max_q]

        return random.choice(best_actions)

    def update(self, previous_state, action,
               reward, next_state, available_moves) -> None:
        previous_state = self.symmetrical_states(previous_state)
        next_state = self.symmetrical_states(next_state)

        old_value = self.q_table.get((previous_state, action), 0)
        # Future rewards based on next state and available actions
        future_rewards = max((self.q_table.get((next_state, move), 0)
                              for move in available_moves), default=0)

        # Update the Q-value using Bellman
        updated_value = (1 - self.learning_rate) * \
            old_value + self.learning_rate * \
            (reward + self.discount_factor * future_rewards)

        self.q_table[(previous_state, action)] = updated_value

    def enhance_reward(self, completed_box, potential_box_opponent) -> float:
        reward = 0
        if completed_box:
            reward += 1
        if potential_box_opponent:
            reward -= 0.5
        return reward

    def save_policy(self, file_path) -> None:
        serialized_q_table = {str(k): v for k, v in self.q_table.items()}
        with open(file_path, 'w') as f:
            json.dump(serialized_q_table, f)

    def load_policy(self, file_path) -> None:
        try:
            with open(file_path, 'r') as f:
                serialized_q_table = json.load(f)
        except FileNotFoundError:
            print("************************************************")
            print(f" FileNotFoundError: Cannot find the {file_path}\n",
                  "Please train the AI agent first and try again.")
            print("************************************************")
            quit()
        else:
            self.q_table = {eval(k): v for k, v in serialized_q_table.items()}

    def symmetrical_states(self, state):
        lines, boxes = state
        transformed_states = self.generate_rotations_and_reflections(lines,
                                                                     boxes)

        # Normalize each state by sorting the lines
        # and making boxes hashable and comparable
        normalized_states = []
        for trans_lines, trans_boxes in transformed_states:
            norm_lines = tuple(sorted(tuple(sorted(line_pair))
                                      for line_pair in trans_lines))
            # Replace None with ' ' in boxes to avoid comparison issues
            norm_boxes = tuple(
                tuple(cell if cell is not None else ' ' for cell in row)
                for row in trans_boxes
            )
            normalized_states.append((norm_lines, norm_boxes))

        # Return the canonical form: the lexicographically smallest
        canonical_state = min(normalized_states)
        return canonical_state

    def generate_rotations_and_reflections(self, lines, boxes) -> list:
        transformations = []
        size = self.grid_size

        transformations.append((lines, boxes))

        def rotate90(lines, boxes) -> tuple[list, list[list]]:
            rotated_lines = []
            for x1, y1, x2, y2 in lines:
                rotated_lines.append((y1, size - x1, y2, size - x2))
            rotated_boxes = [[boxes[size - j - 1][i]
                              for j in range(size)] for i in range(size)]
            return rotated_lines, rotated_boxes

        def reflect_horizontally(lines, boxes) -> tuple[list, list]:
            reflected_lines = []
            for x1, y1, x2, y2 in lines:
                reflected_lines.append((x1, size - y1, x2, size - y2))
            reflected_boxes = [row[::-1] for row in boxes]
            return reflected_lines, reflected_boxes

        # Add rotations (90°, 180°, 270°)
        current_lines, current_boxes = lines, boxes
        for _ in range(3):
            current_lines, current_boxes = rotate90(current_lines,
                                                    current_boxes)
            transformations.append((current_lines, current_boxes))

        # Reflect original horizontally and repeat rotations
        reflected_lines, reflected_boxes = reflect_horizontally(lines, boxes)
        transformations.append((reflected_lines, reflected_boxes))
        current_lines, current_boxes = reflected_lines, reflected_boxes
        for _ in range(3):
            current_lines, current_boxes = rotate90(current_lines,
                                                    current_boxes)
            transformations.append((current_lines, current_boxes))

        return transformations
