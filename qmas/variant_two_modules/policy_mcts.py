import numpy as np
import torch
import os.path
from onpolicy.algorithms.r_mappo.rMAPPOPolicy import R_MAPPOPolicy
from .actor_critic import QmasActor, QmasCritic
from .predictor import Predictor

from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space



# Placeholder for an MCTS implementation

# --- MCTS Node ---
class MCTSNode:
    def __init__(self, parent, action, obs, done, agent, reward=0.0):
        self.parent = parent
        self.action = action
        self.obs = obs  # observation at this node
        self.done = done
        self.agent = agent  # agent id (PettingZoo agent string)
        self.reward = reward
        self.children = dict()  # action -> MCTSNode
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, action_space):
        return len(self.children) == action_space.n

    def best_child(self, c_puct=1.0):
        # UCB1 formula
        best_score = -float('inf')
        best_action = None
        best_child = None
        for a, child in self.children.items():
            if child.visits == 0:
                ucb = float('inf')
            else:
                ucb = child.value / child.visits + c_puct * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-8))
            if ucb > best_score:
                best_score = ucb
                best_action = a
                best_child = child
        return best_action, best_child


class MCTS:
    def __init__(self, env_model, args):
        self.env_model = env_model  # PettingZoo env
        self.args = args
        self.num_simulations = getattr(args, 'mcts_simulations', 25)
        self.max_depth = getattr(args, 'mcts_max_depth', 10)
        self.c_puct = getattr(args, 'mcts_c_puct', 1.0)

    def search(self, obs, available_actions=None):
        # For PettingZoo, we assume obs is the observation for the current agent
        # available_actions is a binary mask for the current agent
        env = self.env_model
        agent = env.agent_selection if hasattr(env, 'agent_selection') else 0
        root = MCTSNode(parent=None, action=None, obs=obs, done=False, agent=agent)

        for _ in range(self.num_simulations):
            self._simulate(env, root, depth=0)

        # Choose the most visited child
        visits = [(a, child.visits) for a, child in root.children.items()]
        if not visits:
            # fallback: random
            if available_actions is not None:
                valid_actions = np.where(available_actions)[0]
                return np.random.choice(valid_actions)
            else:
                return np.random.randint(env.action_space.n)
        best_action = max(visits, key=lambda x: x[1])[0]
        return best_action

    def _simulate(self, env, node, depth):
        # Copy the environment for simulation
        import copy
        sim_env = copy.deepcopy(env)
        agent = node.agent
        obs = node.obs
        done = node.done
        total_reward = 0.0
        current_node = node
        current_depth = depth

        while not done and current_depth < self.max_depth:
            # Expand if not fully expanded
            available_actions = sim_env.action_space(agent)
            if not current_node.is_fully_expanded(available_actions):
                # Expand a random untried action
                tried_actions = set(current_node.children.keys())
                all_actions = list(range(available_actions.n))
                untried = [a for a in all_actions if a not in tried_actions]
                action = np.random.choice(untried)
                obs_, reward, done_, _, _ = self._step_env(sim_env, agent, action)
                child = MCTSNode(parent=current_node, action=action, obs=obs_, done=done_, agent=agent, reward=reward)
                current_node.children[action] = child
                value = self._rollout(sim_env, agent, obs_, done_, current_depth + 1)
                self._backpropagate(child, value + reward)
                return
            # Select best child
            action, child = current_node.best_child(self.c_puct)
            obs_, reward, done_, _, _ = self._step_env(sim_env, agent, action)
            current_node = child
            current_node.obs = obs_
            current_node.done = done_
            current_node.reward = reward
            total_reward += reward
            done = done_
            current_depth += 1
        # If reached max depth or done, backpropagate
        self._backpropagate(current_node, total_reward)

    def _rollout(self, env, agent, obs, done, depth):
        # Simulate until done or max_depth, taking random actions
        total_reward = 0.0
        current_depth = depth
        while not done and current_depth < self.max_depth:
            action_space = env.action_space(agent)
            action = np.random.randint(action_space.n)
            obs, reward, done, _, _ = self._step_env(env, agent, action)
            total_reward += reward
            current_depth += 1
        return total_reward

    def _backpropagate(self, node, value):
        # Propagate value up the tree
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def _step_env(self, env, agent, action):
        # Step the PettingZoo env for a single agent
        # Returns obs, reward, done, info, agent
        env.step(action)
        obs, reward, done, info = env.last()
        return obs, reward, done, info, agent


class QmasPolicy:
    ''' This class implements the QMAS policy with MCTS-based action selection and communication. '''

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.args = args
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # Placeholder for an environment model for MCTS (could be a learned model or simulator)
        self.env_model = None  # Should be set to a model that can simulate environment transitions
        self.mcts = MCTS(self, args)

        # Predictor ensemble
        obs_dim = np.prod(get_shape_from_obs_space(self.obs_space, flatten_dicts=False))
        action_dim = np.prod(get_shape_from_act_space(act_space))
        if args.prediction_ensemble_size > 1:
            print(f"Creating ensemble of {args.prediction_ensemble_size} predictors.")
        self.predictors = [
            Predictor(
                obs_dim,
                action_dim,
                args,
                device=self.device
            ) for _ in range(args.prediction_ensemble_size)
        ]

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs using MCTS.
        Args:
            cent_obs (np.ndarray): centralized input to the critic (unused for MCTS, but kept for API compatibility).
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor (unused for MCTS).
            rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic (unused for MCTS).
            masks: (np.ndarray) denotes points at which RNN states should be reset (unused for MCTS).
            available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled (unused for MCTS).
        Returns:
            values: (torch.Tensor) value function predictions (dummy zeros for compatibility).
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of chosen actions (dummy zeros for compatibility).
            rnn_states_actor: (torch.Tensor) updated actor network RNN states (unchanged).
            rnn_states_critic: (torch.Tensor) updated critic network RNN states (unchanged).
        """
        # For each agent, select an action using MCTS
        n_agents = obs.shape[1] if obs.ndim == 3 else 1
        batch_size = obs.shape[0] if obs.ndim >= 2 else 1
        actions = []
        for b in range(batch_size):
            acts = []
            for i in range(n_agents):
                aa = available_actions[b, i] if available_actions is not None else None
                o = obs[b, i] if n_agents > 1 else obs[b]
                act = self.mcts.search(o, available_actions=aa)
                acts.append(act)
            actions.append(acts)
        actions = np.array(actions)
        # Dummy values and log_probs for compatibility
        values = torch.zeros((batch_size, n_agents, 1), dtype=torch.float32)
        action_log_probs = torch.zeros((batch_size, n_agents, 1), dtype=torch.float32)
        # rnn_states unchanged
        return values, torch.tensor(actions), action_log_probs, rnn_states_actor, rnn_states_critic

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        Args:
            cent_obs (np.ndarray): centralized input to the critic (unused for MCTS, but kept for API compatibility).
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor (unused for MCTS).
            rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic (unused for MCTS).
            action: (np.ndarray) actions whose log probabilites and entropy to compute.
            masks: (np.ndarray) denotes points at which RNN states should be reset (unused for MCTS).
            available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available)
            active_masks: (torch.Tensor) denotes whether an agent is active or dead (unused for MCTS).
        Returns:
            values: (torch.Tensor) value function predictions (dummy zeros for compatibility).
            action_log_probs: (torch.Tensor) log probabilities of the input actions (dummy zeros).
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs (dummy zeros).
        """
        n_agents = obs.shape[1] if obs.ndim == 3 else 1
        batch_size = obs.shape[0] if obs.ndim >= 2 else 1
        values = torch.zeros((batch_size, n_agents, 1), dtype=torch.float32)
        action_log_probs = torch.zeros((batch_size, n_agents, 1), dtype=torch.float32)
        dist_entropy = torch.zeros((batch_size, n_agents, 1), dtype=torch.float32)
        return values, action_log_probs, dist_entropy


    def get_prediction(self, trajectory, visibility_mask=None, prediction_prev=None):
        """
        Get a prediction from the ensemble of predictors.
        Args:
            trajectory: A tensor of shape (T, D), where T is the trajectory length and D is the transition dimension (action + observation).
            visibility_mask: An optional tensor of shape (T,) indicating which timesteps are visible (1) or not (0).
            prediction_prev: An optional tensor of shape (T, D_out) representing the previous prediction to condition on.
        Returns:
            prediction: A tensor of shape (T, D_out) representing the mean prediction across the ensemble.
            uncertainty: A tensor of shape (T, D_out) representing the uncertainty (variance) across the ensemble predictions.
        """

        predictions = []
        for predictor in self.predictors:
            pred = predictor.get_prediction(trajectory.clone(), visibility_mask.clone(), prediction_prev)
            predictions.append(pred.unsqueeze(0))
        predictions = torch.cat(predictions, dim=0)  # Shape: (num_predictors, T, D_out)
        prediction = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)

        return prediction, uncertainty


    def save(self, directory, episode):
        ''' Save the policy and predictors. '''
        os.makedirs(directory, exist_ok=True)
        for i, predictor in enumerate(self.predictors):
            torch.save(predictor.state_dict(), os.path.join(directory, f"predictor{i}.pt"))


    def restore(self, directory):
        ''' Restore the predictors. '''
        for i, predictor in enumerate(self.predictors):
            predictor_state_dict = torch.load(os.path.join(directory, f"predictor{i}.pt"), map_location=self.device)
            # This is hacky - reset the fix_mask here.
            if 'diffuser.fix_mask' in predictor_state_dict:
                predictor_state_dict['diffuser.fix_mask'] = predictor.diffuser.fix_mask
            predictor.load_state_dict(predictor_state_dict)