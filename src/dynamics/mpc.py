import numpy as np
from utils.utils import Utils


class MPC:
    def __init__(self, **kwargs):
        self.control_horizon = kwargs.get("control_horizon", 1)
        self.prediction_horizon = kwargs.get("prediction_horizon", 5)
        self.num_candidates = kwargs.get("num_candidates", 20)
        self.num_iterations = kwargs.get("num_iterations", 4)
        self.velocity_weight = kwargs.get("velocity_weight", 0.02)
        self.distance_weight = kwargs.get("distance_weight", 2.5)
        self.data_collection = kwargs.get("data_collection", False)

        self.utils = Utils()
        self.utils.set_seed()

    def compute_action(self, dynamics, state, goal, prev_action):
        num_links = dynamics.get_num_links()
        shape = (self.prediction_horizon, num_links, 1)

        # Identify 20% of best candidates amongst all candidates
        elite_frac = 0.2
        num_elites = max(1, int(self.num_candidates * elite_frac))

        # Calculate mean and std of actions from elite candidates; initialize with 0 and 1
        mean = np.zeros(shape)
        std = np.ones(shape)

        best_first_action = np.zeros((num_links, 1))
        best_cost = float("inf")

        if self.data_collection:
            total_samples = self.num_candidates * self.num_iterations
            X_chunk = np.zeros((total_samples, 6), dtype=np.float32)
            Y_chunk = np.zeros((total_samples, 4), dtype=np.float32)

        for j in range(self.num_iterations):
            # From elite candidates distribution, sample actions for all new candidates
            samples = np.random.normal(loc=mean, scale=std, size=(self.num_candidates,) + shape)

            costs = np.zeros(self.num_candidates)
            for i in range(self.num_candidates):
                # Compute cost metrics
                actions = samples[i]
                if self.data_collection:
                    idx = j * self.num_candidates + i
                    cost_metrics, old_state, action, new_state = self.compute_cost(
                        state, dynamics, actions, goal
                    )
                    X_chunk[idx, :4] = old_state
                    X_chunk[idx, 4:] = action
                    Y_chunk[idx] = new_state
                else:
                    cost_metrics = self.compute_cost(state, dynamics, actions, goal)
                costs[i] = cost_metrics["final_cost"]

                # Determine best_cost
                if cost_metrics["final_cost"] < best_cost:
                    best_cost_metrics = cost_metrics
                    best_cost = best_cost_metrics["final_cost"]
                    best_first_action = actions[0]
                    # self.print_cost_metrics(best_cost_metrics, dist_now, actions)

            # Determine new elite candidates for next iteration
            elite_idxs = costs.argsort()[:num_elites]
            elites = samples[elite_idxs]
            mean = elites.mean(axis=0)
            std = elites.std(axis=0)

        if self.data_collection:
            return best_first_action, X_chunk, Y_chunk
        return best_first_action

    def compute_cost(self, state, dynamics, actions, goal):
        cost_metrics = {"final_cost": 0.0}
        action = actions[0]
        old_state = state.copy()
        new_state = dynamics.dynamics_step(old_state, action, dynamics.dt)

        # Final distance penalty after all time steps
        dist_final = np.linalg.norm(goal - dynamics.compute_fk(new_state))
        cost_metrics["dist_final_pen"] = dist_final * self.distance_weight
        cost_metrics["final_cost"] += cost_metrics["dist_final_pen"]

        vel_ee_final = np.linalg.norm(dynamics.compute_vel_ee(new_state))
        if vel_ee_final > 0.5:
            cost_metrics["velocity_pen"] = vel_ee_final * self.velocity_weight
            cost_metrics["final_cost"] += cost_metrics["velocity_pen"]

        if self.data_collection:
            return cost_metrics, old_state.flatten(), action.flatten(), new_state.flatten()
        return cost_metrics
