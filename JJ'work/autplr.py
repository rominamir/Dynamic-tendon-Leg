# Callback for managing learning rate adjustments
class AutoLRBayesianCallback(BaseCallback):
    def __init__(self, switch_timesteps, explore_timesteps=125000, exploit_threshold=0.01, lr_decay_factor=0.5, min_lr=1e-6, verbose=0):
        super(AutoLRBayesianCallback, self).__init__(verbose)
        self.switch_timesteps = switch_timesteps
        self.explore_timesteps = explore_timesteps
        self.exploit_threshold = exploit_threshold
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = min_lr
        self.search_interval = 1000  # Interval for applying Bayesian Optimization

        # Separate storage for pre- and post-curriculum phases
        self.pre_curriculum_lrs = []
        self.pre_curriculum_losses = []
        self.post_curriculum_lrs = []
        self.post_curriculum_losses = []
        self.curriculum_switched = False

    def _on_step(self) -> bool:
        # Determine if we are in pre- or post-curriculum phase based on the timestep
        is_pre_curriculum = self.num_timesteps < self.switch_timesteps
        if not is_pre_curriculum and not self.curriculum_switched:
            self.curriculum_switched = True
            print("Switching to post-curriculum learning rate adjustments")

        # Apply learning rate adjustment based on pre- or post-curriculum phase
        if self.num_timesteps % self.search_interval == 0:
            self._bayesian_lr_update(pre_curriculum=is_pre_curriculum)

        return True

    def _bayesian_lr_update(self, pre_curriculum):
        # Get the current reward (or loss) data from the environment
        mean_reward = self.locals['infos'][0].get('episode', {}).get('r')

        # Collect data for Bayesian Optimization
        if mean_reward is not None:
            current_lr = self.model.policy.optimizer.param_groups[0]['lr']
            loss = -mean_reward  # Using negative reward as "loss" for BO
            if pre_curriculum:
                self.pre_curriculum_lrs.append(current_lr)
                self.pre_curriculum_losses.append(loss)
            else:
                self.post_curriculum_lrs.append(current_lr)
                self.post_curriculum_losses.append(loss)

        # Perform Bayesian Optimization if enough data is collected
        data_lrs = self.pre_curriculum_lrs if pre_curriculum else self.post_curriculum_lrs
        data_losses = self.pre_curriculum_losses if pre_curriculum else self.post_curriculum_losses

        if len(data_lrs) >= 5:  # Minimum samples before running BO
            res = gp_minimize(
                self._evaluate_lr(pre_curriculum),
                [(self.min_lr, current_lr)],  # Learning rate range
                x0=[current_lr],
                y0=data_losses,
                n_calls=10,
                random_state=1
            )

            # Apply the new learning rate from Bayesian Optimization
            new_lr = res.x[0]
            for param_group in self.model.policy.optimizer.param_groups:
                param_group['lr'] = new_lr
            stage = "pre-curriculum" if pre_curriculum else "post-curriculum"
            print(f"Updated {stage} learning rate to {new_lr} using Bayesian Optimization")

    def _evaluate_lr(self, pre_curriculum):
        # Return a function that evaluates the loss for Bayesian Optimization based on collected data
        def evaluate(lr):
            simulated_losses = []
            data_lrs = self.pre_curriculum_lrs if pre_curriculum else self.post_curriculum_lrs
            data_losses = self.pre_curriculum_losses if pre_curriculum else self.post_curriculum_losses
            
            for tested_lr, loss in zip(data_lrs, data_losses):
                if abs(tested_lr - lr[0]) < self.exploit_threshold:  # Closer values in lr
                    simulated_losses.append(loss)
            return np.mean(simulated_losses) if simulated_losses else np.inf
        return evaluate
