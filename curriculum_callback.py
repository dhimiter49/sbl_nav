from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """
    Curiiculum callback to change the number of crowd members.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, curriculum, verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.curriculum = curriculum


    def _on_training_start(self) -> None:
        """
        Set the curriculum steps (every how many steps to add a crowd member)
        """
        self.num_crowd_max = self.training_env.get_attr("max_n_crowd")[0]
        self.curriculum_steps = int(
            (self.locals["total_timesteps"]) / self.curriculum / self.num_crowd_max
        )
        print(
            "INFO: A crowd member will be added every", self.curriculum_steps,
            "training steps", "with up to a maximum of", self.num_crowd_max,
            "crowd members."
        )


    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        n_crowd = min(
            1 + int(self.num_timesteps / self.curriculum_steps),
            self.num_crowd_max
        )
        self.training_env.set_attr("n_crowd", n_crowd)
        self.logger.record("n_crowd", n_crowd)


    def _on_step(self) -> bool:
        """
        No effect for each step only at the beginning of a training step.

        :return: If the callback returns False, training is aborted early.
        """
        return True


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
