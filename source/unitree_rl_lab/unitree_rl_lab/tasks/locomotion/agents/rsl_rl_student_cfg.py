from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
)


@configclass
class UnitreeStudentRunnerCfg(RslRlDistillationRunnerCfg):
    """Student distillation runner configuration for Unitree G1 Velocity-Perception.

    Uses StudentTeacherRecurrent: Student (LSTM+MLP) learns to mimic
    Teacher (LSTM+MLP) via online DAgger-style behavior cloning with MSE loss.
    Both student and teacher receive the same height scan + proprioception input.
    """

    class_name: str = "DistillationRunner"

    # ============== WandB Configuration ===========
    logger = "wandb"
    wandb_project = "Unitree_g1_Velocity_Student"
    run_name = "Student_Run"
    experiment_name = "unitree_student"
    # ==============================================

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    empirical_normalization = False

    # ---- 蒸馏不需要 resume，每次从 teacher checkpoint 重新开始 ----
    resume = False

    # ---- obs_groups: policy → Student, teacher → Teacher ----
    obs_groups = {
        "policy": ["policy"],
        "teacher": ["teacher"],
    }

    # ---- Policy: StudentTeacherRecurrent ----
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        class_name="StudentTeacherRecurrent",
        init_noise_std=1.0,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[256, 128],
        teacher_hidden_dims=[256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=True,
    )

    # ---- Algorithm: Distillation (MSE behavior cloning) ----
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        gradient_length=15,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        loss_type="mse",
    )
