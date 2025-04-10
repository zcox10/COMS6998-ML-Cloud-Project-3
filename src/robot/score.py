import numpy as np
import argparse
import time
import torch
import math
from math import pi
import os
from datetime import datetime

from robot.arm_dynamics_teacher import ArmDynamicsTeacher
from src.robot.robot import Robot
from src.robot.render import Renderer

np.set_printoptions(suppress=True)


# part 2 scoring
def reset(arm_teacher, arm_student, torque):
    initial_state = np.zeros((arm_teacher.dynamics.get_state_dim(), 1))  # position + velocity
    initial_state[0] = -math.pi / 2.0
    arm_teacher.set_state(initial_state)
    arm_student.set_state(initial_state)

    action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
    action[0] = torque
    arm_teacher.set_action(action)
    arm_student.set_action(action)

    arm_teacher.set_t(0)
    arm_student.set_t(0)


# part 3 scoring
def get_args():
    parser = argparse.ArgumentParser()
    # Arm
    parser.add_argument("--link_mass", type=float, default=0.1)
    parser.add_argument("--link_length", type=float, default=1)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--time_step", type=float, default=0.01)
    parser.add_argument("--model_dir", type=str, default="models")
    return parser.parse_known_args()


def test(arm, dynamics, goal, renderer, controller, gui, args, dist_limit, time_limit):

    num_steps = round(time_limit / args.time_step)
    initial_state = np.zeros((arm.dynamics.get_state_dim(), 1))  # position + velocity
    initial_state[0] = -math.pi / 2.0

    initial_pos = arm.dynamics.compute_fk(initial_state)
    initial_dist = np.linalg.norm(goal - initial_pos)
    print(f"Initial distance to goal: {initial_dist:.4f}")

    # Controller to reach goals
    arm.reset()
    action = np.zeros((arm.dynamics.get_action_dim(), 1))
    arm.goal = goal
    arm.set_state(initial_state)
    if renderer is not None:
        renderer.plot([(arm, "tab:blue")], 0.0, 0.0)
    for s in range(num_steps):
        state = arm.get_state()
        if s % controller.control_horizon == 0:
            action = controller.compute_action(dynamics, state, goal, action)
            arm.set_action(action)
        arm.advance()
        new_state = arm.get_state()
        pos_ee = arm.dynamics.compute_fk(new_state)
        dist = np.linalg.norm(pos_ee - goal)
        vel_ee = np.linalg.norm(arm.dynamics.compute_vel_ee(state))
        if renderer is not None and gui:
            renderer.plot([(arm, "tab:blue")], vel_ee, dist)

    # TODO: remove `dist`
    if dist < dist_limit[0] and vel_ee < 0.5:
        return "full", pos_ee, vel_ee, dist
    elif dist < dist_limit[1] and vel_ee < 0.5:
        return "partial", pos_ee, vel_ee, dist
    else:
        return "fail", pos_ee, vel_ee, dist


# Take random Goal
def sample_goal():
    goal = np.zeros((2, 1))
    r = np.random.uniform(low=0.05, high=1.95)
    theta = np.random.uniform(low=np.pi, high=2.0 * np.pi)
    goal[0, 0] = r * np.cos(theta)
    goal[1, 0] = r * np.sin(theta)
    return goal


def get_goal(radius, angle):
    angle -= np.pi / 2
    return radius * np.array([np.cos(angle), np.sin(angle)]).reshape(-1, 1)


def score_mpc_learnt_dynamics(controller, arm_student, model_path, gui):
    args, unknown = get_args()
    GOALS = {
        1: [get_goal(1, 0.4), get_goal(1, -0.75)],
        2: [sample_goal() for _ in range(16)],
        3: [
            get_goal(2.2, -1.0),
            get_goal(1.8, -0.25),
            get_goal(1.5, 7.1),
            get_goal(1.3, -0.5),
            get_goal(0.9, 5.1),
        ],
    }

    renderer = None
    if gui:
        renderer = Renderer()
        time.sleep(1)
    # Part2: Evaluate controller with learned dynamics
    score = 0.0
    print("Part2: EVALUATING CONTROLLER + LEARNED DYNAMICS")
    print("-----------------------------------------------")

    pass_trials = 0
    partial_pass_trials = 0
    fail_trials = 0
    total_distance = 0
    for num_links in range(2, 3):
        print("NUM_LINKS:", num_links)
        # Arm
        arm = Robot(
            ArmDynamicsTeacher(
                num_links=num_links,
                link_mass=args.link_mass,
                link_length=args.link_length,
                joint_viscous_friction=args.friction,
                dt=args.time_step,
            )
        )

        # Learnt dynamics
        dynamics = arm_student
        if not os.path.exists(model_path):
            print(f"model not found at {model_path}, skipping tests")
            continue
        try:
            dynamics.init_model(model_path, num_links, args.time_step, device=torch.device("cpu"))
        except Exception as e:
            print(e)
            print(f"Skipping tests")
            continue

        for i, goal in enumerate(GOALS[num_links]):
            print("\nTest ", i + 1)
            try:
                dist_limit = [0.2, 0.3]
                result, pos_ee, vel_ee, dist = test(
                    arm,
                    dynamics,
                    goal,
                    renderer,
                    controller,
                    gui,
                    args,
                    dist_limit=dist_limit,
                    time_limit=2.5,
                )
                total_distance += dist
            except Exception as e:
                print(e)
                continue

            if result == "full":
                print(
                    f"Success! :)\n Goal: {GOALS[num_links][i].reshape(-1).round(3)}, Final position: {pos_ee.reshape(-1).round(3)}, Final velocity: {vel_ee.reshape(-1).round(3)}, Distance to Goal: {round(dist, 3)}"
                )
                print("score:", "0.5/0.5")
                score += 0.5
                pass_trials += 1
            elif result == "partial":
                print(
                    f"Partial Success:|\n Goal: {GOALS[num_links][i].reshape(-1).round(3)}, Final position: {pos_ee.reshape(-1).round(3)}, Final velocity: {vel_ee.reshape(-1).round(3)}, Distance to Goal: {round(dist, 3)}"
                )
                print("score:", "0.3/0.5")
                score += 0.25
                partial_pass_trials += 1
            else:
                print(
                    f"Fail! :(\n Goal: {GOALS[num_links][i].reshape(-1).round(3)}, Final position: {pos_ee.reshape(-1).round(3)}, Final velocity: {vel_ee.reshape(-1).round(3)}, Distance to Goal: {round(dist, 3)}"
                )
                print("score:", "0/0.5")
                fail_trials += 1
    score = (score / 7.5) * 5
    print("       ")
    print("-------------------------")
    print("Part 2 SCORE: ", f"{min(score, 5)}/5")
    print("-------------------------")

    # if renderer is not None:
    #     renderer.plotter.terminate()
    # TODO: remove return for data
    return {
        "score": score,
        "pass_trials": pass_trials,
        "partial_pass_trials": partial_pass_trials,
        "fail_trials": fail_trials,
        "timestamp": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    }
