import os
import numpy as np

from argparse import ArgumentParser
from matplotlib import pyplot as plt

COLORS = [
    # deepmind style
    '#0072B2',
    '#009E73',
    '#D55E00',
    '#CC79A7',
    # '#F0E442',
    '#d73027',  # RED
    # built-in color
    'blue', 'red', 'pink', 'cyan', 'magenta', 'yellow', 'black', 'purple',
    'brown', 'orange', 'teal', 'lightblue', 'lime', 'lavender', 'turquoise',
    'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue', 'green',
    # personal color
    '#313695',  # DARK BLUE
    '#74add1',  # LIGHT BLUE
    '#f46d43',  # ORANGE
    '#4daf4a',  # GREEN
    '#984ea3',  # PURPLE
    '#f781bf',  # PINK
    '#ffc832',  # YELLOW
    '#000000',  # BLACK
]

ENVS = [
    "halfcheetah-random-v0",
    "hopper-random-v0",
    "walker2d-random-v0",
    "halfcheetah-medium-v0",
    "hopper-medium-v0",
    "walker2d-medium-v0",
    "halfcheetah-expert-v0",
    "hopper-expert-v0",
    "walker2d-expert-v0",
    "halfcheetah-medium-expert-v0",
    "hopper-medium-expert-v0",
    "walker2d-medium-expert-v0",
    "halfcheetah-medium-replay-v0",
    "hopper-medium-replay-v0",
    "walker2d-medium-replay-v0",
]


def smooth(y, radius=0, mode='two_sided', valid_only=False):
    '''Smooth signal y, where radius is determines the size of the window.
    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / \
            np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / \
            np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


def plot_fig(args):
    os.makedirs(args.save_dir, exist_ok=True)

    if args.policy == 'td3':
        policies = ("TD3_BC")
    elif args.policy == 'sac':
        policies = ("SAC_BC")
    elif args.policy == 'sac_d':
        policies = ("SAC_BC_d")
    elif args.policy == 'all':
        policies = ("TD3_BC", "SAC_BC", "SAC_BC_d")
    else:
        raise ValueError(f"Invalid policy: {args.policy}")

    for env in ENVS:
        env_save_dir = os.path.join(args.save_dir, env)
        os.makedirs(env_save_dir, exist_ok=True)

        plt.figure()
        for policy_idx, policy in enumerate(policies):
            cur_policy_log_data = []
            steps = np.asarray(range(5000, 505000, 5000))
            for seed in range(5):
                npy_file_path = os.path.join(
                    'results', f"{policy}_{env}_{seed}.npy")
                smooth_log_data = smooth(np.load(npy_file_path)[:len(steps)])
                cur_policy_log_data.append(smooth_log_data)

            cur_policy_log_data = np.asarray(cur_policy_log_data)
            cur_policy_log_mean = np.mean(cur_policy_log_data, axis=0)
            cur_policy_log_std = np.std(cur_policy_log_data, axis=0)

            print(
                f'steps shape: {steps.shape}, cur_policy_log_data shape: {cur_policy_log_data.shape}')

            plt.plot(steps, cur_policy_log_mean,
                     color=COLORS[policy_idx], label=policy)
            plt.fill_between(steps, cur_policy_log_mean - cur_policy_log_std,
                             cur_policy_log_mean + cur_policy_log_std,
                             color=COLORS[policy_idx], alpha=.2)
        plt.legend()
        plt.xlabel('Training Steps')
        plt.ylabel('Episode Return')
        plt.savefig(os.path.join(env_save_dir, f"{args.policy}.pdf"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--policy', type=str,
                        choices=('td3', 'sac', 'all'), required=True)
    parser.add_argument('--save-dir', type=str, default='figs')
    args = parser.parse_args()
    plot_fig(args)
