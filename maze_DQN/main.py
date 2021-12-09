from maze_env import Maze
from RL_brain import DeepQNetwork


def main_learn():
    # 学习步数
    step = 0
    for episode in range(2000):
        # 初始化环境
        state_before = env.reset()

        # DQN主循环
        while True:
            # 刷新环境
            env.update()

            # 根据神经网络选择action
            action = RL.choose_action(state_before)

            # 得到下一步的state, reward, done(是否终止)
            state_next, reward, done = env.next_state(action)

            # DQN存储
            RL.store_tran(state_before, action, reward, state_next)

            # 若步数大于200才开始训练，并且每5步才开始训练一次
            if (step > 200) and (step % 5 == 0):
                RL.learn(reward)

            # 状态转移
            state_before = state_next
            step += 1

            # 如果终止，跳出循环
            if done:
                break

        # end of game
    print("game over")
    env.destroy()


if __name__ == "__main__":
    # 定义变量，env为环境，RL为神经网络
    env = Maze()
    RL = DeepQNetwork(
        env.n_features,
        env.n_action,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_step=200,  # 每 200 步替换一次 target_net 的参数
        memory_size=2000,  # 记忆上限
        output_graph=True  # 是否输出 tensorboard 文件
    )
    # 执行DQN
    env.after(100, main_learn)
    env.mainloop()
    # 查看误差曲线
    RL.plot_cost()
