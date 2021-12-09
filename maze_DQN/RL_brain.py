import  numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network(off-policy)
class DeepQNetwork:
    def __init__(
            self,
            n_features,
            n_actions,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_step=200,
            memory_size=2000,
            batch_size=32,
            output_graph=False
    ):
        self.n_features = n_features  # 环境的特征数量
        self.n_actions = n_actions  # 动作数量
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # lambada
        self.epsilon = e_greedy  # 在epsilon范围内选择value最大值
        self.replace_target_step = replace_target_step  # 参数替换的步数
        self.memory_size = memory_size  # 每个memory的大小
        self.batch_size = batch_size  # 训练batch的size
        self.learn_step_contour = 0  # 总学习步数

        # 初始化存储器[s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # 利用get_collection获得eval和target两个神经网络的参数
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        """
        赋值语句，可以将后面一个参数的值赋给前一个
        """
        # 将e的参数给t
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        # 存储历史cost的值
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        """
        tf.placeholder:一个占位函数，python会为其分配内存，但是要等到session启动的时候
        程序才会真正的将这个网络构架运行，相当于创建了一个镜像网络
        """
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        # 输入值：列的内容为观测到的环境的特点，一共n个，用于计算当前对环境的估计
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        """
        tf.variable_scope(scope_name):用来区分变量(在神经网络中为各种参数)的不同作用域，
        相当于一个命名管理器，创建出来的变量名字为scope_name/variable_name
        """
        # 该层神经网络用于计算q_eval
        with tf.variable_scope('eval_net'):
            """
            TRAINABLE_VARIABLES:会被训练的对象，MODEL_VARIABLES的子集
            MODEL_VARIABLES:变量对象的子集，在模型中被用作接口，GLOBAL_VARIABLES的子集
            GLOBAL_VARIABLES:变量对象的默认集合，会在环境中共享
            tensorflow的collection提供一个全局的存储机制，
            不会受到变量名生存空间的影响。一处保存，到处可取
            """
            # 神经网络参数
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]  # collection的名字
            n_l1 = 10  # 隐藏层神经元的个数
            w_initializer = tf.random_normal_initializer(mean=0, stddev=0.3)  # 正态分布初始化
            b_initializer = tf.constant_initializer(0.1)  # 常数生成器

            # q_eval神经网络的第一层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                """
                tf.nn.relu是线性修正，将小于0的部分去掉
                tf.matmul就是矩阵相乘
                """
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # q_eval神经网络第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

            # q_eval神经网络的损失值，和q_target对比
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

            # 一个优化函数
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        # 该层网络用于计算q_target
        with tf.variable_scope('target_net'):
            # 神经网络的参数
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # q_target的第一层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # q_target的第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    # 存储当前状态， action， 得分，预测状态
    def store_tran(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 将当前状态， action， 得分，预测状态进行水平堆叠
        transition = np.hstack((s, [a, r], s_))
        # 保证index在memory范围内
        index = self.memory_counter % self.memory_size
        # self.memory中加入一行
        self.memory[index, :] = transition
        # 数量+1
        self.memory_counter += 1

    def choose_action(self, state):
        #将state转换成行向量
        state = state[np.newaxis, :]

        # 如果再epsilon内，通过运行得到最大action,否则随机运算
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: state})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self, the_reward):
        # 判断是否需要将两个网络的参数赋值
        if self.learn_step_contour % self.replace_target_step == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 提取相应的数据进行训练
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # 根据sample_index提取batch
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval], feed_dict={
                self.s_: batch_memory[:, -self.n_features:],
                self.s: batch_memory[:, :self.n_features]
            }
        )

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # 取出eval的每一个行为
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        # 取出eval的每一个得分
        reward = batch_memory[:, self.n_features + 1]

        # 取出每一行的最大值
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # 训练
        self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.q_target: q_target
            }
        )

        if self.learn_step_contour % 100 == 0:
            self.cost_his.append(the_reward)
        else:
            self.cost_his[len(self.cost_his) - 1] = self.cost_his[len(self.cost_his) - 1] + the_reward
        self.learn_step_contour += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('reward')
        plt.xlabel('training steps')
        plt.show()
