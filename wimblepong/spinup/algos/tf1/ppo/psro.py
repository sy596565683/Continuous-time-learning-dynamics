import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.tf1.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.utils.ReplayBuffer import FPOHistoryBuffer, FPOEpochBuffer, PPOBuffer
from spinup.utils.test_policy import load_policy_and_env
from spinup.utils.nash_grid import NashGrid
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from wimblepong.wimblepong import WimblepongAutoReset



def make_env():
    return WimblepongAutoReset


def create_env(n_env, seed):
    env = SubprocVecEnv([make_env() for i in range(n_env)])
    return env


def psro(env_fn,
         req_path,
         actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
         vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
         target_kl=0.01, ent_coef=0.1, logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols rew_buf
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure 
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    # print("debug_start_", proc_id())
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    test_env = create_env(16, 1)
    ai_test_env = gym.make("WimblepongSimpleAI-v0")
    
    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    pi, logp, logp_pi, v, logp_all = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    precis_ent = tf.reduce_mean(-logp_all * tf.exp(logp_all)) * env.action_space.n

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph, (1 - clip_ratio) * adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv)) - ent_coef * precis_ent
    v_loss = tf.reduce_mean((ret_ph - v) ** 2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)  # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    def update():
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # psro recorder
    psro_result = []
    psro_path = req_path#  ########
    # print(psro_path)
    psro_iter = []
    policy_list = []
    psro_policy_train = None
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        if len(psro_iter) > 0:
            loc = np.random.choice([i for i in range(len(nash_prob))], p=nash_prob)
            psro_policy_train = policy_list[loc]
            # graph = tf.Graph()
            # with graph.as_default():
            #     if loc ==40:
            #         loc = 39.9
            #     _, psro_policy_train = load_policy_and_env(psro_path, itr=int(loc * 10))
        # print("debug_train collect", proc_id())
        for t in range(local_steps_per_epoch):
            action_input = []
            if len(psro_iter) == 0:
                a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o})
                action_input = a
                a = a[0]
                v_t = v_t[0]
                logp_t = logp_t[0]
            else:
                a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o[0].reshape(1, -1)})
                a_psro = psro_policy_train(o[1][None, :])
                action_input = [a[0], a_psro[0]]
                a = a[0]
            o2, r, d, _ = env.step(action_input)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o[0], a, r[0], v_t, logp_t)
            logger.store(VVals=v_t)

            # Update obs (critical!)
            o = o2

            terminal = d[0] or (ep_len == max_ep_len)
            if terminal or (t == local_steps_per_epoch - 1):
                if not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if d[0] else sess.run(v, feed_dict={x_ph: o[0].reshape(1, -1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0
                if len(psro_iter) > 0:
                    loc = np.random.choice([i for i in range(len(nash_prob))], p=nash_prob)
                    psro_policy_train = policy_list[loc]

                    # if loc ==400:
                    #     loc = 'last'
                    # graph = tf.Graph()
                    # with graph.as_default():
                    #     _, psro_policy_train = load_policy_and_env(psro_path, itr=loc)
        
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, epoch)
            local_test_rounds = int(1000 / num_procs())
            
            # test psro
            psro_iter.append(epoch)
            graph = tf.Graph()
            with graph.as_default():
                _, psro_policy1 = load_policy_and_env(psro_path, itr='last')
            policy_list.append(psro_policy1)
            current_result = np.ones(len(psro_iter))
            for p_i in range(len(psro_iter)):

                if p_i < len(psro_iter) - 20:
                    continue
                
                local_test_rounds = int(1000 / num_procs())
                test_wins = 0
                test_loses = 0

                o_test, r_test, d_test = test_env.reset(), 0, False
                # graph = tf.Graph()
                # with graph.as_default():
                #     _, psro_policy = load_policy_and_env(psro_path, itr=psro_iter[p_i])
                psro_policy = policy_list[p_i]
                for i in range(100000):
                    a_test, v_test, log_pi_test = sess.run(get_action_ops,
                                                           feed_dict={x_ph: o_test[:, 0, :]})
                    a_psro = psro_policy(o_test[:, 1, :])
                    action_input = np.concatenate([np.expand_dims(a_test, -1), np.expand_dims(a_psro, -1)], axis=-1)
                    o_test, r_test, d_test, _ = test_env.step(action_input)
                    for di in range(len(d_test)):
                        if np.max(d_test[di]):
                            if r_test[di][0] > 0:
                                test_wins += 1
                            elif r_test[di][0] < 0:
                                test_loses += 1
                        if test_wins + test_loses > local_test_rounds:  ########
                            break
                    if test_wins + test_loses > local_test_rounds:  ##########
                        break

                score_v = (test_wins - test_loses) / local_test_rounds
                score_v_tot = mpi_statistics_scalar([score_v], with_min_and_max=False)
                current_result[p_i] = score_v_tot[0]
            psro_result.append(current_result)
            psro_matrix = np.zeros([len(psro_iter), len(psro_iter)])
            for i in range(len(psro_iter)):
                for j in range(i+1): # , len(psro_iter)
                    psro_matrix[i][j] = psro_result[i][j]
            player1_payoff = psro_matrix - psro_matrix.T
            psro_payoff = np.concatenate([np.expand_dims(player1_payoff, axis=-1),
                                          np.expand_dims((- player1_payoff), axis=-1)], axis=-1)
            nash_solver = NashGrid(psro_payoff).mixed_strategy_solutions()
            nash_prob = np.zeros([len(psro_iter)])
            for keyi in list(nash_solver[0].keys()):
                nash_prob[keyi] = nash_solver[0][keyi]
            nash_prob = nash_prob/(1e-16 + np.sum(nash_prob))
            
            if len(psro_iter) > 0 and psro_policy_train is None:
                graph = tf.Graph()
                with graph.as_default():
                    _, psro_policy_train = load_policy_and_env(psro_path, itr='last')
            else:
                loc = np.random.choice([i for i in range(len(nash_prob))], p=nash_prob)
                graph = tf.Graph()
                with graph.as_default():
                    if loc ==40:
                        loc = 39.9
                    _, psro_policy_train = load_policy_and_env(psro_path, itr=int(loc * 10))
            if proc_id()==0:
                print(nash_prob, len(policy_list), len(psro_iter),np.expand_dims(psro_matrix - psro_matrix.T, axis=-1).squeeze())
                np.savez("/".join(req_path.split("/")+["nash"+str(epoch)]),nash_prob,player1_payoff)

            test_wins = 0
            test_loses = 0
            for i in range(local_test_rounds):
                o_test, r_test, d_test = ai_test_env.reset(), 0, False
                for i in range(max_ep_len):
                    a_test, v_test, log_pi_test = sess.run(get_action_ops,
                                                           feed_dict={x_ph: np.reshape(o_test, [1, -1])})
                    o_test, r_test, d_test, _ = ai_test_env.step(a_test)
                    if np.max(d_test):
                        if r_test > 0:
                            test_wins += 1
                        elif r_test < 0:
                            test_loses += 1
                        break
        logger.store(EvalWin=test_wins / local_test_rounds)
        logger.store(EvalScore=(test_wins + 0.5 * (local_test_rounds - test_wins - test_loses)) / local_test_rounds)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch * 2)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('EvalWin', average_only=True)
        logger.log_tabular('EvalScore', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='psro')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    psro(lambda: gym.make(args.env), actor_critic=core.mlp_actor_critic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
