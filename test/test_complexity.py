import numpy as np
import process_movement as serve
import train_movement_pattern as train


def test_runtime_multi_dof():
    ''' Time both the MULTI-TASK ELASTIC NET and the BFGS steps of LSDP algorithm
    Time LSDP by changing idx_range or p
    '''
    import multi_dof_lasso as lasso
    reload(lasso)
    date = '16.11.18'
    args = serve.create_default_args(date)
    args.plot = False
    args.ball_file = None
    joint_dict, ball_dict = serve.run_serve_demo(args)
    idx_move = joint_dict['idx_move']
    idx_range = 1
    ex = 0
    idx = np.arange(idx_move[0, ex], idx_move[1, ex], idx_range)
    q = joint_dict['x'][idx, :]
    t = joint_dict['t'][idx]
    t -= t[0]
    train.iter_multi_dof_lasso(t, q, p=500, measure_time=True)


def test_runtime_multi_demo():
    ''' Time cLSDP by changing idx_range or p'''
    import multi_demo_lasso as lasso
    reload(lasso)
    date = '16.11.18'
    examples = [2, 3, 4, 5, 6]
    args = serve.create_default_args(date)
    args.plot = False
    args.date = date
    args.num_examples = 15
    args.ball_file = None
    joint_dict, ball_dict = serve.run_serve_demo(args)
    idx_range = 1  # should be a divisor of 500
    t, Q = process_multi_demo_examples(joint_dict, examples, idx_range)
    train.iter_multi_demo_lasso(t, Q, p=500, measure_time=True)


def process_multi_demo_examples(joint_dict, examples, idx_range):

    idx_move = joint_dict['idx_move']
    # hacky, assuming they are all 1 sec long
    Q = np.zeros((500*7/idx_range, len(examples)))
    intercepts = np.zeros((7, len(examples)))
    for i, ex in enumerate(examples):
        idx = np.arange(idx_move[0, ex], idx_move[1, ex], idx_range)
        q = joint_dict['x'][idx, :]
        intercepts[:, i] = np.mean(q, axis=0)
        q_flat = q.T.flatten()
        Q[:, i] = q_flat
        t = joint_dict['t'][idx]  # assumed the same for each ex
    t -= t[0]

    stacked_intercepts = np.zeros((500*7/idx_range, len(examples)))
    for i, ex in enumerate(examples):
        m = np.tile(intercepts[:, i][:, np.newaxis], (1, 500/idx_range))
        stacked_intercepts[:, i] = m.flatten()
    return t, Q - stacked_intercepts
