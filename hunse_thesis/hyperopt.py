from __future__ import absolute_import

from multiprocessing import Process, Pipe
import sys
import time

import hyperopt
import numpy as np

PY2 = sys.version_info[0] == 2
int_types = (int, long) if PY2 else (int,)


def is_integer(obj):
    return isinstance(obj, int_types + (np.integer,))


def is_number(obj, check_complex=False):
    types = ((float, complex, np.number) if check_complex else
             (float, np.floating))
    return is_integer(obj) or isinstance(obj, types)


def get_vals(trial):
    """Determine hyperparameter values given a ``Trial`` object"""
    # based on hyperopt/base.py:Trials:argmin
    return dict((k, v[0]) for k, v in trial['misc']['vals'].items() if v)


def wrap_cost(cost_fn, timeout=None, iters=1, verbose=0):
    """Wrap cost function to execute trials safely on a separate process.

    Parameters
    ----------
    cost_fn : callable
        The cost function (aka. objective function) to wrap. It follows the
        same specifications as normal Hyperopt cost functions.
    timeout : int
        Time to wait for process to complete, in seconds. If this time is
        reached, the process is re-tried if there are remaining iterations,
        otherwise marked as a failure. If ``None``, wait indefinitely.
    iters : int
        Number of times to allow the trial to timeout before marking it as
        a failure due to timeout.
    verbose : int
        How verbose this function should be. 0 is not verbose, 1 is verbose.

    Example
    -------
    def objective(args):
        case, val = args
        return val**2 if case else val

    space = [hp.choice('case', [False, True]), hp.uniform('val', -1, 1)]
    safe_objective = wrap_cost(objective, timeout=2, iters=2, verbose=1)
    best = hyperopt.fmin(safe_objective, space, max_evals=100)

    Notes
    -----
    Based on code from https://github.com/hyperopt/hyperopt-sklearn
    """
    def _cost_fn(*args, **kwargs):
        _conn = kwargs.pop('_conn')
        try:
            t_start = time.time()
            rval = cost_fn(*args, **kwargs)
            t_done = time.time()

            if not isinstance(rval, dict):
                rval = dict(loss=rval)
            assert 'loss' in rval, "Returned dictionary must include loss"
            loss = rval['loss']
            assert is_number(loss), "Returned loss must be a number type"
            rval.setdefault('status', hyperopt.STATUS_OK if np.isfinite(loss)
                            else hyperopt.STATUS_FAIL)
            rval.setdefault('duration', t_done - t_start)
            rtype = 'return'

        except Exception as exc:
            rval = exc
            rtype = 'raise'

        # -- return the result to calling process
        _conn.send((rtype, rval))

    def wrapper(*args, **kwargs):
        for k in range(iters):
            conn1, conn2 = Pipe()
            kwargs['_conn'] = conn2
            th = Process(target=_cost_fn, args=args, kwargs=kwargs)
            th.start()
            if conn1.poll(timeout):
                fn_rval = conn1.recv()
                th.join()
            else:
                if verbose >= 1:
                    print("TRIAL TIMED OUT (%d/%d)" % (k+1, iters))
                th.terminate()
                th.join()
                continue

            assert fn_rval[0] in ('raise', 'return')
            if fn_rval[0] == 'raise':
                raise fn_rval[1]
            else:
                return fn_rval[1]

        return {'status': hyperopt.STATUS_FAIL,
                'failure': 'timeout'}

    return wrapper
