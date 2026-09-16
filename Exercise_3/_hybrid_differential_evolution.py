"""
differential_evolution: The differential evolution global optimization algorithm
Added by Andrew Nelson 2014
Modified to Hybrid Differential Evolution (HDE) by Igor Araus
"""
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper

from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
                                         NonlinearConstraint, LinearConstraint)
from scipy.sparse import issparse


__all__ = ['hybrid_differential_evolution']


_MACHEPS = np.finfo(np.float64).eps


def hybrid_differential_evolution(func, bounds, args=(), strategy='best1bin',
                                  maxiter=1000, popsize=15, tol=0.01,
                                  mutation=(0.5, 1), recombination=0.7, seed=None,
                                  callback=None, disp=False, polish=True,
                                  init='latinhypercube', atol=0, updating='immediate',
                                  workers=1, constraints=(), x0=None, *,
                                  integrality=None, vectorized=False, 
                                  g_hybrid=None):

    """Finds the global minimum of a multivariate function using a Hybrid
    Differential Evolution algorithm.

    Parameters
    ----------

    g_hybrid : int, optional
        If specified, defines the number of generations after which the
        standard evolution is interrupted to apply a local optimization
        (SLSQP) to the current best individual in the population.

    """

    # using a context manager means that any created Pool objects are
    # cleared up.
    
    with HybridDifferentialEvolutionSolver(func, bounds, args=args,
                                           strategy=strategy,
                                           maxiter=maxiter,
                                           popsize=popsize, tol=tol,
                                           mutation=mutation,
                                           recombination=recombination,
                                           seed=seed, polish=polish,
                                           callback=callback,
                                           disp=disp, init=init, atol=atol,
                                           updating=updating,
                                           workers=workers,
                                           constraints=constraints,
                                           x0=x0,
                                           integrality=integrality,
                                           vectorized=vectorized,
                                           g_hybrid=g_hybrid) as solver:
    
        ret = solver.solve()

    return ret


class HybridDifferentialEvolutionSolver:

    """This class implements the hybrid differential evolution solver"""


    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {'best1bin': '_best1',
                 'randtobest1bin': '_randtobest1',
                 'currenttobest1bin': '_currenttobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2',
                 'rand1bin': '_rand1'}
    _exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'currenttobest1exp': '_currenttobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2'}

    __init_error_msg = ("The population initialization method must be one of "
                        "'latinhypercube' or 'random', or an array of shape "
                        "(S, N) where N is the number of parameters and S>5")


    def __init__(self, func, bounds, args=(),
                 strategy='best1bin', maxiter=1000, popsize=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 maxfun=np.inf, callback=None, disp=False, polish=True,
                 init='latinhypercube', atol=0, updating='immediate',
                 workers=1, constraints=(), x0=None, *, integrality=None,
                 vectorized=False, g_hybrid=None):


        if callable(strategy):
            # a callable strategy is going to be stored in self.strategy anyway
            pass
        elif strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy


        self.g_hybrid = g_hybrid


        self.callback = _wrap_callback(callback, "differential_evolution")
        self.polish = polish

        # set the updating / parallelisation options
        if updating in ['immediate', 'deferred']:
            self._updating = updating

        self.vectorized = vectorized

        # want to use parallelisation, but updating is immediate
        if workers != 1 and updating == 'immediate':
            warnings.warn("differential_evolution: the 'workers' keyword has"
                          " overridden updating='immediate' to"
                          " updating='deferred'", UserWarning, stacklevel=2)
            self._updating = 'deferred'

        if vectorized and workers != 1:
            warnings.warn("differential_evolution: the 'workers' keyword"
                          " overrides the 'vectorized' keyword", stacklevel=2)
            self.vectorized = vectorized = False

        if vectorized and updating == 'immediate':
            warnings.warn("differential_evolution: the 'vectorized' keyword"
                          " has overridden updating='immediate' to updating"
                          "='deferred'", UserWarning, stacklevel=2)
            self._updating = 'deferred'

        # an object with a map method.
        if vectorized:
            def maplike_for_vectorized_func(func, x):
                # send an array (N, S) to the user func,
                # expect to receive (S,). Transposition is required because
                # internally the population is held as (S, N)
                return np.atleast_1d(func(x.T))
            workers = maplike_for_vectorized_func

        self._mapwrapper = MapWrapper(workers)

        # relative and absolute tolerances for convergence
        self.tol, self.atol = tol, atol

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.scale = mutation
        if (not np.all(np.isfinite(mutation)) or
                np.any(np.array(mutation) >= 2) or
                np.any(np.array(mutation) < 0)):
            raise ValueError('The mutation constant must be a float in '
                             'U[0, 2), or specified as a tuple(min, max)'
                             ' where min < max and min, max are in U[0, 2).')

        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination

        # we create a wrapped function to allow the use of map (and Pool.map
        # in the future)
        self.func = _FunctionWrapper(func, args)
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        if isinstance(bounds, Bounds):
            self.limits = np.array(new_bounds_to_old(bounds.lb,
                                                     bounds.ub,
                                                     len(bounds.lb)),
                                   dtype=float).T
        else:
            self.limits = np.array(bounds, dtype='float').T

        if (np.size(self.limits, 0) != 2 or not
                np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence containing finite '
                             'real valued (min, max) pairs for each value'
                             ' in x')

        if maxiter is None:  # the default used to be None
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:  # the default used to be None
            maxfun = np.inf
        self.maxfun = maxfun

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])
        with np.errstate(divide='ignore'):
            # if lb == ub then the following line will be 1/0, which is why
            # we ignore the divide by zero warning. The result from 1/0 is
            # inf, so replace those values by 0.
            self.__recip_scale_arg2 = 1 / self.__scale_arg2
            self.__recip_scale_arg2[~np.isfinite(self.__recip_scale_arg2)] = 0

        self.parameter_count = np.size(self.limits, 1)

        self.random_number_generator = check_random_state(seed)

        # Which parameters are going to be integers?
        if np.any(integrality):
            # # user has provided a truth value for integer constraints
            integrality = np.broadcast_to(
                integrality,
                self.parameter_count
            )
            integrality = np.asarray(integrality, bool)
            # For integrality parameters change the limits to only allow
            # integer values lying between the limits.
            lb, ub = np.copy(self.limits)

            lb = np.ceil(lb)
            ub = np.floor(ub)
            if not (lb[integrality] <= ub[integrality]).all():
                # there's a parameter that doesn't have an integer value
                # lying between the limits
                raise ValueError("One of the integrality constraints does not"
                                 " have any possible integer values between"
                                 " the lower/upper bounds.")
            nlb = np.nextafter(lb[integrality] - 0.5, np.inf)
            nub = np.nextafter(ub[integrality] + 0.5, -np.inf)

            self.integrality = integrality
            self.limits[0, self.integrality] = nlb
            self.limits[1, self.integrality] = nub
        else:
            self.integrality = False

        # check for equal bounds
        eb = self.limits[0] == self.limits[1]
        eb_count = np.count_nonzero(eb)

        # default population initialization is a latin hypercube design, but
        # there are other population initializations possible.
        # the minimum is 5 because 'best2bin' requires a population that's at
        # least 5 long
        # 202301 - reduced population size to account for parameters with
        # equal bounds. If there are no varying parameters set N to at least 1
        self.num_population_members = max(
            5,
            popsize * max(1, self.parameter_count - eb_count)
        )
        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        self._nfev = 0
        # check first str otherwise will fail to compare str with array
        if isinstance(init, str):
            if init == 'latinhypercube':
                self.init_population_lhs()
            elif init == 'sobol':
                # must be Ns = 2**m for Sobol'
                n_s = int(2 ** np.ceil(np.log2(self.num_population_members)))
                self.num_population_members = n_s
                self.population_shape = (self.num_population_members,
                                         self.parameter_count)
                self.init_population_qmc(qmc_engine='sobol')
            elif init == 'halton':
                self.init_population_qmc(qmc_engine='halton')
            elif init == 'random':
                self.init_population_random()
            else:
                raise ValueError(self.__init_error_msg)
        else:
            self.init_population_array(init)

        if x0 is not None:
            # scale to within unit interval and
            # ensure parameters are within bounds.
            x0_scaled = self._unscale_parameters(np.asarray(x0))
            if ((x0_scaled > 1.0) | (x0_scaled < 0.0)).any():
                raise ValueError(
                    "Some entries in x0 lay outside the specified bounds"
                )
            self.population[0] = x0_scaled

        # infrastructure for constraints
        self.constraints = constraints
        self._wrapped_constraints = []

        if hasattr(constraints, '__len__'):
            # sequence of constraints, this will also deal with default
            # keyword parameter
            for c in constraints:
                self._wrapped_constraints.append(
                    _ConstraintWrapper(c, self.x)
                )
        else:
            self._wrapped_constraints = [
                _ConstraintWrapper(constraints, self.x)
            ]
        self.total_constraints = np.sum(
            [c.num_constr for c in self._wrapped_constraints]
        )
        self.constraint_violation = np.zeros((self.num_population_members, 1))
        self.feasible = np.ones(self.num_population_members, bool)

        self.disp = disp

    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Each parameter range needs to be sampled uniformly. The scaled
        # parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the following
        # size:
        segsize = 1.0 / self.num_population_members

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (segsize * rng.uniform(size=self.population_shape)

        # Offset each segment to cover the entire parameter range [0, 1)
                   + np.linspace(0., 1., self.num_population_members,
                                 endpoint=False)[:, np.newaxis])

        # Create an array for population of candidate solutions.
        self.population = np.zeros_like(samples)

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_qmc(self, qmc_engine):
        """Initializes the population with a QMC method.

        QMC methods ensures that each parameter is uniformly
        sampled over its range.

        Parameters
        ----------
        qmc_engine : str
            The QMC method to use for initialization. Can be one of
            ``latinhypercube``, ``sobol`` or ``halton``.

        """
        from scipy.stats import qmc

        rng = self.random_number_generator

        # Create an array for population of candidate solutions.
        if qmc_engine == 'latinhypercube':
            sampler = qmc.LatinHypercube(d=self.parameter_count, seed=rng)
        elif qmc_engine == 'sobol':
            sampler = qmc.Sobol(d=self.parameter_count, seed=rng)
        elif qmc_engine == 'halton':
            sampler = qmc.Halton(d=self.parameter_count, seed=rng)
        else:
            raise ValueError(self.__init_error_msg)

        self.population = sampler.random(n=self.num_population_members)

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_random(self):
        """
        Initializes the population at random. This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.uniform(size=self.population_shape)

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_array(self, init):
        """
        Initializes the population with a user specified population.

        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (S, N), where N is the number of parameters.
            The population is clipped to the lower and upper bounds.
        """
        # make sure you're using a float array
        popn = np.asarray(init, dtype=np.float64)

        if (np.size(popn, 0) < 5 or
                popn.shape[1] != self.parameter_count or
                len(popn.shape) != 2):
            raise ValueError("The population supplied needs to have shape"
                             " (S, len(x)), where S > 4.")

        # scale values and clip to bounds, assigning to population
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)

        self.num_population_members = np.size(self.population, 0)

        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    @property
    def x(self):
        """
        The best solution from the solver
        """
        return self._scale_parameters(self.population[0])

    @property
    def convergence(self):
        """
        The standard deviation of the population energies divided by their
        mean.
        """
        if np.any(np.isinf(self.population_energies)):
            return np.inf
        return (np.std(self.population_energies) /
                (np.abs(np.mean(self.population_energies)) + _MACHEPS))

    def converged(self):
        """
        Return True if the solver has converged.
        """
        if np.any(np.isinf(self.population_energies)):
            return False

        return (np.std(self.population_energies) <=
                self.atol +
                self.tol * np.abs(np.mean(self.population_energies)))

    def solve(self):
        """
        Runs the HybridDifferentialEvolutionSolver.
        """
        nit, warning_flag = 0, False
        status_message = _status_message['success']

        # The population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies.
        # Although this is also done in the evolve generator it's possible
        # that someone can set maxiter=0, at which point we still want the
        # initial energies to be calculated (the following loop isn't run).
        if np.all(np.isinf(self.population_energies)):
            self.feasible, self.constraint_violation = (
                self._calculate_population_feasibilities(self.population))

            # only work out population energies for feasible solutions
            self.population_energies[self.feasible] = (
                self._calculate_population_energies(
                    self.population[self.feasible]))

            self._promote_lowest_energy()

        # do the optimization.
        for nit in range(1, self.maxiter + 1):
            
            
            if self.g_hybrid is not None and nit % self.g_hybrid == 0:
                if self.disp:
                    print(f"HDE: Running local SLSQP optimization at generation {nit}...")

                # Local optimization with SLSQP on the current best individual
                res_local = minimize(
                    self.func,
                    self.x,
                    method='SLSQP',
                    bounds=self.limits.T,
                    constraints=self.constraints
                )

                self._nfev += res_local.nfev

                # Replace the original individual in the population (assuming it improves or keeps optimized feasibility)
                self.population[0] = np.clip(self._unscale_parameters(res_local.x), 0, 1)
                self.population_energies[0] = res_local.fun
                
                # Recalcular factibilidad del nuevo mejor individuo insertado
                if self._wrapped_constraints:
                    scaled_pop0 = self._scale_parameters(np.atleast_2d(self.population[0]))
                    cv = self._constraint_violation_fn(scaled_pop0)
                    self.constraint_violation[0] = cv[0]
                    self.feasible[0] = not (np.sum(cv[0]) > 0)
                else:
                    self.feasible[0] = True
                    self.constraint_violation[0] = np.zeros(1)
            else:
                # evolve the population by a generation
                try:
                    next(self)
                except StopIteration:
                    warning_flag = True
                    if self._nfev > self.maxfun:
                        status_message = _status_message['maxfev']
                    elif self._nfev == self.maxfun:
                        status_message = ('Maximum number of function evaluations'
                                          ' has been reached.')
                    break
            

            if self.disp:
                
                print(f"hybrid_differential_evolution step {nit}: f(x)="
                
                      f" {self.population_energies[0]}"
                      )

            if self.callback:
                c = self.tol / (self.convergence + _MACHEPS)
                res = self._result(nit=nit, message="in progress")
                res.convergence = c
                try:
                    warning_flag = bool(self.callback(res))
                except StopIteration:
                    warning_flag = True

                if warning_flag:
                    status_message = 'callback function requested stop early'

            # should the solver terminate?
            if warning_flag or self.converged():
                break

        else:
            status_message = _status_message['maxiter']
            warning_flag = True

        DE_result = self._result(
            nit=nit, message=status_message, warning_flag=warning_flag
        )

        if self.polish and not np.all(self.integrality):
            # can't polish if all the parameters are integers
            if np.any(self.integrality):
                # set the lower/upper bounds equal so that any integrality
                # constraints work.
                limits, integrality = self.limits, self.integrality
                limits[0, integrality] = DE_result.x[integrality]
                limits[1, integrality] = DE_result.x[integrality]

            polish_method = 'L-BFGS-B'

            if self._wrapped_constraints:
                polish_method = 'trust-constr'

                constr_violation = self._constraint_violation_fn(DE_result.x)
                if np.any(constr_violation > 0.):
                    warnings.warn("differential evolution didn't find a "
                                  "solution satisfying the constraints, "
                                  "attempting to polish from the least "
                                  "infeasible solution",
                                  UserWarning, stacklevel=2)
            if self.disp:
                print(f"Polishing solution with '{polish_method}'")
            result = minimize(self.func,
                              np.copy(DE_result.x),
                              method=polish_method,
                              bounds=self.limits.T,
                              constraints=self.constraints)

            self._nfev += result.nfev
            DE_result.nfev = self._nfev

            # Polishing solution is only accepted if there is an improvement in
            # cost function, the polishing was successful and the solution lies
            # within the bounds.
            if (result.fun < DE_result.fun and
                    result.success and
                    np.all(result.x <= self.limits[1]) and
                    np.all(self.limits[0] <= result.x)):
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                # to keep internal state consistent
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)

        if self._wrapped_constraints:
            DE_result.constr = [c.violation(DE_result.x) for
                                c in self._wrapped_constraints]
            DE_result.constr_violation = np.max(
                np.concatenate(DE_result.constr))
            DE_result.maxcv = DE_result.constr_violation
            if DE_result.maxcv > 0:
                # if the result is infeasible then success must be False
                DE_result.success = False
                DE_result.message = ("The solution does not satisfy the "
                                     f"constraints, MAXCV = {DE_result.maxcv}")

        return DE_result

    def _result(self, **kwds):
        # form an intermediate OptimizeResult
        nit = kwds.get('nit', None)
        message = kwds.get('message', None)
        warning_flag = kwds.get('warning_flag', False)
        result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=self._nfev,
            nit=nit,
            message=message,
            success=(warning_flag is not True),
            population=self._scale_parameters(self.population),
            population_energies=self.population_energies
        )
        if self._wrapped_constraints:
            result.constr = [c.violation(result.x)
                             for c in self._wrapped_constraints]
            result.constr_violation = np.max(np.concatenate(result.constr))
            result.maxcv = result.constr_violation
            if result.maxcv > 0:
                result.success = False

        return result

    def _calculate_population_energies(self, population):
        """
        Calculate the energies of a population.
        """
        num_members = np.size(population, 0)
        # S is the number of function evals left to stay under the
        # maxfun budget
        S = min(num_members, self.maxfun - self._nfev)

        energies = np.full(num_members, np.inf)

        parameters_pop = self._scale_parameters(population)
        try:
            calc_energies = list(
                self._mapwrapper(self.func, parameters_pop[0:S])
            )
            calc_energies = np.squeeze(calc_energies)
        except (TypeError, ValueError) as e:
            # wrong number of arguments for _mapwrapper
            # or wrong length returned from the mapper
            raise RuntimeError(
                "The map-like callable must be of the form f(func, iterable), "
                "returning a sequence of numbers the same length as 'iterable'"
            ) from e

        if calc_energies.size != S:
            if self.vectorized:
                raise RuntimeError("The vectorized function must return an"
                                   " array of shape (S,) when given an array"
                                   " of shape (len(x), S)")
            raise RuntimeError("func(x, *args) must return a scalar value")

        energies[0:S] = calc_energies

        if self.vectorized:
            self._nfev += 1
        else:
            self._nfev += S

        return energies

    def _promote_lowest_energy(self):
        # swaps 'best solution' into first population entry

        idx = np.arange(self.num_population_members)
        feasible_solutions = idx[self.feasible]
        if feasible_solutions.size:
            # find the best feasible solution
            idx_t = np.argmin(self.population_energies[feasible_solutions])
            l = feasible_solutions[idx_t]
        else:
            # no solution was feasible, use 'best' infeasible solution, which
            # will violate constraints the least
            l = np.argmin(np.sum(self.constraint_violation, axis=1))

        self.population_energies[[0, l]] = self.population_energies[[l, 0]]
        self.population[[0, l], :] = self.population[[l, 0], :]
        self.feasible[[0, l]] = self.feasible[[l, 0]]
        self.constraint_violation[[0, l], :] = (
        self.constraint_violation[[l, 0], :])

    def _constraint_violation_fn(self, x):
        """
        Calculates total constraint violation for all the constraints, for a
        set of solutions.
        """
        # how many solution vectors you're calculating constraint violations
        # for
        S = np.size(x) // self.parameter_count
        _out = np.zeros((S, self.total_constraints))
        offset = 0
        for con in self._wrapped_constraints:
            # the input/output of the (vectorized) constraint function is
            # {(N, S), (N,)} --> (M, S)
            # The input to _constraint_violation_fn is (S, N) or (N,), so
            # transpose to pass it to the constraint. The output is transposed
            # from (M, S) to (S, M) for further use.
            c = con.violation(x.T).T

            # The shape of c should be (M,), (1, M), or (S, M). Check for
            # those shapes, as an incorrect shape indicates that the
            # user constraint function didn't return the right thing, and
            # the reshape operation will fail. Intercept the wrong shape
            # to give a reasonable error message. I'm not sure what failure
            # modes an inventive user will come up with.
            if c.shape[-1] != con.num_constr or (S > 1 and c.shape[0] != S):
                raise RuntimeError("An array returned from a Constraint has"
                                   " the wrong shape. If `vectorized is False`"
                                   " the Constraint should return an array of"
                                   " shape (M,). If `vectorized is True` then"
                                   " the Constraint must return an array of"
                                   " shape (M, S), where S is the number of"
                                   " solution vectors and M is the number of"
                                   " constraint components in a given"
                                   " Constraint object.")

            # the violation function may return a 1D array, but is it a
            # sequence of constraints for one solution (S=1, M>=1), or the
            # value of a single constraint for a sequence of solutions
            # (S>=1, M=1)
            c = np.reshape(c, (S, con.num_constr))
            _out[:, offset:offset + con.num_constr] = c
            offset += con.num_constr

        return _out

    def _calculate_population_feasibilities(self, population):
        """
        Calculate the feasibilities of a population.
        """
        num_members = np.size(population, 0)
        if not self._wrapped_constraints:
            # shortcut for no constraints
            return np.ones(num_members, bool), np.zeros((num_members, 1))

        # (S, N)
        parameters_pop = self._scale_parameters(population)

        if self.vectorized:
            # (S, M)
            constraint_violation = np.array(
                self._constraint_violation_fn(parameters_pop)
            )
        else:
            # (S, 1, M)
            constraint_violation = np.array([self._constraint_violation_fn(x)
                                             for x in parameters_pop])
            # if you use the list comprehension in the line above it will
            # create an array of shape (S, 1, M), because each iteration
            # generates an array of (1, M). In comparison the vectorized
            # version returns (S, M). It's therefore necessary to remove axis 1
            constraint_violation = constraint_violation[:, 0]

        feasible = ~(np.sum(constraint_violation, axis=1) > 0)

        return feasible, constraint_violation

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._mapwrapper.__exit__(*args)

    def _accept_trial(self, energy_trial, feasible_trial, cv_trial,
                      energy_orig, feasible_orig, cv_orig):
        """
        Trial is accepted if:
        * it satisfies all constraints and provides a lower or equal objective
          function value, while both the compared solutions are feasible
        - or -
        * it is feasible while the original solution is infeasible,
        - or -
        * it is infeasible, but provides a lower or equal constraint violation
          for all constraint functions.
        """
        if feasible_orig and feasible_trial:
            return energy_trial <= energy_orig
        elif feasible_trial and not feasible_orig:
            return True
        elif not feasible_trial and (cv_trial <= cv_orig).all():
            # cv_trial < cv_orig would imply that both trial and orig are not
            # feasible
            return True

        return False

    def __next__(self):
        """
        Evolve the population by a single generation
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        if np.all(np.isinf(self.population_energies)):
            self.feasible, self.constraint_violation = (
                self._calculate_population_feasibilities(self.population))

            # only need to work out population energies for those that are
            # feasible
            self.population_energies[self.feasible] = (
                self._calculate_population_energies(
                    self.population[self.feasible]))

            self._promote_lowest_energy()

        if self.dither is not None:
            self.scale = self.random_number_generator.uniform(self.dither[0],
                                                              self.dither[1])

        if self._updating == 'immediate':
            # update best solution immediately
            for candidate in range(self.num_population_members):
                if self._nfev > self.maxfun:
                    raise StopIteration

                # create a trial solution
                trial = self._mutate(candidate)

                # ensuring that it's in the range [0, 1)
                self._ensure_constraint(trial)

                # scale from [0, 1) to the actual parameter value
                parameters = self._scale_parameters(trial)

                # determine the energy of the objective function
                if self._wrapped_constraints:
                    cv = self._constraint_violation_fn(parameters)
                    feasible = False
                    energy = np.inf
                    if not np.sum(cv) > 0:
                        # solution is feasible
                        feasible = True
                        energy = self.func(parameters)
                        self._nfev += 1
                else:
                    feasible = True
                    cv = np.atleast_2d([0.])
                    energy = self.func(parameters)
                    self._nfev += 1

                # compare trial and population member
                if self._accept_trial(energy, feasible, cv,
                                      self.population_energies[candidate],
                                      self.feasible[candidate],
                                      self.constraint_violation[candidate]):
                    self.population[candidate] = trial
                    self.population_energies[candidate] = np.squeeze(energy)
                    self.feasible[candidate] = feasible
                    self.constraint_violation[candidate] = cv

                    # if the trial candidate is also better than the best
                    # solution then promote it.
                    if self._accept_trial(energy, feasible, cv,
                                          self.population_energies[0],
                                          self.feasible[0],
                                          self.constraint_violation[0]):
                        self._promote_lowest_energy()

        elif self._updating == 'deferred':
            # update best solution once per generation
            if self._nfev >= self.maxfun:
                raise StopIteration

            # 'deferred' approach, vectorised form.
            # create trial solutions
            trial_pop = np.array(
                [self._mutate(i) for i in range(self.num_population_members)])

            # enforce bounds
            self._ensure_constraint(trial_pop)

            # determine the energies of the objective function, but only for
            # feasible trials
            feasible, cv = self._calculate_population_feasibilities(trial_pop)
            trial_energies = np.full(self.num_population_members, np.inf)

            # only calculate for feasible entries
            trial_energies[feasible] = self._calculate_population_energies(
                trial_pop[feasible])

            # which solutions are 'improved'?
            loc = [self._accept_trial(*val) for val in
                   zip(trial_energies, feasible, cv, self.population_energies,
                       self.feasible, self.constraint_violation)]
            loc = np.array(loc)
            self.population = np.where(loc[:, np.newaxis],
                                       trial_pop,
                                       self.population)
            self.population_energies = np.where(loc,
                                                trial_energies,
                                                self.population_energies)
            self.feasible = np.where(loc,
                                     feasible,
                                     self.feasible)
            self.constraint_violation = np.where(loc[:, np.newaxis],
                                                 cv,
                                                 self.constraint_violation)

            # make sure the best solution is updated if updating='deferred'.
            # put the lowest energy into the best solution position.
            self._promote_lowest_energy()

        return self.x, self.population_energies[0]

    def _scale_parameters(self, trial):
        """Scale from a number between 0 and 1 to parameters."""
        # trial either has shape (N, ) or (L, N), where L is the number of
        # solutions being scaled
        scaled = self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2
        if np.any(self.integrality):
            i = np.broadcast_to(self.integrality, scaled.shape)
            scaled[i] = np.round(scaled[i])
        return scaled

    def _unscale_parameters(self, parameters):
        """Scale from parameters to a number between 0 and 1."""
        return (parameters - self.__scale_arg1) * self.__recip_scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """Make sure the parameters lie between the limits."""
        mask = np.where((trial > 1) | (trial < 0))
        trial[mask] = self.random_number_generator.uniform(size=mask[0].shape)

    def _mutate(self, candidate):
        """Create a trial vector based on a mutation strategy."""
        rng = self.random_number_generator

        if callable(self.strategy):
            _population = self._scale_parameters(self.population)
            trial = np.array(
                self.strategy(candidate, _population, rng=rng), dtype=float
            )
            if trial.shape != (self.parameter_count,):
                raise RuntimeError(
                    "strategy must have signature"
                    " f(candidate: int, population: np.ndarray, rng=None)"
                    " returning an array of shape (N,)"
                )
            return self._unscale_parameters(trial)

        trial = np.copy(self.population[candidate])
        fill_point = rng.choice(self.parameter_count)

        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidate,
                                        self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        if self.strategy in self._binomial:
            crossovers = rng.uniform(size=self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        elif self.strategy in self._exponential:
            i = 0
            crossovers = rng.uniform(size=self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            crossovers[0] = True
            while (i < self.parameter_count and crossovers[i]):
                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1

            return trial

    def _best1(self, samples):
        """best1bin, best1exp"""
        r0, r1 = samples[:2]
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        """rand1bin, rand1exp"""
        r0, r1, r2 = samples[:3]
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _randtobest1(self, samples):
        """randtobest1bin, randtobest1exp"""
        r0, r1, r2 = samples[:3]
        bprime = np.copy(self.population[r0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r1] -
                                self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        """currenttobest1bin, currenttobest1exp"""
        r0, r1 = samples[:2]
        bprime = (self.population[candidate] + self.scale *
                  (self.population[0] - self.population[candidate] +
                   self.population[r0] - self.population[r1]))
        return bprime

    def _best2(self, samples):
        """best2bin, best2exp"""
        r0, r1, r2, r3 = samples[:4]
        bprime = (self.population[0] + self.scale *
                  (self.population[r0] + self.population[r1] -
                   self.population[r2] - self.population[r3]))

        return bprime

    def _rand2(self, samples):
        """rand2bin, rand2exp"""
        r0, r1, r2, r3, r4 = samples
        bprime = (self.population[r0] + self.scale *
                  (self.population[r1] + self.population[r2] -
                   self.population[r3] - self.population[r4]))

        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement. You can't have the original candidate either.
        """
        pool = np.arange(self.num_population_members)
        self.random_number_generator.shuffle(pool)

        idxs = []
        while len(idxs) < number_samples and len(pool) > 0:
            idx = pool[0]
            pool = pool[1:]
            if idx != candidate:
                idxs.append(idx)
        
        return idxs


class _ConstraintWrapper:
    """Object to wrap/evaluate user defined constraints.

    Very similar in practice to `PreparedConstraint`, except that no evaluation
    of jac/hess is performed (explicit or implicit).
    """
    def __init__(self, constraint, x0):
        self.constraint = constraint

        if isinstance(constraint, NonlinearConstraint):
            def fun(x):
                x = np.asarray(x)
                return np.atleast_1d(constraint.fun(x))
        elif isinstance(constraint, LinearConstraint):
            def fun(x):
                if issparse(constraint.A):
                    A = constraint.A
                else:
                    A = np.atleast_2d(constraint.A)

                res = A.dot(x)
                if x.ndim == 1 and res.ndim == 2:
                    res = np.asarray(res)[:, 0]

                return res
        elif isinstance(constraint, Bounds):
            def fun(x):
                return np.asarray(x)
        else:
            raise ValueError("`constraint` of an unknown type is passed.")

        self.fun = fun

        lb = np.asarray(constraint.lb, dtype=float)
        ub = np.asarray(constraint.ub, dtype=float)

        x0 = np.asarray(x0)

        # find out the number of constraints
        f0 = fun(x0)
        self.num_constr = m = f0.size
        self.parameter_count = x0.size

        if lb.ndim == 0:
            lb = np.resize(lb, m)
        if ub.ndim == 0:
            ub = np.resize(ub, m)

        self.bounds = (lb, ub)

    def __call__(self, x):
        return np.atleast_1d(self.fun(x))

    def violation(self, x):
        """How much the constraint is exceeded by."""
        # expect ev to have shape (num_constr, S) or (num_constr,)
        ev = self.fun(np.asarray(x))

        try:
            excess_lb = np.maximum(self.bounds[0] - ev.T, 0)
            excess_ub = np.maximum(ev.T - self.bounds[1], 0)
        except ValueError as e:
            raise RuntimeError("An array returned from a Constraint has"
                               " the wrong shape. If `vectorized is False`"
                               " the Constraint should return an array of"
                               " shape (M,). If `vectorized is True` then"
                               " the Constraint must return an array of"
                               " shape (M, S), where S is the number of"
                               " solution vectors and M is the number of"
                               " constraint components in a given"
                               " Constraint object.") from e

        v = (excess_lb + excess_ub).T
        return v