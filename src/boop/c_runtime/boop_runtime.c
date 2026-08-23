#include "boop_runtime.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
  double normal[BOOP_N];
  double tangent[BOOP_N];
  double compound[BOOP_N];
  double correction[BOOP_N];
  int cg_iterations;
  int cg_stop;
  double cg_initial;
  double cg_final;
  double cg_curvature;
  double cg_rayleigh;
} BoopStep;

static void record(BoopDiagnostics *diag, int k, const BoopWorkspace *work,
                   const BoopModel *model, const BoopStep *step, int procedure,
                   int accepted, double trust);

static double dot(int n, const double *a, const double *b) {
  double value = 0.0;
  for (int i = 0; i < n; ++i)
    value += a[i] * b[i];
  return value;
}

static double norm(int n, const double *a) { return sqrt(dot(n, a, a)); }

static void zero_vector(double *values) {
  for (int i = 0; i < BOOP_N; ++i)
    values[i] = 0.0;
}

static void copy_vector(double *destination, const double *source) {
  for (int i = 0; i < BOOP_N; ++i)
    destination[i] = source[i];
}

static void initialize_step(BoopStep *step) {
  zero_vector(step->normal);
  zero_vector(step->tangent);
  zero_vector(step->compound);
  zero_vector(step->correction);
  step->cg_iterations = 0;
  step->cg_stop = BOOP_CG_ZERO_RADIUS;
  step->cg_initial = 0.0;
  step->cg_final = 0.0;
  step->cg_curvature = NAN;
  step->cg_rayleigh = NAN;
}

static double violation(const BoopModel *model) {
  double value = 0.0;
  for (int i = 0; i < BOOP_M; ++i)
    value = fmax(value, fabs(model->equalities[i]));
  return value;
}

static void project(const BoopWorkspace *work, const BoopModel *model,
                    const double *vector, double *projected) {
  double rhs[BOOP_STORAGE(BOOP_M)] = {0.0};
  double dual[BOOP_STORAGE(BOOP_M)] = {0.0};
  for (int i = 0; i < BOOP_N; ++i)
    projected[i] = work->free_variables[i] ? vector[i] : 0.0;
  for (int row = 0; row < BOOP_M; ++row)
    for (int col = 0; col < BOOP_N; ++col)
      if (work->free_variables[col])
        rhs[row] += model->jacobian[row * BOOP_N + col] * vector[col];
  if (BOOP_M > 0)
    boop_ldl_solve(work->ldl_lower, work->ldl_diagonal, rhs, dual);
  for (int col = 0; col < BOOP_N; ++col)
    if (work->free_variables[col])
      for (int row = 0; row < BOOP_M; ++row)
        projected[col] -= model->jacobian[row * BOOP_N + col] * dual[row];
}

static double boundary_distance(const double *point, const double *direction,
                                double radius) {
  double a = dot(BOOP_N, direction, direction);
  if (a == 0.0)
    return 0.0;
  double b = dot(BOOP_N, point, direction);
  double discriminant =
      fmax(0.0, b * b + a * (radius * radius - dot(BOOP_N, point, point)));
  return fmax(0.0, (-b + sqrt(discriminant)) / a);
}

static void steihaug(const BoopWorkspace *work, const BoopModel *model,
                     const double *gradient, double radius, BoopStep *step) {
  double residual[BOOP_N], direction[BOOP_N], action[BOOP_N], trial[BOOP_N];
  zero_vector(step->tangent);
  step->cg_iterations = 0;
  step->cg_curvature = NAN;
  step->cg_rayleigh = NAN;
  if (radius <= DBL_EPSILON) {
    step->cg_stop = BOOP_CG_ZERO_RADIUS;
    step->cg_initial = step->cg_final = 0.0;
    return;
  }
  project(work, model, gradient, residual);
  for (int i = 0; i < BOOP_N; ++i)
    residual[i] = direction[i] = -residual[i];
  double residual_sq = dot(BOOP_N, residual, residual);
  step->cg_initial = sqrt(residual_sq);
  if (residual_sq <= DBL_EPSILON) {
    step->cg_stop = BOOP_CG_ZERO_RESIDUAL;
    step->cg_final = step->cg_initial;
    return;
  }
  double hessian_norm_sq = 0.0;
  for (int i = 0; i < BOOP_N * BOOP_N; ++i)
    hessian_norm_sq += model->hessian[i] * model->hessian[i];
  double damping = BOOP_TANGENTIAL_DAMPING * sqrt(hessian_norm_sq);
  for (int iteration = 1; iteration <= BOOP_CG_ITERATIONS; ++iteration) {
    for (int row = 0; row < BOOP_N; ++row) {
      action[row] = damping * direction[row];
      for (int col = 0; col < BOOP_N; ++col)
        action[row] += model->hessian[row * BOOP_N + col] * direction[col];
    }
    project(work, model, action, trial);
    copy_vector(action, trial);
    double curvature = dot(BOOP_N, direction, action);
    double direction_sq = dot(BOOP_N, direction, direction);
    step->cg_iterations = iteration;
    step->cg_curvature = curvature;
    step->cg_rayleigh = curvature / direction_sq;
    if (curvature <= BOOP_CURVATURE_FLOOR * direction_sq) {
      double tau = boundary_distance(step->tangent, direction, radius);
      for (int i = 0; i < BOOP_N; ++i)
        step->tangent[i] += tau * direction[i];
      step->cg_stop = BOOP_CG_NEGATIVE_CURVATURE;
      step->cg_final = sqrt(residual_sq);
      return;
    }
    double alpha = residual_sq / curvature;
    for (int i = 0; i < BOOP_N; ++i)
      trial[i] = step->tangent[i] + alpha * direction[i];
    if (dot(BOOP_N, trial, trial) >= radius * radius) {
      double tau = boundary_distance(step->tangent, direction, radius);
      for (int i = 0; i < BOOP_N; ++i)
        step->tangent[i] += tau * direction[i];
      step->cg_stop = BOOP_CG_TRUST_BOUNDARY;
      step->cg_final = sqrt(residual_sq);
      return;
    }
    copy_vector(step->tangent, trial);
    for (int i = 0; i < BOOP_N; ++i)
      trial[i] = residual[i] - alpha * action[i];
    project(work, model, trial, action);
    double new_sq = dot(BOOP_N, action, action);
    if (new_sq <= DBL_EPSILON) {
      step->cg_stop = BOOP_CG_CONVERGED;
      step->cg_final = sqrt(new_sq);
      return;
    }
    for (int i = 0; i < BOOP_N; ++i)
      direction[i] = action[i] + (new_sq / residual_sq) * direction[i];
    copy_vector(residual, action);
    residual_sq = new_sq;
  }
  step->cg_stop = BOOP_CG_ITERATION_LIMIT;
  step->cg_final = sqrt(residual_sq);
}

static bool filter_accepts(BoopWorkspace *work, int *entries,
                           const BoopModel *candidate) {
  double theta = violation(candidate);
  for (int i = 0; i < *entries; ++i) {
    bool objective = candidate->objective + BOOP_FILTER_GAMMA * theta <=
                     work->filter_objective[i];
    bool feasibility =
        work->filter_violation[i] > BOOP_BOUND_TOLERANCE &&
        theta <= (1.0 - BOOP_FILTER_BETA) * work->filter_violation[i];
    if (!(objective || feasibility))
      return false;
  }
  work->filter_objective[*entries] = candidate->objective;
  work->filter_violation[*entries] = theta;
  ++*entries;
  return true;
}

/* Remaining solver routines are intentionally ordinary loops: generated code
   supplies only the NLP evaluator and sparse equality factorization. */

static BoopStatus linearize(BoopWorkspace *work, const BoopModel *model) {
  int active_count = 0;
  for (int i = 0; i < BOOP_N; ++i) {
    work->free_variables[i] = work->active[i] == 0;
    active_count += work->active[i] != 0;
  }
  if (BOOP_M + active_count > BOOP_N)
    return BOOP_INVALID_ACTIVE_SET;
  boop_assemble_gram(model, work->free_variables, work->gram);
  return boop_ldl_factor(work->gram, work->ldl_lower, work->ldl_diagonal);
}

static void normal_step(const BoopWorkspace *work, const BoopModel *model,
                        const double *x, double *normal) {
  double rhs[BOOP_STORAGE(BOOP_M)] = {0.0}, dual[BOOP_STORAGE(BOOP_M)] = {0.0};
  zero_vector(normal);
  for (int i = 0; i < BOOP_N; ++i) {
    if (work->active[i] < 0)
      normal[i] = model->lower[i] - x[i];
    if (work->active[i] > 0)
      normal[i] = model->upper[i] - x[i];
  }
  for (int row = 0; row < BOOP_M; ++row) {
    rhs[row] = -model->equalities[row];
    for (int col = 0; col < BOOP_N; ++col)
      if (!work->free_variables[col])
        rhs[row] -= model->jacobian[row * BOOP_N + col] * normal[col];
  }
  if (BOOP_M > 0)
    boop_ldl_solve(work->ldl_lower, work->ldl_diagonal, rhs, dual);
  for (int col = 0; col < BOOP_N; ++col)
    if (work->free_variables[col])
      for (int row = 0; row < BOOP_M; ++row)
        normal[col] += model->jacobian[row * BOOP_N + col] * dual[row];
}

static int removable_bound(const BoopWorkspace *work, const BoopModel *model) {
  double projected[BOOP_N], rhs[BOOP_STORAGE(BOOP_M)] = {0.0},
                            equality[BOOP_STORAGE(BOOP_M)] = {0.0};
  project(work, model, model->gradient, projected);
  if (norm(BOOP_N, projected) > BOOP_ACTIVE_SET_STATIONARITY_TOLERANCE *
                                    fmax(1.0, norm(BOOP_N, model->gradient)))
    return -1;
  for (int row = 0; row < BOOP_M; ++row)
    for (int col = 0; col < BOOP_N; ++col)
      if (work->free_variables[col])
        rhs[row] -= model->jacobian[row * BOOP_N + col] * model->gradient[col];
  if (BOOP_M > 0)
    boop_ldl_solve(work->ldl_lower, work->ldl_diagonal, rhs, equality);
  int worst = -1;
  double worst_multiplier = -BOOP_BOUND_TOLERANCE;
  for (int col = 0; col < BOOP_N; ++col)
    if (work->active[col] && !work->fixed[col]) {
      bool reached =
          work->active[col] < 0
              ? fabs(work->x[col] - model->lower[col]) <= BOOP_BOUND_TOLERANCE
              : fabs(work->x[col] - model->upper[col]) <= BOOP_BOUND_TOLERANCE;
      if (!reached)
        continue;
      double stationarity = model->gradient[col];
      for (int row = 0; row < BOOP_M; ++row)
        stationarity += model->jacobian[row * BOOP_N + col] * equality[row];
      double multiplier = work->active[col] < 0 ? stationarity : -stationarity;
      if (multiplier < worst_multiplier) {
        worst_multiplier = multiplier;
        worst = col;
      }
    }
  return worst;
}

#if BOOP_NONLINEAR_EQUALITIES
static void correction_step(const BoopWorkspace *work, const BoopModel *model,
                            const BoopModel *bo, const double *bo_x,
                            double radius, double *correction) {
  double rhs[BOOP_STORAGE(BOOP_M)] = {0.0}, dual[BOOP_STORAGE(BOOP_M)] = {0.0};
  zero_vector(correction);
  for (int i = 0; i < BOOP_N; ++i) {
    if (work->active[i] < 0)
      correction[i] = model->lower[i] - bo_x[i];
    if (work->active[i] > 0)
      correction[i] = model->upper[i] - bo_x[i];
  }
  for (int row = 0; row < BOOP_M; ++row) {
    rhs[row] = -bo->equalities[row];
    for (int col = 0; col < BOOP_N; ++col)
      if (!work->free_variables[col])
        rhs[row] -= model->jacobian[row * BOOP_N + col] * correction[col];
  }
  if (BOOP_M > 0)
    boop_ldl_solve(work->ldl_lower, work->ldl_diagonal, rhs, dual);
  for (int col = 0; col < BOOP_N; ++col)
    if (work->free_variables[col])
      for (int row = 0; row < BOOP_M; ++row)
        correction[col] += model->jacobian[row * BOOP_N + col] * dual[row];
  double length = norm(BOOP_N, correction);
  if (length > radius)
    for (int i = 0; i < BOOP_N; ++i)
      correction[i] *= radius / length;
}

static bool box_feasible(const double *x, const BoopModel *model) {
  for (int i = 0; i < BOOP_N; ++i)
    if (x[i] < model->lower[i] - BOOP_BOUND_TOLERANCE ||
        x[i] > model->upper[i] + BOOP_BOUND_TOLERANCE)
      return false;
  return true;
}
#endif

static int blocking_bound(const double *x, const double *step,
                          const BoopModel *model, signed char *active) {
  double first = INFINITY;
  for (int i = 0; i < BOOP_N; ++i)
    if (!active[i]) {
      double fraction = INFINITY;
      if (step[i] > BOOP_BOUND_TOLERANCE && isfinite(model->upper[i]))
        fraction = (model->upper[i] - x[i]) / step[i];
      if (step[i] < -BOOP_BOUND_TOLERANCE && isfinite(model->lower[i]))
        fraction = (model->lower[i] - x[i]) / step[i];
      if (fraction < first)
        first = fraction;
    }
  if (!isfinite(first) || first > 1.0 + BOOP_BOUND_TOLERANCE)
    return 0;
  int changed = 0;
  for (int i = 0; i < BOOP_N; ++i)
    if (!active[i]) {
      double fraction = INFINITY;
      if (step[i] > BOOP_BOUND_TOLERANCE && isfinite(model->upper[i]))
        fraction = (model->upper[i] - x[i]) / step[i];
      if (step[i] < -BOOP_BOUND_TOLERANCE && isfinite(model->lower[i]))
        fraction = (model->lower[i] - x[i]) / step[i];
      if (fabs(fraction - first) <=
          BOOP_BOUND_TOLERANCE * fmax(1.0, fabs(first))) {
        active[i] = step[i] < 0.0 ? -1 : 1;
        changed = 1;
      }
    }
  return changed;
}

static void record(BoopDiagnostics *diag, int k, const BoopWorkspace *work,
                   const BoopModel *model, const BoopStep *step, int procedure,
                   int accepted, double trust) {
  if (!diag)
    return;
  diag->procedure[k] = procedure;
  diag->accepted[k] = accepted;
  diag->objective[k] = model->objective;
  diag->violation[k] = violation(model);
  diag->trust_radius[k] = trust;
  copy_vector(diag->x[k], work->x);
  for (int i = 0; i < BOOP_N; ++i)
    diag->active_bounds[k][i] = work->active[i];
  diag->normal_step_norm[k] = norm(BOOP_N, step->normal);
  diag->tangential_step_norm[k] = norm(BOOP_N, step->tangent);
  diag->correction_step_norm[k] = norm(BOOP_N, step->correction);
  diag->cg_iterations[k] = step->cg_iterations;
  diag->cg_stop_reason[k] = step->cg_stop;
  diag->cg_initial_residual[k] = step->cg_initial;
  diag->cg_final_residual[k] = step->cg_final;
  diag->cg_final_curvature[k] = step->cg_curvature;
  diag->cg_final_rayleigh[k] = step->cg_rayleigh;
}

BoopStatus boop_solve(BoopWorkspace *work, const double *initial,
                      const double *parameters, double *solution,
                      BoopDiagnostics *diag) {
  BoopModel current, bo;
#if BOOP_NONLINEAR_EQUALITIES
  BoopModel soc;
#endif
  int filter_entries = 0;
  double trust = BOOP_INITIAL_TRUST_RADIUS;
  for (int i = 0; i < BOOP_N; ++i) {
    work->active[i] = 0;
    work->fixed[i] = 0;
  }
  BoopStatus status = boop_evaluate(initial, parameters, &current);
  if (status)
    return status;
  for (int i = 0; i < BOOP_N; ++i) {
    work->x[i] = fmin(fmax(initial[i], current.lower[i]), current.upper[i]);
    work->fixed[i] =
        isfinite(current.lower[i]) && isfinite(current.upper[i]) &&
        fabs(current.upper[i] - current.lower[i]) <= BOOP_BOUND_TOLERANCE;
    if (work->fixed[i])
      work->active[i] = -1;
  }
  status = boop_evaluate(work->x, parameters, &current);
  if (status)
    return status;
  /* The projected initial point is the first filter entry. */
  filter_accepts(work, &filter_entries, &current);
  for (int iteration = 0; iteration < BOOP_SQP_ITERATIONS; ++iteration) {
    BoopStep step;
    initialize_step(&step);
    status = boop_evaluate(work->x, parameters, &current);
    if (status)
      return status;
    status = linearize(work, &current);
    if (status)
      return status;
    /* Working-set transitions consume an iteration without changing x. */
    int removable = removable_bound(work, &current);
    if (removable >= 0) {
      work->active[removable] = 0;
      record(diag, iteration, work, &current, &step, BOOP_ACTIVE_SET_UPDATE, 0,
             trust);
      continue;
    }
    /* Walk normally first, then use the remaining radius tangentially. */
    normal_step(work, &current, work->x, step.normal);
    double normal_length = norm(BOOP_N, step.normal);
    if (normal_length > trust) {
      for (int i = 0; i < BOOP_N; ++i)
        step.normal[i] *= trust / normal_length;
      normal_length = trust;
    }
    double gradient_at_normal[BOOP_N];
    for (int row = 0; row < BOOP_N; ++row) {
      gradient_at_normal[row] = current.gradient[row];
      for (int col = 0; col < BOOP_N; ++col)
        gradient_at_normal[row] +=
            current.hessian[row * BOOP_N + col] * step.normal[col];
    }
    steihaug(work, &current, gradient_at_normal,
             sqrt(fmax(0.0, trust * trust - normal_length * normal_length)),
             &step);
    for (int i = 0; i < BOOP_N; ++i)
      step.compound[i] = step.normal[i] + step.tangent[i];
    if (blocking_bound(work->x, step.compound, &current, work->active)) {
      record(diag, iteration, work, &current, &step, BOOP_ACTIVE_SET_UPDATE, 0,
             trust);
      continue;
    }
    /* Evaluate the fixed-cost SOC trial before the uncorrected BO trial. */
    double bo_x[BOOP_N];
#if BOOP_NONLINEAR_EQUALITIES
    double soc_x[BOOP_N];
#endif
    for (int i = 0; i < BOOP_N; ++i)
      bo_x[i] = work->x[i] + step.compound[i];
    status = boop_evaluate(bo_x, parameters, &bo);
    if (status)
      return status;
    bool accepted = false;
    int procedure = BOOP_REJECTED;
    const BoopModel *chosen = &current;
    const double *chosen_x = work->x;
    double chosen_step[BOOP_N] = {0.0};
#if BOOP_NONLINEAR_EQUALITIES
    correction_step(work, &current, &bo, bo_x, trust, step.correction);
    for (int i = 0; i < BOOP_N; ++i)
      soc_x[i] = bo_x[i] + step.correction[i];
    status = boop_evaluate(soc_x, parameters, &soc);
    if (status)
      return status;
    for (int i = 0; i < BOOP_N; ++i)
      chosen_step[i] = step.compound[i] + step.correction[i];
    if (box_feasible(soc_x, &current) &&
        norm(BOOP_N, chosen_step) > DBL_EPSILON &&
        filter_accepts(work, &filter_entries, &soc)) {
      accepted = true;
      procedure = BOOP_BYRD_OMOJOKUN_SOC;
      chosen = &soc;
      chosen_x = soc_x;
    }
#endif
    if (!accepted && norm(BOOP_N, step.compound) > DBL_EPSILON &&
        filter_accepts(work, &filter_entries, &bo)) {
      accepted = true;
      procedure = BOOP_BYRD_OMOJUKUN;
      chosen = &bo;
      chosen_x = bo_x;
      copy_vector(chosen_step, step.compound);
    }
    if (accepted) {
      copy_vector(work->x, chosen_x);
      trust = BOOP_TRUST_EXPAND * norm(BOOP_N, chosen_step);
    } else {
      /* Drop speculative bounds that the rejected trial never reached. */
      trust *= BOOP_TRUST_SHRINK;
      for (int i = 0; i < BOOP_N; ++i)
        if (work->active[i] && !work->fixed[i]) {
          bool reached =
              work->active[i] < 0
                  ? fabs(work->x[i] - current.lower[i]) <= BOOP_BOUND_TOLERANCE
                  : fabs(work->x[i] - current.upper[i]) <= BOOP_BOUND_TOLERANCE;
          if (!reached)
            work->active[i] = 0;
        }
    }
    trust = fmax(trust, DBL_EPSILON);
    record(diag, iteration, work, chosen, &step, procedure, accepted, trust);
  }
  copy_vector(solution, work->x);
  return BOOP_OK;
}
