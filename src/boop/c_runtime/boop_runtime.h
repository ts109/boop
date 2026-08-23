#ifndef BOOP_RUNTIME_H
#define BOOP_RUNTIME_H

#include <stddef.h>

#include "boop_config.h"

#define BOOP_STORAGE(n) ((n) > 0 ? (n) : 1)

typedef enum {
  BOOP_OK = 0,
  BOOP_NONFINITE_EVALUATION = 1,
  BOOP_FACTOR_FAILURE = 2,
  BOOP_INVALID_ACTIVE_SET = 3
} BoopStatus;

typedef enum {
  BOOP_ACTIVE_SET_UPDATE = 0,
  BOOP_BYRD_OMOJOKUN_SOC = 1,
  BOOP_BYRD_OMOJUKUN = 2,
  BOOP_REJECTED = 3
} BoopProcedure;

typedef enum {
  BOOP_CG_ZERO_RADIUS = 0,
  BOOP_CG_ZERO_RESIDUAL = 1,
  BOOP_CG_NEGATIVE_CURVATURE = 2,
  BOOP_CG_TRUST_BOUNDARY = 3,
  BOOP_CG_CONVERGED = 4,
  BOOP_CG_ITERATION_LIMIT = 5
} BoopCGStop;

typedef struct {
  double objective;
  double equalities[BOOP_STORAGE(BOOP_M)];
  double gradient[BOOP_N];
  double hessian[BOOP_N * BOOP_N];
  double jacobian[BOOP_STORAGE(BOOP_M * BOOP_N)];
  double lower[BOOP_N];
  double upper[BOOP_N];
} BoopModel;

typedef struct {
  int procedure[BOOP_SQP_ITERATIONS];
  int accepted[BOOP_SQP_ITERATIONS];
  double x[BOOP_SQP_ITERATIONS][BOOP_N];
  double objective[BOOP_SQP_ITERATIONS];
  double violation[BOOP_SQP_ITERATIONS];
  double trust_radius[BOOP_SQP_ITERATIONS];
  double normal_step_norm[BOOP_SQP_ITERATIONS];
  double tangential_step_norm[BOOP_SQP_ITERATIONS];
  double correction_step_norm[BOOP_SQP_ITERATIONS];
  signed char active_bounds[BOOP_SQP_ITERATIONS][BOOP_N];
  int cg_iterations[BOOP_SQP_ITERATIONS];
  int cg_stop_reason[BOOP_SQP_ITERATIONS];
  double cg_initial_residual[BOOP_SQP_ITERATIONS];
  double cg_final_residual[BOOP_SQP_ITERATIONS];
  double cg_final_curvature[BOOP_SQP_ITERATIONS];
  double cg_final_rayleigh[BOOP_SQP_ITERATIONS];
} BoopDiagnostics;

typedef struct {
  double x[BOOP_N];
  signed char active[BOOP_N];
  unsigned char fixed[BOOP_N];
  unsigned char free_variables[BOOP_N];
  double gram[BOOP_STORAGE(BOOP_M * BOOP_M)];
  double ldl_lower[BOOP_STORAGE(BOOP_M * BOOP_M)];
  double ldl_diagonal[BOOP_STORAGE(BOOP_M)];
  double filter_objective[BOOP_SQP_ITERATIONS + 1];
  double filter_violation[BOOP_SQP_ITERATIONS + 1];
} BoopWorkspace;

BoopStatus boop_evaluate(const double *x, const double *parameters,
                         BoopModel *model);
void boop_assemble_gram(const BoopModel *model,
                        const unsigned char *free_variables, double *gram);
BoopStatus boop_ldl_factor(const double *gram, double *lower, double *diagonal);
void boop_ldl_solve(const double *lower, const double *diagonal,
                    const double *rhs, double *solution);

BoopStatus boop_solve(BoopWorkspace *workspace, const double *initial_guess,
                      const double *parameters, double *solution,
                      BoopDiagnostics *diagnostics);

#endif
