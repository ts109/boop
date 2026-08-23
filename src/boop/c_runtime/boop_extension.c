#include <Python.h>

#include "boop_runtime.h"

#define BOOP_JOIN_INNER(a, b) a##b
#define BOOP_JOIN(a, b) BOOP_JOIN_INNER(a, b)

typedef struct {
  PyObject_HEAD
  BoopDiagnostics diagnostics;
} PyBoopDiagnostics;

typedef struct {
  PyObject_HEAD
  BoopWorkspace workspace;
} PyBoopSolver;

static PyObject *double_list(const double *values, Py_ssize_t length) {
  PyObject *result = PyList_New(length);
  if (!result) return NULL;
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject *value = PyFloat_FromDouble(values[i]);
    if (!value) { Py_DECREF(result); return NULL; }
    PyList_SET_ITEM(result, i, value);
  }
  return result;
}

static PyObject *int_list(const int *values, Py_ssize_t length) {
  PyObject *result = PyList_New(length);
  if (!result) return NULL;
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject *value = PyLong_FromLong(values[i]);
    if (!value) { Py_DECREF(result); return NULL; }
    PyList_SET_ITEM(result, i, value);
  }
  return result;
}

static PyObject *diagnostics_x(PyBoopDiagnostics *self, void *closure) {
  (void)closure;
  PyObject *outer = PyList_New(BOOP_SQP_ITERATIONS);
  if (!outer) return NULL;
  for (int i = 0; i < BOOP_SQP_ITERATIONS; ++i) {
    PyObject *row = double_list(self->diagnostics.x[i], BOOP_N);
    if (!row) { Py_DECREF(outer); return NULL; }
    PyList_SET_ITEM(outer, i, row);
  }
  return outer;
}

static PyObject *diagnostics_active_bounds(PyBoopDiagnostics *self, void *closure) {
  (void)closure;
  PyObject *outer = PyList_New(BOOP_SQP_ITERATIONS);
  if (!outer) return NULL;
  for (int i = 0; i < BOOP_SQP_ITERATIONS; ++i) {
    PyObject *row = PyList_New(BOOP_N);
    if (!row) { Py_DECREF(outer); return NULL; }
    for (int j = 0; j < BOOP_N; ++j) {
      PyObject *value = PyLong_FromLong(self->diagnostics.active_bounds[i][j]);
      if (!value) { Py_DECREF(row); Py_DECREF(outer); return NULL; }
      PyList_SET_ITEM(row, j, value);
    }
    PyList_SET_ITEM(outer, i, row);
  }
  return outer;
}

static PyObject *diagnostics_procedure(PyBoopDiagnostics *self, void *closure) {
  (void)closure;
  static const char *names[] = {"active-set-update", "byrd-omojukun-soc", "byrd-omojukun", "rejected"};
  PyObject *result = PyList_New(BOOP_SQP_ITERATIONS);
  if (!result) return NULL;
  for (int i = 0; i < BOOP_SQP_ITERATIONS; ++i) {
    PyObject *name = PyUnicode_FromString(names[self->diagnostics.procedure[i]]);
    if (!name) { Py_DECREF(result); return NULL; }
    PyList_SET_ITEM(result, i, name);
  }
  return result;
}

#define DOUBLE_GETTER(name, field) \
  static PyObject *name(PyBoopDiagnostics *self, void *closure) { (void)closure; return double_list(self->diagnostics.field, BOOP_SQP_ITERATIONS); }
#define INT_GETTER(name, field) \
  static PyObject *name(PyBoopDiagnostics *self, void *closure) { (void)closure; return int_list(self->diagnostics.field, BOOP_SQP_ITERATIONS); }

DOUBLE_GETTER(diagnostics_objective, objective)
DOUBLE_GETTER(diagnostics_violation, violation)
DOUBLE_GETTER(diagnostics_trust_radius, trust_radius)
DOUBLE_GETTER(diagnostics_normal_step_norm, normal_step_norm)
DOUBLE_GETTER(diagnostics_tangential_step_norm, tangential_step_norm)
DOUBLE_GETTER(diagnostics_correction_step_norm, correction_step_norm)
DOUBLE_GETTER(diagnostics_cg_initial_residual, cg_initial_residual)
DOUBLE_GETTER(diagnostics_cg_final_residual, cg_final_residual)
DOUBLE_GETTER(diagnostics_cg_final_curvature, cg_final_curvature)
DOUBLE_GETTER(diagnostics_cg_final_rayleigh, cg_final_rayleigh)
INT_GETTER(diagnostics_accepted, accepted)
INT_GETTER(diagnostics_cg_iterations, cg_iterations)
INT_GETTER(diagnostics_cg_stop_reason, cg_stop_reason)

static PyGetSetDef diagnostics_getset[] = {
  {"procedure", (getter)diagnostics_procedure, NULL, "Iteration procedures.", NULL},
  {"accepted", (getter)diagnostics_accepted, NULL, "Acceptance flags.", NULL},
  {"x", (getter)diagnostics_x, NULL, "Iteration vectors.", NULL},
  {"objective", (getter)diagnostics_objective, NULL, "Objective history.", NULL},
  {"violation", (getter)diagnostics_violation, NULL, "Equality violation history.", NULL},
  {"trust_radius", (getter)diagnostics_trust_radius, NULL, "Trust-radius history.", NULL},
  {"normal_step_norm", (getter)diagnostics_normal_step_norm, NULL, "Normal-step norms.", NULL},
  {"tangential_step_norm", (getter)diagnostics_tangential_step_norm, NULL, "Tangential-step norms.", NULL},
  {"correction_step_norm", (getter)diagnostics_correction_step_norm, NULL, "SOC norms.", NULL},
  {"active_bounds", (getter)diagnostics_active_bounds, NULL, "Active-bound history.", NULL},
  {"cg_iterations", (getter)diagnostics_cg_iterations, NULL, "CG iteration counts.", NULL},
  {"cg_stop_reason", (getter)diagnostics_cg_stop_reason, NULL, "CG stop codes.", NULL},
  {"cg_initial_residual", (getter)diagnostics_cg_initial_residual, NULL, "Initial CG residuals.", NULL},
  {"cg_final_residual", (getter)diagnostics_cg_final_residual, NULL, "Final CG residuals.", NULL},
  {"cg_final_curvature", (getter)diagnostics_cg_final_curvature, NULL, "Final CG curvatures.", NULL},
  {"cg_final_rayleigh", (getter)diagnostics_cg_final_rayleigh, NULL, "Final CG Rayleigh quotients.", NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject PyBoopDiagnosticsType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = BOOP_MODULE_STRING ".SolverDiagnostics",
  .tp_basicsize = sizeof(PyBoopDiagnostics),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Lazy view of a native BoopDiagnostics structure.",
  .tp_getset = diagnostics_getset,
};

static int read_vector(PyObject *object, Py_ssize_t expected, double *destination, const char *name) {
  PyObject *sequence = PySequence_Fast(object, name);
  if (!sequence) return -1;
  if (PySequence_Fast_GET_SIZE(sequence) != expected) {
    PyErr_Format(PyExc_ValueError, "%s must contain %zd values", name, expected);
    Py_DECREF(sequence); return -1;
  }
  PyObject **items = PySequence_Fast_ITEMS(sequence);
  for (Py_ssize_t i = 0; i < expected; ++i) {
    destination[i] = PyFloat_AsDouble(items[i]);
    if (PyErr_Occurred()) { Py_DECREF(sequence); return -1; }
  }
  Py_DECREF(sequence); return 0;
}

static PyObject *solver_call(PyBoopSolver *self, PyObject *args, PyObject *kwargs) {
  static char *keywords[] = {"initial_guess", "parameters", "diagnostics", NULL};
  PyObject *initial_object;
  PyObject *parameter_object = NULL;
  int diagnostics_requested = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op:__call__", keywords, &initial_object, &parameter_object, &diagnostics_requested)) return NULL;
  double initial[BOOP_N], parameters[BOOP_STORAGE(BOOP_P)] = {0.0}, solution[BOOP_N];
  if (read_vector(initial_object, BOOP_N, initial, "initial_guess") < 0) return NULL;
  if (parameter_object) {
    if (read_vector(parameter_object, BOOP_P, parameters, "parameters") < 0) return NULL;
  } else if (BOOP_P != 0) {
    PyErr_Format(PyExc_ValueError, "parameters must contain %d values", BOOP_P); return NULL;
  }
  PyBoopDiagnostics *native = NULL;
  if (diagnostics_requested) {
    native = PyObject_New(PyBoopDiagnostics, &PyBoopDiagnosticsType);
    if (!native) return NULL;
  }
  BoopStatus status;
  Py_BEGIN_ALLOW_THREADS
  status = boop_solve(&self->workspace, initial, parameters, solution, native ? &native->diagnostics : NULL);
  Py_END_ALLOW_THREADS
  if (status != BOOP_OK) {
    Py_XDECREF(native); PyErr_Format(PyExc_RuntimeError, "native Boop solver failed with status %d", status); return NULL;
  }
  PyObject *solution_object = double_list(solution, BOOP_N);
  if (!solution_object) { Py_XDECREF(native); return NULL; }
  if (!native) return solution_object;
  PyObject *result = PyTuple_New(2);
  if (!result) { Py_DECREF(solution_object); Py_DECREF(native); return NULL; }
  PyTuple_SET_ITEM(result, 0, solution_object); PyTuple_SET_ITEM(result, 1, (PyObject *)native);
  return result;
}

static PyTypeObject PyBoopSolverType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = BOOP_MODULE_STRING ".Solver",
  .tp_basicsize = sizeof(PyBoopSolver),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Compiled Boop solver.",
  .tp_call = (ternaryfunc)solver_call,
  .tp_new = PyType_GenericNew,
};

static PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  .m_name = BOOP_MODULE_STRING,
  .m_doc = "Generated Boop solver.",
  .m_size = -1,
};

PyMODINIT_FUNC BOOP_JOIN(PyInit_, BOOP_MODULE_TOKEN)(void) {
  if (PyType_Ready(&PyBoopSolverType) < 0 || PyType_Ready(&PyBoopDiagnosticsType) < 0) return NULL;
  PyObject *result = PyModule_Create(&module);
  if (!result) return NULL;
  Py_INCREF(&PyBoopSolverType);
  if (PyModule_AddObject(result, "Solver", (PyObject *)&PyBoopSolverType) < 0) {
    Py_DECREF(&PyBoopSolverType); Py_DECREF(result); return NULL;
  }
  Py_INCREF(&PyBoopDiagnosticsType);
  if (PyModule_AddObject(result, "SolverDiagnostics", (PyObject *)&PyBoopDiagnosticsType) < 0) {
    Py_DECREF(&PyBoopDiagnosticsType); Py_DECREF(result); return NULL;
  }
  return result;
}
