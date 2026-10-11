/* Natural spline derivatives and tensor-product Hermite evaluation. */
static void lut_factor(int n, const double *nodes, double *diagonal) {
  diagonal[0] = 1.0;
  for (int i = 1; i < n - 1; ++i) {
    double left = nodes[i] - nodes[i - 1];
    double right = nodes[i + 1] - nodes[i];
    diagonal[i] = 2.0 * (left + right)
                  - (i == 1 ? 0.0 : left * (left / diagonal[i - 1]));
  }
}

static void lut_slopes(int n, const double *nodes, const double *values,
                       int stride, double *slopes, const double *diagonal,
                       double *rhs) {
  rhs[0] = 0.0;
  for (int i = 1; i < n - 1; ++i) {
    double left = nodes[i] - nodes[i - 1];
    double right = nodes[i + 1] - nodes[i];
    rhs[i] = 6.0 * ((values[(i + 1) * stride] - values[i * stride]) / right -
                    (values[i * stride] - values[(i - 1) * stride]) / left)
             - (left / diagonal[i - 1]) * rhs[i - 1];
  }
  rhs[n - 1] = 0.0;
  for (int i = n - 2; i >= 1; --i)
    rhs[i] = (rhs[i] - (nodes[i + 1] - nodes[i]) * rhs[i + 1]) / diagonal[i];
  for (int i = 0; i < n - 1; ++i) {
    double h = nodes[i + 1] - nodes[i];
    slopes[i * stride] = (values[(i + 1) * stride] - values[i * stride]) / h
                         - h * (2.0 * rhs[i] + rhs[i + 1]) / 6.0;
  }
  double h = nodes[n - 1] - nodes[n - 2];
  slopes[(n - 1) * stride] = (values[(n - 1) * stride] - values[(n - 2) * stride]) / h
                             + h * (rhs[n - 2] + 2.0 * rhs[n - 1]) / 6.0;
}

BoopStatus boop_prepare_luts(BoopWorkspace *work, const BoopLutInput *inputs) {
  if (!inputs) return BOOP_INVALID_LUT;
  for (int k = 0; k < BOOP_LUT_COUNT; ++k) {
    int nx = boop_lut_shapes[k][0], ny = boop_lut_shapes[k][1];
    const BoopLutInput *input = inputs + k;
    if (!input->nodes[0] || !input->values || (ny > 1 && !input->nodes[1]))
      return BOOP_INVALID_LUT;
    for (int axis = 0; axis < (ny > 1 ? 2 : 1); ++axis) {
      int n = axis ? ny : nx;
      const double *nodes = input->nodes[axis];
      for (int i = 0; i < n; ++i)
        if (!isfinite(nodes[i]) || (i && !(nodes[i] > nodes[i - 1])) ||
            (i && !isfinite(nodes[i] - nodes[i - 1]))) return BOOP_INVALID_LUT;
    }
    work->lut_nodes[k][0] = input->nodes[0];
    work->lut_nodes[k][1] = input->nodes[1];
    int count = nx * ny;
    double *f = work->lut_fields + boop_lut_offsets[k];
    double *fx = f + count;
    for (int i = 0; i < count; ++i) {
      if (!isfinite(input->values[i])) return BOOP_INVALID_LUT;
      f[i] = input->values[i];
    }
    lut_factor(nx, input->nodes[0], work->lut_scratch);
    for (int j = 0; j < ny; ++j)
      lut_slopes(nx, input->nodes[0], f + j, ny, fx + j,
                 work->lut_scratch, work->lut_scratch + BOOP_LUT_MAX_AXIS);
    if (ny > 1) {
      double *fy = fx + count, *fxy = fy + count;
      lut_factor(ny, input->nodes[1], work->lut_scratch);
      for (int i = 0; i < nx; ++i) {
        lut_slopes(ny, input->nodes[1], f + i * ny, 1, fy + i * ny,
                   work->lut_scratch, work->lut_scratch + BOOP_LUT_MAX_AXIS);
        lut_slopes(ny, input->nodes[1], fx + i * ny, 1, fxy + i * ny,
                   work->lut_scratch, work->lut_scratch + BOOP_LUT_MAX_AXIS);
      }
    }
    for (int i = 0; i < (ny > 1 ? 4 : 2) * count; ++i)
      if (!isfinite(f[i])) return BOOP_INVALID_LUT;
  }
  return BOOP_OK;
}

static int lut_interval(const double *nodes, int n, double query) {
  int lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    int mid = lo + (hi - lo) / 2;
    if (query < nodes[mid]) hi = mid;
    else lo = mid;
  }
  return lo;
}

/* Value and slope basis at each endpoint; differentiate in physical coordinates. */
static void lut_basis(double t, double h, int order, double basis[4]) {
  static const double coefficients[4][4] = {
    {1, 0, -3, 2}, {0, 1, -2, 1}, {0, 0, 3, -2}, {0, 0, -1, 1}
  };
  for (int b = 0; b < 4; ++b) {
    double value = 0.0;
    for (int power = 3; power >= order; --power) {
      double c = coefficients[b][power];
      for (int d = 0; d < order; ++d) c *= power - d;
      value = value * t + c;
    }
    for (int d = 0; d < order; ++d) value /= h;
    basis[b] = (b % 2 ? h : 1.0) * value;
  }
}

double boop_lut_eval(const BoopWorkspace *work, int k, int dx, int dy,
                            double x, double y) {
  if (!isfinite(x) || !isfinite(y)) return NAN;
  int nx = boop_lut_shapes[k][0], ny = boop_lut_shapes[k][1];
  const double *xn = work->lut_nodes[k][0];
  int i = lut_interval(xn, nx, x);
  double hx = xn[i + 1] - xn[i], bx[4];
  lut_basis((x - xn[i]) / hx, hx, dx, bx);
  int count = nx * ny;
  const double *fields = work->lut_fields + boop_lut_offsets[k];
  double result = 0.0;
  if (ny == 1) {
    for (int a = 0; a < 4; ++a)
      result += bx[a] * fields[(a % 2) * count + i + a / 2];
  } else {
    const double *yn = work->lut_nodes[k][1];
    int j = lut_interval(yn, ny, y);
    double hy = yn[j + 1] - yn[j], by[4];
    lut_basis((y - yn[j]) / hy, hy, dy, by);
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b)
        result += bx[a] * by[b] * fields[((a % 2) + 2 * (b % 2)) * count
                                         + (i + a / 2) * ny + j + b / 2];
  }
  return result;
}
