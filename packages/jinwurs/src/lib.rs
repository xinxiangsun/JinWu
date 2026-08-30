//! jinwurs — Rust-accelerated kernels for jinwu

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Core rebinning: distribute original-bin counts into new uniform bins.
#[pyfunction]
#[pyo3(signature = (orig_counts, orig_errors, orig_left, orig_right, orig_width, orig_exposure, new_edges))]
fn rebin_counts_core<'py>(
    py: Python<'py>,
    orig_counts: PyReadonlyArray1<'py, f64>,
    orig_errors: Option<PyReadonlyArray1<'py, f64>>,
    orig_left: PyReadonlyArray1<'py, f64>,
    orig_right: PyReadonlyArray1<'py, f64>,
    orig_width: PyReadonlyArray1<'py, f64>,
    orig_exposure: Option<PyReadonlyArray1<'py, f64>>,
    new_edges: PyReadonlyArray1<'py, f64>,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let oc = orig_counts.as_slice().unwrap();
    let ol = orig_left.as_slice().unwrap();
    let or_ = orig_right.as_slice().unwrap();
    let ow = orig_width.as_slice().unwrap();
    let ne = new_edges.as_slice().unwrap();

    let oe: Option<&[f64]> = orig_errors.as_ref().map(|a| a.as_slice().unwrap());
    let oexpo: Option<&[f64]> = orig_exposure.as_ref().map(|a| a.as_slice().unwrap());

    let n_orig = oc.len();
    let m = ne.len().saturating_sub(1);
    let have_exposure = oexpo.is_some();

    let mut counts = vec![0.0_f64; m];
    let mut var = vec![0.0_f64; m];
    let mut exposure = vec![0.0_f64; m];

    for i in 0..n_orig {
        let a = ol[i];
        let b = or_[i];
        if b <= ne[0] || a >= ne[m] {
            continue;
        }

        // binary search for overlapping new-bin range
        let j0 = ne.partition_point(|&e| e <= a).saturating_sub(1);
        let j1 = ne.partition_point(|&e| e < b).saturating_sub(1);
        let j0 = j0.min(m.saturating_sub(1));
        let j1 = j1.min(m.saturating_sub(1));

        let owi = ow[i];
        let ci = oc[i];
        let ei = oe.map(|e| e[i]).unwrap_or_else(|| ci.max(0.0).sqrt());

        for j in j0..=j1 {
            let new_l = ne[j];
            let new_r = ne[j + 1];
            let overlap = (b.min(new_r) - a.max(new_l)).max(0.0);
            if overlap <= 0.0 {
                continue;
            }
            let frac = overlap / owi;
            let contrib = ci * frac;
            counts[j] += contrib;
            var[j] += (ei * frac).powi(2);

            if have_exposure {
                exposure[j] += oexpo.unwrap()[i] * frac;
            } else {
                exposure[j] += overlap;
            }
        }
    }

    let c = PyArray1::from_vec(py, counts).unbind();
    let v = PyArray1::from_vec(py, var).unbind();
    let e = PyArray1::from_vec(py, exposure).unbind();
    (c, v, e)
}

/// Finalize rebinned counts into counts space or rate space.
#[pyfunction]
fn rebin_finalize<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray1<'py, f64>,
    var: PyReadonlyArray1<'py, f64>,
    exposure: PyReadonlyArray1<'py, f64>,
    method: &str,
    empty_nan: bool,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let c = counts.as_slice().unwrap();
    let v = var.as_slice().unwrap();
    let e = exposure.as_slice().unwrap();
    let m = c.len();

    let mut value = vec![0.0_f64; m];
    let mut error = vec![0.0_f64; m];

    if method == "sum" {
        for j in 0..m {
            value[j] = c[j];
            error[j] = v[j].sqrt();
        }
    } else {
        for j in 0..m {
            let denom = if e[j] > 0.0 { e[j] } else { f64::NAN };
            value[j] = c[j] / denom;
            error[j] = v[j].sqrt() / denom;
        }
    }

    if empty_nan {
        for j in 0..m {
            if e[j] == 0.0 {
                value[j] = f64::NAN;
                error[j] = f64::NAN;
            }
        }
    }

    (PyArray1::from_vec(py, value).unbind(), PyArray1::from_vec(py, error).unbind())
}

#[pymodule]
fn jinwurs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rebin_counts_core, m)?)?;
    m.add_function(wrap_pyfunction!(rebin_finalize, m)?)?;
    Ok(())
}
