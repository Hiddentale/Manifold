use glam::Vec3;

/// Local metric parameters at a world position.
#[derive(Copy, Clone, Debug)]
pub struct MetricParams {
    pub minkowski_exponent: f32,
    pub warpedness: f32,
}

impl MetricParams {
    pub const EUCLIDEAN: Self = Self {
        minkowski_exponent: 2.0,
        warpedness: 0.0,
    };
}

/// A localized region where the metric deviates from Euclidean.
#[derive(Clone, Debug)]
pub struct Anomaly {
    pub center: Vec3,
    pub inner_radius: f32,
    pub outer_radius: f32,
    pub target_minkowski_exponent: f32,
    pub active: bool,
}

/// Spatial field that maps any world position to its local metric.
/// Default is Euclidean everywhere. Anomalies warp the metric in their radius.
pub struct MetricField {
    anomalies: Vec<Anomaly>,
}

impl MetricField {
    pub fn new() -> Self {
        Self { anomalies: Vec::new() }
    }

    pub fn add_anomaly(&mut self, anomaly: Anomaly) {
        self.anomalies.push(anomaly);
    }

    pub fn anomalies(&self) -> &[Anomaly] {
        &self.anomalies
    }

    pub fn sample_metric_at_pos(&self, pos: Vec3) -> MetricParams {
        let mut inv_minkowski_exponent_sum = 0.0_f32;
        let mut weight_sum = 0.0_f32;
        let mut max_warpedness = 0.0_f32;

        for anomaly in &self.anomalies {
            if !anomaly.active {
                continue;
            }
            let distance_to_anomaly = (pos - anomaly.center).length();
            if distance_to_anomaly >= anomaly.outer_radius {
                continue;
            }

            let t = if distance_to_anomaly <= anomaly.inner_radius {
                1.0
            } else {
                let raw = 1.0 - (distance_to_anomaly - anomaly.inner_radius) / (anomaly.outer_radius - anomaly.inner_radius);
                raw * raw * (3.0 - 2.0 * raw) // smoothstep
            };

            inv_minkowski_exponent_sum += t * inv_p(anomaly.target_minkowski_exponent);
            weight_sum += t;
            max_warpedness = max_warpedness.max(t);
        }

        if weight_sum < 1e-6 {
            return MetricParams::EUCLIDEAN;
        }

        // Weighted average of anomaly contributions + Euclidean baseline
        let euclidean_weight = (1.0 - weight_sum).max(0.0);
        let total_inv_p = inv_minkowski_exponent_sum + euclidean_weight * inv_p(2.0);
        let total_weight = weight_sum + euclidean_weight;
        let result_minkowski_exponent = from_inv_p(total_inv_p / total_weight);

        MetricParams {
            minkowski_exponent: result_minkowski_exponent,
            warpedness: max_warpedness,
        }
    }
}

/// Lp (Minkowski) distance between two points for a given p-value.
pub fn minkowski_distance(a: Vec3, b: Vec3, p: f32) -> f32 {
    let d = (a - b).abs();
    if p >= 50.0 {
        d.x.max(d.y).max(d.z)
    } else if p <= 1.0 {
        d.x + d.y + d.z
    } else {
        (d.x.powf(p) + d.y.powf(p) + d.z.powf(p)).powf(1.0 / p)
    }
}

/// Interpolate between two p-values in reciprocal space.
/// This keeps the blend finite even when one endpoint is Chebyshev (p=inf).
pub fn lerp_p(a: f32, b: f32, t: f32) -> f32 {
    let result = inv_p(a) + (inv_p(b) - inv_p(a)) * t;
    from_inv_p(result)
}

/// Speed multiplier for movement in direction `dir` under metric with exponent `p`.
/// Returns the ratio of Euclidean distance to metric distance for a unit step.
/// > 1.0 means faster (Chebyshev diagonal), < 1.0 means slower (Manhattan diagonal).
pub fn metric_speed_scale(dir: Vec3, p: f32) -> f32 {
    if (p - 2.0).abs() < 0.01 {
        return 1.0;
    }
    let unit = dir.normalize_or_zero();
    if unit == Vec3::ZERO {
        return 1.0;
    }
    let euclidean = unit.length(); // always 1.0 for normalized
    let metric = minkowski_distance(Vec3::ZERO, unit, p);
    if metric < 1e-6 {
        return 1.0;
    }
    euclidean / metric
}

fn inv_p(p: f32) -> f32 {
    if p >= 50.0 {
        0.0
    } else {
        1.0 / p
    }
}

fn from_inv_p(inv: f32) -> f32 {
    if inv <= 0.01 {
        50.0
    } else {
        1.0 / inv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    // --- minkowski_distance ---

    #[test]
    fn euclidean_distance_p2() {
        let d = minkowski_distance(Vec3::ZERO, Vec3::new(3.0, 4.0, 0.0), 2.0);
        assert!(approx(d, 5.0), "expected 5.0, got {d}");
    }

    #[test]
    fn manhattan_distance_p1() {
        let d = minkowski_distance(Vec3::ZERO, Vec3::new(3.0, 4.0, 5.0), 1.0);
        assert!(approx(d, 12.0), "expected 12.0, got {d}");
    }

    #[test]
    fn chebyshev_distance_p_large() {
        let d = minkowski_distance(Vec3::ZERO, Vec3::new(3.0, 7.0, 5.0), 50.0);
        assert!(approx(d, 7.0), "expected 7.0, got {d}");
    }

    #[test]
    fn distance_is_symmetric() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(5.0, 6.0, 7.0);
        for p in [1.0, 1.5, 2.0, 3.0, 50.0] {
            let d1 = minkowski_distance(a, b, p);
            let d2 = minkowski_distance(b, a, p);
            assert!(approx(d1, d2), "asymmetric for p={p}: {d1} != {d2}");
        }
    }

    #[test]
    fn distance_zero_for_same_point() {
        let a = Vec3::new(5.0, 3.0, 1.0);
        for p in [1.0, 2.0, 50.0] {
            let d = minkowski_distance(a, a, p);
            assert!(approx(d, 0.0), "expected 0 for p={p}, got {d}");
        }
    }

    #[test]
    fn manhattan_geq_euclidean_geq_chebyshev() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(5.0, 7.0, 9.0);
        let d1 = minkowski_distance(a, b, 1.0);
        let d2 = minkowski_distance(a, b, 2.0);
        let dinf = minkowski_distance(a, b, 50.0);
        assert!(d1 >= d2, "Manhattan {d1} < Euclidean {d2}");
        assert!(d2 >= dinf, "Euclidean {d2} < Chebyshev {dinf}");
    }

    // --- lerp_p ---

    #[test]
    fn lerp_p_at_zero_returns_first() {
        assert!(approx(lerp_p(2.0, 50.0, 0.0), 2.0));
    }

    #[test]
    fn lerp_p_at_one_returns_second() {
        assert!(approx(lerp_p(2.0, 50.0, 1.0), 50.0));
    }

    #[test]
    fn lerp_p_midpoint_is_between_inputs() {
        let mid = lerp_p(1.0, 50.0, 0.5);
        assert!(mid > 1.0 && mid < 50.0, "midpoint {mid} not between 1 and 50");
    }

    #[test]
    fn lerp_p_between_euclidean_and_manhattan() {
        let mid = lerp_p(2.0, 1.0, 0.5);
        assert!(mid > 1.0 && mid < 2.0, "midpoint {mid} not between 1 and 2");
    }

    // --- MetricField::sample ---

    #[test]
    fn empty_field_returns_euclidean() {
        let field = MetricField::new();
        let params = field.sample_metric_at_pos(Vec3::ZERO);
        assert!(approx(params.minkowski_exponent, 2.0));
        assert!(approx(params.warpedness, 0.0));
    }

    #[test]
    fn at_anomaly_center_returns_target_p() {
        let mut field = MetricField::new();
        field.add_anomaly(Anomaly {
            center: Vec3::ZERO,
            inner_radius: 10.0,
            outer_radius: 50.0,
            target_minkowski_exponent: 1.0,
            active: true,
        });
        let params = field.sample_metric_at_pos(Vec3::ZERO);
        assert!(
            approx(params.minkowski_exponent, 1.0),
            "expected p=1.0, got {}",
            params.minkowski_exponent
        );
        assert!(approx(params.warpedness, 1.0));
    }

    #[test]
    fn outside_anomaly_returns_euclidean() {
        let mut field = MetricField::new();
        field.add_anomaly(Anomaly {
            center: Vec3::ZERO,
            inner_radius: 10.0,
            outer_radius: 50.0,
            target_minkowski_exponent: 1.0,
            active: true,
        });
        let params = field.sample_metric_at_pos(Vec3::new(100.0, 0.0, 0.0));
        assert!(approx(params.minkowski_exponent, 2.0));
        assert!(approx(params.warpedness, 0.0));
    }

    #[test]
    fn gradient_zone_is_between_target_and_euclidean() {
        let mut field = MetricField::new();
        field.add_anomaly(Anomaly {
            center: Vec3::ZERO,
            inner_radius: 10.0,
            outer_radius: 50.0,
            target_minkowski_exponent: 1.0,
            active: true,
        });
        let params = field.sample_metric_at_pos(Vec3::new(30.0, 0.0, 0.0));
        assert!(
            params.minkowski_exponent > 1.0 && params.minkowski_exponent < 2.0,
            "gradient p={} not between 1 and 2",
            params.minkowski_exponent
        );
        assert!(params.warpedness > 0.0 && params.warpedness < 1.0);
    }

    #[test]
    fn inactive_anomaly_is_ignored() {
        let mut field = MetricField::new();
        field.add_anomaly(Anomaly {
            center: Vec3::ZERO,
            inner_radius: 10.0,
            outer_radius: 50.0,
            target_minkowski_exponent: 1.0,
            active: false,
        });
        let params = field.sample_metric_at_pos(Vec3::ZERO);
        assert!(approx(params.minkowski_exponent, 2.0));
    }

    #[test]
    fn two_overlapping_anomalies_blend() {
        let mut field = MetricField::new();
        // Manhattan anomaly at origin
        field.add_anomaly(Anomaly {
            center: Vec3::new(-20.0, 0.0, 0.0),
            inner_radius: 10.0,
            outer_radius: 40.0,
            target_minkowski_exponent: 1.0,
            active: true,
        });
        // Chebyshev anomaly nearby
        field.add_anomaly(Anomaly {
            center: Vec3::new(20.0, 0.0, 0.0),
            inner_radius: 10.0,
            outer_radius: 40.0,
            target_minkowski_exponent: 50.0,
            active: true,
        });
        let params = field.sample_metric_at_pos(Vec3::ZERO);
        assert!(
            params.minkowski_exponent > 1.0 && params.minkowski_exponent < 50.0,
            "overlap p={} not intermediate",
            params.minkowski_exponent
        );
    }

    #[test]
    fn p_decreases_monotonically_toward_manhattan_center() {
        let mut field = MetricField::new();
        field.add_anomaly(Anomaly {
            center: Vec3::ZERO,
            inner_radius: 10.0,
            outer_radius: 50.0,
            target_minkowski_exponent: 1.0,
            active: true,
        });
        let mut prev_minkowski_exponent = 2.0;
        for i in 1..=10 {
            let dist = 50.0 - (i as f32 * 4.0); // walk from outer edge inward
            let params = field.sample_metric_at_pos(Vec3::new(dist, 0.0, 0.0));
            assert!(
                params.minkowski_exponent <= prev_minkowski_exponent + EPSILON,
                "p increased at dist={dist}: {} > {prev_minkowski_exponent}",
                params.minkowski_exponent
            );
            prev_minkowski_exponent = params.minkowski_exponent;
        }
    }

    // --- metric_speed_scale ---

    #[test]
    fn euclidean_metric_gives_unit_scale() {
        let scale = metric_speed_scale(Vec3::new(1.0, 0.0, 1.0), 2.0);
        assert!(approx(scale, 1.0), "expected 1.0, got {scale}");
    }

    #[test]
    fn chebyshev_diagonal_is_faster() {
        let scale = metric_speed_scale(Vec3::new(1.0, 0.0, 1.0), 50.0);
        assert!(scale > 1.0, "Chebyshev diagonal scale {scale} should be > 1.0");
    }

    #[test]
    fn manhattan_diagonal_is_slower() {
        let scale = metric_speed_scale(Vec3::new(1.0, 0.0, 1.0), 1.0);
        assert!(scale < 1.0, "Manhattan diagonal scale {scale} should be < 1.0");
    }

    #[test]
    fn cardinal_direction_unaffected_by_metric() {
        // Along a single axis, all Lp metrics agree (distance = |x|)
        for p in [1.0, 2.0, 50.0] {
            let scale = metric_speed_scale(Vec3::new(1.0, 0.0, 0.0), p);
            assert!(approx(scale, 1.0), "cardinal scale for p={p} should be 1.0, got {scale}");
        }
    }
}
