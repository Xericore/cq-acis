use super::*;

fn word(out: &mut Vec<u8>, value: i64, width: usize) {
    out.extend_from_slice(&value.to_le_bytes()[..width]);
}
fn name(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&[13, value.len() as u8]);
    out.extend_from_slice(value.as_bytes());
}
fn fixture(width: usize, flags: i64, extra: Option<(&str, Vec<u8>)>) -> Vec<u8> {
    let mut out = if width == 4 {
        b"ASM BinaryFile4".to_vec()
    } else {
        b"ASM BinaryFile8".to_vec()
    };
    for n in [23200, 0, 0, flags] {
        word(&mut out, n, width);
    }
    for text in ["test", "ASM", "date"] {
        out.extend_from_slice(&[7, text.len() as u8]);
        out.extend_from_slice(text.as_bytes());
    }
    for n in [1.0_f64, 1e-6, 1e-10] {
        out.push(6);
        out.extend_from_slice(&n.to_le_bytes());
    }
    name(&mut out, "asmheader");
    out.push(12);
    word(&mut out, -1, width);
    out.push(4);
    word(&mut out, -1, width);
    out.push(17);
    if let Some((kind, fields)) = extra {
        name(&mut out, kind);
        out.push(12);
        word(&mut out, -1, width);
        out.push(4);
        word(&mut out, -1, width);
        out.extend(fields);
        out.push(17);
    }
    name(&mut out, "End-of-ASM-data");
    out
}
fn parse(data: &[u8]) -> Result<SabDocument> {
    parse_sab(data, "test-domain", &SabLimits::default())
}

#[test]
fn four_and_eight_byte_words_preserve_source_and_signed_values() {
    for width in [4, 8] {
        let value = if width == 8 {
            i64::MIN + 7
        } else {
            i32::MIN as i64 + 7
        };
        let mut fields = vec![4];
        word(&mut fields, value, width);
        fields.push(12);
        word(&mut fields, 0, width);
        let data = fixture(width, 0, Some(("future-attribute", fields)));
        let doc = parse(&data).unwrap();
        assert_eq!(doc.header.integer_width, width as u8);
        let raw = doc.model.entities()[1].raw();
        let span = raw.source.as_ref().unwrap();
        assert_eq!(
            raw.raw_data.as_deref(),
            Some(&data[span.start_offset..span.end_offset])
        );
        assert_eq!(
            raw.values,
            vec![
                AcisValue::Integer(value.into()),
                AcisValue::Reference(EntityRef(0))
            ]
        );
        assert!(raw.record.is_none());
    }
}
#[test]
fn every_truncation_without_history_is_rejected() {
    let data = fixture(4, 0, None);
    for end in 0..data.len() {
        assert!(parse(&data[..end]).is_err(), "accepted prefix {end}");
    }
    let mut trailing = data.clone();
    trailing.push(0);
    assert!(parse(&trailing).is_err());
    let mut unknown = data;
    unknown[15..19].copy_from_slice(&23201i32.to_le_bytes());
    assert!(parse(&unknown).unwrap_err().message.contains("unsupported"));
}
#[test]
fn unknown_tags_bad_lengths_and_unbalanced_scopes_fail() {
    for fields in [
        vec![255],
        vec![16],
        vec![15],
        vec![9, 255, 255, 255, 255],
        vec![7, 2, 255, 255],
    ] {
        assert!(parse(&fixture(4, 0, Some(("future", fields)))).is_err());
    }
}
#[test]
fn references_cannot_resolve_into_opaque_history() {
    let mut ref_one = vec![12];
    word(&mut ref_one, 2, 4);
    let mut data = fixture(4, 1, Some(("future", ref_one)));
    let end = data.len() - 17;
    data.truncate(end);
    name(&mut data, "delta_state");
    data.extend_from_slice(b"opaque history");
    assert!(parse(&data).is_err());
    let mut data = fixture(4, 1, None);
    data.truncate(data.len() - 17);
    name(&mut data, "delta_state");
    data.extend_from_slice(b"opaque history");
    let doc = parse(&data).unwrap();
    assert!(doc.history.is_some());
    assert_eq!(doc.model.entities().len(), 2);
    data[27..31].copy_from_slice(&0u32.to_le_bytes());
    assert!(parse(&data).is_err());
}
#[test]
fn byte_record_token_string_and_depth_limits_are_enforced() {
    let data = fixture(4, 0, None);
    for limits in [
        SabLimits {
            max_bytes: data.len() - 1,
            ..Default::default()
        },
        SabLimits {
            max_records: 0,
            ..Default::default()
        },
        SabLimits {
            max_tokens: 1,
            ..Default::default()
        },
        SabLimits {
            max_string_bytes: 1,
            ..Default::default()
        },
    ] {
        assert!(parse_sab(&data, "test", &limits).is_err());
    }
    let deep = fixture(4, 0, Some(("future", vec![15, 15, 16, 16])));
    assert!(parse_sab(
        &deep,
        "test",
        &SabLimits {
            max_depth: 1,
            ..Default::default()
        }
    )
    .is_err());
    assert!(parse_sab(
        &data,
        "test",
        &SabLimits {
            max_records: 1,
            ..Default::default()
        }
    )
    .is_ok());
}
#[test]
fn schema_extensions_remain_raw_instead_of_being_silently_ignored() {
    let mut fields = vec![12];
    word(&mut fields, -1, 4);
    fields.push(19);
    for n in [1.0_f64, 2.0, 3.0] {
        fields.extend_from_slice(&n.to_le_bytes());
    }
    let doc = parse(&fixture(4, 0, Some(("point", fields.clone())))).unwrap();
    assert!(matches!(doc.model.entities()[1], Entity::Point(_)));
    fields.push(11);
    let doc = parse(&fixture(4, 0, Some(("point", fields)))).unwrap();
    assert!(matches!(doc.model.entities()[1], Entity::Raw(_)));
    assert!(doc
        .model
        .diagnostics()
        .iter()
        .any(|d| d.code == "sab.entity_schema_unsupported"));
}

fn real(fields: &mut Vec<u8>, x: f64) {
    fields.push(6);
    fields.extend_from_slice(&x.to_le_bytes());
}
fn vector(fields: &mut Vec<u8>, xyz: [f64; 3]) {
    fields.push(19);
    for x in xyz {
        fields.extend_from_slice(&x.to_le_bytes());
    }
}
#[test]
fn analytic_sab_fields_preserve_scale_signed_radius_and_all_bytes() {
    for width in [4, 8] {
        let mut prefix = vec![12];
        word(&mut prefix, -1, width);
        let mut cone = prefix.clone();
        for v in [[0., 0., 0.], [0., 0., 1.], [4., 0., 0.]] {
            vector(&mut cone, v);
        }
        real(&mut cone, 0.5);
        cone.extend([11, 11]);
        for x in [0.6, -0.8, 13.] {
            real(&mut cone, x);
        }
        cone.extend([11; 5]);
        let parsed = parse(&fixture(width, 0, Some(("cone-surface", cone.clone())))).unwrap();
        let Entity::ConeSurface(e) = &parsed.model.entities()[1] else {
            panic!("cone not decoded")
        };
        assert_eq!(
            (e.reference_radius, e.parameter_scale, e.cos_half_angle),
            (4., 13., -0.8)
        );
        assert_eq!(e.evaluate(0., 5.).unwrap(), Vec3::from((7., 0., -4.)));
        cone.push(11);
        assert!(matches!(
            parse(&fixture(width, 0, Some(("cone-surface", cone))))
                .unwrap()
                .model
                .entities()[1],
            Entity::Raw(_)
        ));
        let mut sphere = prefix.clone();
        vector(&mut sphere, [1., 2., 3.]);
        real(&mut sphere, -2.);
        vector(&mut sphere, [1., 0., 0.]);
        vector(&mut sphere, [0., 0., 1.]);
        sphere.extend([11; 5]);
        let doc = parse(&fixture(width, 0, Some(("sphere-surface", sphere)))).unwrap();
        let Entity::SphereSurface(e) = &doc.model.entities()[1] else {
            panic!("sphere not decoded")
        };
        assert_eq!(e.radius, -2.);
        assert_eq!(e.pole, Vec3::from((0., 0., 1.)));
        assert_eq!(e.evaluate(0., 0.).unwrap(), Vec3::from((3., 2., 3.)));
        let mut torus = prefix;
        vector(&mut torus, [0., 0., 0.]);
        vector(&mut torus, [0., 0., 1.]);
        real(&mut torus, 4.);
        real(&mut torus, -1.);
        vector(&mut torus, [1., 0., 0.]);
        torus.extend([11; 5]);
        let doc = parse(&fixture(width, 0, Some(("torus-surface", torus)))).unwrap();
        let Entity::TorusSurface(e) = &doc.model.entities()[1] else {
            panic!("torus not decoded")
        };
        assert_eq!(e.minor_radius, -1.);
        assert_eq!(e.evaluate(0., 0.).unwrap(), Vec3::from((5., 0., 0.)));
    }
}

// Independently specified quadratic rational quarter-circle, embedded profile
// 22601. These are synthetic wire-format tests, not native vendor oracles.
fn explicit_curve_values() -> Vec<AcisValue> {
    use AcisValue::{Bytes as B, Float as F, Integer as I, Reference as R, String as S};
    vec![
        R(EntityRef(-1)),
        B(vec![11]),
        B(vec![15]),
        S("exact_int_cur".into()),
        I(22601.into()),
        I(0.into()),
        S("nurbs".into()),
        I(2.into()),
        I(0.into()),
        I(2.into()),
        F(0.),
        I(2.into()),
        F(1.),
        I(2.into()),
        F(1.),
        F(0.),
        F(0.),
        F(1.),
        F(1.),
        F(1.),
        F(0.),
        F(0.5_f64.sqrt()),
        F(0.),
        F(1.),
        F(0.),
        F(1.),
        F(0.),
        S("null_surface".into()),
        S("null_surface".into()),
        S("nullbs".into()),
        S("nullbs".into()),
        B(vec![11]),
        B(vec![11]),
        I(0.into()),
        I(0.into()),
        I(0.into()),
        I(0.into()),
        B(vec![10]),
        F(1.),
        B(vec![10]),
        F(0.),
        I(0.into()),
        I(0.into()),
        B(vec![16]),
        B(vec![10]),
        F(0.),
        B(vec![10]),
        F(1.),
    ]
}

#[test]
fn subtype_alias_uses_nested_definition_and_its_own_envelope() {
    use AcisValue::{Bytes as B, Float as F, Integer as I, Reference as R, String as S};
    let direct = explicit_curve_values();
    let end = direct.iter().position(|v| *v == B(vec![16])).unwrap() + 1;
    let mut owner = RawEntity::new(0, "pcurve");
    owner.values = vec![
        R(EntityRef(-1)),
        I(0.into()),
        B(vec![11]),
        B(vec![15]),
        S("exp_par_cur".into()),
    ];
    owner.values.extend_from_slice(&direct[2..end]);
    owner.values.push(B(vec![16]));
    let mut alias = RawEntity::new(1, "intcurve-curve");
    alias.values = vec![
        R(EntityRef(-1)),
        B(vec![11]),
        B(vec![15]),
        S("ref".into()),
        I(1.into()),
        B(vec![16]),
        B(vec![10]),
        F(0.2),
        B(vec![10]),
        F(0.8),
    ];
    alias.raw_data = Some(vec![1, 2, 3]);
    let model = AcisModel::new(
        AcisMetadata {
            save_version: Some(22700.into()),
            ..Default::default()
        },
        vec![Entity::Raw(owner), Entity::Raw(alias.clone())],
        vec![],
    )
    .unwrap();
    let resolved = crate::subtypes::SubtypeResolver::new(std::sync::Arc::new(model.clone()))
        .resolve(alias.index)
        .unwrap()
        .unwrap();
    assert_eq!(resolved.definition.index, 1);
    assert_eq!(resolved.definition.parent, Some(0));
    let Entity::BSplineCurve(curve) = resolved.geometry else {
        panic!()
    };
    assert_eq!(curve.raw, alias);
    assert_eq!(curve.parameter_range.unwrap().lower, Some(0.2));
    let p = curve.evaluate(0.5).unwrap();
    assert!((p.x - 0.5_f64.sqrt()).abs() < 1e-12);
    assert!((p.y - 0.5_f64.sqrt()).abs() < 1e-12);
    let mut reversed = alias.clone();
    reversed.values[1] = B(vec![10]);
    let mut wrong_kind = alias.clone();
    wrong_kind.type_name = "spline-surface".into();
    let mut wrong_target = alias;
    wrong_target.values[4] = I(0.into());
    for bad in [reversed, wrong_kind, wrong_target] {
        let bad_model = AcisModel::new(
            model.metadata().clone(),
            vec![model.entities()[0].clone(), Entity::Raw(bad)],
            vec![],
        )
        .unwrap();
        assert!(
            crate::subtypes::SubtypeResolver::new(std::sync::Arc::new(bad_model))
                .resolve(1)
                .is_err()
        );
    }
}
fn encode_values(values: &[AcisValue], width: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in values {
        match value {
            AcisValue::Reference(v) => {
                bytes.push(12);
                word(&mut bytes, v.0, width);
            }
            AcisValue::Integer(v) => {
                bytes.push(4);
                word(&mut bytes, v.to_string().parse().unwrap(), width);
            }
            AcisValue::Float(v) => {
                bytes.push(6);
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            AcisValue::String(s) => {
                bytes.extend_from_slice(&[7, s.len() as u8]);
                bytes.extend_from_slice(s.as_bytes());
            }
            AcisValue::Bytes(b) => bytes.extend_from_slice(b),
            _ => panic!("unsupported test value"),
        }
    }
    bytes
}
fn explicit_fixture(values: &[AcisValue], width: usize, version: i64) -> Vec<u8> {
    let mut data = fixture(
        width,
        0,
        Some(("intcurve-curve", encode_values(values, width))),
    );
    data[15..15 + width].copy_from_slice(&version.to_le_bytes()[..width]);
    data
}
#[test]
fn explicit_asm_curve_preserves_bytes_and_matches_rational_circle() {
    for width in [4, 8] {
        let values = explicit_curve_values();
        let data = explicit_fixture(&values, width, 22700);
        let model = parse(&data).unwrap().model;
        let Entity::BSplineCurve(curve) = &model.entities()[1] else {
            panic!("not decoded")
        };
        assert_eq!(curve.raw.values, values);
        let span = curve.raw.source.as_ref().unwrap();
        assert_eq!(
            curve.raw.raw_data.as_deref(),
            Some(&data[span.start_offset..span.end_offset])
        );
        for t in [0., 0.1, 0.5, 0.9, 1.] {
            let p = curve.evaluate(t).unwrap();
            assert!((p.x.hypot(p.y) - 1.).abs() < 1e-12);
        }
        assert!((curve.evaluate(0.5).unwrap().x - 0.5_f64.sqrt()).abs() < 1e-12);
        assert_eq!(curve.multiplicities, vec![3, 3]);
        assert!(curve.evaluate(-0.1).is_err());
        assert!(curve.evaluate(f64::NAN).is_err());
    }
}
#[test]
fn unqualified_explicit_spline_records_stay_raw_with_source_diagnostics() {
    let base = explicit_curve_values();
    let mut cases = vec![];
    for (index, value) in [
        (1, AcisValue::Bytes(vec![10])),
        (4, AcisValue::Integer(22602.into())),
        (5, AcisValue::Integer(1.into())),
        (7, AcisValue::Integer(usize::MAX.into())),
        (8, AcisValue::Integer(2.into())),
        (9, AcisValue::Integer(4097.into())),
        (11, AcisValue::Integer(1000000.into())),
        (21, AcisValue::Float(-1.)),
        (26, AcisValue::Float(0.001)),
        (27, AcisValue::String("plane".into())),
        (41, AcisValue::Integer(1.into())),
    ] {
        let mut v = base.clone();
        v[index] = value;
        cases.push(v);
    }
    // Negative degree instead of usize::MAX is serializable in both word sizes.
    cases[3][7] = AcisValue::Integer((-1).into());
    let mut trailing = base.clone();
    trailing.push(AcisValue::Integer(1.into()));
    cases.push(trailing);
    for values in cases {
        let model = parse(&explicit_fixture(&values, 8, 22700)).unwrap().model;
        assert!(matches!(&model.entities()[1],Entity::Raw(raw) if raw.values==values));
        assert!(model
            .diagnostics()
            .iter()
            .any(|d| d.code == "sab.entity_schema_unsupported"));
    }
    let old = parse(&explicit_fixture(&base, 4, 22600)).unwrap();
    assert!(matches!(old.model.entities()[1], Entity::Raw(_)));
    for length in 0..base.len() {
        let raw = RawEntity {
            values: base[..length].to_vec(),
            ..RawEntity::new(0, "intcurve-curve")
        };
        assert!(
            !matches!(crate::asm_nurbs::decode(&raw, 22700), Ok(Some(_))),
            "accepted prefix {length}"
        );
    }
}

#[test]
fn explicit_asm_surface_decodes_tensor_order_and_refuses_trailer_changes() {
    use AcisValue::{Bytes as B, Float as F, Integer as I, Reference as R, String as S};
    let mut values = vec![
        R(EntityRef(-1)),
        B(vec![11]),
        B(vec![15]),
        S("exact_spl_sur".into()),
        I(22601.into()),
        I(0.into()),
        S("nurbs".into()),
        I(1.into()),
        I(1.into()),
        I(0.into()),
        I(0.into()),
        I(0.into()),
        I(0.into()),
        I(2.into()),
        I(2.into()),
    ];
    for _ in 0..2 {
        values.extend([F(0.), I(1.into()), F(1.), I(1.into())]);
    }
    for p in [
        [0., 0., 0., 1.],
        [2., 0., 0., 2.],
        [0., 3., 0., 1.],
        [2., 3., 4., 2.],
    ] {
        values.extend(p.map(F));
    }
    values.push(F(0.));
    for _ in 0..6 {
        values.push(I(0.into()));
    }
    values.push(B(vec![11]));
    for _ in 0..2 {
        values.extend([B(vec![10]), F(1.), B(vec![10]), F(0.)]);
    }
    values.extend([
        I(0.into()),
        B(vec![16]),
        B(vec![11]),
        B(vec![11]),
        B(vec![11]),
        B(vec![11]),
    ]);
    for width in [4, 8] {
        let mut data = fixture(
            width,
            0,
            Some(("spline-surface", encode_values(&values, width))),
        );
        data[15..15 + width].copy_from_slice(&22700_i64.to_le_bytes()[..width]);
        let doc = parse(&data).unwrap();
        let Entity::BSplineSurface(s) = &doc.model.entities()[1] else {
            panic!("not decoded")
        };
        assert_eq!(s.raw.values, values);
        for (u, v) in [(0., 0.), (1., 1.), (0.2, 0.7)] {
            let p = s.evaluate(u, v).unwrap();
            let a = 2. * u / (1. + u);
            assert!(
                (p.x - 2. * a).abs() < 1e-12
                    && (p.y - 3. * v).abs() < 1e-12
                    && (p.z - 4. * a * v).abs() < 1e-12
            );
        }
    }
    let raw = RawEntity {
        values: values.clone(),
        ..RawEntity::new(0, "spline-surface")
    };
    for length in 0..values.len() {
        let mut truncated = raw.clone();
        truncated.values.truncate(length);
        assert!(!matches!(
            crate::asm_nurbs::decode(&truncated, 22700),
            Ok(Some(_))
        ));
    }
    let mut future = raw.clone();
    future.values.push(I(1.into()));
    assert!(crate::asm_nurbs::decode(&future, 22700).is_err());
    let mut finite = raw.clone();
    let index = finite.values.len() - 4;
    finite.values.splice(index..index + 1, [B(vec![10]), F(0.)]);
    let Some(Entity::BSplineSurface(surface)) = crate::asm_nurbs::decode(&finite, 22700).unwrap()
    else {
        panic!("finite chart not decoded")
    };
    assert_eq!(surface.u_range.unwrap().lower, Some(0.));
    finite.values[index + 1] = F(-0.1);
    assert!(crate::asm_nurbs::decode(&finite, 22700).is_err());
    let mut periodic = raw;
    periodic.values[9] = I(2.into());
    assert!(crate::asm_nurbs::decode(&periodic, 22700).is_err());
}

// Transform records in both forms. D0_TRANSFORM_RECORD is a real binary record
// (an AutoCAD 2018 drawing, ASM 4-byte words) and D0_VALUES its 12 stored
// doubles, independently decoded (as f64 bit patterns; hex-float in comments).
// STAIRS_TEXT_TRANSFORM_RECORD is a real text record of the same exporter.
const D0_TRANSFORM_RECORD: &str = concat!(
    "0d097472616e73666f726d0cffffffff04ffffffff",
    "1418c67a91d2a9a6bf1d363e861786ee3ffcb3f32e1e01d33f",
    "14bdd23137c2308c3f15aa5153e3fbd2bfc9b18e3d808eee3f",
    "143991789931f7ef3fa4fe68bfdfbba73f0000000000c0dbbc",
    "14ff2e67e051e65e40a52a7802cffd55c03e72d539e40ba740",
    "06000000000000f03f",
    "0a0b0b11",
);
const D0_VALUES: [u64; 12] = [
    0xbfa6a9d2917ac618, // -0x1.6a9d2917ac618p-5
    0x3fee8617863e361d, // 0x1.e8617863e361dp-1
    0x3fd3011e2ef3b3fc, // 0x1.3011e2ef3b3fcp-2
    0x3f8c30c23731d2bd, // 0x1.c30c23731d2bdp-7
    0xbfd2fbe35351aa15, // -0x1.2fbe35351aa15p-2
    0x3fee8e803d8eb1c9, // 0x1.e8e803d8eb1c9p-1
    0x3feff73199789139, // 0x1.ff73199789139p-1
    0x3fa7bbdfbf68fea4, // 0x1.7bbdfbf68fea4p-5
    0xbcdbc00000000000, // -0x1.bc00000000000p-50
    0x405ee651e0672eff, // 0x1.ee651e0672effp+6
    0xc055fdcf02782aa5, // -0x1.5fdcf02782aa5p+6
    0x40a70be439d5723e, // 0x1.70be439d5723ep+11
];
const STAIRS_TEXT_TRANSFORM_RECORD: &str = concat!(
    "0d097472616e73666f726d0cffffffff04ffffffff",
    "125e000000",
    "3120302030203020312030203020302031203137352e3230353135313833303639323834",
    "202d3333362e393530373632303037323439393820313035362031206e6f5f726f746174",
    "65206e6f5f7265666c656374206e6f5f736865617220",
    "11",
);
const STAIRS_TEXT_VALUES: [u64; 12] = [
    0x3ff0000000000000, // 1
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x3ff0000000000000, // 1
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x3ff0000000000000, // 1
    0x4065e6909a92714a, // 175.20515183069284
    0xc0750f365238f6b0, // -336.95076200724998
    0x4090800000000000, // 1056
];
fn hex_bytes(hex: &str) -> Vec<u8> {
    assert!(hex.len() % 2 == 0);
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}
/// A 4-byte ASM document whose record 1 is `record` (identifier through 0x11).
fn document_with(record: &[u8]) -> Vec<u8> {
    let mut data = fixture(4, 0, None);
    data.truncate(data.len() - 17); // drop the End-of-ASM-data identifier
    data.extend_from_slice(record);
    name(&mut data, "End-of-ASM-data");
    data
}
fn binary_fields(vectors: [[f64; 3]; 4], scale: f64, flags: [bool; 3]) -> Vec<u8> {
    let mut fields: Vec<u8> = Vec::new();
    for xyz in vectors {
        fields.push(20);
        for x in xyz {
            fields.extend_from_slice(&x.to_le_bytes());
        }
    }
    real(&mut fields, scale);
    fields.extend(flags.map(|f| if f { 10u8 } else { 11u8 }));
    fields
}
fn text_fields(text: &str) -> Vec<u8> {
    let mut fields = vec![18];
    word(&mut fields, text.len() as i64, 4);
    fields.extend_from_slice(text.as_bytes());
    fields
}
fn transform_of(data: &[u8]) -> TransformEntity {
    let doc = parse(data).unwrap();
    match &doc.model.entities()[1] {
        Entity::Transform(t) => t.clone(),
        other => panic!("transform not decoded: {}", other.raw().type_name),
    }
}
const IDENTITY_AT_10_20_30: [[f64; 3]; 4] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [10., 20., 30.]];

#[test]
fn binary_transform_decodes() {
    let record = hex_bytes(D0_TRANSFORM_RECORD);
    let data = document_with(&record);
    let doc = parse(&data).unwrap();
    let Entity::Transform(t) = &doc.model.entities()[1] else {
        panic!("binary transform not decoded")
    };
    assert_eq!(t.matrix_values.map(f64::to_bits), D0_VALUES);
    assert_eq!(t.scale, 1.);
    let span = t.raw.source.as_ref().unwrap();
    assert_eq!(&data[span.start_offset..span.end_offset], &record[..]);
    assert!(doc
        .model
        .diagnostics()
        .iter()
        .all(|d| d.code != "sab.entity_schema_unsupported"));
    // Stored order is the row layout: x maps to the first vector, the origin to the translation.
    let [a, b, c, ..] = t.matrix_values;
    assert_eq!(t.transform_vector(Vec3::from((1., 0., 0.))), Vec3::from((a, b, c)));
    let [.., x, y, z] = t.matrix_values;
    assert_eq!(t.transform_point(Vec3::from((0., 0., 0.))), Vec3::from((x, y, z)));
}

#[test]
fn binary_transform_flags_recorded_not_applied() {
    for flags in [
        [false, false, false],
        [true, false, false],
        [false, true, false],
        [false, false, true],
        [true, true, true],
    ] {
        let fields = binary_fields(IDENTITY_AT_10_20_30, 1., flags);
        let t = transform_of(&fixture(4, 0, Some(("transform", fields))));
        assert_eq!((t.rotated, t.reflected, t.sheared), (flags[0], flags[1], flags[2]));
        assert_eq!(
            t.matrix_values,
            [1., 0., 0., 0., 1., 0., 0., 0., 1., 10., 20., 30.]
        );
        assert_eq!(t.linear_determinant(), 1.);
        assert_eq!(t.transform_point(Vec3::from((1., 2., 3.))), Vec3::from((11., 22., 33.)));
    }
    let d0 = transform_of(&document_with(&hex_bytes(D0_TRANSFORM_RECORD)));
    assert_eq!((d0.rotated, d0.reflected, d0.sheared), (true, false, false));
}

#[test]
fn text_transform_unchanged() {
    // The row-vector example of tests/geometry.rs, read through the SAB text form.
    let text = "0 1 0 1 0 0 0 0 1 10 20 30 2 rotate reflect no_shear";
    let t = transform_of(&fixture(4, 0, Some(("transform", text_fields(text)))));
    assert_eq!(t.matrix_values, [0., 1., 0., 1., 0., 0., 0., 0., 1., 10., 20., 30.]);
    assert_eq!(t.scale, 2.);
    assert_eq!((t.rotated, t.reflected, t.sheared), (true, true, false));
    assert_eq!(t.linear_determinant(), -8.);
    assert_eq!(t.transform_vector(Vec3::from((1., 2., 3.))), Vec3::from((4., 2., 6.)));
    assert_eq!(t.transform_point(Vec3::from((1., 2., 3.))), Vec3::from((14., 22., 36.)));
    // A real text record: a pure translation, all flags off.
    let record = hex_bytes(STAIRS_TEXT_TRANSFORM_RECORD);
    let data = document_with(&record);
    let t = transform_of(&data);
    assert_eq!(t.matrix_values.map(f64::to_bits), STAIRS_TEXT_VALUES);
    assert_eq!(t.scale, 1.);
    assert_eq!((t.rotated, t.reflected, t.sheared), (false, false, false));
    let span = t.raw.source.as_ref().unwrap();
    assert_eq!(&data[span.start_offset..span.end_offset], &record[..]);
}

#[test]
fn text_transform_flags_available() {
    // The Python model exposes these flags as rotate/reflect/shear for both forms.
    let words = "1 0 0 0 1 0 0 0 1 175.20515183069284 -336.95076200724998 1056 1";
    let plain = transform_of(&fixture(
        4,
        0,
        Some(("transform", text_fields(&format!("{words} no_rotate no_reflect no_shear ")))),
    ));
    assert_eq!(plain.matrix_values.map(f64::to_bits), STAIRS_TEXT_VALUES);
    for (flags, expected) in [
        ("rotate no_reflect no_shear", (true, false, false)),
        ("no_rotate reflect no_shear", (false, true, false)),
        ("no_rotate no_reflect shear", (false, false, true)),
        ("rotate reflect shear", (true, true, true)),
    ] {
        let fields = text_fields(&format!("{words} {flags}"));
        let t = transform_of(&fixture(4, 0, Some(("transform", fields))));
        assert_eq!((t.rotated, t.reflected, t.sheared), expected);
        assert_eq!(
            t.matrix_values.map(f64::to_bits),
            plain.matrix_values.map(f64::to_bits)
        );
        assert_eq!(t.scale.to_bits(), plain.scale.to_bits());
    }
    let fields = text_fields(&format!("{words} rotated no_reflect no_shear"));
    let doc = parse(&fixture(4, 0, Some(("transform", fields)))).unwrap();
    assert!(matches!(doc.model.entities()[1], Entity::Raw(_)));
}

#[test]
fn binary_transform_malformed_stays_raw() {
    let full = binary_fields(IDENTITY_AT_10_20_30, 1., [false; 3]);
    let mut three_vectors = full.clone();
    three_vectors.drain(75..100); // drop the translation vector
    let missing_boolean = full[..full.len() - 1].to_vec();
    let mut trailing = full.clone();
    trailing.push(11);
    let mut nan_vector = IDENTITY_AT_10_20_30;
    nan_vector[1][2] = f64::NAN;
    let mut infinite_translation = IDENTITY_AT_10_20_30;
    infinite_translation[3][0] = f64::INFINITY;
    for fields in [
        three_vectors,
        missing_boolean,
        trailing,
        binary_fields(nan_vector, 1., [false; 3]),
        binary_fields(infinite_translation, 1., [false; 3]),
        binary_fields(IDENTITY_AT_10_20_30, f64::NAN, [false; 3]),
    ] {
        let data = fixture(4, 0, Some(("transform", fields)));
        let doc = parse(&data).unwrap();
        let Entity::Raw(raw) = &doc.model.entities()[1] else {
            panic!("malformed binary transform was decoded")
        };
        assert_eq!(raw.type_name, "transform");
        let span = raw.source.as_ref().unwrap();
        assert_eq!(raw.raw_data.as_deref(), Some(&data[span.start_offset..span.end_offset]));
        assert!(doc
            .model
            .diagnostics()
            .iter()
            .any(|d| d.code == "sab.entity_schema_unsupported" && d.entity_index == Some(1)));
    }
}
