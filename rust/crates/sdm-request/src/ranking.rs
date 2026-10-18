// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A ranking asked as the questions the model serves, and its answers put
//! back together (T2.8): the client's `decompose._per_candidate` and
//! `_as_literal`, and `SDMClient._ranked` and `_how_sure`.
//!
//! The endpoint has no ranking task. A ranking over candidates is asked as
//! one binary question per distinct candidate, whether the entity reaches it
//! in the window, written from the query's final aggregation
//! ([`Ranked::question`]), each candidate written into its question by the
//! type of the column ranked ([`literal`]). The answers are put back
//! together as one row per entity asked about, each candidate scored by the
//! probability of its question's positive class, best first, at most `top_k`
//! ([`rank`]), laid out as an Arrow table of candidate positions and scores
//! ([`table`]); a host gives each candidate back as the value it was given
//! as.
//!
//! Each refusal is a code and fields naming the candidate by its position
//! among those given, never its value; the host words it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, ListBuilder, StringBuilder};
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};

use crate::predict::{Entity, Part, id_columns};

/// A window of the final aggregation, as the query writes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Window {
    /// `None` for an open start, which the client cannot write a question
    /// of.
    pub start: Option<i64>,
    pub end: i64,
    /// The unit's name as the query writes it (`days`).
    pub unit: String,
}

/// What every candidate's question is written from: the final
/// aggregation's target, as the query writes it, filter and all, and the
/// table and column it names; its window; and the entity clause, filter and
/// all (`_for_each`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ranked {
    pub target: String,
    /// The table and the column ranked, each unquoted.
    pub table: String,
    pub column: String,
    pub window: Option<Window>,
    pub entity: String,
}

impl Ranked {
    /// Whether the rows ranked are chosen by a filter (`' WHERE ' in
    /// ranked`), which the client cannot combine with each candidate's own.
    pub fn filtered(&self) -> bool {
        self.target.contains(" WHERE ")
    }

    /// The question asked of one candidate, written as `literal`: whether
    /// the entity reaches a row of the ranked table holding it in the
    /// window, `PREDICT COUNT(<table>.* WHERE <target>=<literal>, <start>,
    /// <end>, <unit>) > 0 FOR EACH ...`; `quote` spells the table as the
    /// query language names it. None where the window has no start, which
    /// the client cannot write (a `TypeError` there).
    pub fn question(&self, literal: &str, quote: fn(&str) -> String) -> Option<String> {
        let Window {
            start: Some(start),
            end,
            unit,
        } = self.window.as_ref()?
        else {
            return None;
        };
        Some(format!(
            "PREDICT COUNT({}.* WHERE {}={literal}, {start}, {end}, {unit}) > 0 {}",
            quote(&self.table),
            self.target,
            self.entity,
        ))
    }
}

/// One candidate as the host gave it: a value of its Arrow column.
#[derive(Debug, Clone, PartialEq)]
pub enum Candidate {
    /// A null, or a float NaN: no row is named.
    Missing,
    Bool(bool),
    /// An integer of any Arrow integer type.
    Int(i128),
    /// A float of any Arrow float type, as the double it is.
    Float(f64),
    /// A decimal: its unscaled value and scale.
    Decimal {
        value: i128,
        scale: i8,
    },
    Text(String),
    /// A value of another type, by the type's name.
    Other(String),
}

impl Candidate {
    /// A key equal for two candidates Python's `dict.fromkeys` holds as
    /// one: a number by its exact value, whatever its type (`1`, `1.0`,
    /// `True` and `Decimal('1.00')` are one), text by its text, a missing
    /// value one key; none for a NaN or a value of another type, which no
    /// other equals.
    fn key(&self) -> Option<String> {
        let trimmed = |text: String| -> String {
            let text = if text.contains('.') {
                text.trim_end_matches('0').trim_end_matches('.').to_owned()
            } else {
                text
            };
            if text == "-0" { "0".to_owned() } else { text }
        };
        match self {
            Candidate::Missing => Some("z".to_owned()),
            Candidate::Bool(b) => Some(format!("n{}", u8::from(*b))),
            Candidate::Int(n) => Some(format!("n{n}")),
            Candidate::Float(f) if !f.is_finite() => {
                (!f.is_nan()).then(|| format!("i{}", f.is_sign_negative()))
            }
            Candidate::Float(f) => Some(format!("n{}", trimmed(format!("{f:.1100}")))),
            Candidate::Decimal { value, scale } => {
                Some(format!("n{}", trimmed(decimal_text(*value, *scale))))
            }
            Candidate::Text(text) => Some(format!("s{text}")),
            Candidate::Other(_) => None,
        }
    }
}

/// A candidate as the host gave it, from its contract scalar: a null or a
/// NaN missing; a decimal past 128 bits, or a scalar of a type a host does
/// not write a candidate as, another type, by its name.
pub fn read(scalar: &Scalar) -> Candidate {
    match scalar {
        Scalar::Null => Candidate::Missing,
        Scalar::Bool(b) => Candidate::Bool(*b),
        Scalar::Int8(v) => Candidate::Int((*v).into()),
        Scalar::Int16(v) => Candidate::Int((*v).into()),
        Scalar::Int32(v) => Candidate::Int((*v).into()),
        Scalar::Int64(v) => Candidate::Int((*v).into()),
        Scalar::UInt8(v) => Candidate::Int((*v).into()),
        Scalar::UInt16(v) => Candidate::Int((*v).into()),
        Scalar::UInt32(v) => Candidate::Int((*v).into()),
        Scalar::UInt64(v) => Candidate::Int((*v).into()),
        Scalar::Float32(v) => float((*v).into()),
        Scalar::Float64(v) => float(*v),
        Scalar::Decimal128 { scale, value, .. } => Candidate::Decimal {
            value: *value,
            scale: *scale,
        },
        Scalar::Decimal256 { scale, value, .. } => match value.to_i128() {
            Some(value) => Candidate::Decimal {
                value,
                scale: *scale,
            },
            None => Candidate::Other("d256".to_owned()),
        },
        Scalar::Utf8(text) => Candidate::Text(text.clone()),
        Scalar::Binary(_) => Candidate::Other("binary".to_owned()),
        Scalar::Date32(_) => Candidate::Other("date32".to_owned()),
        Scalar::Timestamp { .. } => Candidate::Other("ts".to_owned()),
        Scalar::Duration { .. } => Candidate::Other("duration".to_owned()),
        Scalar::Time64 { .. } => Candidate::Other("time64".to_owned()),
    }
}

/// A float, a NaN missing (`pd.isna`).
fn float(value: f64) -> Candidate {
    if value.is_nan() {
        Candidate::Missing
    } else {
        Candidate::Float(value)
    }
}

/// A candidate as the result's `rankings` names it (T2.8): the JSON text of
/// its value, an integer's digits, a float as Python's `repr` writes it
/// (`float_repr`), a decimal's digits at its scale, text as a JSON string;
/// none for a value no question is asked of (missing, a boolean, another
/// type).
pub fn json_text(candidate: &Candidate, float_repr: fn(f64) -> String) -> Option<String> {
    match candidate {
        Candidate::Int(n) => Some(n.to_string()),
        Candidate::Float(f) if f.is_finite() => Some(float_repr(*f)),
        Candidate::Decimal { value, scale } => Some(decimal_text(*value, *scale)),
        Candidate::Text(text) => Some(
            String::from_utf8_lossy(&sdm_wire::json::Value::text(text).to_bytes()).into_owned(),
        ),
        _ => None,
    }
}

/// The distinct candidates, each with its position among those given (its
/// first), in the order first given (`dict.fromkeys`).
pub fn distinct(candidates: &[Candidate]) -> Vec<(u32, &Candidate)> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut kept: Vec<(u32, &Candidate)> = Vec::new();
    for (position, candidate) in (0u32..).zip(candidates) {
        // A value with no key equals none but itself, and is kept.
        let new = candidate.key().is_none_or(|key| seen.insert(key));
        if new {
            kept.push((position, candidate));
        }
    }
    kept
}

/// A candidate refused, by rule, naming it by its position.
fn refused(rule: &'static str, position: u32) -> CoreError {
    CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        "a candidate cannot be written into the question asked of it",
    )
    .with_field("field", Detail::text("candidates"))
    .with_field("rule", Detail::text(rule))
    .with_field("candidate", Detail::Scalar(Scalar::UInt32(position)))
}

/// Whether Python's `str.isspace` holds of `c`: Unicode white space, and
/// the four separators U+001C to U+001F.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// A decimal's text as `format(value, 'f')` writes it: its digits, at its
/// scale.
fn decimal_text(value: i128, scale: i8) -> String {
    let negative = value < 0;
    let digits = value.unsigned_abs().to_string();
    let sign = if negative { "-" } else { "" };
    if scale <= 0 {
        let zeros = "0".repeat(usize::from(scale.unsigned_abs()));
        return if value == 0 {
            format!("{sign}0")
        } else {
            format!("{sign}{digits}{zeros}")
        };
    }
    let scale = usize::from(scale.unsigned_abs());
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (whole, fraction) = padded.split_at(padded.len() - scale);
    format!("{sign}{whole}.{fraction}")
}

/// Whether `float` is exactly the decimal `value` at `scale`
/// (`Decimal(float(value)) == value`), from the double's exact expansion.
fn exactly(float: f64, value: i128, scale: i8) -> bool {
    if float == 0.0 || value == 0 {
        return float == 0.0 && value == 0;
    }
    let trimmed = |text: String| -> String {
        if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            text
        }
    };
    trimmed(format!("{float:.1100}")) == trimmed(decimal_text(value, scale))
}

/// A float's text as `format(Decimal(str(value)), 'f')` writes it, from its
/// shortest repr as Python writes it (`repr`): positional, with no
/// exponent.
fn positional(repr: &str) -> String {
    let Some((mantissa, exponent)) = repr.split_once('e') else {
        return repr.to_owned();
    };
    let Ok(exponent) = exponent.parse::<i64>() else {
        return repr.to_owned();
    };
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fraction}");
    // Where the point falls, counted in digits from the left.
    let point = i64::try_from(whole.len()).unwrap_or(0) + exponent;
    let len = i64::try_from(digits.len()).unwrap_or(0);
    if point <= 0 {
        let zeros = "0".repeat(usize::try_from(-point).unwrap_or(0));
        format!("{sign}0.{zeros}{digits}")
    } else if point >= len {
        let zeros = "0".repeat(usize::try_from(point - len).unwrap_or(0));
        format!("{sign}{digits}{zeros}")
    } else {
        let at = usize::try_from(point).unwrap_or(0);
        format!("{sign}{}.{}", &digits[..at], &digits[at..])
    }
}

/// A whole double's digits, as `str(int(value))` writes them.
fn whole_text(value: f64) -> String {
    if value == 0.0 {
        "0".to_owned()
    } else {
        format!("{value:.0}")
    }
}

/// The candidate at `position` as the query language writes it, by the
/// dtype of the column ranked (`dtype`, its name on the wire, none where
/// the schema does not hold it), as the client's `_as_literal` writes it:
/// an integer as its digits; a whole float as its integer's; a decimal as
/// the column's type takes it (an integer column's integer, a float
/// column's double, else its digits as text), refused where that is
/// another value; text in double quotes; a number on a text column as its
/// digits in double quotes. Refused (`INVALID_REQUEST`, field
/// `candidates`, naming the rule and the candidate): a missing value
/// (`candidate_missing`), a boolean (`candidate_bool`), a decimal its
/// column's type does not hold exactly (`candidate_not_exact`), a float not
/// finite or not whole (`candidate_not_whole`), and text holding a quote, a
/// comma, a bracket or a backslash, or only white space
/// (`candidate_unwritable`); a value of any other type, which this core does
/// not write yet (`UNSUPPORTED_REQUEST`, rule `candidate_type`).
/// `float_repr` is Python's `repr` of a double.
pub fn literal(
    candidate: &Candidate,
    position: u32,
    dtype: Option<&str>,
    float_repr: fn(f64) -> String,
) -> Result<String, CoreError> {
    let mut value = candidate.clone();
    match &value {
        Candidate::Missing => return Err(refused("candidate_missing", position)),
        Candidate::Bool(_) => return Err(refused("candidate_bool", position)),
        Candidate::Other(name) => {
            return Err(CoreError::new(
                ErrorKind::Unsupported,
                ErrorCode::UnsupportedRequest,
                "a candidate is of a type this core does not write into a question yet",
            )
            .with_field("field", Detail::text("candidates"))
            .with_field("rule", Detail::text("candidate_type"))
            .with_field("candidate", Detail::Scalar(Scalar::UInt32(position)))
            .with_field("type", Detail::text(name)));
        }
        Candidate::Decimal {
            value: digits,
            scale,
        } => {
            let (digits, scale) = (*digits, *scale);
            let dtype = dtype.map_or_else(|| decimal_dtype(digits, scale), str::to_owned);
            value = match dtype.as_str() {
                "int64" => {
                    let unit = 10_i128.checked_pow(u32::from(scale.max(0).unsigned_abs()));
                    match unit {
                        Some(unit) if digits % unit == 0 => Candidate::Int(digits / unit),
                        _ => {
                            return Err(refused("candidate_not_exact", position)
                                .with_field("dtype", Detail::text(&dtype)));
                        }
                    }
                }
                "float64" => {
                    let text = decimal_text(digits, scale);
                    match text.parse::<f64>() {
                        Ok(float) if exactly(float, digits, scale) => Candidate::Float(float),
                        _ => {
                            return Err(refused("candidate_not_exact", position)
                                .with_field("dtype", Detail::text(&dtype)));
                        }
                    }
                }
                _ => Candidate::Text(decimal_text(digits, scale)),
            };
        }
        _ => {}
    }
    if dtype == Some("string") {
        match &value {
            Candidate::Float(f) if !f.is_finite() => {
                return Err(refused("candidate_not_whole", position));
            }
            Candidate::Float(f) => value = Candidate::Text(positional(&float_repr(*f))),
            Candidate::Int(n) => value = Candidate::Text(n.to_string()),
            _ => {}
        }
    }
    match value {
        Candidate::Int(n) => Ok(n.to_string()),
        Candidate::Float(f) if !f.is_finite() || f.fract() != 0.0 => {
            Err(refused("candidate_not_whole", position))
        }
        Candidate::Float(f) => Ok(whole_text(f)),
        Candidate::Text(text) => {
            if text.chars().any(|c| "\"'\\,()".contains(c)) || text.chars().all(is_space) {
                return Err(refused("candidate_unwritable", position));
            }
            Ok(format!("\"{text}\""))
        }
        _ => Err(refused("candidate_missing", position)),
    }
}

/// The dtype a decimal's column travels as where the schema does not say
/// (`_decimal_dtype` of the one value): `float64` for a fraction, `int64`
/// for a whole number of fewer than 18 digits or one int64 holds, else
/// `string`.
fn decimal_dtype(value: i128, scale: i8) -> String {
    let unit = 10_i128.checked_pow(u32::from(scale.max(0).unsigned_abs()));
    match unit {
        Some(unit) if value % unit != 0 => "float64".to_owned(),
        Some(unit) => {
            if i64::try_from(value / unit).is_ok() {
                "int64".to_owned()
            } else {
                "string".to_owned()
            }
        }
        None => "float64".to_owned(),
    }
}

/// One row of a ranking: the entity asked about on it, and its candidates,
/// best first, at most `top_k`, each as the result names it ([`json_text`]),
/// with its score.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub row_index: i64,
    pub id: Entity,
    pub ranked: Vec<(String, f64)>,
}

/// How sure the endpoint was that the row's entity reaches the candidate:
/// the probability of the positive class, spelled `true`, `True`, `TRUE` or
/// `1` (`_how_sure`); none where the answer gives no such class.
fn how_sure(probabilities: Option<&Vec<(String, f64)>>) -> Option<f64> {
    let given = probabilities?;
    ["true", "True", "TRUE", "1"]
        .iter()
        .find_map(|spelling| given.iter().find(|(class, _)| class == spelling))
        .map(|(_, p)| *p)
}

/// The answers to a ranking's questions put back together (`_ranked`):
/// each candidate's answer, by its position among those given and as the
/// result names it, in the order asked. One row per row asked about, keyed by its row and entity, in the
/// order the answers first name it; each candidate scored by how sure its
/// answer was ([`how_sure`]), an answer that does not say refused
/// (`INTERNAL_ERROR`, naming the candidate and the classes it gave); sorted
/// best first, ties in the order asked, a NaN last (register PD-1), and
/// cut to `top_k`.
pub fn rank(answers: &[(u32, String, Part)], top_k: Option<u32>) -> Result<Vec<Row>, CoreError> {
    let mut rows: Vec<Row> = Vec::new();
    let mut at: BTreeMap<(i64, String), usize> = BTreeMap::new();
    for (candidate, named, part) in answers {
        for answer in &part.rows {
            let Some(score) = how_sure(answer.probabilities.as_ref()) else {
                let mut classes: Vec<&str> = answer
                    .probabilities
                    .iter()
                    .flatten()
                    .map(|(class, _)| class.as_str())
                    .collect();
                classes.sort_unstable();
                return Err(CoreError::new(
                    ErrorKind::Internal,
                    ErrorCode::InternalError,
                    "the endpoint answered a candidate's question without saying how sure it was",
                )
                .with_field("field", Detail::text("probabilities"))
                .with_field("candidate", Detail::Scalar(Scalar::UInt32(*candidate)))
                .with_field(
                    "classes",
                    Detail::List(
                        classes
                            .into_iter()
                            .map(|c| Scalar::Utf8(c.to_owned()))
                            .collect(),
                    ),
                ));
            };
            let key = (answer.row_index, format!("{:?}", answer.id));
            let row = *at.entry(key).or_insert_with(|| {
                rows.push(Row {
                    row_index: answer.row_index,
                    id: answer.id.clone(),
                    ranked: Vec::new(),
                });
                rows.len() - 1
            });
            rows[row].ranked.push((named.clone(), score));
        }
    }
    for row in &mut rows {
        // Stable, best first, as Python's sort with `reverse=True` keeps
        // equal scores in the order asked; a NaN last (PD-1).
        row.ranked
            .sort_by(|(_, a), (_, b)| match (a.is_nan(), b.is_nan()) {
                (false, false) => b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal),
                (a_nan, b_nan) => a_nan.cmp(&b_nan),
            });
        if let Some(k) = top_k
            && k > 0
        {
            row.ranked.truncate(k as usize);
        }
    }
    Ok(rows)
}

/// A ranking's result: one row per entity asked about, ordered by
/// `row_index` (stable), with `row_index`, `id` (and `id_as_written`) as
/// [`crate::predict::table`] writes them, `rankings` (list of utf8, each
/// candidate as the result names it, best first), `scores` (list of
/// float64, each one's score, in the same order) and `answer_index`, the
/// row's place in the order the answers named it.
pub fn table(rows: &[Row], text_ids: bool) -> Result<RecordBatch, CoreError> {
    let internal = || {
        CoreError::new(
            ErrorKind::Internal,
            ErrorCode::InternalError,
            "the ranking does not lay out",
        )
    };
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| rows[i].row_index);
    let mut fields = vec![Field::new("row_index", DataType::Int64, false)];
    let mut columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter_values(
        order.iter().map(|&i| rows[i].row_index),
    ))];
    let ids: Vec<&Entity> = order.iter().map(|&i| &rows[i].id).collect();
    let (id_fields, id_columns) = id_columns(&ids, text_ids);
    fields.extend(id_fields);
    columns.extend(id_columns);
    let mut rankings = ListBuilder::new(StringBuilder::new());
    let mut scores = ListBuilder::new(Float64Builder::new());
    for &i in &order {
        for (candidate, score) in &rows[i].ranked {
            rankings.values().append_value(candidate);
            scores.values().append_value(*score);
        }
        rankings.append(true);
        scores.append(true);
    }
    let rankings = rankings.finish();
    let scores = scores.finish();
    fields.push(Field::new("rankings", rankings.data_type().clone(), false));
    columns.push(Arc::new(rankings));
    fields.push(Field::new("scores", scores.data_type().clone(), false));
    columns.push(Arc::new(scores));
    fields.push(Field::new("answer_index", DataType::Int64, false));
    columns.push(Arc::new(Int64Array::from_iter_values(
        order.iter().map(|&i| i64::try_from(i).unwrap_or(i64::MAX)),
    )));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(|_| internal())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repr(x: f64) -> String {
        // Enough of Python's repr for these tests.
        match x {
            1.0 => "1.0".into(),
            1e16 => "1e+16".into(),
            1.5e-7 => "1.5e-07".into(),
            _ => format!("{x}"),
        }
    }

    fn written(candidate: Candidate, dtype: Option<&str>) -> Result<String, String> {
        literal(&candidate, 0, dtype, repr).map_err(|e| {
            e.fields
                .get("rule")
                .map(|r| format!("{r:?}"))
                .unwrap_or_default()
        })
    }

    #[test]
    fn a_candidate_is_written_as_the_client_writes_it() {
        assert_eq!(written(Candidate::Int(7), Some("int64")), Ok("7".into()));
        assert_eq!(
            written(Candidate::Int(7), Some("string")),
            Ok("\"7\"".into())
        );
        assert_eq!(
            written(Candidate::Float(3.0), Some("int64")),
            Ok("3".into())
        );
        assert_eq!(
            written(Candidate::Float(-0.0), Some("int64")),
            Ok("0".into())
        );
        assert_eq!(
            written(Candidate::Float(1e20), Some("int64")),
            Ok("100000000000000000000".into())
        );
        assert!(written(Candidate::Float(2.5), Some("int64")).is_err());
        assert_eq!(
            written(Candidate::Float(1.0), Some("string")),
            Ok("\"1.0\"".into())
        );
        assert_eq!(
            written(Candidate::Float(1e16), Some("string")),
            Ok("\"10000000000000000\"".into())
        );
        assert_eq!(
            written(Candidate::Float(1.5e-7), Some("string")),
            Ok("\"0.00000015\"".into())
        );
        assert_eq!(
            written(Candidate::Text("a b".into()), Some("string")),
            Ok("\"a b\"".into())
        );
        assert!(written(Candidate::Text("a,b".into()), None).is_err());
        assert!(written(Candidate::Text(" \u{1f}".into()), None).is_err());
        assert!(written(Candidate::Missing, None).is_err());
        assert!(written(Candidate::Bool(true), None).is_err());
        let decimal = |value, scale| Candidate::Decimal { value, scale };
        assert_eq!(written(decimal(500, 2), Some("int64")), Ok("5".into()));
        assert!(written(decimal(510, 2), Some("int64")).is_err());
        assert_eq!(written(decimal(500, 2), Some("float64")), Ok("5".into()));
        assert!(written(decimal(1, 1), Some("float64")).is_err());
        assert!(written(decimal(5, 1), Some("float64")).is_err());
        assert_eq!(
            written(decimal(150, 2), Some("string")),
            Ok("\"1.50\"".into())
        );
        assert_eq!(
            written(decimal(-5, 3), Some("string")),
            Ok("\"-0.005\"".into())
        );
        assert_eq!(written(decimal(5, 0), None), Ok("5".into()));
    }

    #[test]
    fn candidates_are_kept_once_in_the_order_first_given() {
        let given = vec![
            Candidate::Int(2),
            Candidate::Int(1),
            Candidate::Int(2),
            Candidate::Missing,
            Candidate::Missing,
        ];
        let kept: Vec<u32> = distinct(&given).into_iter().map(|(p, _)| p).collect();
        assert_eq!(kept, vec![0, 1, 3]);
        let floats = vec![Candidate::Float(0.0), Candidate::Float(-0.0)];
        assert_eq!(distinct(&floats).len(), 1);
    }

    #[test]
    fn candidates_equal_in_python_are_one_whatever_their_type() {
        let given = vec![
            Candidate::Int(1),
            Candidate::Float(1.0),
            Candidate::Bool(true),
            Candidate::Decimal {
                value: 100,
                scale: 2,
            },
            Candidate::Text("1".into()),
            Candidate::Float(0.5),
            Candidate::Decimal { value: 5, scale: 1 },
            Candidate::Float(f64::NAN),
            Candidate::Float(f64::NAN),
        ];
        let kept: Vec<u32> = distinct(&given).into_iter().map(|(p, _)| p).collect();
        assert_eq!(kept, vec![0, 4, 5, 7, 8]);
    }

    #[test]
    fn a_candidate_is_named_by_the_json_text_of_its_value() {
        assert_eq!(json_text(&Candidate::Int(-3), repr), Some("-3".into()));
        assert_eq!(
            json_text(&Candidate::Float(1e16), repr),
            Some("1e+16".into())
        );
        assert_eq!(
            json_text(
                &Candidate::Decimal {
                    value: 150,
                    scale: 2
                },
                repr
            ),
            Some("1.50".into())
        );
        assert_eq!(
            json_text(&Candidate::Text("a\"b".into()), repr),
            Some("\"a\\\"b\"".into())
        );
        assert_eq!(json_text(&Candidate::Missing, repr), None);
    }

    fn answered(scores: &[(i64, f64)]) -> Part {
        Part {
            model: "m".into(),
            request_id: "r".into(),
            metadata: sdm_wire::json::Value::Object(vec![]),
            rows: scores
                .iter()
                .map(|(row, p)| crate::predict::Row {
                    row_index: *row,
                    forecast_step: None,
                    answer_index: 0,
                    id: Entity::Integer(*row),
                    prediction: crate::predict::Answer::Bool(true),
                    probabilities: Some(vec![("False".into(), 1.0 - p), ("True".into(), *p)]),
                    scores: None,
                    rankings: None,
                })
                .collect(),
            answers: crate::predict::AnswerType::Bool,
            text_ids: false,
        }
    }

    #[test]
    fn a_ranking_is_best_first_ties_in_order_asked_a_nan_last_cut_to_k() {
        let answers = vec![
            (0, "\"a\"".to_owned(), answered(&[(0, 0.2), (1, f64::NAN)])),
            (1, "\"b\"".to_owned(), answered(&[(0, 0.9), (1, 0.5)])),
            (2, "\"c\"".to_owned(), answered(&[(0, 0.2), (1, 0.7)])),
        ];
        let rows = rank(&answers, Some(2)).unwrap();
        fn named(row: &Row) -> Vec<&str> {
            row.ranked.iter().map(|(c, _)| c.as_str()).collect()
        }
        assert_eq!(named(&rows[0]), ["\"b\"", "\"a\""]);
        assert_eq!(named(&rows[1]), ["\"c\"", "\"b\""]);
        let all = rank(&answers, None).unwrap();
        assert_eq!(named(&all[1]), ["\"c\"", "\"b\"", "\"a\""]);
        // An answer with no positive class's probability is refused.
        let mut silent = answered(&[(0, 0.5)]);
        silent.rows[0].probabilities = Some(vec![("x".into(), 0.5)]);
        let error = rank(&[(4, "4".into(), silent)], None).unwrap_err();
        assert_eq!(error.code, ErrorCode::InternalError);
    }

    #[test]
    fn a_float_is_written_without_an_exponent() {
        assert_eq!(positional("1e+16"), "10000000000000000");
        assert_eq!(positional("-1.5e+16"), "-15000000000000000");
        assert_eq!(positional("1.5e-07"), "0.00000015");
        assert_eq!(positional("1.25e+01"), "12.5");
        assert_eq!(positional("0.1"), "0.1");
    }
}
