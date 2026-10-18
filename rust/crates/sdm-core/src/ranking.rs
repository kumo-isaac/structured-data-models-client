// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A ranking (`RANK TOP k`), asked as one binary question per candidate and
//! answered as one ranking (T2.8), as the reference's client decomposes it
//! (`decompose.parts_of`, `PredictiveQuery._in_parts`,
//! `SDMClient._answer_in_parts`), the questions written and the answers put
//! back together by [`sdm_request::ranking`].
//!
//! `materialize` refuses, in the client's order, what the client refuses
//! of a ranking before it gathers anything: a prediction statistic, a
//! ranking over no aggregation or over filtered rows, no candidates, none
//! left once each is kept once, more than the host's limit (`most_parts`),
//! a window with no start, a candidate that cannot be written into its
//! question, and a `top_k` that is not a number of candidates the model
//! ranks; then gathers each question in turn, as the client gathers each
//! part with a materializer of its own, a refusal naming the candidate it
//! happened in (`ranking_candidate`, its position among those given, and
//! `ranking_parts`).
//!
//! `predict` refuses outputs a ranking does not give, checks every question
//! before any is sent, leaving out a candidate whose context holds one
//! answer only (named by a notice, `CANDIDATES_UNSCORED`, or refused when
//! no candidate is left), asks each question left as the client asks a
//! classification, and puts the answers back together as one ranking.

use sdm_contract::ops::{
    Materialized, OperationResult, QueryFacts, Ranking, RankingPart, RequestState, SamplerMemo,
};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_request::ranking::{self, Candidate};
use sdm_wire::json::Value;

use crate::materialize::{
    Gathered, Inputs, MOST_PARTS, REQUEST_STATE_VERSION, attach_request, one,
};
use crate::pql::RankingPlan;
use crate::run::Asker;

/// A refusal of the call, `INVALID_REQUEST`, naming the field and the
/// rule.
fn refused(message: &str, field: &str, rule: &str) -> CoreError {
    CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        message,
    )
    .with_field("field", Detail::text(field))
    .with_field("rule", Detail::text(rule))
}

fn count(n: usize) -> Detail {
    Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)))
}

/// The refusal of candidates on a question that ranks nothing
/// (`parts_of`), before anything of it is gathered.
pub(crate) fn candidates_on_no_ranking() -> CoreError {
    refused(
        "candidates are the rows a ranking is over, and this question ranks nothing",
        "candidates",
        "candidates_not_ranked",
    )
}

/// A member of the call's option values, as the host gave it.
fn given<'a>(inputs: &'a Inputs, name: &str) -> Option<&'a Value> {
    inputs.values.as_ref().and_then(|values| values.get(name))
}

/// A ranking gathered: each distinct candidate's question, in the order
/// the candidates were given, each gathered as the client's `_in_parts`
/// gathers a part.
pub(crate) async fn materialize(
    asker: Asker,
    inputs: Inputs,
    plan: RankingPlan,
    facts: Option<QueryFacts>,
) -> Result<OperationResult, CoreError> {
    // `parts_of`: a statistic is a regression's, and each part is a
    // classification.
    if given(&inputs, "prediction_statistic").is_some() {
        return Err(refused(
            "prediction_statistic applies only to a regression, and a ranking is ordered by how \
             likely each candidate is",
            "prediction_statistic",
            "ranking_statistic",
        ));
    }
    // `_per_candidate`, in its order.
    let Some(ranked) = plan.ranked else {
        return Err(refused(
            "a ranking is asked one candidate at a time, about what an entity reaches in a \
             window, and this one ranks a column instead",
            "query",
            "ranking_column",
        ));
    };
    if ranked.filtered() {
        return Err(refused(
            "the rows to rank are chosen by a filter, which each candidate's question cannot \
             combine with its own",
            "query",
            "ranking_filter",
        ));
    }
    let Some(given_candidates) = &inputs.candidates else {
        return Err(refused(
            "a ranking is over candidates, which have to be named",
            "candidates",
            "candidates_missing",
        )
        .with_field("column", Detail::text(&ranked.column)));
    };
    let candidates: Vec<Candidate> = given_candidates.iter().map(ranking::read).collect();
    let kept = ranking::distinct(&candidates);
    if kept.is_empty() {
        return Err(refused(
            "candidates is empty, so there is nothing to rank",
            "candidates",
            "candidates_empty",
        ));
    }
    let most = inputs.most_parts.unwrap_or(MOST_PARTS);
    if kept.len() > most as usize {
        return Err(refused(
            "a ranking asks one question per candidate, and these are more than the host's limit",
            "candidates",
            "too_many_parts",
        )
        .with_field("candidates", count(kept.len()))
        .with_field("limit", Detail::Scalar(Scalar::UInt32(most))));
    }
    // How the column ranked travels (`_how_the_column_travels`), which
    // decides how each candidate is written.
    let dtype = inputs
        .description
        .tables
        .iter()
        .find(|t| t.name == ranked.table)
        .and_then(|t| t.columns.iter().find(|c| c.name == ranked.column))
        .map(|c| c.dtype.wire_name());
    if ranked.window.as_ref().is_none_or(|w| w.start.is_none()) {
        return Err(sdm_engine::not_yet("open_window"));
    }
    let mut questions = Vec::with_capacity(kept.len());
    for (position, candidate) in &kept {
        let literal = ranking::literal(candidate, *position, dtype, crate::pql::float_text)?;
        let question = ranked
            .question(&literal, crate::pql::quote_name)
            .ok_or_else(|| sdm_engine::not_yet("open_window"))?;
        let named = ranking::json_text(candidate, crate::pql::float_text)
            .ok_or_else(|| crate::run::internal("a candidate written has no JSON text"))?;
        questions.push((*position, named, question));
    }
    // `_candidates_to_rank`: the call's `top_k`, else the query's.
    let top_k = match given(&inputs, "top_k") {
        Some(value) => Some(sdm_request::options::top_k(value)?),
        None => plan
            .top_k
            .as_deref()
            .map(|digits| digits.parse::<u32>().unwrap_or(u32::MAX)),
    };
    // Each question gathered with the call's values but `top_k`, which is
    // the whole ranking's, each by a materializer of its own, so with no
    // memo of another's reads; the materializer's own memo is left as it
    // was handed, as the client's per-part materializers leave it.
    let values = inputs.values.clone().map(|mut values| {
        values.remove("top_k");
        values
    });
    let handed = inputs.memo.clone().unwrap_or_else(|| SamplerMemo {
        source: inputs.source.clone(),
        ..SamplerMemo::default()
    });
    let parts = u64::try_from(questions.len()).unwrap_or(u64::MAX);
    let mut gathered = Vec::with_capacity(questions.len());
    for (position, named, question) in questions {
        let mut asked = inputs.clone();
        asked.query = question;
        asked.memo = None;
        asked.candidates = None;
        asked.values.clone_from(&values);
        let tagged = |e: CoreError| {
            e.with_field(
                "ranking_candidate",
                Detail::Scalar(Scalar::UInt32(position)),
            )
            .with_field("ranking_parts", Detail::Scalar(Scalar::UInt64(parts)))
        };
        let done = one(&asker, asked).await;
        asker.keep(handed.clone());
        match done.map_err(tagged)? {
            Gathered::Done(result) => return Ok(result),
            Gathered::Request(payload, query) => {
                gathered.push((position, named, payload, query));
            }
        }
    }
    let mut attached = Vec::with_capacity(gathered.len());
    for (position, named, payload, query) in gathered {
        let (payload, query) = attach_request(&asker, payload, &query)?;
        attached.push(RankingPart {
            candidate: position,
            value: named,
            payload,
            query,
            request_id: None,
        });
    }
    let first = attached.remove(0);
    Ok(OperationResult::Materialize(Materialized {
        state: RequestState {
            version: REQUEST_STATE_VERSION,
            payload: first.payload,
            query: first.query,
            forecast: None,
            batches: None,
            ranking: Some(Ranking {
                top_k,
                value: first.value,
                parts: attached,
            }),
        },
        query: facts,
    }))
}

/// The outputs a ranking gives back, and what each question is asked for:
/// how sure each answer was is what orders the candidates.
const RANKING_OUTPUTS: [&str; 2] = ["prediction", "rankings"];
const RANKING_PARTS: [&str; 2] = ["prediction", "probabilities"];

/// The notice of the candidates a ranking leaves out, each context sampled
/// for it holding one answer only, which a host words naming them.
pub const CANDIDATES_UNSCORED: &str = "CANDIDATES_UNSCORED";

/// A ranking's question as `predict` asks it: the candidate, by its
/// position among those given and as the result names it, and the
/// question's payload, query and the id its call carries.
pub(crate) type Question = (u32, String, Vec<u8>, String, String);

/// A ranking predicted (`_answer_in_parts` for a ranking): `questions` in
/// the order asked, the first the state's own.
pub(crate) async fn predict(
    asker: Asker,
    inputs: crate::predict::Inputs,
    questions: Vec<Question>,
    top_k: Option<u32>,
) -> Result<OperationResult, CoreError> {
    use sdm_request::predict::{body, preflight, prepare};

    // `_refuse_outputs_a_ranking_cannot_give`.
    if !inputs.default_outputs {
        if inputs.outputs.is_empty() {
            return Err(refused(
                "outputs must name at least one field",
                "outputs",
                "ranking_outputs",
            ));
        }
        let unknown: Vec<&String> = inputs
            .outputs
            .iter()
            .filter(|name| !RANKING_OUTPUTS.contains(&name.as_str()))
            .collect();
        if !unknown.is_empty() {
            return Err(refused(
                "a ranking gives back its prediction and rankings, and an output named is \
                 neither",
                "outputs",
                "ranking_outputs",
            )
            .with_field(
                "unsupported",
                Detail::List(unknown.iter().map(|n| Scalar::Utf8((*n).clone())).collect()),
            ));
        }
    }
    let parts = u64::try_from(questions.len()).unwrap_or(u64::MAX);
    let tagged = |position: u32| {
        move |e: CoreError| {
            e.with_field(
                "ranking_candidate",
                Detail::Scalar(Scalar::UInt32(position)),
            )
            .with_field("ranking_parts", Detail::Scalar(Scalar::UInt64(parts)))
        }
    };
    let asked: Vec<String> = RANKING_PARTS.iter().map(|f| (*f).to_owned()).collect();
    // `_candidates_no_context_teaches`: every question checked as the
    // client prepares and plans it before any is sent; one whose context
    // holds one answer only is left out, and any other refusal ends the
    // ranking, said of its candidate.
    let mut cold = Vec::new();
    for (position, _, payload, query, request_id) in &questions {
        let checked = (|| -> Result<(), CoreError> {
            let payload = crate::predict::payload(payload)?;
            let outputs = prepare(&payload, &inputs.model, &asked, false)?;
            let payloads = [payload];
            preflight(&payloads, &outputs, query, &inputs.model)?;
            body(
                &payloads[0],
                &inputs.model,
                request_id,
                &outputs,
                &inputs.inference_asked,
            )?;
            Ok(())
        })();
        if let Err(error) = checked {
            let one_class = matches!(
                error.fields.get("classes_in_context"),
                Some(Detail::Scalar(Scalar::UInt64(1)))
            );
            if !one_class {
                return Err(tagged(*position)(error));
            }
            cold.push(*position);
        }
    }
    let positions = |at: &[u32]| Detail::List(at.iter().map(|p| Scalar::UInt32(*p)).collect());
    if cold.len() == questions.len() {
        return Err(CoreError::new(
            ErrorKind::Unsupported,
            ErrorCode::UnsupportedRequest,
            "the context gathered for every candidate holds one answer only, so there is \
             nothing to rank and nothing was sent",
        )
        .with_field("model", Detail::text(&inputs.model))
        .with_field("rule", Detail::text("ranking_unscored"))
        .with_field("candidates", positions(&cold))
        .with_field("classes_in_context", count(1)));
    }
    if !cold.is_empty() {
        asker.warn(sdm_contract::CoreWarning {
            code: CANDIDATES_UNSCORED.to_owned(),
            message: "candidates are left out of the ranking, unscored".to_owned(),
            fields: [("candidates".to_owned(), positions(&cold))].into(),
        });
    }
    // Each question left asked as a classification, in order, a refusal
    // said of its candidate, with how many were answered before it.
    let mut answers = Vec::with_capacity(questions.len() - cold.len());
    for (position, named, payload, query, request_id) in questions {
        if cold.contains(&position) {
            continue;
        }
        let asked_part = crate::predict::Inputs {
            payload,
            query,
            request_id,
            outputs: asked.clone(),
            default_outputs: false,
            forecast: None,
            ranking: None,
            ..inputs.clone()
        };
        let answered = answers.len();
        let (part, _) = crate::predict::one(&asker, &asked_part)
            .await
            .map_err(|e| {
                let e = tagged(position)(e);
                if answered == 0 {
                    e
                } else {
                    e.with_field("parts_answered", count(answered))
                }
            })?;
        answers.push((position, named, part));
    }
    let rows = ranking::rank(&answers, top_k)?;
    let (_, _, first) = &answers[0];
    let laid = ranking::table(&rows, first.text_ids)?;
    let result = asker.attach(crate::run::Attachment::Arrow(crate::run::ArrowData::new(
        laid.schema(),
        vec![laid],
    )))?;
    // The ranking's result names no metadata, as the client's does not.
    let metadata = asker.attach(crate::run::Attachment::Bytes(b"{}".to_vec()))?;
    let outputs = if inputs.default_outputs {
        RANKING_OUTPUTS.iter().map(|f| (*f).to_owned()).collect()
    } else {
        inputs.outputs.clone()
    };
    Ok(OperationResult::Predict(sdm_contract::ops::Predicted {
        result,
        model: first.model.clone(),
        request_id: first.request_id.clone(),
        metadata,
        outputs,
        unscored_candidates: cold,
    }))
}
