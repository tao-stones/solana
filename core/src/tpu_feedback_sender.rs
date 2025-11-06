use {
    crossbeam_channel::{unbounded, Receiver, Sender},
    solana_perf::packet::BytesPacket,
    solana_runtime_transaction::transaction_with_meta::TransactionWithMeta,
    std::{
        collections::{hash_map::Entry, HashMap},
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
        thread::Builder,
    },
};

/// TPU Feedback with arbitraty score of transaction that went through packing process
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct TpuFeedback {
    // first 8-byte of transaction signature. Streamer caches the mapping between transaction id to
    // connection id. Streamer aggregates transaction score per connection, use that to QoS future
    // bandwidth
    transaction_id: u64,
    // for this experiment, the score is normalized between 0 and 1:
    // - `0` indicates no fault by the client, not punishable;
    // - `1` indicates maximum fault, such as invalid signature, are punishable;
    score: u8,
    // for https://github.com/anza-xyz/agave/issues/7853, feedback
    // includes the total fee a landed transaction pays;
    // Note, it is not the reward (which burns part of fee), nor the priority
    // (which considering total transction CU)
    // not-landed transction will have value `0`
    total_fee: u64,
}

/// Represents the lifecycle of a transaction as it moves through the TPU.
///
/// A transaction transitions through several states, with possible fall-out
/// errors at each stage. This enum defines all valid states and the
/// corresponding failure reasons.
///
/// State transitions:
///
/// * **Received** → Initial state after a transaction is accepted by the TPU.
///   - On success, transitions to **Queued**.
///   - On failure, exits the pipeline with an [`IngestingError`].
///
/// * **Queued** → Transaction has passed ingestion and is waiting to be scheduled.
///   - On success, transitions to **Scheduled**.
///   - On failure, exits the pipeline with a [`SchedulingError`].
///
/// * **Scheduled** → Transaction has been assigned for execution.
///   - On success, transitions to **Processed** (executed but not yet committed)
///     or **Committed** (successfully included in a block).
///   - On failure, exits the pipeline with a [`ProcessingError`].
///
/// This enum captures both the normal state progression and all
/// possible fall-out conditions encountered along the way.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) enum PackingEvent {
    Received,
    IngestingFailed(IngestingError),
    Queued,
    SchedulingFailed(SchedulingError),
    Scheduled,
    ProcessingFailed(ProcessingError),
    Processed,
    Committed(TotalFee),
}

type TotalFee = u64;

#[derive(Clone, Debug)]
pub(crate) enum IngestingError {
    // Sigverify stage marked packet "discarded" often due to malformatting, invalid signatures etc
    DiscardedBySigverify,

    // transaction sanization failure
    SanitizationFailure,

    // at early stage of banking, transaction is validated if has more than max account lock
    LockValidationFailure,

    // compute budget limits are parsed and validated at early stage of banking
    ComputeBudgetFailure,

    // fee payer account is checked
    FeePayerFailure,
}

#[derive(Clone, Debug)]
pub(crate) enum SchedulingError {
    // NOTE - at prototyping stage, scheduling errors are not supported; Can find ways to support
    // external schedulers. The failures with scheduling can be deducted from diff between "Queued"
    // and "scheduled".
}

#[derive(Clone, Debug)]
pub(crate) enum ProcessingError {
    // NOTE - at prototyping stage, transaction being processed is seem as good case, as clinet
    // sent valid transaction good for runtime. Processing errors are not considered as punishable
    // failures. In next steps, can consider TransactionError
    // (https://github.com/anza-xyz/solana-sdk/blob/master/transaction-error/src/lib.rs#L15)
    // some of them could be cause of punishment
}

impl PackingEvent {
    fn is_initial_state(&self) -> bool {
        matches!(self, PackingEvent::Received)
    }

    // transaction will be drop out of banking pipeline if error, so all errors here are considered
    // the final state, as well as processed and committed, to generate score and feedback to
    // streamer.
    fn is_final_state(&self) -> bool {
        matches!(
            self,
            PackingEvent::Processed
                | PackingEvent::Committed(_)
                | PackingEvent::IngestingFailed(_)
                | PackingEvent::SchedulingFailed(_)
                | PackingEvent::ProcessingFailed(_)
        )
    }

    /// Assigns an arbitrary score to each final packing event.
    ///
    /// This score provides a simple measurement for the Streamer to grade clients
    /// and potentially limit the bandwidth of future connections. This experimental
    /// version of the algorithm focuses primarily on punishable events.  
    ///
    /// The score is normalized between 0 and 1:
    /// - `0` indicates no fault by the client ("not your fail")  
    /// - `1` indicates maximum fault ("you are not good")
    ///
    // NOTE - if the idea of "score"' works, it can be implemented for public `trait` that
    // does:
    //   - validate score (should have a min/max range)
    //   - allows different parts (sigverify stage, banking stage, external scheduler etc) to
    //     clasify events in their own way.
    fn score(&self) -> u8 {
        if matches!(
            self,
            PackingEvent::IngestingFailed(_)
                | PackingEvent::SchedulingFailed(_)
                | PackingEvent::ProcessingFailed(_)
        ) {
            return 1;
        } else {
            return 0;
        }
    }

    /// return True if `self` can transit to new `event`
    fn is_updateble(&self, event: &PackingEvent) -> bool {
        match self {
            PackingEvent::Received
                if (matches!(event, PackingEvent::IngestingFailed(_))
                    || matches!(event, PackingEvent::Queued)) =>
            {
                true
            }
            PackingEvent::Queued
                if (matches!(event, PackingEvent::SchedulingFailed(_))
                    || matches!(event, PackingEvent::Scheduled)) =>
            {
                true
            }
            PackingEvent::Scheduled
                if (matches!(event, PackingEvent::ProcessingFailed(_))
                    || matches!(event, PackingEvent::Scheduled)  // tx can be sent to re-schedule
                    || matches!(event, PackingEvent::Processed)
                    || matches!(event, PackingEvent::Committed(_))) =>
            {
                true
            }
            PackingEvent::IngestingFailed(_)
            | PackingEvent::SchedulingFailed(_)
            | PackingEvent::ProcessingFailed(_)
            | PackingEvent::Processed
            | PackingEvent::Committed(_)
            | _ => false,
        }
    }

    // Committed event comes with total fee a landed transaction pays,
    // all other events have `0`
    fn total_fee(&self) -> TotalFee {
        match self {
            PackingEvent::Committed(total_fee) => *total_fee,
            _ => 0,
        }
    }
}

/// internal message TpuFeedbackSender sends from banking stage threads to backend service threads
/// for packing event processing.
#[derive(Debug)]
struct TransactionPackingEvent {
    // quick and cheap way of identifying a transaction, shared method with Streamer
    transaction_id: u64,
    // packing event that can be used to grade sender connection quality
    packing_event: PackingEvent,
}

/// TpuFeedbackSender: TPU feedback mechanism for transaction quality scoring.
///
/// The TPU receives client transactions and attempts to include them in blocks.
/// However, some clients may send malformed or low-quality transactions that are
/// unlikely to be included and can congest the pipeline.
///
/// `TpuFeedbackSender` aggregates transaction packing events, scores each
/// transaction's quality, and sends feedback scores back to the Streamer.
/// The Streamer can then apply these scores to perform QoS
/// adjustments for client connections in the future.
///
///
/// Lifespan: TPU creates crossbeam channel, give Receiver to streamer, use Sender to create
/// TpuFeedbackSender, then owns it. When TPU teardown, it drops TpuFeedbackSender, which in turn
/// drops its internal sender that cause spawned thread to exit; it also drops crossbeam Sender to
/// close channel to Streamer.
//
// NOTE - if this poc approved to yield effective benifits, we should consider replacing Channel
// with adn atomic score owned by Streamer, shared to various components that allows to submit
// score for transactions. Such components can implements their owner scoring system within bonds.
pub(crate) struct TpuFeedbackSender {
    // internal channel to send packing event to service thread for processing
    transaction_packing_event_sender: Sender<TransactionPackingEvent>,
}

impl TpuFeedbackSender {
    pub(crate) fn new(
        tpu_feedback_sender: Sender<TpuFeedback>,
        exit_signal: Arc<AtomicBool>,
    ) -> Self {
        // use `unbounded` to not to block sender sending from hot path
        let (transaction_packing_event_sender, transaction_packing_event_receiver) =
            unbounded::<TransactionPackingEvent>();

        Builder::new()
            .name("solTpuFeedback".to_string())
            .spawn(move || {
                Self::packing_event_receiving_service(
                    transaction_packing_event_receiver,
                    tpu_feedback_sender,
                    exit_signal,
                );
            })
            .unwrap();

        Self {
            transaction_packing_event_sender,
        }
    }

    // Public function - light weight
    /// TPU calls to report an packing event for packet
    // NOTE - this is a temp hack to fetch "transaction id" (first 8-bytes of signature) from raw bytes;
    //        Streamer will need something to identify transaction, it should be the same method
    //        here.
    pub(crate) fn on_packet_event(&self, packet: &BytesPacket, event: PackingEvent) {
        // NOTE - let unwrap fail hard during try-out
        // grab first 8-bytes of signature from packet bytes, first byte is number of signatures.
        // NOTE - this doesn't work well in txv1
        let transaction_id = u64::from_be_bytes(packet.buffer()[1..9].try_into().unwrap());

        let transaction_packing_event = TransactionPackingEvent {
            transaction_id: transaction_id,
            packing_event: event,
        };

        let _ = self
            .transaction_packing_event_sender
            .send(transaction_packing_event);
    }

    /// TPU calls to report an packing event for transaction
    pub(crate) fn on_transaction_event(&self, tx: &impl TransactionWithMeta, event: PackingEvent) {
        // TODO - can check if tx is 1) not-discarded, 2) Non-vote, 3) and staked TXs;
        //            because the feedback is assumed to only apply to tpu traffic

        // NOTE - let unwrap fail hard during try-out
        let transaction_id = u64::from_be_bytes(tx.signature().as_ref()[..8].try_into().unwrap());

        let transaction_packing_event = TransactionPackingEvent {
            transaction_id: transaction_id,
            packing_event: event,
        };

        let _ = self
            .transaction_packing_event_sender
            .send(transaction_packing_event);
    }

    /// the event receviing service - heavy lifting
    /// - receives transaction packing event
    /// - if (!in_cache && !event::Received), then log and drop
    /// - advance state
    /// - if failed to advance state (something wrong), log and call self.OnFinal() to send event back
    ///   to streamer;
    /// - if advanced state is final state, call self.OnFinal()
    /// - else, cache is updated
    ///
    fn packing_event_receiving_service(
        transaction_packing_event_receiver: Receiver<TransactionPackingEvent>,
        tpu_feedback_sender: Sender<TpuFeedback>,
        exit_signal: Arc<AtomicBool>,
    ) {
        let mut transaction_packing_event_aggregator = TransactionPackingEventAggregator::default();

        // blocking receiving, break at disconected
        for tx_packing_event in transaction_packing_event_receiver.iter() {
            // if received packing event results a feedback, send it to Streamer
            if let Some(tpu_feedback) =
                transaction_packing_event_aggregator.on_event(tx_packing_event)
            {
                let result = tpu_feedback_sender.send(tpu_feedback.clone());
                println!("TpuFeedback: send {:?}, result: {:?}", tpu_feedback, result);
            }

            if exit_signal.load(Ordering::Relaxed) {
                break;
            }
        }
    }
}

/// A kitch sink for now, it:
/// - tracks transaction's packing event
/// - scores tx when reaches finale event (landed, dropped, errored out)
#[derive(Default)]
struct TransactionPackingEventAggregator {
    // `cache` stores stats for transactions actively being packed, eg transactions were received
    // from sigverify, and haven't reached its finale.
    // TODO - maybe still safer to limit cache size to avoid eating up memory in high TPS env
    cache: HashMap<u64, PackingEvent>,
}

impl TransactionPackingEventAggregator {
    pub fn on_event(&mut self, tx_packing_event: TransactionPackingEvent) -> Option<TpuFeedback> {
        let TransactionPackingEvent {
            transaction_id,
            ref packing_event,
        } = tx_packing_event;

        let updated = match self.cache.entry(transaction_id) {
            Entry::Vacant(entry) => {
                if packing_event.is_initial_state() {
                    entry.insert(packing_event.clone());
                    true
                } else {
                    println!(
                        "TpuFeedback: failed to initiate tx state: {:?}",
                        tx_packing_event
                    );
                    false
                }
            }
            Entry::Occupied(mut entry) => {
                if entry.get().is_updateble(&packing_event) {
                    entry.insert(packing_event.clone());
                    true
                } else {
                    println!(
                        "TpuFeedback: failed to transit from {:?} to {:?}",
                        entry.get(),
                        tx_packing_event
                    );
                    false
                }
            }
        };

        if updated {
            println!("TpuFeedback: updated {:?}", tx_packing_event);
        }

        if updated && packing_event.is_final_state() {
            self.cache.remove(&transaction_id);
            Some(TpuFeedback {
                transaction_id: transaction_id,
                score: packing_event.score(),
                total_fee: packing_event.total_fee(),
            })
        } else {
            None
        }
    }
}
