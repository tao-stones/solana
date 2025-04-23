#[path = "receive_and_buffer_utils.rs"]
mod utils;
use {
    criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput},
    crossbeam_channel::{unbounded, Receiver, Sender},
    jemallocator::Jemalloc,
    solana_core::banking_stage::{
        scheduler_messages::{ConsumeWork, FinishedConsumeWork},
        transaction_scheduler::{
            greedy_scheduler::{GreedyScheduler, GreedySchedulerConfig},
            prio_graph_scheduler::{PrioGraphScheduler, PrioGraphSchedulerConfig},
            receive_and_buffer::{
                ReceiveAndBuffer, SanitizedTransactionReceiveAndBuffer,
                TransactionViewReceiveAndBuffer,
            },
            scheduler::{PreLockFilterAction, Scheduler},
            scheduler_metrics::{SchedulerCountMetrics, SchedulerTimingMetrics},
            transaction_state::TransactionState,
            transaction_state_container::StateContainer,
        },
    },
    solana_runtime_transaction::transaction_with_meta::TransactionWithMeta,
    std::time::{Duration, Instant},
};

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

// a bench consumer worker that quickly drain work channel, then send a OK back via completed-work
// channel
// NOTE: Avoid creating PingPong within bench iter since joining threads at its eol would
// introducing variance to bench timing.
#[allow(dead_code)]
struct PingPong {
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl PingPong {
    fn new<Tx: TransactionWithMeta + Send + Sync + 'static>(
        work_receivers: Vec<Receiver<ConsumeWork<Tx>>>,
        completed_work_sender: Sender<FinishedConsumeWork<Tx>>,
    ) -> Self {
        let mut threads = Vec::with_capacity(work_receivers.len());

        for receiver in work_receivers {
            let completed_work_sender_clone = completed_work_sender.clone();

            let handle = std::thread::spawn(move || {
                Self::service_loop(receiver, completed_work_sender_clone);
            });
            threads.push(handle);
        }

        Self { threads }
    }

    fn service_loop<Tx: TransactionWithMeta + Send + Sync + 'static>(
        work_receiver: Receiver<ConsumeWork<Tx>>,
        completed_work_sender: Sender<FinishedConsumeWork<Tx>>,
    ) {
        while let Ok(work) = work_receiver.recv() {
            if completed_work_sender
                .send(FinishedConsumeWork {
                    work,
                    retryable_indexes: vec![],
                })
                .is_err()
            {
                // kill this worker if finished_work channel is broken
                break;
            }
        }
    }
}

struct BenchEnv<Tx: TransactionWithMeta + Send + Sync + 'static> {
    #[allow(dead_code)]
    pingpong_worker: PingPong,
    filter_1: fn(&[&Tx], &mut [bool]),
    filter_2: fn(&TransactionState<Tx>) -> PreLockFilterAction,
    consume_work_senders: Vec<Sender<ConsumeWork<Tx>>>,
    finished_consume_work_receiver: Receiver<FinishedConsumeWork<Tx>>,
}

impl<Tx: TransactionWithMeta + Send + Sync + 'static> BenchEnv<Tx> {
    fn new() -> Self {
        let num_workers = 4;

        let (consume_work_senders, consume_work_receivers) =
            (0..num_workers).map(|_| unbounded()).unzip();
        let (finished_consume_work_sender, finished_consume_work_receiver) = unbounded();
        let pingpong_worker = PingPong::new(consume_work_receivers, finished_consume_work_sender);

        Self {
            pingpong_worker,
            filter_1: Self::test_pre_graph_filter,
            filter_2: Self::test_pre_lock_filter,
            consume_work_senders,
            finished_consume_work_receiver,
        }
    }

    fn test_pre_graph_filter(_txs: &[&Tx], results: &mut [bool]) {
        results.fill(true);
    }

    fn test_pre_lock_filter(_tx: &TransactionState<Tx>) -> PreLockFilterAction {
        PreLockFilterAction::AttemptToSchedule
    }
}

// to support both schedulers
macro_rules! timing_scheduler {
    ($scheduler:expr, $container:expr, $filter_1:expr, $filter_2:expr) => {{
        let start = Instant::now();
        {
            while !$container.is_empty() {
                $scheduler
                    .receive_completed(black_box(&mut $container))
                    .unwrap();

                $scheduler
                    .schedule(black_box(&mut $container), $filter_1, $filter_2)
                    .unwrap();
            }
        }
        start.elapsed()
    }};
}

fn bench_scheduler_impl<T: ReceiveAndBuffer + utils::ReceiveAndBufferCreator>(
    c: &mut Criterion,
    bench_name: &str,
) where
    <T as ReceiveAndBuffer>::Transaction: 'static,
{
    let mut group = c.benchmark_group("bench_scheduler");
    group.sample_size(10);

    let scheduler_types: Vec<(bool, &str)> =
        vec![(true, "greedy_scheduler"), (false, "prio_graph_scheduler")];
    //solana_core::banking_stage::TOTAL_BUFFERED_PACKETS took too long
    let tx_counts: Vec<(usize, &str)> = vec![(16 * 1024, "16K_txs")];
    let ix_counts: Vec<(usize, &str)> = vec![
        (1, "single_ix"),
        (utils::MAX_INSTRUCTIONS_PER_TRANSACTION, "max_ixs"),
    ];
    // TODO - add tx accounts conflict config

    for (is_greedy_scheduler, scheduler_desc) in scheduler_types {
        for (ix_count, ix_count_desc) in &ix_counts {
            for (tx_count, tx_count_desc) in &tx_counts {
                let bench_name =
                    format!("{bench_name}/{scheduler_desc}/{ix_count_desc}/{tx_count_desc}");
                group.throughput(Throughput::Elements(*tx_count as u64));
                group.bench_function(&bench_name, |bencher| {
                    bencher.iter_custom(|iters| {
                        let utils::ReceiveAndBufferSetup {
                            txs,
                            sender,
                            mut container,
                            mut receive_and_buffer,
                            decision,
                        }: utils::ReceiveAndBufferSetup<T> =
                            utils::setup_receive_and_buffer(*tx_count, *ix_count, 0.0, true);

                        let mut execute_time: Duration = std::time::Duration::ZERO;
                        for _i in 0..iters {
                            // setup new Scheduler and reset/refill Container for each iteration of execution
                            if sender.send(txs.clone()).is_err() {
                                panic!("Unexpectedly dropped receiver!");
                            }
                            container.clear();
                            let mut count_metrics = SchedulerCountMetrics::default();
                            let mut timing_metrics = SchedulerTimingMetrics::default();
                            let res = receive_and_buffer.receive_and_buffer_packets(
                                &mut container,
                                &mut timing_metrics,
                                &mut count_metrics,
                                &decision,
                            );
                            assert!(res.unwrap() == *tx_count && !container.is_empty());

                            let bench_env: BenchEnv<T::Transaction> = BenchEnv::new();
                            let elapsed = if is_greedy_scheduler {
                                let scheduler = GreedyScheduler::new(
                                    bench_env.consume_work_senders.clone(),
                                    bench_env.finished_consume_work_receiver.clone(),
                                    GreedySchedulerConfig::default(),
                                );
                                let mut scheduler = black_box(scheduler);
                                timing_scheduler!(
                                    scheduler,
                                    container,
                                    bench_env.filter_1,
                                    bench_env.filter_2
                                )
                            } else {
                                let scheduler = PrioGraphScheduler::new(
                                    bench_env.consume_work_senders.clone(),
                                    bench_env.finished_consume_work_receiver.clone(),
                                    PrioGraphSchedulerConfig::default(),
                                );
                                let mut scheduler = black_box(scheduler);
                                timing_scheduler!(
                                    scheduler,
                                    container,
                                    bench_env.filter_1,
                                    bench_env.filter_2
                                )
                            };

                            execute_time = execute_time.saturating_add(elapsed);
                        }
                        execute_time
                    })
                });
            }
        }
    }
}

fn bench_greedy_scheduler(c: &mut Criterion) {
    bench_scheduler_impl::<SanitizedTransactionReceiveAndBuffer>(c, "sdk_transaction");
    bench_scheduler_impl::<TransactionViewReceiveAndBuffer>(c, "transaction_view");
}

criterion_group!(benches, bench_greedy_scheduler,);
criterion_main!(benches);
