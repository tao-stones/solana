use {
    criterion::{black_box, criterion_group, criterion_main, Criterion},
    crossbeam_channel::{unbounded, Receiver, Sender},
    jemallocator::Jemalloc,
    solana_core::banking_stage::{
        scheduler_messages::{ConsumeWork, FinishedConsumeWork, MaxAge},
        transaction_scheduler::{
            greedy_scheduler::{GreedyScheduler, GreedySchedulerConfig},
            prio_graph_scheduler::{PrioGraphScheduler, PrioGraphSchedulerConfig},
            scheduler::Scheduler,
            transaction_state::SanitizedTransactionTTL,
            transaction_state_container::{StateContainer, TransactionStateContainer},
        },
        TOTAL_BUFFERED_PACKETS,
    },
    solana_runtime_transaction::{
        runtime_transaction::RuntimeTransaction, transaction_with_meta::TransactionWithMeta,
    },
    solana_sdk::{
        compute_budget::ComputeBudgetInstruction,
        hash::Hash,
        message::Message,
        pubkey::Pubkey,
        signature::Keypair,
        signer::Signer,
        system_instruction,
        transaction::{SanitizedTransaction, Transaction},
    },
    std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

// A non-contend low-prio tx, aka Tracer, is tag with this requested_loaded_accounts_data_size_limit
const TAG_NUMBER: u32 = 1234;

fn is_tracer<Tx: TransactionWithMeta + Send + Sync + 'static>(tx: &Tx) -> bool {
    matches!(
        tx.compute_budget_instruction_details()
            .requested_loaded_accounts_data_size_limit(),
        Some(TAG_NUMBER)
    )
}

// TODO - the goal is to measure and compare performance between different Schedulers,
// and observe the behavior/pattern of Scheduler;
// - performance: time, througput
// - behavior:
//   - how many time it loops container;
//   - when non-contend low-pri is picked;
//   - behavior/perform if block limits reached
//
// In order to do that, we need:
// - dummy worker that quickly drain channel to minimize pressure that can potentially impact
// Scheduler `send` works
// - identically prefilled container for each benck loops.

// TODO - transaction factory, to build container scenarios
// - contending / competing TX with non-contend low prio tx at bottom
// - prio distribution doesn't matter since "insert" to container will sort them
fn build_non_contend_transactions(count: usize) -> Vec<RuntimeTransaction<SanitizedTransaction>> {
    let mut transactions = Vec::with_capacity(count);
    // non-contend low-prio tx is first received
    transactions.push(build_tracer_transaction());

    let compute_unit_price = 1_000;
    const MAX_TRANSFERS_PER_TX: usize = 58;

    for _n in 1..count {
        let payer = Keypair::new();
        let to_pubkey = Pubkey::new_unique();
        let mut ixs = system_instruction::transfer_many(
            &payer.pubkey(),
            &vec![(to_pubkey, 1); MAX_TRANSFERS_PER_TX],
        );
        let prioritization = ComputeBudgetInstruction::set_compute_unit_price(compute_unit_price);
        ixs.push(prioritization);
        let message = Message::new(&ixs, Some(&payer.pubkey()));
        let tx = Transaction::new(&[payer], message, Hash::default());
        let transaction = RuntimeTransaction::from_transaction_for_tests(tx);

        transactions.push(transaction);
    }

    transactions
}

fn build_fully_contend_transactions(count: usize) -> Vec<RuntimeTransaction<SanitizedTransaction>> {
    let mut transactions = Vec::with_capacity(count);
    // non-contend low-prio tx is first received
    transactions.push(build_tracer_transaction());

    let compute_unit_price = 1_000;
    const MAX_TRANSFERS_PER_TX: usize = 58;

    let to_pubkey = Pubkey::new_unique();
    for _n in 1..count {
        let payer = Keypair::new();
        let mut ixs = system_instruction::transfer_many(
            &payer.pubkey().clone(),
            &vec![(to_pubkey, 1); MAX_TRANSFERS_PER_TX],
        );
        let prioritization = ComputeBudgetInstruction::set_compute_unit_price(compute_unit_price);
        ixs.push(prioritization);
        let message = Message::new(&ixs, Some(&payer.pubkey()));
        let tx = Transaction::new(&[payer], message, Hash::default());
        let transaction = RuntimeTransaction::from_transaction_for_tests(tx);

        transactions.push(transaction);
    }

    transactions
}

// Tracer is a non-contend low-prio transfer transaction, it'd usually be inserted into the bottom
// of Container ddue to its low prio, but it should be scheduled early since it is non-contend for
// better UX.
fn build_tracer_transaction() -> RuntimeTransaction<SanitizedTransaction> {
    let payer = Keypair::new();
    let to_pubkey = Pubkey::new_unique();
    let mut ixs = vec![system_instruction::transfer(&payer.pubkey(), &to_pubkey, 1)];
    ixs.push(ComputeBudgetInstruction::set_compute_unit_price(4));
    ixs.push(ComputeBudgetInstruction::set_loaded_accounts_data_size_limit(TAG_NUMBER));
    let message = Message::new(&ixs, Some(&payer.pubkey()));
    let tx = Transaction::new(&[payer], message, Hash::default());
    RuntimeTransaction::from_transaction_for_tests(tx)
}

struct BenchContainer<Tx: TransactionWithMeta> {
    container: TransactionStateContainer<Tx>,
}

impl<Tx: TransactionWithMeta> BenchContainer<Tx> {
    fn new(capacity: usize) -> Self {
        Self {
            container: TransactionStateContainer::with_capacity(capacity),
        }
    }

    fn fill_container(&mut self, transactions: impl Iterator<Item = Tx>) {
        for transaction in transactions {
            let compute_unit_price = transaction
                .compute_budget_instruction_details()
                .sanitize_and_convert_to_compute_budget_limits(
                    &solana_feature_set::FeatureSet::default(),
                )
                .unwrap()
                .compute_unit_price;

            let transaction_ttl = SanitizedTransactionTTL {
                transaction,
                max_age: MaxAge::MAX,
            };
            // NOTE - setting transaction cost to be `0` for now, so it doesn't bother block_limits
            // when scheduling.
            const TEST_TRANSACTION_COST: u64 = 0;
            if self.container.insert_new_transaction(
                transaction_ttl,
                compute_unit_price,
                TEST_TRANSACTION_COST,
            ) {
                unreachable!("test is setup to fill the Container to fullness");
            }
        }
    }
}

#[derive(Debug, Default)]
struct BenchStats {
    bench_iter_count: usize,
    num_of_scheduling: usize,
    // worker reports:
    num_works: Arc<AtomicUsize>,
    num_transaction: Arc<AtomicUsize>, // = bench_iter_count * container_capacity
    tracer_placement: Arc<AtomicUsize>, // > 0
    // from scheduler().result:
    num_scheduled: usize, // = num_transaction
}

impl BenchStats {
    fn print_and_reset(&mut self) {
        println!("{:?}", self);
        self.num_works.swap(0, Ordering::Relaxed);
        self.num_transaction.swap(0, Ordering::Relaxed);
        self.tracer_placement.swap(0, Ordering::Relaxed);
        self.bench_iter_count = 0;
        self.num_of_scheduling = 0;
        self.num_scheduled = 0;
    }
}

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
        num_works: Arc<AtomicUsize>,
        num_transaction: Arc<AtomicUsize>,
        tracer_placement: Arc<AtomicUsize>,
    ) -> Self {
        let mut threads = Vec::with_capacity(work_receivers.len());

        for receiver in work_receivers {
            let completed_work_sender_clone = completed_work_sender.clone();
            let num_works_clone = num_works.clone();
            let num_transaction_clone = num_transaction.clone();
            let tracer_placement_clone = tracer_placement.clone();

            let handle = std::thread::spawn(move || {
                Self::service_loop(
                    receiver,
                    completed_work_sender_clone,
                    num_works_clone,
                    num_transaction_clone,
                    tracer_placement_clone,
                );
            });
            threads.push(handle);
        }

        Self { threads }
    }

    fn service_loop<Tx: TransactionWithMeta + Send + Sync + 'static>(
        work_receiver: Receiver<ConsumeWork<Tx>>,
        completed_work_sender: Sender<FinishedConsumeWork<Tx>>,
        num_works: Arc<AtomicUsize>,
        num_transaction: Arc<AtomicUsize>,
        tracer_placement: Arc<AtomicUsize>,
    ) {
        // NOTE: will blocking recv() impact benchmark quality? Perhaps making worker threads
        // hot spinning?
        while let Ok(work) = work_receiver.recv() {
            num_works.fetch_add(1, Ordering::Relaxed);
            let mut tx_count =
                num_transaction.fetch_add(work.transactions.len(), Ordering::Relaxed);
            if tracer_placement.load(Ordering::Relaxed) == 0 {
                work.transactions.iter().for_each(|tx| {
                    tx_count += 1;
                    if is_tracer(tx) {
                        tracer_placement.store(tx_count, Ordering::Relaxed)
                    }
                });
            }

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
    pingpong_worker: PingPong,
    filter_1: fn(&[&Tx], &mut [bool]),
    filter_2: fn(&Tx) -> bool,
    consume_work_senders: Vec<Sender<ConsumeWork<Tx>>>,
    finished_consume_work_receiver: Receiver<FinishedConsumeWork<Tx>>,
}

impl<Tx: TransactionWithMeta + Send + Sync + 'static> BenchEnv<Tx> {
    fn new(stats: &mut BenchStats) -> Self {
        let num_workers = 4;

        let (consume_work_senders, consume_work_receivers) =
            (0..num_workers).map(|_| unbounded()).unzip();
        let (finished_consume_work_sender, finished_consume_work_receiver) = unbounded();
        let pingpong_worker = PingPong::new(
            consume_work_receivers,
            finished_consume_work_sender,
            stats.num_works.clone(),
            stats.num_transaction.clone(),
            stats.tracer_placement.clone(),
        );

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

    fn test_pre_lock_filter(_tx: &Tx) -> bool {
        true
    }

    fn run(
        &self,
        mut scheduler: impl Scheduler<Tx>,
        mut container: TransactionStateContainer<Tx>,
        stats: &mut BenchStats,
    ) {
        // each bench measurement is to schedule everything in the container
        while !container.is_empty() {
            let result = scheduler
                .schedule(&mut container, self.filter_1, self.filter_2)
                .unwrap();

            // do some VERY QUICK stats collecting to print/assert at end of bench
            stats.num_of_scheduling += 1;
            stats.num_scheduled += result.num_scheduled;
        }

        stats.bench_iter_count += 1;
    }
}

fn bench_empty_container(c: &mut Criterion) {
    let mut stats = BenchStats::default();
    let bench_env: BenchEnv<RuntimeTransaction<SanitizedTransaction>> = BenchEnv::new(&mut stats);

    c.benchmark_group("bench_empty_container")
        .bench_function("sdk_transaction_type", |bencher| {
            bencher.iter_with_setup(
                || {
                    let bench_container = BenchContainer::new(0);
                    let scheduler = PrioGraphScheduler::new(
                        bench_env.consume_work_senders.clone(),
                        bench_env.finished_consume_work_receiver.clone(),
                        PrioGraphSchedulerConfig::default(),
                    );
                    (scheduler, bench_container.container)
                },
                |(scheduler, container)| {
                    black_box(bench_env.run(scheduler, container, &mut stats));
                    //stats.print_and_reset();
                },
            )
        });
    stats.print_and_reset();
}

fn bench_prio_graph_non_contend_transactions(c: &mut Criterion) {
    let capacity = TOTAL_BUFFERED_PACKETS;
    let mut stats = BenchStats::default();
    let bench_env: BenchEnv<RuntimeTransaction<SanitizedTransaction>> = BenchEnv::new(&mut stats);

    c.benchmark_group("bench_prio_graph_non_contend_transactions")
        .sample_size(10)
        .bench_function("sdk_transaction_type", |bencher| {
            bencher.iter_with_setup(
                || {
                    let mut bench_container = BenchContainer::new(capacity);
                    bench_container
                        .fill_container(build_non_contend_transactions(capacity).into_iter());
                    let scheduler = PrioGraphScheduler::new(
                        bench_env.consume_work_senders.clone(),
                        bench_env.finished_consume_work_receiver.clone(),
                        PrioGraphSchedulerConfig::default(),
                    );
                    (scheduler, bench_container.container)
                },
                |(scheduler, container)| {
                    black_box(bench_env.run(scheduler, container, &mut stats));
                    //stats.print_and_reset();
                },
            )
        });

    stats.print_and_reset();
}

fn bench_prio_graph_fully_contend_transactions(c: &mut Criterion) {
    let capacity = TOTAL_BUFFERED_PACKETS;
    let mut stats = BenchStats::default();
    let bench_env: BenchEnv<RuntimeTransaction<SanitizedTransaction>> = BenchEnv::new(&mut stats);

    c.benchmark_group("bench_prio_graph_fully_contend_transactions")
        .sample_size(10)
        .bench_function("sdk_transaction_type", |bencher| {
            bencher.iter_with_setup(
                || {
                    let mut bench_container = BenchContainer::new(capacity);
                    bench_container
                        .fill_container(build_fully_contend_transactions(capacity).into_iter());
                    let scheduler = PrioGraphScheduler::new(
                        bench_env.consume_work_senders.clone(),
                        bench_env.finished_consume_work_receiver.clone(),
                        PrioGraphSchedulerConfig::default(),
                    );
                    (scheduler, bench_container.container)
                },
                |(scheduler, container)| {
                    black_box(bench_env.run(scheduler, container, &mut stats));
                    //stats.print_and_reset();
                },
            )
        });

    stats.print_and_reset();
}

fn bench_greedy_non_contend_transactions(c: &mut Criterion) {
    let capacity = TOTAL_BUFFERED_PACKETS;
    let mut stats = BenchStats::default();
    let bench_env: BenchEnv<RuntimeTransaction<SanitizedTransaction>> = BenchEnv::new(&mut stats);

    c.benchmark_group("bench_greedy_non_contend_transactions")
        .sample_size(10)
        .bench_function("sdk_transaction_type", |bencher| {
            bencher.iter_with_setup(
                || {
                    let mut bench_container = BenchContainer::new(capacity);
                    bench_container
                        .fill_container(build_non_contend_transactions(capacity).into_iter());
                    let scheduler = GreedyScheduler::new(
                        bench_env.consume_work_senders.clone(),
                        bench_env.finished_consume_work_receiver.clone(),
                        GreedySchedulerConfig::default(),
                    );
                    (scheduler, bench_container.container)
                },
                |(scheduler, container)| {
                    black_box(bench_env.run(scheduler, container, &mut stats));
                    //stats.print_and_reset();
                },
            )
        });

    stats.print_and_reset();
}

fn bench_greedy_fully_contend_transactions(c: &mut Criterion) {
    let capacity = TOTAL_BUFFERED_PACKETS;
    let mut stats = BenchStats::default();
    let bench_env: BenchEnv<RuntimeTransaction<SanitizedTransaction>> = BenchEnv::new(&mut stats);

    c.benchmark_group("bench_greedy_fully_contend_transactions")
        .sample_size(10)
        .bench_function("sdk_transaction_type", |bencher| {
            bencher.iter_with_setup(
                || {
                    let mut bench_container = BenchContainer::new(capacity);
                    bench_container
                        .fill_container(build_fully_contend_transactions(capacity).into_iter());
                    let scheduler = GreedyScheduler::new(
                        bench_env.consume_work_senders.clone(),
                        bench_env.finished_consume_work_receiver.clone(),
                        GreedySchedulerConfig::default(),
                    );
                    (scheduler, bench_container.container)
                },
                |(scheduler, container)| {
                    black_box(bench_env.run(scheduler, container, &mut stats));
                    //stats.print_and_reset();
                },
            )
        });

    stats.print_and_reset();
}

criterion_group!(
    benches,
    bench_empty_container,
    bench_prio_graph_non_contend_transactions,
    bench_prio_graph_fully_contend_transactions,
    bench_greedy_non_contend_transactions,
    bench_greedy_fully_contend_transactions,
);
criterion_main!(benches);
