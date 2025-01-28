use {
    criterion::{black_box, criterion_group, criterion_main, Criterion},
    crossbeam_channel::{unbounded, Receiver, Sender},
    jemallocator::Jemalloc,
    solana_core::banking_stage::{
        TOTAL_BUFFERED_PACKETS,
        immutable_deserialized_packet::ImmutableDeserializedPacket,
        scheduler_messages::{ConsumeWork, FinishedConsumeWork, MaxAge},
        transaction_scheduler::{
            prio_graph_scheduler::{PrioGraphScheduler, PrioGraphSchedulerConfig},
            transaction_state::SanitizedTransactionTTL,
            transaction_state_container::{StateContainer, TransactionStateContainer},
        },
    },
    solana_runtime_transaction::{
        runtime_transaction::RuntimeTransaction, transaction_with_meta::TransactionWithMeta,
    },
    solana_sdk::{
        compute_budget::ComputeBudgetInstruction,
        hash::Hash,
        message::Message,
        packet::Packet,
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

// TODO - the goal is to measure the performance, and observe the behavior/pattern, of Scheduler;
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

println!("prep non-contend txs: {:?}", transactions.len());

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

println!("prep full-contend txs: {:?}", transactions.len());

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
        let mut n: usize = 0;
        for transaction in transactions {
            let compute_unit_price = transaction
                .compute_budget_instruction_details()
                .sanitize_and_convert_to_compute_budget_limits(
                    &solana_feature_set::FeatureSet::default(),
                )
                .unwrap()
                .compute_unit_price;

            let packet = Arc::new(
                ImmutableDeserializedPacket::new(
                    Packet::from_data(None, transaction.to_versioned_transaction()).unwrap(),
                )
                .unwrap(),
            );
            let transaction_ttl = SanitizedTransactionTTL {
                transaction,
                max_age: MaxAge::MAX,
            };
            // NOTE - setting transaction cost to be `0` for now, so it doesn't bother block_limits
            // when scheduling.
            const TEST_TRANSACTION_COST: u64 = 0;
            if self.container.insert_new_transaction(
                transaction_ttl,
                packet,
                compute_unit_price,
                TEST_TRANSACTION_COST,
            ) {
                assert!(false);
                println!("fail fill container: remaining cap {:?}, nth {:?}", self.container.remaining_capacity(), n);
            }
            n += 1;
        }
        println!("==== inserted {} transactions to container ====", n);
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
    num_scheduled: usize,  // = num_transaction
}

// a bench consumer worker that quickly drain work channel, then send a OK back via completed-work
// channel
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
                        println!("==== tracer found! {:?}, {:?}, {:?}", num_works.load(Ordering::Relaxed), tx_count, num_transaction.load(Ordering::Relaxed));
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

    fn join(self) {
        for thread in self.threads {
            thread.join().unwrap();
        }
    }
}

// setup Scheduler with bench accessories: pingpong worker, filters and status
struct BenchSetup<Tx: TransactionWithMeta + Send + Sync + 'static> {
    scheduler: PrioGraphScheduler<Tx>,
    pingpong_worker: PingPong,
    stats: BenchStats,
    filter_1: fn(&[&Tx], &mut [bool]),
    filter_2: fn(&Tx) -> bool,
}

impl<Tx: TransactionWithMeta + Send + Sync + 'static> BenchSetup<Tx> {
    fn new() -> Self {
        let stats = BenchStats::default();

        let num_workers = 4;

        let (consume_work_senders, consume_work_receivers) =
            (0..num_workers).map(|_| unbounded()).unzip();
        let (finished_consume_work_sender, finished_consume_work_receiver) = unbounded();
        let scheduler = PrioGraphScheduler::new(
            consume_work_senders,
            finished_consume_work_receiver,
            PrioGraphSchedulerConfig::default(),
        );
        let pingpong_worker = PingPong::new(
            consume_work_receivers,
            finished_consume_work_sender,
            stats.num_works.clone(),
            stats.num_transaction.clone(),
            stats.tracer_placement.clone(),
        );

        Self {
            scheduler,
            pingpong_worker,
            stats,
            filter_1: Self::test_pre_graph_filter,
            filter_2: Self::test_pre_lock_filter,
        }
    }

    fn test_pre_graph_filter(_txs: &[&Tx], results: &mut [bool]) {
        results.fill(true);
    }

    fn test_pre_lock_filter(_tx: &Tx) -> bool {
        true
    }

    fn run(&mut self, mut container: TransactionStateContainer<Tx>) {
        // each bench measurement is to schedule everything in the container
        while !container.is_empty() {
            let result = self
                .scheduler
                .schedule(&mut container, self.filter_1, self.filter_2)
                .unwrap();

            // do some VERY QUICK stats collecting to print/assert at end of bench
            self.stats.num_of_scheduling += 1;
            self.stats.num_scheduled += result.num_scheduled;
        }

        self.stats.bench_iter_count += 1;
    }

    fn print_stats(self) {
        drop(self.scheduler);
        self.pingpong_worker.join();
        println!("{:?}", self.stats);
    }
}

fn bench_empty_container(c: &mut Criterion) {
    let mut bench_setup: BenchSetup<RuntimeTransaction<SanitizedTransaction>> =
        BenchSetup::new();

    c.benchmark_group("bench_empty_container")
        .bench_function("sdk_transaction_type", |bencher| {
            bencher.iter_with_setup(
                || {
                    let bench_container = BenchContainer::new(0);
                    bench_container.container
                },
                |container| {
                    black_box(bench_setup.run(container));
                },
            )
        });

    bench_setup.print_stats();
}

fn bench_non_contend_transactions(c: &mut Criterion) {
    let capacity = TOTAL_BUFFERED_PACKETS;
    let mut bench_setup: BenchSetup<RuntimeTransaction<SanitizedTransaction>> = BenchSetup::new();

    c.benchmark_group("bench_non_contend_transactions")
        .sample_size(10)
        .bench_function("sdk_transaction_type", |bencher| {
            bencher.iter_with_setup(
                || {
                    let mut bench_container = BenchContainer::new(capacity);
                    bench_container
                        .fill_container(build_non_contend_transactions(capacity).into_iter());
                    bench_container.container
                },
                |container| {
                    black_box(bench_setup.run(container));
                },
            )
        });

    bench_setup.print_stats();
}

fn bench_fully_contend_transactions(c: &mut Criterion) {
    let capacity = 10000; //TOTAL_BUFFERED_PACKETS;
    let mut bench_setup: BenchSetup<RuntimeTransaction<SanitizedTransaction>> = BenchSetup::new();

    c.benchmark_group("bench_fully_contend_transactions")
        .sample_size(10)
        .bench_function("sdk_transaction_type", |bencher| {
            bencher.iter_with_setup(
                || {
                    let mut bench_container = BenchContainer::new(capacity);
                    bench_container
                        .fill_container(build_fully_contend_transactions(capacity).into_iter());
                    bench_container.container
                },
                |container| {
                    black_box(bench_setup.run(container));
                },
            )
        });

    bench_setup.print_stats();
}

criterion_group!(
    benches,
//    bench_empty_container,
//    bench_non_contend_transactions,
    bench_fully_contend_transactions,
);
criterion_main!(benches);
