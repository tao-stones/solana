use {
    criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput},
    crossbeam_channel::{unbounded, Receiver, Sender},
    jemallocator::Jemalloc,
    solana_core::banking_stage::{
        scheduler_messages::{ConsumeWork, FinishedConsumeWork, MaxAge},
        transaction_scheduler::{
            greedy_scheduler::{GreedyScheduler, GreedySchedulerConfig},
            prio_graph_scheduler::{PrioGraphScheduler, PrioGraphSchedulerConfig},
            scheduler::{PreLockFilterAction, Scheduler},
            transaction_state::TransactionState,
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
        instruction::Instruction,
        message::Message,
        pubkey::Pubkey,
        signature::Keypair,
        signer::Signer,
        system_instruction,
        transaction::{SanitizedTransaction, Transaction},
    },
    std::time::{Duration, Instant},
};

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

trait TestTxsBuilder {
    fn build(
        &self,
        count: usize,
        is_single_payer: bool,
    ) -> Vec<RuntimeTransaction<SanitizedTransaction>> {
        let mut transactions = Vec::with_capacity(count);

        let compute_unit_price = 1_000;
        let single_payer = Keypair::new();
        for _n in 0..count {
            let payer = if is_single_payer {
                &single_payer
            } else {
                &Keypair::new()
            };
            let mut ixs = self.prepare_instructions(payer);
            let prioritization =
                ComputeBudgetInstruction::set_compute_unit_price(compute_unit_price);
            ixs.push(prioritization);
            let message = Message::new(&ixs, Some(&payer.pubkey()));
            let tx = Transaction::new(&[payer], message, Hash::default());
            let transaction = RuntimeTransaction::from_transaction_for_tests(tx);

            transactions.push(transaction);
        }

        transactions
    }

    fn prepare_instructions(&self, payer: &Keypair) -> Vec<Instruction>;
}

struct SimpleTransferTxBuilder;
impl TestTxsBuilder for SimpleTransferTxBuilder {
    fn prepare_instructions(&self, payer: &Keypair) -> Vec<Instruction> {
        let to_pubkey = Pubkey::new_unique();
        vec![system_instruction::transfer(&payer.pubkey(), &to_pubkey, 1)]
    }
}

struct ManyTransfersTxBuilder;
impl TestTxsBuilder for ManyTransfersTxBuilder {
    fn prepare_instructions(&self, payer: &Keypair) -> Vec<Instruction> {
        const MAX_TRANSFERS_PER_TX: usize = 35;
        let to = Vec::from_iter(
            std::iter::repeat_with(|| (Pubkey::new_unique(), 1)).take(MAX_TRANSFERS_PER_TX),
        );
        system_instruction::transfer_many(&payer.pubkey(), &to)
    }
}

struct PackedNoopsTxBuilder;
impl TestTxsBuilder for PackedNoopsTxBuilder {
    fn prepare_instructions(&self, _payer: &Keypair) -> Vec<Instruction> {
        // Creating noop instructions to maximize the number of instructions per
        // transaction. We can fit up to 355 noops.
        const MAX_INSTRUCTIONS_PER_TRANSACTION: usize = 355;
        let program_id = Pubkey::new_unique();
        (0..MAX_INSTRUCTIONS_PER_TRANSACTION)
            .map(|_| Instruction::new_with_bytes(program_id, &[], vec![]))
            .collect()
    }
}

fn fill_container<Tx: TransactionWithMeta>(
    transactions: impl Iterator<Item = Tx>,
) -> TransactionStateContainer<Tx> {
    let mut container = TransactionStateContainer::with_capacity(TOTAL_BUFFERED_PACKETS);
    for transaction in transactions {
        let compute_unit_price =
            transaction
                .compute_budget_instruction_details()
                .sanitize_and_convert_to_compute_budget_limits(
                    &agave_feature_set::FeatureSet::default(),
                )
                .unwrap()
                .compute_unit_price;

        // NOTE - setting transaction cost to be `0` for now, so it doesn't bother block_limits
        // when scheduling.
        const TEST_TRANSACTION_COST: u64 = 0;
        if container.insert_new_transaction(
            transaction,
            MaxAge::MAX,
            compute_unit_price,
            TEST_TRANSACTION_COST,
        ) {
            unreachable!("test is setup to fill the Container to fullness");
        }
    }
    container
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

    fn run(
        &self,
        scheduler: &mut impl Scheduler<Tx>,
        container: &mut TransactionStateContainer<Tx>,
    ) {
        // each bench measurement is to schedule everything in the container
        while !container.is_empty() {
            scheduler.receive_completed(container).unwrap();

            scheduler
                .schedule(container, self.filter_1, self.filter_2)
                .unwrap();
        }
    }
}

fn bench_prio_graph_scheduler(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_prio_graph_scheduler_with_sdk_transactions");
    group.sample_size(10);

    let txs_builders: Vec<(&str, Box<dyn TestTxsBuilder>)> = vec![
        ("simple_transfer", Box::new(SimpleTransferTxBuilder)),
        ("many_transfer", Box::new(ManyTransfersTxBuilder)),
        ("packed_noop", Box::new(PackedNoopsTxBuilder)),
    ];

    let conflict_conditions: Vec<(&str, bool)> =
        vec![("non-conflict", false), ("full-conflict", true)];

    let tx_counts: Vec<(&str, usize)> = vec![
        ("empty_container", 0),
        ("full_container", TOTAL_BUFFERED_PACKETS),
    ];

    for (txs_builder_type, txs_builder) in &txs_builders {
        for (conflict_condition_type, conflict_condition) in &conflict_conditions {
            for (tx_count_type, tx_count) in &tx_counts {
                let bench_name =
                    format!("{txs_builder_type}/{conflict_condition_type}/{tx_count_type}");
                group.throughput(Throughput::Elements(*tx_count as u64));
                group.bench_function(&bench_name, |bencher| {
                    bencher.iter_custom(|iters| {
                        let mut execute_time: Duration = std::time::Duration::ZERO;
                        for _i in 0..iters {
                            // setup new Scheduler and Container for each iteration of execution
                            let bench_env: BenchEnv<RuntimeTransaction<SanitizedTransaction>> =
                                BenchEnv::new();
                            let mut container = fill_container(
                                txs_builder
                                    .build(*tx_count, *conflict_condition)
                                    .into_iter(),
                            );
                            let mut scheduler = PrioGraphScheduler::new(
                                bench_env.consume_work_senders.clone(),
                                bench_env.finished_consume_work_receiver.clone(),
                                PrioGraphSchedulerConfig::default(),
                            );

                            // execute with custom timing
                            let start = Instant::now();
                            {
                                bench_env.run(black_box(&mut scheduler), black_box(&mut container));
                            }
                            execute_time = execute_time.saturating_add(start.elapsed());
                        }
                        execute_time
                    })
                });
            }
        }
    }
}

fn bench_greedy_scheuler(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_greedy_scheduler_with_sdk_transactions");
    group.sample_size(10);

    let txs_builders: Vec<(&str, Box<dyn TestTxsBuilder>)> = vec![
        ("simple_transfer", Box::new(SimpleTransferTxBuilder)),
        ("many_transfer", Box::new(ManyTransfersTxBuilder)),
        ("packed_noop", Box::new(PackedNoopsTxBuilder)),
    ];

    let conflict_conditions: Vec<(&str, bool)> =
        vec![("non-conflict", false), ("full-conflict", true)];

    let tx_counts: Vec<(&str, usize)> = vec![
        ("empty_container", 0),
        ("full_container", TOTAL_BUFFERED_PACKETS),
    ];

    for (txs_builder_type, txs_builder) in &txs_builders {
        for (conflict_condition_type, conflict_condition) in &conflict_conditions {
            for (tx_count_type, tx_count) in &tx_counts {
                let bench_name =
                    format!("{txs_builder_type}/{conflict_condition_type}/{tx_count_type}");
                group.throughput(Throughput::Elements(*tx_count as u64));
                group.bench_function(&bench_name, |bencher| {
                    bencher.iter_custom(|iters| {
                        let mut execute_time: Duration = std::time::Duration::ZERO;
                        for _i in 0..iters {
                            // setup new Scheduler and Container for each iteration of execution
                            let bench_env: BenchEnv<RuntimeTransaction<SanitizedTransaction>> =
                                BenchEnv::new();
                            let mut container = fill_container(
                                txs_builder
                                    .build(*tx_count, *conflict_condition)
                                    .into_iter(),
                            );
                            let mut scheduler = GreedyScheduler::new(
                                bench_env.consume_work_senders.clone(),
                                bench_env.finished_consume_work_receiver.clone(),
                                GreedySchedulerConfig::default(),
                            );

                            // execute with custom timing
                            let start = Instant::now();
                            {
                                bench_env.run(black_box(&mut scheduler), black_box(&mut container));
                            }
                            execute_time = execute_time.saturating_add(start.elapsed());
                        }
                        execute_time
                    })
                });
            }
        }
    }
}

criterion_group!(benches, bench_prio_graph_scheduler, bench_greedy_scheuler,);
criterion_main!(benches);
