use {
    solana_compute_budget_instruction::compute_budget_instruction_details::ComputeBudgetInstructionDetails,
};

#[derive(Debug)]
#[cfg_attr(feature = "dev-context-only-utils", derive(Clone))]
pub struct TransactionConfigValues {
    pub priority_fee_lamports: u64,
    pub compute_unit_limit: u32,
    pub loaded_accounts_data_size_limit: u32,
    pub requested_heap_size: u32,
}

#[derive(Debug)]
#[cfg_attr(feature = "dev-context-only-utils", derive(Clone))]
pub enum TransactionConfigSource {
    LegacyAndV0( ComputeBudgetInstructionDetails ),
    V1( TransactionConfigValues ),
}
