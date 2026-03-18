use {
    agave_feature_set::FeatureSet,
    solana_compute_budget::compute_budget_limits::{
        ComputeBudgetLimits, MAX_COMPUTE_UNIT_LIMIT, MAX_HEAP_FRAME_BYTES,
        MAX_LOADED_ACCOUNTS_DATA_SIZE_BYTES, MIN_HEAP_FRAME_BYTES,
    },
    solana_compute_budget_instruction::compute_budget_instruction_details::ComputeBudgetInstructionDetails,
    solana_transaction_error::{TransactionError, TransactionResult as Result},
    std::num::NonZeroU32,
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
    LegacyAndV0(ComputeBudgetInstructionDetails),
    V1(TransactionConfigValues),
}

impl TransactionConfigValues {
    fn sanitize_and_convert_to_compute_budget_limits(&self) -> Result<ComputeBudgetLimits> {
        if self.compute_unit_limit > MAX_COMPUTE_UNIT_LIMIT {
            return Err(TransactionError::SanitizeFailure);
        }

        if self.loaded_accounts_data_size_limit > MAX_LOADED_ACCOUNTS_DATA_SIZE_BYTES.into() {
            return Err(TransactionError::SanitizeFailure);
        }

        if !(MIN_HEAP_FRAME_BYTES..=MAX_HEAP_FRAME_BYTES).contains(&self.requested_heap_size)
            || !self.requested_heap_size.is_multiple_of(1024)
        {
            return Err(TransactionError::SanitizeFailure);
        }

        Ok(ComputeBudgetLimits {
            updated_heap_bytes: self.requested_heap_size,
            compute_unit_limit: self.compute_unit_limit,
            compute_unit_price: self.priority_fee_lamports,
            loaded_accounts_bytes: NonZeroU32::new(self.loaded_accounts_data_size_limit)
                .ok_or(TransactionError::InvalidLoadedAccountsDataSizeLimit)?,
        })
    }
}

impl TransactionConfigSource {
    pub fn sanitize_and_convert_to_compute_budget_limits(
        &self,
        feature_set: &FeatureSet,
    ) -> Result<ComputeBudgetLimits> {
        match self {
            TransactionConfigSource::LegacyAndV0(details) => {
                details.sanitize_and_convert_to_compute_budget_limits(feature_set)
            }
            TransactionConfigSource::V1(config) => {
                config.sanitize_and_convert_to_compute_budget_limits()
            }
        }
    }
}
