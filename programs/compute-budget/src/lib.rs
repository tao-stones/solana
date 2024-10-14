use solana_program_runtime::declare_process_instruction;

// compute budget instructions are processed before VM, therefore do not have execution cost
pub const DEFAULT_COMPUTE_UNITS: u64 = 0;

declare_process_instruction!(Entrypoint, DEFAULT_COMPUTE_UNITS, |_invoke_context| {
    // Do nothing, compute budget instructions handled by the runtime
    Ok(())
});
